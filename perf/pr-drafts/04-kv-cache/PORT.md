# PORT.md: PR 4 (low-bit KV cache) onto origin/master 4d4db90

## Upstream today (verified with `git show origin/master:...`)

- KV cache = `kc` / `vc` (int8 or fp16) + `ksc` / `vsc` (fp16 scale per 32) in the uniform layout `[slot][kv head][ctx][HD]`,
  chosen by one bool `kvq8` (6 occurrences in each of `qwen38_mk.cu` and `qwen38_mixed.cu`, 1 in `prefill.cuh`). `kv_store_q8` /
  `kv_store_f16` at qwen38_mk.cu:1145 / 1166 and qwen38_mixed.cu:1269 / 1290; the row store at 1393 / 1519. Context:
  `MKEngine::Impl` allocates the cache once (our equivalent block is `Impl` ~line 6535 of our qwen38_mk.cu: `kv_spec`,
  `kvb`, `kvp`, rings). `--kv f16|q8|auto` in `main.cu`.
- Decode attention: `attn_core` / `attn_core_t` only (the simple loop). Upstream has NO pipelined q8 decode attention
  (`attn_core_q8_t`, PR 2). Max T = 5 (`constexpr int MAXT = 5`; `MK_MAXT=8` is PR 1).
- Prefill attention: `pk_attn` in `prefill.cuh` (192 threads); no `prefill_attn.cuh` (PR 3 adds it).
- Two engines, duplicated code: every KV hunk below lands in BOTH `qwen38_mk.cu` and `qwen38_mixed.cu`. The KV code does not
  depend on quant type, so the two copies can be produced by one script and diffed. The mixed engine is sm_80+ only; the plain
  engine also builds for sm_75 (45031ed ... 5397a05: `cuda_compat.cuh` with synchronous copies, split k8 MMAs, no L2 hints).

## Hard dependencies on other PRs (port order: 2 -> 3 -> 1 -> 4 -> 5)

- PR 2: the pipelined q8 decode attention (`attn_core_q8_t<T, MIX>`, cp.async rings, `aq_issue`, `AQ_NSKM`): 680d412 builds the
  MIX form on it; abe5556 / f64ff51 / ceb7d44 / 2ea89b6 / eb133a5 edit it. Without PR 2, low-bit slots could only run through
  `attn_core_t` (fp16 conversion in `kv_stage`, f138d36's form: 2.0-2.2x slower per call than the MIX kernel at 38k / 101.6k).
- PR 3: pk_attn2 / pk_attn8 in `prefill_attn.cuh` (f138d36, 251849d, 380d2dd, 9b05adb, 23905b4, 71ac710, 1548b89 edit them).
- PR 1: `MK_MAXT=8` / the wide pass (5245602 dispatches `attn_core_q8m` there; the host guard that refuses `n_draft > 4` with a
  low-bit side is removed). Without PR 1 keep that guard.
- PR 5: 3d84349's `EXA` template parameter (MK_EXACT). Recommend moving 3d84349 into PR 5 or porting it with the EXA parameter
  and its call sites removed (S).

## Per change

| change (commits) | lands in | both engines | collisions with upstream | effort / risk |
|---|---|---|---|---|
| `MK_KV` parse, per-slot `kvb[2][17]`, packed slots `kvp` / `kvs`, rotated flags `kvr` / `kvrv`, allocation + log lines (f138d36, 097aee7, 380d2dd) | `Impl` construction block + `HostBuf`/`cB` struct fields in each engine | yes | upstream `kvq8` bool stays as the "q8 family" flag; `kc`/`vc` byte counts change from `rows*HD` to packed sums; the auto-mode condition (`want_kv` forces q8) | M, low |
| `kv_store` for q4..q7 (`kv_store_qn`), rotated K / V (`fwht256`), `attn_combine` / `pk_attn_out` un-rotation (f138d36, 380d2dd) | the row-store site (qwen38_mk.cu ~1393, mixed ~1519) + `attn_combine` + prefill `pk_attn_prep` / `pk_attn_out` in `prefill.cuh` | yes | upstream edited `pk_attn_prep` (line 467) only for sm75 | M, medium (touches every K / V write; byte-compare all-q8 vs old build is the gate) |
| pipelined MIX decode attention (`attn_core_q8m`, `kv_rows_expand`, `KvFmt`, `kv_tile_src`) (680d412, abe5556, f64ff51, ceb7d44, 2ea89b6, eb133a5) | inside PR 2's `attn_core_q8_t`, both engines | yes | none beyond PR 2 | L (smem budget 99 KB, m / l / alpha moved into pad columns; spills: mk_cycle frame 816 -> 832 B in 680d412) , high: the kernel is in the persistent megakernel's phase code, register/smem pressure changes with any other PR |
| window ring (`MK_KV_WIN`, `kv_ws`, `kv_wsm`, `kv_src`, `k_ring_refill` -> `ring_rewind`, `MKEngine::truncate`) (9b05adb, f292074, 5245602 slack) | `Impl`, decode + prefill attention, `MKEngine::rewind/truncate` in capi/server paths | yes | upstream checkpoint/restore code: our `k_ring_refill` hooks the restore (`mk_prefill_ck`, PR 5/6 territory). Upstream has one checkpoint and `MKEngine::prefill` on non-empty sequences; check where its rewind lives before hooking | M, medium (server semantics) |
| pk_attn8 low-bit / rotated K (`<QROW, KLOW>`), C3a fix, pk_attn2 loaders `a2_ld_q8` / `a2_st_q8` (251849d, 23905b4, 71ac710, 1548b89, 2ea89b6) | `prefill_attn.cuh` (PR 3) + dispatch (host chooses `KLOW` instantiation per slot) | yes (kernel is shared) | none (new file) | M, medium: `<_, true>` was +17% over the q8 kernel and costs registers; keep q8 slots on the old instantiation (71ac710 measured 2-3% default-prefill loss otherwise) |
| 5245602 low-bit at T > 5 | wide pass dispatch + ring slack `kvrc` | yes | PR 1 | S |
| 91b0551 KV emulation + `--kld-specs` | `main.cu`, `qwen38_mk.h` (`set_kv_emu`), `kv_store_emu` in both engines | yes | upstream main.cu `--kld` already exists; `--kld-specs` extends it | M, low; optional: could be dropped from the PR (test tooling) |
| `MK_MTP_WIN` (380d2dd item 2) | `attn_dims_w`, `mtp_ws` | yes | negative result; recommend NOT porting | skip |
| 3d84349 K6 / V5 instantiation | `prefill_attn.cuh` + host dispatch | yes | needs PR 5's EXA | S after PR 5 |

## Turing (sm_75)
- Plain engine only. The low-bit formats are storage formats and the simple `attn_core_t` / `pk_attn` path would work on
  Turing (`kv_stage` with synchronous copies), but the fast paths do not: `attn_core_q8m` uses cp.async rings and 99 KB smem
  (Turing has 64 KB), and pk_attn8 uses `mma.m16n8k32` s8. So: on `QWEN_MK_SM75` either refuse `MK_KV` with a clear message
  (simplest, recommended for the first port) or port only the storage + `attn_core_t` fallback. Nothing here was run on Turing.
- The ring + 640-row bucket logic (`ATT_BUCKET`) is arithmetic only; it ports unchanged.

## Sequencing and gates for the compiling port
1. Port the format parser and storage (all-q8 default) first; gate: all-q8 greedy ids + KLD mixed byte-identical to upstream.
2. Add kernels per commit group; gates per group are in PR.md "Correctness" (C 32k KLD, rotated D 32k KLD, tinv d0 == d4 == d7).
3. The compile cost: the EMU build needed ~17-18 GB ptxas; the mixed engine's compile is already ~11 GiB per upstream README.
   Compile the KV changes for the plain engine first; do the mixed engine second.
4. Total effort for PR 4 after PRs 1-3 and 5 are in: L (about 1,600 added lines in two TUs). Biggest risks: (a) the decode kernel's
   smem/register budget interacts with whatever PR 1 / PR 2 settle on; (b) the ring refill semantics vs upstream's checkpoint
   restore; (c) double maintenance of the same hunks in two engines.
