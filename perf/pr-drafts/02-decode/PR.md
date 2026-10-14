# PR 2: megakernel: faster decode kernels (pipelined q8 attention, T-invariant attention partition, Q6_K / alpha-beta layouts, spill fixes)

Status of this text: draft for L-Forster/open-jet, written against origin/master 4d4db90 (two engines: `qwen38_mk.cu` =
plain Q4_K_M, `qwen38_mixed.cu` = mixed-quant). The numbers were measured on our single-file tree (fabce52 history); the
port onto the two engines (PORT.md) has to be re-gated before posting. This PR has no dependency on PR 1; PR 1 depends on
this one (its wide verify attention is built on `attn_core_q8`).

## Summary

Decode with the q8 KV cache gets a pipelined attention kernel that streams the cache at 650-750 GB/s instead of about
290 GB/s at 100K context, with bit-identical output: +18.7% decode at 101.6K depth and +7.3% at 38K on an RTX 3090. The
split-K attention partition now depends on the query position only, so greedy output no longer changes with the draft depth
(at `-d 0..4` it did). Several smaller exact changes cut local-memory spills and bytes: the decode attention prologue stops
spilling (mk_cycle's stack frame 2,848 -> 992 B, 223 MiB less reserved local memory), the gate / up step uses two rows at
every T, ffn_down at T = 5 splits its K halves out of line, Q6_K blocks shrink from 224 to 210 bytes, and the UD
`ssm_alpha` / `ssm_beta` matrices stay int8 (-66 MiB). None of it changes output (all checks below are byte-identical ids
and KLD), apart from the partition change, which moves 4 top-1 decisions in the KLD run (166 vs 162 mismatches; within noise).

## What changed

### Pipelined decode attention for the q8 KV cache (ad79468, 8316b32, 7996cfa)
- `attn_core_q8<T>`: the old `attn_core_t` ran five dependent stages per 32-position tile (q8 -> fp16 with I2F, S MMA chain,
  softmax with two rows per warp in sequence, P.V) with five block barriers: 4.5 us per tile against a 1.2 us tensor floor,
  about 290 GB/s of KV. The new kernel keeps the per-row arithmetic and restructures the schedule: a three-deep software
  pipeline with one barrier per tile (interval j: S(j) on warps 0-3, softmax of tile j - 1 on warps 12-15 with a quad per row,
  O += P(j - 2) V(j - 2) on warps 4-11), separate `cp.async` rings for the raw q8 K (3 stages) and V (4 stages) tiles, no fp16
  K / V tiles (both MMA B operands are converted from int8 in registers with `q8_to_f16`'s rounding), P.V B fragments that let
  one 32-bit q8 load per position feed four n8 tiles. The softmax keeps `attn_core_t`'s float operations and butterfly sum
  order, so the partials are bit-identical.
- 8316b32: the next tiles' copies are issued from warps 4-15 only (the S warps carry the longest MMA chains): 5-6% shorter
  intervals at T = 3-5. Adds `bench/dattn_bench.cu` (verbatim copies of both kernels on a random q8 cache: bitwise comparison
  of the split-K partials, T-invariance check, time per call).
- 7996cfa `MK_DA_ROUTE` (compile-time, default 1): chunks of a single tile (context <= `ATT_BUCKET` = 640) run `attn_core_t`,
  because the pipeline fill costs more than it saves there (21.5 vs 17.8 us per call at 35-290 positions). One call site per
  kernel (`attn_core_t` is force-inlined; a second site grew mk_cycle's frame by 112 B).
- Runtime switch `MK_DATTN=0`: the previous kernel everywhere (same output).
- Shared memory of the kernel: 92,672 B. Only for the all-q8 cache; the fp16 cache keeps `attn_core_t`.

### T-invariant attention partition (535e80c; moved here from PR 1)
`attn_dims(L)`: chunk size `32 * ceil(L / 640)` positions (whole K/V tiles), `nch = ceil(L / cs) <= 20`, so the chunk
boundaries below a query position depend on that position only (positions past a row's limit are exact zeros in the online
softmax and the combine). `cs` changes only when L crosses a multiple of 640 (`ATT_BUCKET`); the host caps T (`att_cap_T`) so
one launch never straddles such a boundary (step, prefill mk loop, score, score_logits); cost: once per 640 positions one
cycle verifies fewer drafts. Tiles per item: unchanged for L > 1280, 1 tile (was 2) for L <= 640. Before this, greedy ids at
`-d 1..4` differed from `-d 0` (bench prompt 1 diverged at token 90 / 319 / 308 for d1+d3 / d2 / d4). It is also what
`attn_core_q8` is built around (whole tiles per chunk) and what makes PR 1's variable-depth gate safe for greedy output.

### Spill and frame-size fixes (370b529, a8d7c0a, 2b8605c)
- 370b529: `attn_core_t`'s prologue carried `rc[]` (RoPE, assigned for lanes < 8 only) and `qv[]` (rows < R only) and a pointer
  into `qv` across the item loop in local memory. Unconditional loads (rows >= R clamp to R-1, every lane loads a valid RoPE
  row) and a copy instead of the pointer: `attn_core<T>` spills 768 B -> 0, mk_cycle's frame 2,848 -> 992 B. The driver
  reserves local memory for every thread slot (frame x 1,536 x 82 SMs), so the engine now also sets the CUDA stack-size limit
  to 0 (`MK_STACK_LIMIT=<bytes>` overrides; no kernel here uses a dynamic stack): reservation 342 -> 119 MiB. (The commit
  also contains MK_MEMLOG / MK_KV_EMU_SPEC hunks that belong to PR 6 / PR 4; they are not part of this PR.)
- a8d7c0a (verify track's 50e4de4 without its probe knobs): the K-halves of the T = 5 ffn_down split fill run in
  `__noinline__ half_rows`; `resid32<5,17,*>` hot-loop spills of 136-188 B -> 0.
- 2b8605c: the gate / up step uses two rows per matrix at every T (the 4-row step at T <= 2 spilled 44-160 B in `rows16` /
  `ffn_gu_any` for Q5_K / Q6_K gate / up and was slower even for Q4_K pairs), and ffn_down's whole-K loop runs out of line at
  T = 2..4 (`MK_FD_NOINL`, compile-time bit mask, default 0x1c = T 2..4; T = 1 and the K = 5120 / 6144 roles got slower out of
  line and stay inline).
- 58ce261: the L2 prefetch of the first weight rows before the activation fill (`pf_rows`) is removed from resid32, ffn_gu,
  dn_proj, attn_proj and lm_head (measured null by an independent probe; the next norm weight prefetch stays).

### Smaller weights (a001f2e, 1d63ab7)
- a001f2e: Q6_K in two streams per row: 192 B of values + an 18 B per block meta stream (pairs: sc even | sc odd | d even |
  d odd) = 210 B per block like GGUF instead of 224. `prefill_gemm.cuh`'s Q6_K reader follows. Bit-identical.
- 1d63ab7: UD's Q8_0 `ssm_alpha` / `ssm_beta` stay int8 rows plus fp16 scales; the kernels form `(q * d) * attn_norm`, which is
  the folded fp32 value the loader used to build. `MK_AB_F32=1` restores the fp32 rows. F32 files (plain Q4_K_M) take the old
  path unchanged. `prefill.cuh` (`pk_ab`) and the decode `dn_proj` read it.

## Results

RTX 3090, Qwen3.8-27B UD-Q4_K_XL unless noted; A-B-A = arms interleaved in one `gpu_lock.sh --timing` hold.

Pipelined q8 attention (8316b32 state, build lca_r2 vs 5f69600) [notes/agent_lcattn.md "Final results"]:

| | before | after | change |
|---|---|---|---|
| attn_core, us per call at 101.6K (`MK_PROFILE`) | 739.6 / 740.8 (299 GB/s) | 338.0 (654 GB/s) | -54% |
| mtp_attn_core, us per call at 101.6K | 572.6 / 571.7 (386 GB/s) | 296.0 (747 GB/s) | -48% |
| in-kernel ms per launch at 101.6K | 45.41 / 45.41 | 38.13 | -7.28 ms |
| attn_core, us per call at 38K | 289.5 / 293.9 (284 GB/s) | 147.2 (562 GB/s) | -50% |
| in-kernel ms per launch at 38K | 36.35 / 36.50 | 33.87 | -2.56 ms |
| decode tok/s at 101.6K (ctx 120K, `-d 4`, 256 tokens) | 104.25 / 103.67 | 123.40 | +18.7% |
| decode tok/s at 38K (ctx 40K) | 114.21 / 113.95 | 122.38 | +7.3% |
| `bench_mk`, fp16 KV, 8K ctx | 109.63 / 109.71 | 109.49 | no measurable change |

Microbenchmark (`bench/dattn_bench.cu`, q8 cache, 101.6K depth, us per call, old -> new) [commit 8316b32]: T=1 553 -> 310,
T=2 545 -> 298, T=3 748 -> 333, T=4 763 -> 340, T=5 769 -> 339 (650-740 GB/s); at 38K: T=1 243 -> 151, T=3 307 -> 155,
T=5 317 -> 163. Bit-identical in 18 configurations (n = 0 .. 101,600).

Short context, q8 cache (ctx 120000, bench prompts 0-3, 512 tokens, `-d 4`): the first version was -0.45% (about half in
`attn_core` on one-tile chunks, half a ptxas side effect: roughly 30 unrelated functions re-allocate registers, not avoidable
by call structure). 7996cfa routes one-tile chunks to `attn_core_t`: decode +0.5% (17.854 / 17.812 s -> 17.743 s summed over
the 4 prompts), in-kernel 31.31 -> 31.20 ms per launch; 38K / 101.6K unchanged [notes/agent_dattn2.md, commit 7996cfa].
Not measured: Q4_K_M with the q8 cache at long context (only exactness was checked), other GPUs.

T-invariant partition (535e80c) [notes/agent_spec.md "Step 1"]: phase-level, prompt 0, `-n 256`, one hold: mean ms per launch
31.62 -> 31.78 at `-d 4` and 25.66 -> 25.72 at `-d 0` (+0.2%; attention total -0.01 ms: more, smaller chunks make the core faster
and the combine slightly slower). `bench_mk` A-B-A-B 105.99 / 106.29 -> 105.14 / 105.65 tok/s is not a speed comparison (the old
build's `-d 4` text differs from its `-d 0` text, so acceptance differs per prompt).

Spill and prefetch changes:
- 370b529: local-memory reservation 342 -> 119 MiB (-223 MiB; 104 MiB with the later 864 B frame) [notes/agent_vram.md "FINAL
  RESULTS"]; the VRAM track as a whole (about 20 items) was -1,770 MiB at 120K ctx and +1.4% decode; this item's own speed
  effect is "in b3", i.e. not separated.
- a8d7c0a (measured on its predecessor 50e4de4, same code without probe knobs) [notes/agent_verify.md "Results table"]: UD
  `bench.py -d 4` A-B-A 105.92 tok/s, +1.1% vs the mean of the A / A' base runs, ffn_down T=5 109.8 -> 95.7 us (-0.9 ms per T=5 cycle);
  plain Q4_K_M +3.0% (115.42 -> 118.91 tok/s, accepted counts identical).
- 2b8605c [commit message]: UD ffn_gu T=1 10.17 -> 9.03 ms, T=2 10.87 -> 9.39 ms (per pass); `-d 0` bench A-B-A 39.44 / 41.24 /
  39.44 tok/s (+4.6%); ffn_down per pass T=2 5.55 -> 5.32, T=3 5.84 -> 5.59, T=4 5.98 -> 5.81 ms; `-d 4` greedy unchanged
  (110.98 / 110.93 / 110.88). Q4_K_M: ffn_gu T=1 8.26 -> 8.04, T=2 8.59 -> 8.23 ms; `-d 0` +1.5%, fixed `-d 1` +2.2%.
- 58ce261 [commit message]: greedy bench A-B-A 111.65 / 111.85 / 111.75 (no change); `MK_PROFILE` `-d 4` launch mean 31.224 / 31.065 /
  31.243 ms (-0.54%; part of it code layout).
- Five-commit MTP-cut series (17e775f, f58e21e, 2b8605c, c0a1783, 58ce261; the first, second and fourth are in PR 1)
  vs 134ba4c [notes/agent_mtpcut.md "Final results"]: greedy bench `-d 4` 109.41 / 109.43 -> 111.77 (+2.15%), `-d 0` +5.1%,
  Q4_K_M +0.95%.

Weights:
- a001f2e: -182 MiB on UD (commit message, includes the Q3_K / IQ3_S re-encodes at that time); the VRAM table lists -201 MiB at
  120K ctx; decode +1.5% for Q6_K two streams + alpha/beta together (A-B-A 107.53 / 109.08 / 107.44) [notes/agent_vram.md]. Adds
  24-96 B of spills in `resid32<T,*,Q6_K>` (still faster overall).
- 1d63ab7: -66 MiB, about 70 MB less read per cycle; bit-identical [commit message; notes/agent_vram.md].

Not measured: any of this on Blackwell, Ada or Turing.

## Correctness

Reference: llama.cpp logits (`~/kld/ud.ref` "mixed", `ud32k.ref` 32k set; `q4km.ref`); "byte-identical" = the KLD result lines
are identical, not just within noise.
- attn_core_q8 (ad79468, 8316b32): greedy ids identical at `-d 0` / `-d 4` (bench prompts 0-3, 512 tokens, and a 38K prompt on
  q8), vs the previous build; KLD mixed and 32k KLD byte-identical (UD); Q4_K_M KLD mixed byte-identical; `MK_DATTN=0` ids
  identical; split-K partials bitwise-equal to `attn_core_t` in `dattn_bench` (18 configs).
- 7996cfa: ids d0 == d4 == before (q8), 38K ids identical, KLD mixed and 32k byte-identical, ptxas identical to the previous build.
- 535e80c: greedy ids identical at `-d 0..4` on 4 bench prompts (`-n 512`) and prompts 1 / 3 at `-n 1600` (crossing 640 / 1280);
  KLD mixed 98.752% top-1 (166 mismatches) / 0.00114 vs base 98.782% (162) / 0.00115 (the 4-position difference is the
  partition change; one-time near-tie changes).
- 370b529, a001f2e, 1d63ab7: KLD mixed and prefill byte-identical, greedy ids identical (UD and Q4_K_M) [notes/agent_vram.md].
- 2b8605c, 58ce261, a8d7c0a: "exact" - per-row accumulation order independent of the rows per step; ids and accepted counts at
  `-d 0` / `-d 4` identical, KLD mixed byte-identical.

## Commits

```
535e80c megakernel: T-invariant attention split-K partition (greedy output independent of draft depth)       [moved in from PR 1]
ad79468 megakernel: attn_core_q8, a pipelined decode attention for the all-q8 KV cache (bit-identical, ~2x faster at long context)
8316b32 megakernel: attn_core_q8 issues the next tiles' copies from warps 4-15 only; bench/dattn_bench.cu
d78252e megakernel README: attn_core_q8 (pipelined q8 decode attention, MK_DATTN=0 for the previous kernel), ...
7996cfa megakernel: decode attention chunks of one K/V tile run attn_core_t (MK_DA_ROUTE)
370b529 megakernel: decode attention prologue without conditionally assigned / pointer-aliased arrays ... and stack-size limit 0
a001f2e megakernel: Q6_K in two streams per row, 192 B of values + an 18 B/block meta stream ...
1d63ab7 megakernel: Q8_0 ssm_alpha / ssm_beta (UD) kept as int8 rows + fp16 scales ...
2b8605c megakernel: 2-row gate / up step at every T; ffn_down rows out of line at T = 2..4
a8d7c0a megakernel: T=5 ffn_down split-fill K-halves in __noinline__ half_rows ...
58ce261 megakernel: drop the pre-fill L2 prefetch of the first weight rows (pf_rows)
```

Moved: 535e80c in from PR 1 (the whole-tile partition and `ATT_BUCKET` are prerequisites of `attn_core_q8` / `MK_DA_ROUTE`, and
PR 1's wide attention needs `attn_core_q8`; this removes a circular dependency). f58e21e (sampled lm_head epilogues) out to
PR 1: it rewrites code that exists only after PR 1's gate / sampled drafts (the staged draft q rows and LSE sums), so
in PR 2 it would depend on PR 1. 370b529 carries unrelated MEMLOG / KV_EMU hunks (see above): split them when porting.
d78252e (README) is superseded by README.md in this directory.

## Notes for the reviewer

- Default on: everything here (it is all exact apart from the partition's one-time near-tie changes). Run-time switches:
  `MK_DATTN=0` (old attention kernel), `MK_STACK_LIMIT=<bytes>`, `MK_AB_F32=1` (fp32 alpha/beta rows). Compile-time:
  `MK_DA_ROUTE` (default 1), `MK_FD_NOINL` (default 0x1c).
- `attn_core_q8` needs 92,672 B of shared memory and `cp.async` rings; the q8 cache only. Turing's 64 KiB path keeps
  `attn_core_t` (see PORT.md). The kernel handles T <= 5 (R = 6T <= 32 rows); PR 1 adds the 48-row form for T up to 8.
- At very short contexts (<= 640 positions) the routing keeps the old kernel, so the attention gain only shows at longer
  context (measured at 38K and 101.6K).
- Q6_K two-stream layout adds a few spill bytes to the Q6_K residual GEMV; net speed still positive in our A-B-A.
- Known open items from the notes: `resid32<2,17,Q5_K>` still spills 96-128 B (T = 2 is rare: under 0.5% of sampled cycles with
  the gate), transposed P.V (exact, est. under 1%), int8 Q.K for decode (-10% kernel time at T >= 3 but not exact: not ported),
  Q8_0 and IQ3_S FM layouts (PR 1's area).
- The decode gains were measured with the q8 KV cache (120K-context server config); the fp16 cache keeps `attn_core_t`.
