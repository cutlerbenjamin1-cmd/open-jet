# PORT.md: PR 3 (prefill) onto origin/master 4d4db90

Layout upstream (verified with `git show origin/master:...`): `qwen38_mk.cu` (3163 lines, plain Q4_K_M) and `qwen38_mixed.cu`
(3436 lines, mixed quants) each `#include "prefill.cuh"` (line 2122 / 2244); `prefill.cuh` pulls in `prefill_gemm_mixed.cuh`
when building the mixed engine and `prefill_gemm.cuh` otherwise (lines 8-10). There is NO `prefill_attn.cuh` upstream: pk_attn,
`pk_attn_prep`, `pk_attn_out` live in `prefill.cuh` (pk_attn at line 507, 192 threads, `AT_K = QWEN_MK_SM75 ? 16 : 32`).
`prefill_gemm.cuh` (436 lines) has `BN = QWEN_MK_SM75 ? 64 : 128`; `prefill_gemm_mixed.cuh` (451 lines) has `BN = 128` and the
mixed-quant types (TIQ4 etc.). The host side of both engines has its own copy of `Impl::mm`, the `mmh` launch switch,
`pf_prof`, `pf_init` (scratch allocation: `hp.xh = dalloc<__half>(cap * FF)`).

Our tree has one 8,499-line `qwen38_mk.cu` + `prefill_attn.cuh` + one `prefill_gemm.cuh`. Every host-side hunk of ours therefore
lands twice (once per engine) unless it is Q4_K_M-only or mixed-only.

Dependencies on other PRs (our tree order matters; these commits were written on top of them):

- bd8a22e / 167b509 (item 5) use `LAY` (wide-verify FM weight layout), `FMT`, `FM6`, `MK_WV_Q6F`, `wv.cuh` types from the PR 1 / 2
  wide-verify commits (38521b7, 5e653f7, 15c8003, ...). The pipelined fragment code is written as `HalfW<TYPE, LAY, ...>`;
  the `LAY = 0` path is the plain layout and is what upstream has. Port the `LAY = 0` half first; `LAY = 1` comes with PR 1.
- bd8a22e's y1 shrink is conditioned on `MK_EXACT` (PR 5). Without PR 5, port it as an unconditional no-op or leave it out.
- 0e19805 and 568e86b touch `Impl::mm` / `pf_prof` areas that PR 5 / 6 also edit (checkpointing, `mk_prefill_ck`).

## Per change

### 1. pk_attn2 (fbae2d3, 6b320e4) and 2. pk_attn8 (60af318, da119d6)
- Lands in: a new `megakernel/prefill_attn.cuh` (copy as is, 572 + 433 lines before cleanup) included from `prefill.cuh`,
  and the dispatch in the host prefill loop (`pk_attn<true>/<false>` call sites: qwen38_mk.cu ~2924, qwen38_mixed.cu ~3197).
  The kernels themselves are engine-independent (they read `cB`: kc / vc / ksc / vsc, rope/hd constants, `kvq8`).
  Both engines: yes (identical).
- Collisions: upstream's `prefill.cuh` pk_attn (line 507-672) is the code fbae2d3 deleted (-171 lines) and moved. Upstream
  edited that file only for sm75 (45031ed): `AT_K` 16 on Turing, `AT_LD`, `AT_SMEM <= 65536` static_assert, and `cuda_compat.cuh`
  routing of `mma16816`. The port keeps pk_attn in place for sm75 builds (`#if QWEN_MK_SM75`) and uses pk_attn2 / pk_attn8 only
  for sm_80+. `pk_attn_prep` / `pk_attn_out` stay (our pk_attn_out later grows a rotated-V un-rotate for PR 4: keep the
  function signatures stable). Host: `MKEngine::Impl` env parsing for `MK_ATTN`, `MK_ATTN_SPLIT`, `MK_ATTN_PV16`,
  `MK_ATTN_S8`, `a2_plan` (tail split: partial buffers reuse `y1`; its split count depends on `cap * FF` of `y1`, which is why
  non-exact mode must keep `y1` at FF) and `pk_attn2_comb`. `cudaFuncSetAttribute` for the new kernels (99,840 B smem).
- sm75: pk_attn8 uses `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` and cp.async 3-stage rings: neither exists on Turing
  (`cuda_compat.cuh` only provides m16n8k16 fp16 -> split to k8, and synchronous copies). pk_attn2 needs 93 KB smem and 12 warps
  (384 threads): above Turing's 64 KB. So Turing keeps upstream's pk_attn; guard with `QWEN_MK_SM75`. Not tested.
- Effort / risk: M (mechanical move) + S (dispatch). Risk: low for correctness (byte-compare to old kernel in exact mode is a
  built-in gate); medium for the register/spill budget (168 regs; compile the whole TU and diff ptxas output against upstream).
- Gates for the later compiling port: exact mode (`MK_ATTN_PV16=0 MK_ATTN_SPLIT=0`) prefill KLD byte-identical to upstream's
  pk_attn; KLD mixed/32k vs `ud.ref`/`ud32k.ref`/`q4km.ref`; `bench/attn_bench.cu`; compute-sanitizer on it.

### 3. pk_glu coalesced (568e86b)
- Lands in: qwen38_mixed.cu only (`pk_glu` at line 2248; Q4_K_M never runs pk_glu: all its gate / up pairs are same-type and
  fused). `bench/mmq_bench.cu` hunk: add `iq4_lut` define (upstream mmq_bench.cu may already compile; check).
- Collision: none known (upstream took pk_glu from our #19 unchanged). If item 5 is ported, pk_glu only runs with
  `MK_PF_GLU_MIX=0`; still worth taking (S) because it is exact and one hunk.
- sm75: engine is sm_80+ only. Effort S, risk low.

### 4. Narrow token tiles (0e19805)
- Lands in: both engines' prefill GEMM headers (`prefill_gemm.cuh` and `prefill_gemm_mixed.cuh`: `mmh<TYPE, GLU, NB>` with `HWT`,
  `HNI`, `XIT`, `HM_X`, epilogue loops and `hm_smem` derived from NB instead of the global `BN`) and the host (`Impl::pf_nb`,
  every `mmh` launch site with `grid(m.N/BM, ceil(ntok/NB))`, all `cudaFuncSetAttribute` lines, `MK_PF_BN`).
- Collision: upstream sm75 defines `BN = QWEN_MK_SM75 ? 64 : 128` and sizes the epilogue static_assert with BN; NB must default to
  that BN and `pf_nb` must be disabled (or restricted to NB <= 64) on Turing, whose 64 KB smem and sync copies change the cost
  model (the constants 0.80 / 0.95 / 1.44 and "NB 32 fits two CTAs/SM" are sm_86 measurements). The mixed header has a separate
  copy of `mmh` (+15 lines vs plain): do the two edits as one diff generated by script.
- Effort: M (two copies; 27 new instantiations; the ptxas build of the mixed engine is ~11 GiB of RAM per upstream README, more
  instantiations raise it: ours peaked at 21.7-23.1 GB for the whole tree). Risk: low (byte-identical outputs across NB are
  checked by `notes/ttft/gemm_nb.cu`, a ready-made unit test to bring along).

### 5. Pipelined fragments (S4), mixed-type fused gate / up (V2), Q5_K super-block buffer (bd8a22e, 167b509)
- Lands in: `prefill_gemm.cuh` (S4 + Q5_K buffer: Q4_K_M uses Q4_K, Q5_K, Q6_K, Q8_0) and `prefill_gemm_mixed.cuh` (S4 + Q5_K +
  IQ4_XS + V2). V2 and `MK_GLU_MIX_PAIRS` are mixed-engine only. Host: `glu_mix`, `glu_mix_attr`, `MK_PF_GLU_MIX`, profile
  category `gemm_ffn_gum`, `pf_init` (`all_fused` -> y1 shrink), both engines for the S4 launches.
- Collision: the diff against our base is 501 + 132 lines of `prefill_gemm.cuh` and it rewrites `mmh` wholesale (per-type code
  into `HalfW`), so it conflicts with any upstream change to `mmh`/`HT<TYPE>`/`WT<TYPE>` in either header; upstream changes to
  these since 2993f47: only the mixed copy (210eda0) and sm75 (45031ed `BN`, sync copies via `cuda_compat.cuh`). The mixed
  header already contains our #19 UD types, so the rewrite should transplant cleanly there; the plain header needs Turing review.
  Upstream's sm75 path uses synchronous copies in place of cp.async: S4's pipelining assumes cp.async stage groups and the Q5_K
  double buffer uses a commit group per super-block; on Turing take upstream's code and skip S4 (guard `QWEN_MK_SM75`).
- Needs from other PRs: `LAY` / `FMT` / `FM6` (PR 1 / 2) as discussed above; V2's "FM Q6_K / FM Q6_K and row-major Q3_K sides"
  assume them. Port with `LAY = 0` only first; the 11 UD pairs then still fuse (check which of them need `LAY = 1`: the notes
  list FM Q6_K sides, so the plain-layout version may fuse fewer pairs).
- Effort: L (largest rewrite in this PR; 500 lines x two headers; needs GPU gates: microbench `notes/pf/gb/gb.cu` byte compare, KLD
  prefill byte-identical, tinv 12/12). Risk: medium (register pressure: the notes say 0 changed spills across 638 functions, but
  that was with our tree's other kernels; the notes list mmh<IQ4_XS,GLU> at 200 registers after an older note had it at 255: re-check the new fused mixed kernels).

### 6. xh sizing (bbf4cd0)
- Lands in: both engines' `pf_init`/scratch block (qwen38_mk.cu:2781, qwen38_mixed.cu:3040): `hp.xh = dalloc<__half>(cap * 2 * D)`.
  Requires that every GEMM input has K <= 2 D (true for the MTP `eh` input; check upstream's mixed engine for any FF-wide input
  into `xh` — `xh2` is the FF-wide one). S, risk low.

## Order for the port
6 (S), 3 (S), then 1 + 2 (with the sm75 guard), then 4, then 5 (last; depends on layouts from PR 1).
Upstream Turing CI: only a model-less `bench/turing_check.cu`; extend it to build with the new headers under `-DQWEN_MK_SM75=1`
(syntax only) before merging.
