# Title

megakernel: faster prefill (int8 Q.K^T attention, pipelined GEMM fragments, mixed-type fused gate / up, short-prompt tiles)

# Summary

Prompt processing is faster on the RTX 3090: a 101.6k-token prompt prefills in 83.98 s instead of 97.99 s on UD-Q4_K_XL
(+16.7%), and 38k tokens go from 1341.5 to 1474.6 tok/s (+9.9%); the plain Q4_K_M model gains +13.8% / +6.0% at the same
prompts. Most of it is the prefill attention kernel (a 12-warp restructure, then Q.K^T on int8 tensor cores straight from
the q8 KV cache); the rest is in the prefill GEMMs (software-pipelined dequantised fragments, a fused gate / up kernel for
mixed-type pairs, a Q5_K super-block buffer) and in short prompts (32 / 64-token GEMM tiles, so a chat turn of 40 new
tokens costs 103 ms instead of 144 ms). Everything is exact except two attention switches (fp16 P.V and the tail split,
and the int8 Q.K^T), which are on by default and measured KLD-neutral against llama.cpp; `MK_ATTN_S8=0`, or `MK_ATTN_PV16=0
MK_ATTN_SPLIT=0`, give back the previous numerics. No VRAM cost; two small VRAM savings (-14 MiB, -44 MiB in exact mode).

# What changed

## 1. Prefill attention: pk_attn2 (fbae2d3, 6b320e4)

New file `prefill_attn.cuh` (the kernels moved out of `prefill.cuh`). pk_attn keeps the CTA tile (16 query positions x one
kv head) but runs 12 warps: each q head gets a warp pair that splits S by key positions and P.V by head-dim halves, so the
O accumulator is 64 registers instead of 128 and each SM sub-partition has 3 warps. K / V tiles load during the MMAs; q8
tiles convert to fp16 with `0x6400` byte_perm + HSUB2 / HMUL2 (same rounding as I2F). Bit-identical to the old kernel in
exact mode.

- Tail split: the items of the last partial wave (256 CTAs on 82 SMs at cap 1024 = 3.12 waves) are split into tile ranges
  and merged by `pk_attn2_comb` (the partials live in `y1`). Not bit-exact; KLD-neutral.
- fp16 P.V: P.V accumulated in fp16 per 32-position tile, then in fp32. The largest |v| in the V cache over a 101K prompt is
  111, so a tile sum overflows fp16 only above 2047.
- Env vars: `MK_ATTN_SPLIT=0` (no tail split), `MK_ATTN_PV16=0` (fp32 P.V; with `MK_ATTN_SPLIT=0` the output is
  byte-identical to the old kernel), `MK_ATTN=1` (select the old kernel). Defaults: split on, PV16 on.
- `bench/attn_bench.cu`: standalone correctness / throughput check of pk_attn vs pk_attn2 on a random KV cache.

## 2. pk_attn8: int8 Q.K^T on the q8 cache (60af318, da119d6)

S = Q K^T on int8 tensor cores (`mma.m16n8k32` s8) straight from the cache's int8 K. Q is quantised per (row, 32-dim block)
in the prologue and kept in registers; each block's int32 partial is scaled by q scale x k scale (I2FP + FFMA). Raw q8 K / V
tiles stream through a 3-stage cp.async ring; the fp16 V tile is double-buffered (V(it + 1) converts during tile it), so
there is one block barrier per tile. Scores are in log2 units (log2 e folded into the Q scales) and P = exp2(s - m); the split
partials keep m in natural-log units for `pk_attn2_comb`. 168 registers, 0 spills, 99,840 B smem, 1 CTA / SM (384 threads).

- Env var: `MK_ATTN_S8` = 1 (default; q8 cache with PV16), 2 (one Q scale per row: ~5% faster kernel, +1.2% end to end at
  101.6k, 1.6x the Q quantisation noise in the synthetic test), 0 (pk_attn2). Exact mode (`MK_ATTN_PV16=0 MK_ATTN_SPLIT=0`)
  does not use it.
- `bench/attn_bench.cu` gains the S8 variants (`AB_VARS`, `TRUEREF=1` = error against an unquantised cache).
- Profile category: fused-GLU layers (`gemm_ffn_gu`) and two-GEMM gate / up layers (`gemm_ffn_gu2`) reported separately.

## 3. pk_glu coalesced (568e86b)

`pk_glu` (silu(gate) * up for the UD layers whose gate and up types differ) read 16 consecutive floats per thread with
scalar loads; with `-dlcm=cg` each warp load touched 32 L2 sectors (1.1-1.3 ms per 1024-token launch, 3.0 s of a 101.6k UD
prefill). Now one float4 of g and u per thread, two 4-byte stores; same per-element arithmetic. Also makes
`bench/mmq_bench.cu` compile again (`iq4_lut` define). This only matters while some gate / up pair is unfused (superseded for
the UD pairs by item 5, still used when `MK_PF_GLU_MIX=0`).

## 4. Narrow token tiles for short prompts (0e19805)

`pf::mmh` padded the token dimension to 128 columns: a 42-token prefill ran its GEMMs at 2.1 us per 64-K stage on 86 padding
columns, and the N = 5120 GEMMs (ffn_down, ssm_out, attn_output) used 40 of 82 SMs. `mmh<TYPE, GLU, NB>` now takes NB = 32 /
64 / 128 tokens per CTA (warp tile, X-tile copy, epilogue and smem follow NB; BM 128, 8 warps and the pipeline depth are
unchanged). `Impl::pf_nb` picks NB per GEMM from a wave cost model (lone-CTA stage time 0.80 / 0.95 / 1.44 for NB 32 / 64 /
128; NB 32 fits two CTAs per SM): 32 below 33 tokens, 32 / 64 up to 64 tokens, 64 / 128 up to 128, mostly 128 above; chunks
> 384 tokens always use 128 (the original kernel). `MK_PF_BN=32|64|128` forces one (tests / tuning). Bit-identical for every
NB (same fp16 products and sums per output element). Registers: NB 128 195-214 (unchanged), NB 64 119-134, NB 32 79-96, 0 spills.

## 5. Pipelined GEMM fragments and mixed-type fused gate / up (bd8a22e, 167b509)

- S4 software pipeline in `mmh`: the per-type code moves into `HalfW<TYPE, LAY, ROWS, ...>`; the next stage's scales and fp16
  A fragments are made between this stage's MMAs, each warp scales its own rows (`__syncwarp`, no CTA barrier), and the
  fp16 -> fp32 adds of a 128-k group are interleaved with the next group's first MMAs (the fp32 add sequence is unchanged).
  `frag_g()` is the per-kernel MMA group after which the fragments are made.
- V2: `mmh<TYPE, true, NB, LAY, TYPE2, LAY2>`: warps 0-3 compute gate rows of TYPE, warps 4-7 the up rows of TYPE2, each half
  with its own stage layout. The 11 UD-Q4_K_XL mixed gate / up pairs (`MK_GLU_MIX_PAIRS`) run one fused launch instead of two
  GEMMs into fp32 `y0` / `y1` + `pk_glu`. `MK_PF_GLU_MIX=0` turns it off. Profile category `gemm_ffn_gum`.
- Q5_K super-block buffer (`SBT<TYPE>`): the header + qh (48 B / row, the same for the 4 stages of a super-block) go once per
  super-block into a double buffer; each stage holds qs only (32 B / row), so Q5_K runs 4 cp.async stages instead of 3
  (IQ4_XS forced to 3 stages measured 2-5% slower). The fused-GLU epilogue reads the staged tile as float4 (it had 4-way
  smem bank conflicts); mixed-type GLU kernels at NB 32 stay <= 48 KB (2 CTAs / SM).
- Scratch: in `MK_EXACT` mode, when every gate / up pair is fused, `y1` is `[cap][VD]` instead of `[cap][FF]` (-44 MiB at cap
  1024); non-exact mode keeps FF (the attention split count depends on the capacity). `MK_PF_Y1=1` forces FF.

## 6. Prefill `xh` sizing (bbf4cd0)

`hp.xh` is `[cap][2 D]` (its largest GEMM input is the MTP `eh`, K = 2 D) instead of `[cap][FF]`: -14 MiB at cap 1024
(notes/vram_opportunities.md). Adds a `mem_mark("before prefill scratch")` line for MK_MEMLOG.

# Results

All on an RTX 3090 (no Ada / Blackwell numbers), `--timing` holds (warm-up first, then base | new | base [| new]),
Unsloth Qwen3.8-27B. "UD" = UD-Q4_K_XL, "Q4_K_M" = the plain model. Prefill = one prompt through `qwen38_mk` CLI, tok/s of
the prompt. Not measured: anything on sm_75 / sm_89 / sm_120, the default `MK_WV=0` build for items 5-6.

## Attention (pk_attn2 + PV16 + split; pk_attn8; pk_glu)

Baseline for the first table is the original `pk_attn` (2993f47 lineage); for the second, 78f1b07 (= pk_attn2 merged).

| model | prompt / cap | base | pk_attn2 default | delta | source |
|---|---|---|---|---|---|
| UD | 38,049 / 2048 | 1282.2 tok/s | 1488.0 | +16.1% | notes/agent_prefill.md "FINAL RESULTS (prefill track)" |
| UD | 101,584 / 1024 | 794.0 tok/s | 1124.7 | +41.6% | same |
| Q4_K_M | 38,049 / 2048 | 1217.3 | 1400.7 | +15.1% | same |
| Q4_K_M | 101,584 / 1024 | 772.4 | 1088.1 | +40.9% | same |

| model | prompt / cap | base 78f1b07 | pk_attn8 + pk_glu (568e86b) | delta | source |
|---|---|---|---|---|---|
| UD | 38,049 / 2048 | 1341.5 tok/s | 1474.6 | +9.9% | notes/agent_prefill2.md "FINAL RESULTS (prefill2 track)" |
| UD | 101,584 / 1024 | 1036.7 (97.99 s) | 1209.7 (83.98 s) | +16.7% | same |
| Q4_K_M | 38,049 / 2048 | 1409.2 | 1494.4 | +6.0% | same (pf2_fin = final: Q4_K_M never runs pk_glu) |
| Q4_K_M | 101,584 / 1024 | 1096.2 (92.68 s) | 1247.9 (81.41 s) | +13.8% | same |

2 base + 2 new runs per hold after a warm-up; runs within a hold agree to 0.1-0.6%. Split of the UD gain: attention
+5.9% / +13.2%, pk_glu +3.8% / +3.1% at 38k / 101.6k. Exact variants of pk_attn2 (UD): fp32 P.V with the split only 1455.7
tok/s at 38k / 2048 (+13.5%), 1080.7 at 101.6k / 1024 (+36.1%). Attention kernel alone (microbench, 1024 q at n0 = 100k):
pk_attn 87.1 ms -> pk_attn2 60.3 (exact) -> 47.4 (+ tail split) -> 39.8 (+ PV16, final form after the conversion changes)
[agent_prefill.md]; in the later track, 42.6 ms (pk_attn2 PV16 as the baseline there) -> pk_attn8 30.2 ms (59 -> 84
TFLOPS-equivalent) [agent_prefill2.md]. (The two tracks' absolute times differ slightly in build and clock; the microbench
baselines are not the same run.) In the engine (UD 101.6k, cap 1024, MK_PF_PROFILE)
attention takes 79.60 s (base, 59% of kernel time) -> 38.10 s (pk_attn2) -> 25.8 s (pk_attn8)
[agent_prefill.md "Prefill profile"; agent_prefill2.md "FINAL RESULTS"].

## Short prompts (0e19805)

Daily config (ctx 120000, `MK_PF_CAP=1024`, q8 KV, `-d 4`, UD-Q4_K_XL), `turns.py`, median of 3 in one process after a
warm-up, A-B-A in one hold; "turn" = the server's chat-turn path (rewind to the checkpoint + prefill of N new tokens)
[notes/agent_ttft.md "TTFT A-B-A"]. The column includes two items that are not in this PR (F9 async checkpoint
save / restore, in PR 5/6); the agent's attribution is F9 = ~6 ms (fresh) / ~12 ms (turn) and F2 (this PR) = the rest:
40-64 tokens ~30 ms, 100-128 ~8-12 ms, 300 ~4 ms, 1000 ~0.

| new tokens | turn base | turn new | fresh base | fresh new |
|---|---|---|---|---|
| 40 | 144.4 / 144.2 ms | 103.1 | 139.8 / 139.3 | 104.1 |
| 100 | 149.6 / 148.9 | 127.9 | 144.6 / 144.3 | 126.8 |
| 300 | 279.9 / 279.5 | 265.1 | 276.5 / 275.9 | 266.3 |
| 1000 | 645.2 / 642.7 | 631.4 | 638.6 / 635.5 | 628.9 |

Long prefill is unaffected (38k / cap 2048: 25.93 / 26.01 s -> 25.81 / 25.77 s). The per-stage fixed cost (~0.6 us) is why
narrow tiles are not used above 128 tokens (two 64-token tiles cost ~1.2-1.3x one 128 tile).

## Pipelined GEMMs, mixed fused gate / up, Q5_K buffer (bd8a22e, 167b509)

[notes/agent_pf.md; LEDGER "From agent_pf"]. Baseline pf0 = 21523be. Profile first (UD, cap 1024, `MK_PF_PROFILE`): GEMMs
are 77.7% of a 38k prefill (17.95 of 23.11 s) and 61.8% at 101.6k (48.05 of 77.72 s); 96-108 TFLOPS = 66-72% of the
fp16-accumulate peak at the real clock (prefill runs power-bound at ~1770-1860 MHz, 425-475 W of 480 W).

- Microbench (`notes/pf/gb/gb.cu`, 1024 tokens, ~1850 MHz, launcher layouts, byte compare): GEMM time 0.921-0.980 of base
  (Q5_K 0.95, Q6_K / Q3_K 0.93, Q4_K 0.98, iq4 GLU 1.00); mixed fused vs two GEMMs + pk_glu 0.84-0.95.
- End to end, A-B-A pf0 | pf2 | pf0 in one `--timing` hold per context, B vs mean of A: prefill 38k -4.2% (exact off) /
  -4.0% (exact on), 101.6k -3.1% / -3.0% (79.4 -> 76.9 s, 80.7 -> 78.3 s) of prefill time. TTFT (exact mode, c120000, warm)
  383 / 1355 / 5897 tokens -4.6 / -4.2 / -4.0% (240 -> 229, 783 -> 750, 3290 -> 3159 ms).
- At deploy (HANDOFF "r6c", prep_r6c.out): prefill 38k -2.9%, 101.6k -2.6% (a separate run of the same commit).
- VRAM: exact-mode prefill scratch 309.4 -> 265.4 MiB (-44 MiB). bbf4cd0: -14 MiB at cap 1024.
- Not measured: the default `MK_WV=0` build, Q4_K_M, decode (no decode code changed).

## End state, for context

Deployed build vs llama.cpp b11548 (tuned, q8_0 KV, MTP): prefill 101.6k 78.0 s vs 101.0 s, 38k 23.5 s vs 29.65 s
[notes/main/sweep3/RESULTS.md, candidate A; the llama column is the same file, UD-Q4_K_XL, ctx 120k, single server run].

# Correctness

Reference = llama.cpp batched-logit references (`ud.ref` 2k ctx mixed set, `ud32k.ref` 31k ctx, `q4km.ref`). Format:
top-1 agreement / mean KL / mismatches.

- pk_attn2 default vs base (agent_prefill.md): UD mixed 98.782% / 0.00115 / 162 -> 98.774% / 0.00111 / 163; prefill path at
  2k 98.677% / 0.00122 / 176 -> 98.729% / 0.00122 / 169; prefill at 31k 98.974% / 0.00231 / 42 -> 98.925% / 0.00228 / 44;
  Q4_K_M mixed 98.729% / 0.00131 / 169 -> 98.729% / 0.00128 / 169. Exact mode (PV16=0, SPLIT=0): byte-identical.
- pk_attn8 final vs base 78f1b07 (agent_prefill2.md): UD mixed 98.707% / 0.00113 / 172 -> 98.692% / 0.00113 / 174; prefill @ 31k
  98.925% / 0.00228 / 44 -> 98.974% / 0.00232 / 42; prefill (2k) 98.729% / 0.00122 / 169 -> 98.744% / 0.00124; Q4_K_M mixed
  98.834% / 0.00130 / 155 -> 98.789% / 0.00134 / 161 (6 more mismatches of 13,299, KL +0.00004 = +3%, inside the spread of the
  other non-exact variants of this kernel: reported as is). Exact mode and `MK_ATTN_S8=0`: byte-identical to base.
- Greedy output of the non-exact default forks from base after ~150 tokens on a 38k prompt (near-tie), as any non-bit-exact
  change does (agent_prefill.md).
- compute-sanitizer on the attention microbench (64 @ 1000, 100 @ 2000 with S = 17 split, 33 @ 500): racecheck 0 hazards,
  synccheck 0, memcheck 0.
- ptxas: pk_attn8 `<false>` / `<true>` 168 registers, 0 spills; 332 other functions unchanged, mk_cycle 128 registers.
- pk_glu (568e86b): UD prefill KLD log byte-identical.
- Narrow tiles (0e19805), build ttft_fin vs 5f69600: KLD result sections byte-identical for mixed, prefill, and prefill with
  `MK_PF_BN=32` / `64` forced on UD; Q4_K_M mixed byte-identical and prefill with `MK_PF_BN=32` byte-identical; greedy ids
  (4 prompts x 512 tokens) identical at d0 / d4 with identical accepted counts; 36 prefills x 33 ids (fresh and rewind turns)
  identical. Kernel level: NB 32 / 64 outputs byte-identical to NB 128 for every shape, type and token count tested.
  ptxas: 372 common functions, 0 changed.
- S4 / V2 / Q5_K buffer (bd8a22e, 167b509): all bit-identical: microbench byte compare; prefill-path KLD (default and
  `MK_EXACT=1`) byte-identical to pf0; KLD mixed byte-identical; tinv (greedy ids, q8, prompts 0-3, d0 / d4 / d7) 12 / 12 equal;
  exact-reuse probe (q8 + D') bitwise equal with identical fresh digests; ptxas: 0 changed spills in common functions, mk_cycle 128.

# Commits

Squashed: see commits.txt (9 commits: fbae2d3, 6b320e4, 60af318, da119d6, 568e86b, 0e19805, bd8a22e, 167b509, bbf4cd0).
Totals (sum of per-commit numstat): 7 files, +1982 / -576. No commits moved in or out. Note: bd8a22e / 167b509 were written on
top of the wide-verify weight layout (`LAY`, `FMT`, `FM6`, from the PR 1 / 2 commits 38521b7 and later) and on `MK_EXACT` (PR 5);
see PORT.md. Item 4 (0e19805) is the ttft track's F2; the other ttft commits (F6 / F9 / F10 / F11) belong to PR 5 / 6.

# Notes for the reviewer

- Default behaviour changes (non-exact, KLD-neutral as measured above): fp16 P.V, the tail split, int8 Q.K^T on the q8 cache.
  Three switches restore the old numerics (`MK_ATTN_S8=0`, `MK_ATTN_PV16=0`, `MK_ATTN_SPLIT=0`; `MK_ATTN=1` the old kernel,
  which could be dropped). Items 3-6 are exact.
- pk_attn8 requires the q8 cache (the default above ~80K tokens in open-jet's 120k server); with an fp16 cache pk_attn2 runs.
- Smem: pk_attn2 93.0 KB (was 82.5), pk_attn8 99,840 B per block, 1 block / SM as before. No VRAM change for attention (split
  partials reuse `y1`).
- sm_75: `mma.m16n8k32` s8 is not available on Turing (the compat layer splits k16 into two k8 MMAs; k32 s8 is not covered);
  pk_attn8 needs a fallback to pk_attn2 / pk_attn there. Nothing here was run on Turing.
- Open items from the notes: the per-tile phases still run in lockstep (more overlap needs registers: o 64 + Q fragments 32
  + scales 16 at 384 threads); GEMMs are ~60% of a 101.6k prefill (52 s): BK = 128 / fewer stage barriers, stream overlap of the
  small k / v GEMMs; per-stage barrier ~7% of stall samples; persistent CTAs (prologue / epilogue ~2.3% of a K 5120 tile);
  `frag_g` re-tune at Q5_K with 4 stages. An int8 GEMM was judged not justified by the numbers (agent_prefill2.md).
- Nsight Compute was unusable under WSL for the attention work (ERR_NVGPUCTRPERM) but worked later with kernel replay (agent_pf.md).
