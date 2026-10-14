# README.md changes for PR 3 (prefill)

Written against origin/master (4d4db90) `megakernel/README.md`. Three edits.

## Edit 1: Build / run, the CLI-flags paragraph

Where: in "Build / run", in the paragraph that starts "CLI flags: `-d` drafts per cycle", after
"`MK_PF_CAP=1024` (prefill chunks of 1024 tokens; default 2048, rounded down to a multiple of 128 and clamped to 128-2048),"
and before "`MK_EMBD_DEVICE=1`". Insert:

> `MK_ATTN_S8=0` (q8 cache: Q.K^T on fp16 tensor cores instead of int8; `=2`: one Q scale per row instead of per 32 dims),
> `MK_ATTN_PV16=0` / `MK_ATTN_SPLIT=0` (prefill attention with fp32-accumulated P.V / without the tail split; with both, the
> output is bit-identical to `MK_ATTN=1`, the original prefill attention kernel), `MK_PF_BN=32|64|128` (force the token
> tile of the prefill GEMMs; default: chosen per GEMM from the chunk length), `MK_PF_GLU_MIX=0` (UD-Q4_K_XL: two GEMMs +
> `pk_glu` instead of the fused mixed-type gate / up kernel), `MK_PF_Y1=1` (keep the full-size `y1` scratch in `MK_EXACT` mode),

(`MK_PF_Y1` and the `MK_EXACT` mention only apply once PR 5 is in; drop that item otherwise. `MK_PF_GLU_MIX` exists only
in the mixed engine.)

## Edit 2: Prefill, the GEMM bullet and the Attention bullet

Where: "## Prefill". Replace the "GEMMs on fp16 tensor cores" bullet's last two sentences, and the one-line "Attention" bullet.

GEMM bullet (replace from "~100-108 TFLOPS ..." to the end of the bullet):

> ~100-108 TFLOPS on all four quant types (`bench/mmq_bench.cu`). Each k-stage is software-pipelined: the next stage's
> scales and fp16 A fragments are produced between this stage's MMAs, and the fp16 -> fp32 adds of one 128-k group are
> interleaved with the next group's first MMAs (the fp32 add order is unchanged, so the output is bit-identical). Q5_K keeps
> a super-block's header and high bits in a double buffer so it runs 4 cp.async stages. The token tile is 32, 64 or 128
> columns: prompts and chat turns of up to 128 new tokens use narrow tiles (cost model in `Impl::pf_nb`), which cuts a
> 40-token turn from 144 to 103 ms. FFN gate and up share one GEMM whose epilogue writes `silu(gate) * up` as the down
> projection's fp16 input; on UD-Q4_K_XL this also holds for the 11 gate / up pairs of different types (one half of the
> CTA per type), instead of two GEMMs and an elementwise pass.

Attention bullet (replace the whole bullet):

> - **Attention**: causal flash attention over the KV cache on tensor cores (`prefill_attn.cuh`). A block takes 16 query
>   positions of one KV head (6 q heads x 16 rows); two warps per q head split S by key positions and P.V by head-dim halves,
>   so 12 warps keep 64 fp32 accumulators each and K / V of the next phase load during the MMAs. P.V accumulates in fp16 per
>   32-position tile, then in fp32. The q-tiles of the last, partial wave are split along the keys and merged. With the q8
>   cache, Q.K^T runs on int8 tensor cores straight from the cache (`pk_attn8`): Q is quantised per 32 dims like the cache,
>   each 32-dim block's int32 partial is scaled by q scale x k scale, the raw q8 K / V tiles stream through a 3-stage cp.async
>   ring and V is converted for the next tile while the current one computes. ~84 TFLOPS-equivalent at 100K context
>   (`bench/attn_bench.cu`); the attention of a 101.6K-token prompt takes 25.8 s instead of 79.6 s with the original kernel.
>   KLD against llama.cpp is unchanged within noise (see the KLD tables); `MK_ATTN_S8=0`, `MK_ATTN_PV16=0` and
>   `MK_ATTN_SPLIT=0` restore the earlier numerics step by step.

## Edit 3: Results (optional), a prefill row set

Where: end of the "Results" section's prefill table, if upstream keeps one (it has a prefill table in the current README;
update its megakernel numbers with a run of the merged code, do not paste ours: ours are RTX 3090, UD-Q4_K_XL / Q4_K_M, with the
base/new pairs in PR.md). For reference, measured in this PR: UD 101,584-token prompt cap 1024 1036.7 -> 1209.7 tok/s,
38,049-token cap 2048 1341.5 -> 1474.6; Q4_K_M 1096.2 -> 1247.9 and 1409.2 -> 1494.4 (pk_attn2 / pk_attn8 / pk_glu only;
GEMM items add another 3-4% at 38k-101.6k on UD).

## Files list

Where: "## Files". Change the `prefill.cuh` / `prefill_gemm.cuh` line to
"`prefill.cuh` / `prefill_attn.cuh` / `prefill_gemm.cuh` - batched prefill kernels, causal attention and the quantised fp16 GEMM"
(mixed engine: `prefill_gemm_mixed.cuh` as upstream names it), and add `bench/attn_bench.cu` to the bench list if upstream
lists benches individually.
