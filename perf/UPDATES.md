# Updates after the 2026-10-14 benchmark

[RESULTS.md](RESULTS.md) was measured on the branch as of 2026-10-14 (engine commit "merge exp-pf ..."). The engine changes
below went in on 2026-10-15. Each one is **bitwise exact**: greedy token ids identical at draft depths 0 / 4 / 7, KLD logs
against llama.cpp byte-identical (mixed prompts, prefill path, prefill with exact reuse, 32k with the low-bit KV preset),
plain Q4_K_M ids identical, server cache reuse == fresh prefill, VRAM unchanged. Same build flags as [BUILD.md](BUILD.md).

They came from a pass over the kernels the megakernel shipped with (the DeltaNet core, the grid barrier, the projection
phases) rather than the ones we had already rewritten.

## Changes

| | change | where | effect (RTX 3090, UD-Q4_K_XL, -d 7) |
|---|---|---|---|
| K1 | dn_proj: the alpha / beta rows go to the warps of each block that have no qkv / z tile (they used to share warps with a tile and formed the phase's tail) | `wv.cuh` dn_proj_wv | dn_proj 82.6 -> 80.1 us per layer; arrival spread 10.7 -> 7.6 us |
| K2 | wide-verify residual combine: the last-arriving warp loads every token's partial sums first, then runs the per-token epilogues (same add order) | `wv.cuh` resid_wv | resid_o 43.2 -> 40.6, resid_sout 42.1 -> 39.7, ffn_down 82.5 -> 80.5 us |
| K6 | Q8_0 tiles in the attention projections: 16-byte loads (each 32-byte sector read once instead of twice), and each Q8_0 k / v tile split into two 8-row items (each row fed as both MMA rows, so results are bit-identical) | `wv.cuh` | attn_proj 88.5 -> 78.0 us per layer at T = 8 |
| K7 | DeltaNet recurrent state stored in a coalesced per-head layout (each warp instruction reads / writes 512 contiguous bytes); the prefill scan uses the same layout | `qwen38_mk.cu` dn_core, `prefill.cuh` pk_dn_scan | dn_core 26.7 -> 18.3 us per layer |
| K8 | prefill: RMS-norm + quantise reads x once; the DeltaNet conv prep stages the conv taps once per 4 tokens | `prefill.cuh` | those kernels -29% / -27%; no measurable change in a whole prefill |
| K4 | measured the grid barrier: 0.55 us (82 blocks), ~0.18 ms per cycle. Two flag-broadcast variants were slower, so the upstream barrier stays | `qwen38_mk.cu` | `MK_GSB=N qwen38_mk ... --bench-phases` runs the micro-benchmark |

K6 and K7 apply unchanged to upstream's Q4_K_M engine (same kernels).

## Decode speed, A-B-A on the same card

Decode from saved post-prefill states (`--state-dir`), 255 tokens, -d 7, q8 KV, exact-reuse mode, one timing hold per form,
base | new | base, two passes. Accepted-draft counts are identical in every run (same tokens), so the difference is time
per cycle only.

| step | 38k context | 101.6k context |
|---|---|---|
| K1 + K2 vs the 2026-10-14 engine | +1.02% / +0.98% | +1.23% / +1.58% |
| K6 + K7 + K8 vs K1 + K2 | +2.61% / +1.98% | +2.35% / +2.46% |
| together | ~+3.3% | ~+3.8% |

In tok/s at 101.6k: 186.4 -> 188.7-189.3 -> 193.4. Short prompts (4 prompts x 2 reps, 512 tokens): +1.6% and +1.7% for the
two steps. These are this machine's A-B-A numbers; the end-to-end tables in RESULTS.md were not re-run.

## Switches (defaults are the measured configuration)

Runtime env: `MK_K1=1`, `MK_K2=1`, `MK_K6=4` (bit 1: a K-split variant, KLD-gated, opt-in; bit 2: spread items over
blocks, opt-in), `MK_K8=6` (bit 1: pk_ab on a second stream, opt-in, unmeasured). Compile time: `MK_K7=1`, `MK_K6_LD=1`,
`MK_K2_TB=8`, `MK_GSYNC=0`.

## Measured, not merged (yet)

- Splitting the grid barrier between the DeltaNet projection and the DeltaNet core: blocks take heads in the order they
  finish the projection, and each head replays its already-accepted drafts before waiting for the new inputs. Exact (same
  gates as above), dn_proj + dn_core -3 us per layer, but only +0.6% decode at 38k and within noise at 101.6k, for a new
  synchronisation pattern. Kept out of this branch for now.

## Tried and dropped

- Skipping the DeltaNet replay after a fully accepted draft chain (keep the final state as a shadow copy): +144 MiB VRAM
  and 0.45-0.7% slower than without it, because the extra code cost the DeltaNet core registers even when switched off.
