# Updates after the 2026-10-14 benchmark

[RESULTS.md](RESULTS.md) was measured on the branch as of 2026-10-14 (engine commit "merge exp-pf ..."). What went in
since, newest first:

- [2026-10-16: image input (vision), CPU encoder](#2026-10-16-image-input-vision-cpu-encoder): the server takes images;
  text-only prompts run exactly the code they ran before; no extra VRAM.
- [2026-10-15: kernel pass](#2026-10-15-kernel-pass): bitwise-exact kernel changes, ~+3.3% decode at 38k, ~+3.8% at
  101.6k (A-B-A).

The end-to-end tables in RESULTS.md were not re-run after either.

## 2026-10-16: Image input (vision), CPU encoder

The server takes images like `llama-server --mmproj`: OpenAI `image_url` parts (base64 `data:` URLs), with a Qwen-VL
projector (`qwen3vl_merger`, e.g. unsloth's `mmproj-BF16.gguf`). The vision tower runs on the CPU, so the GPU memory
budget is unchanged: idle VRAM at 120k context, q8 KV, is 21,875-21,920 MiB (22,096 MiB before). Seven commits
(`megakernel vision: ...`), merged as one. Full description: [megakernel/README.md, "Images (vision)"](../megakernel/README.md#images-vision).

- **Engine:** an image's positions take the encoder's fp32 rows instead of token embeddings (`MKImage`,
  `mk_prefill_img`; the MTP block takes them too, as llama.cpp does) and llama.cpp's Qwen-VL M-RoPE positions (IMROPE),
  done by rewriting rows of the RoPE table for the image spans: no attention or decode kernel changes. Checkpoints,
  rewinds and exact reuse work across images.
- **Prompt and preprocessing** are llama-server's bit for bit (stb_image, smart resize, padded bilinear, normalisation):
  same token counts and pixels as `llama-server --mmproj`.
- **Server reuse** stops at the first image that is not the same file (content hash) at the same place; decoded /
  encoded images are cached; a warm-up encode runs at startup.
- **Encoder:** `vision/vit_cpu.cpp`, the tower in fp32 with AVX2 / FMA kernels (the default where the CPU has AVX2 + FMA;
  otherwise llama.cpp's libmtmd encoder).

**Enable:** `make vision`, then `mk_server.py ... --mmproj mmproj-BF16.gguf` ([BUILD.md](BUILD.md#images) has the
flags). In open-jet, `OPENJET_MEGAKERNEL_MMPROJ=<mmproj GGUF>` adds `--mmproj` when `libqwen38vision.so` is built.

**Upstream:** the same work ported onto upstream's two-engine layout (master 72ba4a7, both `qwen38_mk.cu` and
`qwen38_mixed.cu`), text-only behaviour identical to master on both engines: [L-Forster/open-jet#23](https://github.com/L-Forster/open-jet/pull/23).

### Gates: the text path is unchanged

Greedy token ids identical at draft depths 0 / 4 / 7; KLD logs against llama.cpp byte-identical (mixed prompts, prefill
path, prefill with exact reuse, 32k with the low-bit KV preset D'); server cache reuse == fresh prefill; VRAM as above.

### Image prompts against llama.cpp

Mean KL of the megakernel's next-token distributions from llama.cpp's, same prompt and same image rows (llama.cpp's own
rows from the BF16 mmproj; UD-Q4_K_XL, q8 KV; `tools/vision_ref`, then `qwen38_mk --kld --kld-images`). 116-223 scored
positions per prompt. The same engine's text-only level is 0.0012.

| path | photo | code screenshot | 1024-token image | wide image | two images | image between text |
|---|---|---|---|---|---|---|
| prefill | 0.00088 | 0.00064 | 0.0017 | 0.0018 | 0.0011 | 0.0059 |
| prefill + decode | 0.0011 | 0.00069 | 0.0017 | 0.0016 | 0.0012 | 0.0064 |

- Top-token agreement 97.7-100%.
- The between-text figure is dominated by one position (max 0.92-1.03 nats) where llama.cpp's batched decode differs
  from its own one-token decode; with `vision_ref --text-batch 1` llama.cpp matches the megakernel there.
- Control: without the image rows (placeholder embeddings) the KL is 0.26-0.52.
- Against llama-server on 7 image cases (greedy): prompt tokens identical 7 / 7. Server scenarios (multi-turn image
  conversation reused == sent fresh; a different image at the same place is not reused; a rewind into an image re-feeds
  it): all OK with both encoders.

### Encoder: accuracy and speed

Against an fp64 forward of the tower (`vision/vit_ref.py`, 5 images of 196-1024 tokens):

| encoder | rel. L2 error of the rows | min row cosine |
|---|---|---|
| vit_cpu, BF16 mmproj | 4e-6 .. 4e-5 | 1.000000 |
| libmtmd, BF16 mmproj (BF16 GEMM inputs, fp16 K / V) | 4e-3 .. 2.7e-2 | 0.903 |
| libmtmd, Q8_0 mmproj | 1.3e-2 .. 8.4e-2 | 0.889 |

At the output (teacher-forced on llama.cpp's distributions given the fp64 rows), vit_cpu's rows score the same as the
fp64 rows (mean KL 0.0006-0.0016), libmtmd's BF16 rows up to 2.5x that, Q8_0 rows up to 7x. Use the BF16 mmproj:
vit_cpu widens either file to fp32, so a Q8_0 file is no faster, only less accurate.

Encode time, Ryzen 9 5900XT (Zen 3, 16 cores / 32 threads), BF16 mmproj, back-to-back encodes:

| image | tokens | vit_cpu | libmtmd |
|---|---|---|---|
| 448 x 448 | 196 | 0.51 s | 0.80 s |
| 640 x 480 | 300 | 0.78 s | 1.29 s |
| 1024 x 1024 | 1024 | 3.56 s | 6.07 s |

**Correction to megakernel/README.md:** the README says the first encode after the server's load costs ~0.5 s more and
the warm-up encode at startup absorbs it. That is incomplete. In the server, a new 300-token image that arrives after
the CPU has been idle takes 1.2-1.5 s here: the host's cores need about a second of all-core load to ramp up from idle
(spinning them up beforehand shortens the encode by about the spin time, so there is no net gain). Back-to-back encodes
take the 0.78 s above; an image already in the cache (`--image-cache`) costs nothing.

Not included: an AVX-512 / BF16 path for Zen 4 (faster but lossy, and untested here); video, audio, image URLs other than
`data:`.

## 2026-10-15: Kernel pass

The engine changes below are each **bitwise exact**: greedy token ids identical at draft depths 0 / 4 / 7, KLD logs
against llama.cpp byte-identical (mixed prompts, prefill path, prefill with exact reuse, 32k with the low-bit KV preset),
plain Q4_K_M ids identical, server cache reuse == fresh prefill, VRAM unchanged. Same build flags as [BUILD.md](BUILD.md).

They came from a pass over the kernels the megakernel shipped with (the DeltaNet core, the grid barrier, the projection
phases) rather than the ones we had already rewritten.

### Changes

| | change | where | effect (RTX 3090, UD-Q4_K_XL, -d 7) |
|---|---|---|---|
| K1 | dn_proj: the alpha / beta rows go to the warps of each block that have no qkv / z tile (they used to share warps with a tile and formed the phase's tail) | `wv.cuh` dn_proj_wv | dn_proj 82.6 -> 80.1 us per layer; arrival spread 10.7 -> 7.6 us |
| K2 | wide-verify residual combine: the last-arriving warp loads every token's partial sums first, then runs the per-token epilogues (same add order) | `wv.cuh` resid_wv | resid_o 43.2 -> 40.6, resid_sout 42.1 -> 39.7, ffn_down 82.5 -> 80.5 us |
| K6 | Q8_0 tiles in the attention projections: 16-byte loads (each 32-byte sector read once instead of twice), and each Q8_0 k / v tile split into two 8-row items (each row fed as both MMA rows, so results are bit-identical) | `wv.cuh` | attn_proj 88.5 -> 78.0 us per layer at T = 8 |
| K7 | DeltaNet recurrent state stored in a coalesced per-head layout (each warp instruction reads / writes 512 contiguous bytes); the prefill scan uses the same layout | `qwen38_mk.cu` dn_core, `prefill.cuh` pk_dn_scan | dn_core 26.7 -> 18.3 us per layer |
| K8 | prefill: RMS-norm + quantise reads x once; the DeltaNet conv prep stages the conv taps once per 4 tokens | `prefill.cuh` | those kernels -29% / -27%; no measurable change in a whole prefill |
| K4 | measured the grid barrier: 0.55 us (82 blocks), ~0.18 ms per cycle. Two flag-broadcast variants were slower, so the upstream barrier stays | `qwen38_mk.cu` | `MK_GSB=N qwen38_mk ... --bench-phases` runs the micro-benchmark |

K6 and K7 apply unchanged to upstream's Q4_K_M engine (same kernels).

### Decode speed, A-B-A on the same card

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

### Switches (defaults are the measured configuration)

Runtime env: `MK_K1=1`, `MK_K2=1`, `MK_K6=4` (bit 1: a K-split variant, KLD-gated, opt-in; bit 2: spread items over
blocks, opt-in), `MK_K8=6` (bit 1: pk_ab on a second stream, opt-in, unmeasured). Compile time: `MK_K7=1`, `MK_K6_LD=1`,
`MK_K2_TB=8`, `MK_GSYNC=0`.

### Measured, not merged (yet)

- Splitting the grid barrier between the DeltaNet projection and the DeltaNet core: blocks take heads in the order they
  finish the projection, and each head replays its already-accepted drafts before waiting for the new inputs. Exact (same
  gates as above), dn_proj + dn_core -3 us per layer, but only +0.6% decode at 38k and within noise at 101.6k, for a new
  synchronisation pattern. Kept out of this branch for now.

### Tried and dropped

- Skipping the DeltaNet replay after a fully accepted draft chain (keep the final state as a shadow copy): +144 MiB VRAM
  and 0.45-0.7% slower than without it, because the extra code cost the DeltaNet core registers even when switched off.
