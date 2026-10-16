# Qwen3.8-27B megakernel: performance branch

This branch is the engine we run daily on one RTX 3090: upstream open-jet at 2993f47 plus our UD-Q4_K_XL support and
VRAM work (merged upstream as #19 / #20) and about 95 further commits of engine work, each gated before it went in
(greedy output identical across draft depths, KL divergence against llama.cpp's own next-token distributions, A-B-A
timing on the same card). One engine source, `megakernel/qwen38_mk.cu`, serves both the plain Q4_K_M file and
unsloth's UD-Q4_K_XL. Since 2026-10-16 the server also takes images (`--mmproj`): the vision tower runs on the CPU, so
it costs no VRAM, and text-only prompts run exactly what they ran before ([UPDATES.md](UPDATES.md#2026-10-16-image-input-vision-cpu-encoder)).

- [BUILD.md](BUILD.md): how to build and run the configuration we measure (images included), and how to check it.
- [RESULTS.md](RESULTS.md): the benchmark results (one session, 2026-10-14), with charts; [PR_TABLES.md](PR_TABLES.md) has
  the same numbers as compact tables; raw data in [data/](data/).
- [UPDATES.md](UPDATES.md): what went in after that benchmark, with gates and numbers: 2026-10-15 exact engine changes
  (~+3.3% decode at 38k, ~+3.8% at 101.6k, A-B-A); 2026-10-16 image input (CPU encoder, image rows + M-RoPE in the engine).
- [pr-drafts/](pr-drafts/SUMMARY.md): the work grouped into 7 reviewable changes, each with a description, the numbers,
  a README section and a porting map onto upstream's current two-engine layout (origin/master 4d4db90); the image
  input is an upstream PR (row 8).
- The benchmark harness is on the separate branch `bench-suite` (based on upstream master; new files only, under
  `megakernel/bench/suite/`).

Bracketed sources in the drafts, such as `[notes/agent_kvq.md ...]`, refer to our lab notes, which are not in the repo.
All numbers are from one RTX 3090 (24 GB, Ampere, sm_86) unless stated otherwise.
