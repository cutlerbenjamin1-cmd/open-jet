# Qwen3.8-27B megakernel: performance branch

This branch is the engine we run daily on one RTX 3090: upstream open-jet at 2993f47 plus our UD-Q4_K_XL support and
VRAM work (merged upstream as #19 / #20) and about 85 further commits of engine work, each gated before it went in
(greedy output identical across draft depths, KL divergence against llama.cpp's own next-token distributions, A-B-A
timing on the same card). One engine source, `megakernel/qwen38_mk.cu`, serves both the plain Q4_K_M file and
unsloth's UD-Q4_K_XL.

- [BUILD.md](BUILD.md): how to build and run the configuration we measure, and how to check it.
- [RESULTS.md](RESULTS.md): the benchmark results (one session, 2026-10-14), with charts; [PR_TABLES.md](PR_TABLES.md) has
  the same numbers as compact tables; raw data in [data/](data/).
- [UPDATES.md](UPDATES.md): exact engine changes after that benchmark (2026-10-15: ~+3.3% decode at 38k, ~+3.8% at 101.6k),
  with their gates and A-B-A numbers.
- [pr-drafts/](pr-drafts/SUMMARY.md): the work grouped into 7 reviewable changes, each with a description, the numbers,
  a README section and a porting map onto upstream's current two-engine layout (origin/master 4d4db90).
- The benchmark harness is on the separate branch `bench-suite` (based on upstream master; new files only, under
  `megakernel/bench/suite/`).

Bracketed sources in the drafts, such as `[notes/agent_kvq.md ...]`, refer to our lab notes, which are not in the repo.
All numbers are from one RTX 3090 (24 GB, Ampere, sm_86) unless stated otherwise.
