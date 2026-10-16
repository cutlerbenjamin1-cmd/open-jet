# Proposed change set (2026-10-14)

Written from the commit history and our lab notes; numbers spot-checked against the notes. No code has been ported
to upstream's current layout yet. Upstream is at 4d4db90 (two engines, #21 sampled drafts, Turing): see UPSTREAM_STATE.md.

| # | PR | Depends on | Files |
|---|---|---|---|
| 1 | megakernel: speculative decoding rework (draft policy, wide tensor-core verify, up to 7 drafts, cheaper MTP) | 2 | 01-spec-verify/ |
| 2 | megakernel: faster decode kernels (pipelined q8 attention, T-invariant attention partition, Q6_K / alpha-beta layouts, spill fixes) | - | 02-decode/ |
| 3 | megakernel: faster prefill (int8 Q.K^T attention, pipelined GEMM fragments, mixed-type fused gate / up, short-prompt tiles) | (1 for wv layouts) | 03-prefill/ |
| 4 | megakernel: opt-in low-bit KV cache (MK_KV per-slot q4..q7, K / V rotation, MK_KV_WIN q8 recent window, presets D / D') | 2, 3, 1 | 04-kv-cache/ |
| 5 | megakernel: prompt reuse across edits (DeltaNet checkpoint pool), faster chat turns, optional bitwise-exact reuse (MK_EXACT) | 3, 4 (for 5b exact) | 05-prompt-reuse/ |
| 6 | megakernel: fast model load, CLI --batch and state files, native Q3_K / IQ3_S / IQ4_NL, penalties, watchdog, MEMLOG | 1, 4, 5 (state keys) | 06-load-memory/ |
| 7 | bench: head-to-head, quality and long-context benchmark suite (megakernel/bench/suite) | - | 07-bench-suite/ + branch bench-suite |
| 8 | megakernel: image input (Qwen-VL vision), encoded on the CPU, no extra VRAM (added 2026-10-16) | - | upstream PR [#23](https://github.com/L-Forster/open-jet/pull/23) |

Row 8 is not a draft: it was written directly against upstream master 72ba4a7 (both engines, the vision library, server,
open-jet hook, tests) and opened upstream; its description and results are on the PR. On this branch it is the
2026-10-16 entry of ../UPDATES.md.

Each code PR folder: PR.md (description, results with sources, correctness, commits, reviewer notes), README.md (section for
upstream's current megakernel/README.md), PORT.md (where each change lands in qwen38_mk.cu / qwen38_mixed.cu, collisions,
sm75, S / M / L effort), commits.txt.

Port order proposed by the agents: 2 -> 3 -> 1 -> 4 -> 5 (5a pool / async / server first, 5b MK_EXACT later) -> 6; 7 any time.
Suggested splits for review size: PR 1 into (a) policy + MTP cuts, (b) MK_WV=1, (c) MK_MAXT=8; PR 5 into 5a / 5b.

Commit moves vs INDEX.md: 535e80c 1 -> 2; f58e21e 2 -> 1; 3d84349 4 -> 5; 370b529 (PR 2) and a730f00 (PR 6) carry hunks
for PR 4 / 6 (MK_KV_EMU, MEMLOG) to split out when porting.

#21 reconciliation (PR 1): keep #21's sampled-draft mechanism and MK_DRAFT_SAMPLING; add our residual fallback fix (1e23ba6),
Ctl::dsmp, the confidence gate and MK_POLICY on top. Penalties (PR 6) must also apply to the draft distribution.

Bench suite (PR 7): branch `bench-suite` of this fork, 9 commits on upstream 4d4db90 (17 / 17 pure-Python tests pass on Windows Python and WSL Python 3.12; bash -n ok; all LF; no
machine paths). Prompt files are rebuilt from the repo's sources (token counts differ from our 38,049 / 101,586). Dataset
licenses are stated as on the cards when written: check each card before upstream. Results: see ../RESULTS.md.

Open before anything goes upstream: the port itself (compile + GPU gates per PR on both engines), Turing guards, Q4_K_M
measurements for items only measured on UD, the final-run numbers in each Results section.
