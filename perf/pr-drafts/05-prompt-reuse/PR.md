# megakernel: prompt reuse across edits (DeltaNet checkpoint pool), faster chat turns, optional bitwise-exact reuse (MK_EXACT)

## Summary

The server today keeps one DeltaNet checkpoint (at the end of the last prompt, in device memory). A request that changes anything before that point, such as an edited earlier message, a changed system prompt tail or a thinking-mode follow-up that re-renders the previous turn, prefills the whole conversation again. This PR keeps a pool of checkpoints in pinned host memory, saves them inside the normal prefill pass at the places a chat client edits (generation prompt, each message start, every 4096 tokens), and lets the server restore the last checkpoint at or before the point where the new prompt diverges. Saving and restoring are moved off the prompt's critical path, which also shortens every chat turn and the first prompt of a process. An opt-in exact mode (`MK_EXACT=1`, `mk_server --exact-reuse`) makes a reused request answer bit for bit as the same request prefilled from token 0.

## What changed

**1. DeltaNet checkpoint in pinned host memory (b655092).** State, conv ring and h ring (~0.16-0.17 GB) are saved to and restored from pinned host memory instead of a device copy: -159 MiB VRAM, ~13 ms PCIe copy per prompt / rewind (commit message). `MK_CK_DEVICE=1` keeps the old device-to-device copies.

**2. Allocation at construction (5057c03).** The pinned checkpoint and the batched-prefill scratch (`pf_init`) are allocated in the engine constructor; `cudaHostAlloc` of the checkpoint took 118-529 ms inside the first prompt under WSL2. No device code change.

**3. Checkpoint save / restore off the critical path (a5b6c41, 032a1b0).** Save: device-to-device snapshot into the idle prefill scratch (~0.4 ms), then the device-to-host copy on a non-blocking stream under the first decode launch. Restore: host-to-device in 5 groups of DeltaNet layers on a copy stream, each group waited for only before its first layer's `dn_prep`; the restore is issued right after the prefill's token upload (032a1b0, because the later synchronous copies queued behind the 0.17 GB transfer). Watchdog waits cover the new events. Same bytes in the same order. With `MK_PF_CAP` below ~640 the scratch is too small for the snapshot and the copy reads the live state.

**4. Thinking-mode follow-up turns (646a6a5; same change as 6c45530 in the notes).** `mk_server` rewinds to a checkpoint at the last `<|im_start|>` (the generation prompt) instead of the end of the prompt: a thinking-mode follow-up re-renders that turn (`<think>\n\n</think>` instead of `<think>\n`), so the end of the prompt is not a usable restore point. `prefill(tokens, ck_at)`, C API `mk_prefill_ck`.

**5. Checkpoints saved inside one prefill pass, and a pool (bd894d1, 0c899f7).** `prefill(tokens, ck_at)` used to split the prompt into two `pf_run`s around `ck_at` (a host wait for the copy plus a second pass over the weights: +75 ms per short chat prompt). Now `pf_run` saves checkpoints on the way: `pk_dn_scan` takes a row offset and separate in / out state, the scan of each DeltaNet layer is split at a checkpoint position on its 32-token grid, the state at that point is staged in `hp.xm` (idle until the chunk's MTP part) and copied to the host per layer on a save stream; `pk_ck_conv` / `pk_ck_hist` save the conv and h rings. A checkpoint off the 32-grid (the exact generation-prompt position) is computed by a side scan from the grid point before it, so saving a checkpoint does not change what the prefill computes (0c899f7). Pool: `MK_CK_SLOTS` (default 8) x 0.17 GB pinned host, no device memory; eviction by coverage (old checkpoints thin out geometrically, recent ones stay dense). New API: `prefill(tokens, cks, ck_exact)`, `checkpoints()`, `rewind_to(pos)`; C: `mk_prefill_cks`, `mk_checkpoints`, `mk_rewind_to`. `MK_PF_CAP` below ~170 keeps the old two-pass split; in-prefill checkpoints need 2 staging slots (`MK_PF_CAP` >= ~340).

**6. Server (a191db9).** `load_prompt` asks for checkpoints at the generation prompt (exact), at each message start (`--ck-msgs`, default 1) and every `--ck-every` tokens (default 4096, snapped to the grid, 0 = off), and on a prompt that diverges inside the processed sequence rewinds to the last checkpoint at or before the divergence. Libraries without the pool API keep the single-checkpoint behaviour. Unit tests of the choice against a fake engine in `tests/test_megakernel_backend.py`.

**7. Exact reuse, `MK_EXACT=1` / `mk_server --exact-reuse` (4c23523, 4ca650d, ade2195, 43d67cd, 1db6d85, 5f30a10, + 3d84349 moved in from PR 4).** Default off; default numerics are unchanged. Reasons a reused request differed from a fresh one (agent_reuse.md): the tail-split plan of the prefill attention depends on the chunk's row count; the `MK_KV_WIN` ring / main split was taken per chunk; ring refill after a restore is lossy; the previous reply's rows (and each prompt's last row) come from the decode megakernel; an off-grid checkpoint restarts the DeltaNet 32-grid. In exact mode: `pk_attn8<_, _, EXA>` computes every prefill attention row as an in-order fold of absolute key segments (`MK_EXACT_SEG`, default 4096 positions, each from a fresh online softmax), so any work split gives the same bits; a split item is a head unit plus single-segment units chosen by a host list-scheduling estimate (`ex_plan`; `MK_EXACT_SPLIT=0` = whole items only, a test knob); the KV window ring / main split is taken per query tile and ring slots run in 640-position pieces; checkpoints sit on the absolute 32-token grid, are saved only inside the batched prefill, and carry the q8 window rings (pinned host); the server never appends to decoded rows, it restores the last checkpoint at or before the divergence (so the previous reply is re-prefilled). 3d84349 adds `KB` / `VB` template parameters to `pk_attn8` (compile-time K6 / V5 instantiation for the D / D' KV presets); its signature `pk_attn8<QROW, KLOW, EXA, KB, VB>` depends on the `EXA` parameter above (moved from PR 4).

| Variable / flag | Default | Effect |
|---|---|---|
| `MK_CK_SLOTS` | 8 | pooled checkpoints, each ~0.17 GB pinned host (D' + exact: +106 MiB per slot) |
| `MK_CK_DEVICE` | 0 | 1 = checkpoints stay in device memory (device-to-device copies on stream 0) |
| `MK_EXACT` | 0 | 1 = bitwise-exact reuse (see above) |
| `MK_EXACT_SEG` | 4096 | key-segment length of the exact attention fold |
| `MK_EXACT_SPLIT` | on | 0 = no split of exact attention items (test knob) |
| `mk_server --ck-every N` | 4096 | checkpoint every N prompt tokens (0 = off) |
| `mk_server --ck-msgs 0/1` | 1 | checkpoint at each message start |
| `mk_server --exact-reuse` | off | sets `MK_EXACT=1` before the engine is built |

## Results

All on one RTX 3090, UD-Q4_K_XL, ctx 120000, q8 KV, `MK_PF_CAP` 1024, greedy, server on one port, A-B-A in one GPU hold unless "single run" is stated.

| Measurement | Before | After | Source |
|---|---|---|---|
| Server TTFT, 22-token chat prompt, prompt_ms (A-B-A, 9 requests each) | 173.8 / 175.6 ms | 99.1 ms | agent_ckpt.md "Server v2"; LEDGER "From agent_ckpt" (base ed3b0f8; branch exp-ckpt a191db9) |
| Engine prefill, fresh chat prompt 22..3000 tokens | 172..1925 ms | 72-75 ms less at every size | agent_ckpt.md "v2" (ckturns, one hold) |
| Thinking turn (rewind to generation prompt + ~60-340 new tokens) | 181..350 ms | 107..275 ms | agent_ckpt.md "v1 functional" |
| 101.6k prompt after a 38k prompt it starts with | 84.4 s, 0 reused | 59.2 s, 36,864 reused (cap-aligned restore point) | agent_ckpt.md "Part 2 scenarios" (single run per build) |
| 101.6k from scratch (pool on) | 84.7 s | 84.4 s | same |
| Edit a user message 2 turns back, 11.7k-token chat | 7.31 s, 0 reused | 0.10 s, 11,650 reused | same |
| Edit the last line of a 9.1k-token system prompt | 5.72 s, 0 reused | 0.11 s, 9,088 reused | same |
| Same two edits, final deployed build vs llama.cpp b11548 (same session, UD-Q4_K_XL, 120k, q8 KV, MTP 4) | llama.cpp 0.33 s (11,638 reused) / 0.96 s (8,102 reused) | 0.10 s / 0.10 s (this work, build 9f25c2f which includes the pool) | notes/main/h2h2/RESULTS.md (one session, llama.cpp reuses at its nearest context checkpoint) |
| First prompt of a process (40 tokens, cold) | 352..401 ms | 113 ms (F6: ~210-290 ms of lazy allocation moved to load) | agent_ttft.md "F6" (turns.py, non-timing hold) |
| Rewind call (restore) | 6.4 ms | 0.02 ms (turn at 40 tokens 144.4 -> 103.1 ms includes the tile change F2, which is PR 3) | agent_ttft.md "TTFT A-B-A" |
| VRAM | | -159 MiB (checkpoint to host); server up 21,651 MiB in all three builds of the pool A-B-A; MK_MEMLOG device allocations identical line by line | b655092 message; agent_ckpt.md |
| Host RAM / load | 0.17 GB pinned | 1.34 GB pinned (+1.17 GB), load +1.1 s (13.7 -> 14.8 s) | agent_ckpt.md |
| Plain long prefill cost of the pool (CLI pf38k, cap 2048, 4 interleaved rounds) | 25.74 s | 25.92 s (+0.7%); +0.3% with `MK_CK_SLOTS=1`; server pf38k 26.06 / 25.99 -> 25.92 s | agent_ckpt.md "Plain-case cost" |

Exact mode (`MK_EXACT=1`) cost, 98k q8, cap 1024, exact vs default, A-B-A (agent_reuse.md "reuse2 Summary", LEDGER "From agent_reuse2"): fresh prefill +1.3%; restore + 32 / 128 / 512 tokens -2.6% / -1.6% / +0.7%. D' (`MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048`) at 98k: fresh +2.2%, -2.2% / -1.2% / +1.6%. 38k q8: fresh +1.2%, restores -7.7% / -3.7% / -0.5%. Earlier round (r6): fresh +2.7%, +2.7% / +5.0% / +11.0%. Server, 11.6k chat, -d 7: a verbatim follow-up pays the re-prefill of the previous reply (167 vs 78 ms; a 675-token reply kept with its reasoning 469-477 vs 81-86 ms); edits cost the same. 3d84349: D' prefill -2.5% at 101.6k, -1.3% at 38k (CLI A-B-A); q8 unchanged (38k A-B-A 23.74 / 23.71 / 23.75 s). Decode unchanged (A-B-A 95.53 vs 95.51 tok/s, accepted counts identical).

## Correctness

* Pool, default mode (agent_ckpt.md): greedy ids d0 / d4, prompts 0-3, identical to base, d4 == d0, accepted counts identical (217/235, 361/525, 270/663, 368/513); KLD (mixed path, ud.ref) byte-identical from "top token agrees" on (98.692% / 0.00113). A prompt prefilled with checkpoints gives ids identical to one without (22..3000 tokens, attention split off and on); checkpoint content equals the old single checkpoint's (12 prefills x 33 ids, `MK_ATTN_SPLIT=0`). Async save/restore: ids identical on 16 + 36 prefills (agent_ttft.md); also identical for `MK_PF_CAP=128`, `MK_CK_DEVICE=1`, `MK_WATCHDOG=0`.
* Server unit tests: 28 passed in tests/test_megakernel_backend.py with 4 new `load_prompt` tests (agent_ckpt.md "v1 functional"); the test file has changed upstream since, see PORT.md.
* Server outputs: equal to base in the long-prefill scenarios; in the 3-turn chat scenario the first turns differ in text because the old server path regridded the DeltaNet scan at `ck_at` and the new one computes what the CLI computes (near-tie flips). In the default mode reuse == fresh text holds when the restore point is cap-aligned (36,864 case) and the edit scenarios (True / True); it does not hold in general (long greedy thinking drifts after a few hundred characters in the old build as well). That is what `MK_EXACT` fixes.
* Exact mode (agent_reuse.md, LEDGER "From agent_reuse", "From agent_reuse2"): a state-digest probe (`mk_state_digest`: every K / V row, ring row, DeltaNet / conv state, h ring) is bitwise equal between a reused and a fresh prefill in every scenario tried: q8 and D', caps 1024 / 2048, `MK_EXACT_SEG` 2048 / 4096, abandoned replies up to 1,200 tokens, 38k, split == no split, second partial buffer. Server: kvd7's D' c120000 repro 904 / 904 == fresh, q8 -d 4 / -d 7, edit and thinking-follow-up scenarios reuse == fresh. Default path with the exact code present: tinv q8 d0/4/7 and KLD mixed byte-identical to the build before (447a87c); ptxas of the default instantiations unchanged. Exact-mode tinv q8 and D' d0 == d4 == d7. KLD of the exact path within noise: q8 mixed 98.662% / 0.00118 (vs 98.654 / 0.00116), q8 32k 99.121% / 0.00231, D' 32k 99.096% / 0.00230 (D' default 98.974 / 0.00239).

## Commits

Squashed into this PR (14 + 1 moved in), oldest first:

b655092 DeltaNet checkpoint in pinned host memory; 5057c03 allocate checkpoint and prefill scratch at construction (F6); a5b6c41 checkpoint save / restore off the critical path (F9); 032a1b0 issue the restore after the token upload; 646a6a5 rewind checkpoint at the generation prompt; bd894d1 checkpoints saved inside one prefill pass, and a pool; 0c899f7 saving a checkpoint no longer changes the prefill's arithmetic; a191db9 mk_server reuses the last checkpoint before the divergence; 4c23523 MK_EXACT=1 (WIP); 4ca650d MK_EXACT absolute key segments; ade2195 even cuts of whole segments; 43d67cd split v2; 1db6d85 segments of 4096, two partial buffers; 5f30a10 folds outside the tile loop.

Moved in: 3d84349 (pk_attn8 KB / VB template parameters), from PR 4: moved from PR 4, depends on MK_EXACT (its `pk_attn8<QROW, KLOW, EXA, KB, VB>` signature builds on the `EXA` parameter introduced here).
Moved out: none. Note: 646a6a5 and 6c45530 in the notes are the same change. The F10 / F11 commits of the same track (78873a0, fa27118, 9297c43) are in PR 6; the small-tile GEMM commit 0e19805 is in PR 3.

## Notes for the reviewer

* Pool, async copies and the generation-prompt checkpoint are on by default and change no numerics (see Correctness). The costs: +1.17 GB pinned host RAM (`MK_CK_SLOTS=1` reduces it; +0.3% CLI pf38k), load +1.1 s, and the first decode step after a prompt 0.3-1.0 ms slower (the checkpoint copy runs under it).
* `MK_EXACT` and `--exact-reuse` are opt-in. The project's own launcher turns it on by default; the recommended default for upstream is a maintainer's choice: it costs the re-prefill of the previous reply on verbatim follow-ups (see Results) and buys reuse == fresh bit for bit.
* Exact mode has dependencies: the exact attention is a mode of `pk_attn8` (PR 3, int8 prefill attention) and of the KV window rings (`MK_KV_WIN`, PR 4). If those land later, split this PR: the pool, async copies and the server (items 1-6) stand alone on the plain prefill; item 7 follows PR 3 and PR 4. See PORT.md.
* sm_75: the pool and copies use plain `cudaHostAlloc` / `cudaMemcpyAsync` and small kernels (`pk_ck_conv`, `pk_ck_hist`), no sm_86-only instructions; the exact attention rides on PR 3's kernel and inherits its Turing status. Nothing was tested on Turing.
* Known limits and open items from the notes: the exact attention kernel still costs +2.6% per tile over the plain kernel (spills 24-62 B in the EXA instantiations; fold traffic 192 KB per segment per whole item); `MK_EXACT_SEG` 2048 not tried at long context; the long-chunk tail balance at segment granularity; a single pinned block for all slots (instead of 8 allocations) recovered ~0.05 s of 0.18 s on pf38k (noise level) and was not committed; "settle" re-prefill of the previous reply while idle was tried and dropped (open-jet strips reasoning from the history, so thinking turns diverge at the generation prompt anyway).
* The engine holds one sequence; the pool serves one conversation at a time. Alternating conversations on one server were not measured in the notes.
