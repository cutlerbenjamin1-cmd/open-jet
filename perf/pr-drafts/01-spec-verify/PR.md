# PR 1: megakernel: speculative decoding rework (draft policy, wide tensor-core verify, up to 7 drafts, cheaper MTP)

Status of this text: draft for L-Forster/open-jet, written against origin/master 4d4db90 (two engines: qwen38_mk.cu =
plain Q4_K_M, qwen38_mixed.cu = mixed-quant). The code in our tree (fabce52) is a single 8,499-line qwen38_mk.cu, so the
numbers below were measured on that tree; the port onto the two engines (PORT.md) has to be re-gated before posting.
Depends on PR 2 (02-decode: pipelined q8 decode attention `attn_core_q8` and the T-invariant attention partition). See
"Commits" for the two commits that moved between PR 1 and PR 2 to make the dependency one-directional.

## Summary

This changes how the engine chooses, produces and verifies MTP drafts, and widens the verify pass. (1) A confidence gate
on the MTP's own top-1 probability decides on the device how many drafts a cycle keeps, replacing the fixed "4, then 3
after a rejection" rule; the policy is runtime-configurable (`MK_POLICY`). (2) With the T-invariant attention partition
(PR 2) greedy output no longer depends on the draft depth, which is what lets the gate change T from cycle to cycle
without changing the text. (3) `-DMK_WV=1` (opt-in, compile-time) runs every main-pass projection of a verify cycle on an
int8 tensor-core GEMM over a load-time fragment-ordered weight layout, so a verify pass of 6-8 rows costs little more
than 5; `-DMK_MAXT=8` then allows `-d 5..7` (up to 7 drafts). (4) The MTP block is cheaper per pass (re-quantised to Q4_K
at load, no gathered copy of the draft head, K/V-only catch-up rows, embedding/M0 phases spread over more blocks).
On an RTX 3090 with UD-Q4_K_XL the default-on parts give +2.9% greedy and +11.7..13.5% sampled-thinking decode, and the
opt-in wide-verify build (`-DMK_WV=1 -DMK_MAXT=8`, `-d 7`, q8 KV) gives +17.6% greedy / +19.5% sampled thinking over the
previous build without it (9f25c2f at `-d 4`, one session; on the same build, `-d 7` vs `-d 4` alone is +9.4% greedy;
numbers and conditions below).

## Relation to #21 (sampled MTP drafts, af416bd + 6f11649)

Our b9349ce / 943c1bc / 1e23ba6 (written 2026-10-11, before #21) implement the same speculative-sampling scheme as #21.
This PR is meant to sit on top of #21, not replace it: keep #21's mechanism and add what it does not have.

Same in both:
- At temp > 0 the draft token is sampled from q, the draft head's top-32 passed through the same top_k / top_p / min_p /
  temperature transform as the target (`cand_probs`), instead of taking the argmax. Verification accepts draft d at row i
  with min(1, p_i(d) / q_i(d)); on rejection the row's token is drawn from norm(max(0, p_i - q_i)) over the target's
  candidates, and the rows after it are dropped. Greedy (temp 0) is untouched. The target distribution is exact.
- The draft head runs lm_head's TOPK path, ids mapped through `draft_ids`; q is stored as normalised probabilities next
  to the candidates; the draft uses its own random stream keyed by its absolute position, separate from the accept /
  residual streams `2 * (n + i)` and `2 * (n + i) + 1`.
- On by default at temp > 0; an off switch (#21: env `MK_DRAFT_SAMPLING=greedy`; ours: policy key `h1=0`).
- Both found that the draft vocabulary can hold a token twice (`dv_add` pads the draft rows to an even count with a copy
  of the last added token) and handle it (details below).
- Effect size agrees across GPUs: #21 reports 33.0 -> 38.5 tok/s (+16.7%, n=10) at temp 1.0 on an RTX PRO 4000 Blackwell
  SFF; we measured +11.7% (95% CI +6.5..+17.3) for sampled drafts alone on an RTX 3090 (agent_spec.md, "final_eval").

Differences (ours vs #21):

| | #21 on master | ours |
|---|---|---|
| where q lives | `dcand`, carved out of `cand` (`2 * MAXT * TOPK_MAX`), global memory | `cB.dq` (own buffer); after f58e21e staged in shared memory (SM_DQ) for `sample_targets` |
| duplicate draft rows | de-duplicated before the draft's q is built (keep the first entry; 6f11649) | kept; `dq_prob` sums all entries of a token at verify time (q is then the distribution over rows, and the proposal probability of a token is the sum). Both are exact because the stored q is the distribution actually sampled from |
| residual fallback when sum(max(0, p - q)) <= 0 | `cand_draw(c, p, K, -1, u)`: draws from p including the rejected draft d; if it returns d, `main_control` counts the row as accepted and reads a stale target | 1e23ba6: draw from p excluding d. Reachable only through float rounding (p = q on the support, about 1e-7 per rejection) but it is a correctness bug in the rare case |
| flag marking how the drafts were made | the verifying launch's `P.draft_prob` | `Ctl::dsmp`, written by the draft pass: robust if sampling parameters change between the launch that drafted and the one that verifies |
| draft depth / early stop | fixed rule, depth 4 -> 3 after a rejection | confidence gate (below) |
| random salt for the draft stream | `seed ^ 0x85ebca6b`, counter `3 * (mtp_P + di + 1)` | `seed ^ 0xD1B54A32D192ED03`, counter = absolute position. Sampled text for a given seed differs between the two; distributions are unaffected |
| exactness evidence | 2-token distribution over 250 seeds vs the no-draft engine: TV 0.004 | chi-square over 2000 + 2000 independent seeds x 5 tokens (min p 0.47 of 7 tests; point-mass control 0.47) and a randomized-PIT check of 17,393 committed sampled tokens (top-1 rate +0.02 pp, z +0.10; mid-PIT z -0.54) |

Proposed reconciliation (details in PORT.md): rebase onto master, drop our H1 hunks (`Launch::spec_smp`, `cB.dq`,
`draft_sample`, `dq_prob`, `Ctl::dsmp`'s sampling meaning, policy key `h1`), keep #21's `draft_prob` / `dcand` /
`MK_DRAFT_SAMPLING`, port the 1e23ba6 fix into `residual_draw`, and build the gate and policy on top. Keep `MK_DRAFT_SAMPLING`
as the user-facing switch (policy `h1=0` becomes an alias or goes away).

## What changed

### Draft policy and confidence gate (b9349ce, 943c1bc; default on)
- The draft `lm_head` computes the log-sum-exp of the draft logits (`lm_head<.., LSE>`). The last block derives the MTP's own
  top-1 probability `conf = exp(best - lse)` over the draft vocabulary and the running product over the cycle's drafts
  (`Ctl::dconf`, `dchain`).
- On-device early stop (`Launch::gate_drop[]`, `gate_stop[]`, `gate_chain`): per draft index k, `f < gate_drop[k]` drops that
  draft and stops, `f < gate_stop[k]` keeps it and stops (f = conf, or the running product with `chain=1`). All blocks
  leave the draft loop after the barrier once the last block set `Ctl::nd_stop`; `Ctl::nd_prod` tells the host how many
  drafts exist. The host verifies `min(nd, nd_prod)` drafts next cycle.
- Default policy at `MK_MAXT=5`: always up to `n_draft` drafts (`depth=1`), chain gate on, stop after draft k >= 2 when the
  chain product is below 0.4, drop the 4th draft (the T = 5 verify row) when it is below 0.4, sampled drafts at temp > 0.
  The previous behaviour is `MK_POLICY="depth=0,drop=0,stop=0,chain=0,h1=0"` (reproduces the old texts exactly).
- `MK_POLICY` (env, `key=value,...`), `MKEngine::set_policy`, C API `mk_set_policy`, `mk_last()` (T / drafts / accepted of the
  last step). Keys: `depth=0|1|2`, `dmin`, `drop`, `stop`, `dropK`, `stopK` (K = 1..MAXT-1), `chain=0|1|2`, `h1=0|1`,
  `mtpq=0|1` (only with `MK_MTP_Q4=ab`), and for `MK_MAXT > 5` builds `droplast`, `dctx=N` (from position N on at most 4
  drafts; default off), `g7=0|1`, `blk=0|1`.
- `MK_SPEC_LOG` (compile switch, `-DMK_SPEC_LOG=1`) + `MK_SPEC_LOG_FILE=<path>`: one JSON line per decode cycle (drafts with
  the MTP's top-32 and lse, verify targets, the main top-32 rows when sampling). Analysis only; the policy thresholds were
  fitted on these logs offline.

### Sampled drafts (b9349ce H1, 1e23ba6) - to be reconciled with #21, see above.

### Depth-independent greedy output (535e80c) - lives in PR 2 now
The split-K attention partition depends on the query position only (chunks are whole 32-position K/V tiles, size
`32 * ceil(L / 640)`; the host caps T so one launch never straddles a multiple of 640). Without it greedy ids at `-d 0..4`
differ (bench prompt 1 diverges from `-d 0` at token 90 (d1, d3), 319 (d2), 308 (d4) on the pre-change build). The gate
makes T content-dependent, so this property is a precondition of the policy here. The text and numbers are in PR 2.

### Wide verify: int8 tensor-core GEMM for the verify pass (`-DMK_WV=1`, opt-in, default 0)
- New `wv.cuh` / `wv_host.cuh`: an int8 `mma.m16n8k32` verify GEMM for every main-pass matrix of a cycle: dn_proj, attn_proj
  (qkv / z), ffn_gu (GLU epilogue), out / ssm_out and ffn_down (split-K 4, ordered last-arrival combine). It serves every
  T including `-d 0`; it is T-invariant but not bit-identical to the dp4a GEMV (different, fixed summation order).
- Load-time, in-place, lossless re-layout (no extra VRAM): Q4_K / Q5_K / IQ4_XS into a fragment-ordered tile-major layout
  ("FM": each (16-row tile, super-block) unit contiguous, every lane load one 16-byte LDG), IQ4_NL / Q8_0 and the offset
  types (Q6_K / Q3_K / IQ3_S for ffn_down) tile-major. The re-layout adds about 10.8 s to model load (one-time).
- Why: wv with the engine's row-major layout read as 16-row tiles ran at 450-600 GB/s vs 800-870 for dp4a at T = 1; the FM
  layout brings wv to 0.98-1.09x of dp4a T = 1 time at T = 1 and 1.04-1.17x at T = 8 (agent_wvgemm.md "wvgemm2").
- Prefill (`prefill_gemm.cuh`, `mmh<..., LAY=1>`) reads the wv layout whole (FMT stage layout, a675751), so one copy of the
  weights serves both decode and prefill.
- `MK_WV_FMK` (compile, default 1): Q4_K / Q5_K fragment-ordered. 0 = tile-major (prefill reads whole chunks, but decode
  loses: -5.8% at `-d 0`, gain gone at `-d 4`).
- `MK_WV_OUT` (compile, default 0, MAXT > 5): out / ssm_out of the offset types on wv. Measured slower (+1.4% per cycle at
  38k, +-0 at 101.6k); kept for reference.
- `MK_WV_Q6F` (compile, default 0, MAXT > 5): every main-layer Q6_K in an FM Q6_K unit (3360 B per 16-row tile and
  super-block, meta inline) on wv; ffn_down's Q6_K output is bit-identical to the tile-major path, out / ssm_out / gate / up
  / qkv / z / k / v leave dp4a (KLD-gated). Used in the deployed build (`-DMK_WV_Q6F=1`).
- Build memory: MK_WV builds first exceeded 25-30 GB in ptxas (`gemv_pairs` instantiated per (T, type, call-site lambda) was
  47% of the PTX). `PairOut` (one store functor in MK_WV builds) cut PTX by 28%. The merged tree still needed
  `LIMIT_GB=30` to build (24.9 GB peak).

### Seven drafts: `-DMK_MAXT=8` (opt-in, needs `MK_WV=1`; dc91306, a675751, a5631b7)
- `MK_MAXT` (compile, default 5, 6..8 requires `MK_WV=1`) = most tokens of one verify pass. T = 6..8 run as one padded
  8-row pass (`run_wide`, one extra instantiation). Every main-layer ffn_down is on wv (dp4a ffn_down cannot hold T >= 6),
  conv ring 16 slots, gate arrays `[MAXT - 1]`, lm_head misc slots, SM_CAND assert.
- Attention at T > 5: `attn_core_w48` (447a87c) handles all 48 query rows (6 heads x T <= 8) in one pass over K/V in 99,072 B
  of shared memory (3 K + 3 V stages); the first form ran two head-group passes (K/V read twice). `MK_W48=0` keeps the
  two-pass form; `MK_W48S=1` (default) keeps the call's scalars and per-item thread indices in shared memory (805ce7e; spill
  fix, exact). all-q8 KV cache only in this PR.
- Gates for depth > 5 (a5631b7, policy `g7`, default on at depth > 5): stop after drafts 2-4 below chain 0.25, drop draft 5
  below 0.3 (stay at T <= 5), stop after drafts 5-6 below 0.3, never drop the deepest draft (T = 6..8 cost the same). Any
  explicit `stop` / `drop` / `droplast` key (or `g7=0`) restores the previous gates. Chosen by an offline search on
  open-policy spec logs, confirmed live.
- `dctx=N`: from position N on at most 4 drafts. Default off (no context cap needed up to 101.6k).
- -d 5..7 needs the q8 KV cache (`--kv q8`). The CLI/server `-d` limit becomes `MAXT - 1`.

### Cheaper MTP passes (default on)
- 17e775f `MK_MTP_Q4` (default 1; 0 = stored types; `ab` loads both, policy key `mtpq` swaps them): the MTP block
  (`blk.64` + `nextn.eh_proj`) matrices stored wider than Q4_K (Q5_K / Q6_K / Q8_0) are dequantised and re-quantised to Q4_K
  at load with `ggml_quantize_chunk` (no imatrix). Only the drafts change; the main model verifies them, so greedy output
  and the sampled distribution are unchanged; acceptance moves by under a point.
- d487e7e: the draft head reads lm_head rows through `draft_ids` instead of a gathered copy of the rows (-158 MiB VRAM,
  bit-identical).
- c0a1783: the embedding prologue and the MTP M0 phase run on 10 blocks per row instead of 1-5 (exact).
- 624803c `MK_MTP_KVC` (default on; env `MK_MTP_KVC=0` off; MAXT > 5 + `MK_WV` builds only): of a cycle's catch-up rows the
  ones before the last 5 only need their MTP-slot K/V, so they run a K/V-only pass (`mtp_kv_pass`) instead of a second
  full pass. Drafts are bit-identical.
- f58e21e (moved in from PR 2): sampled `lm_head` epilogues: bitonic top-32 merges with a (value, list * 32 + rank) key
  that reproduces the sequential `topk_insert` result entry for entry, candidates and q staged in shared memory, parallel
  load of the draft head's per-block (max, sum exp). Exact.

### Opt-in, default off
- `blk=1` (624803c, a5631b7): block verification (Sun et al., arXiv 2403.10444, Alg. 2) for sampled drafts
  (`sample_targets_blk`), MAXT > 5 builds. `chain=2`: gate on the drafted token's own q probability (offline: not better;
  may be removed).

## Results

All numbers: RTX 3090, Qwen3.8-27B UD-Q4_K_XL unless noted; "A-B-A" = arms interleaved inside one `gpu_lock.sh --timing`
hold (cross-session numbers are not comparable; notes/LEDGER.md "Measurement facts").

Gate + sampled drafts (default policy) vs the old rule (`4 -> 3`, argmax drafts) [notes/agent_spec.md "Step 5", commit 943c1bc]:
- greedy `bench_mk` A-B-A-B, 4 prompts, `-d 4`, same text in both arms: 103.04 / 103.50 -> 106.55 / 106.06 tok/s (+2.9%).
- sampled thinking (temp 1 / top_p 0.95 / top_k 20, 6 prompts x seeds 1000-1003 x 768 tokens, 3 arms interleaved per block,
  3 GPU holds): old 83.79 tok/s, sampled drafts with the old depth rule 93.63 (+11.7%, 95% CI +6.5..+17.3), default 95.09
  (+13.5%, CI +10.0..+17.2); tokens per cycle 2.564 -> 2.898, ms per cycle 30.60 -> 30.48, drafts made 21,394 -> 17,690.
  The independent half of the seeds (1002-1003, 12 blocks): +11.7% (CI +7.8..+16.7). An earlier +15% was partly a slow
  text draw of the old arm.
- Plain Q4_K_M: no measurable change (116.18 / 116.58 -> 116.35 tok/s, texts differ; a clean A-B-A was not run).

MTP cost cuts [notes/agent_mtpcut.md "Final results", commits 17e775f, c0a1783, f58e21e; 2b8605c and 58ce261 are in PR 2]:
- All five commits vs 134ba4c, one hold each: greedy bench `-d 4` 109.41 / 109.43 -> 111.77 (+2.15%); `-d 0` +5.1%;
  sampled thinking items 2-5 +1.55% (token-identical runs); Q4_K_M +0.95%.
- 17e775f alone, greedy A-B-A in one hold (`MK_MTP_Q4=0 / 1 / 0`): 109.10 / 110.81 / 109.29 (+1.48%); accepted 63.4% ->
  62.8%; sampled A/B (36 paired runs, arms interleaved) +2.8% (SE 1.8%), ms per cycle -1.33%. VRAM -106.8 MiB (UD),
  -50 MB (Q4_K_M); +2.3-2.7 s at load (8 matrices; Q4_K_M: 3 matrices, 0.9 s).
- c0a1783: embed 15 -> 6 us, mtp_m0 91 -> 45 us per launch (-54.5 us per cycle, 0.17%).
- f58e21e: epilogue tails -197 us per cycle; sampled +0.40% (paired SE 0.02%, 12 token-identical runs).
- d487e7e: -158 MiB (commit message; the vram table lists -160 MiB at 120k ctx, notes/agent_vram.md "FINAL RESULTS").

Wide verify, `-DMK_WV=1`, `MK_MAXT=5` (M1) vs the same tree with MK_WV=0, A-B-A [LEDGER "From agent_wvplumb3", commit 2292859 / a675751]:
- decode `-d 4` +7.0%, `-d 0` +3.2%, 38k +12.6%, 101.6k +9.6%; prefill 38k +3.0% faster, 101.6k +2.5% faster (the first
  reader, which split each 16-B FM chunk into two 8-B copies, was -4.2% / -3.6%; a675751 fixes it). Bench `-d 4` 118.0 ->
  126.3 tok/s, `-d 0` 42.7 -> 44.1 (2292859 message, pre-reader-fix build).
- Plain Q4_K_M: `-d 4` +8.7%.
- Load: +10.8 s once (re-layout).

Depth 7, `-DMK_WV=1 -DMK_MAXT=8` [LEDGER "From agent_wvplumb3"]: `-d 7` vs `-d 4` on the same build: bench greedy +9.4%, spec_bench2
greedy +2.9% / sampled thinking +5.1%, decode at 38k / 64k / 101.6k +16.5 / +10.6 / +16.5%. T = 8 costs about 3.5 ms
over T = 5 (MTP about 0.5 ms per extra draft + about 2 ms main pass).

Deployed build vs the previous deployed build (9f25c2f), one session, 120k ctx, q8 KV, `-d 4` (A: `MK_WV=1`) and `-d 7`
(B: `MK_WV=1 MK_MAXT=8`) [notes/main/sweep3/RESULTS.md]:

| metric | 9f25c2f | A | B |
|---|---|---|---|
| greedy decode, short prompts, tok/s | 117.9 | 127.3 (+7.9%) | 138.7 (+17.6%) |
| sampled thinking, tok/s | 105.7 | 117.3 (+11.0%) | 126.3 (+19.5%) |
| decode at 11.6k | 109.9 | 120.2 (+9.4%) | 126.7 (+15.3%) |
| decode at 38k | 97.0 | 103.7 (+6.9%) | 102.7 (+5.8%) |
| decode at 101.6k | 96.8 | 100.3 (+3.7%) | 105.7 (+9.2%) |

and B vs llama.cpp tuned (MTP 4): greedy 1.71x, sampled thinking 1.94x, 101.6k decode 1.98x.

W48 and later depth-7 work, UD, `-d 7`, A-B-A [LEDGER "From agent_kvd7", "From agent_spill", "From agent_spill2", "From agent_mtp"]:
- 447a87c one pass vs two head-group passes at 101.6k, q8: 41.17 -> 38.03 ms per cycle (+8.2% tok/s); `MK_PROFILE` attn_core
  q8 554 -> 395 us per call. 805ce7e: leaf spill stalls 0.48% -> 0.07% of the cycle (ncu), no measurable wall-time change.
- 9fed197 `MK_WV_Q6F`: ms per cycle at 101.6k q8 -1.5%, 38k q8 -2.6% (ids differ), short q8 -1.9%; load +0.6 s (10.7 vs 10.1 s).
- 624803c `MK_MTP_KVC`: 38k +0.9%, 101.6k +0.7% (same binary, identical drafts).
- a5631b7 g7 gates (with KVC and `blk=1`) vs 005c902: 38k +2.0%, 101.6k +4.1% greedy (same ids); sampled within noise.
- `blk=1` alone: offline exact expectation +0.8..1.1% tokens per cycle (default gate), +2.0% with the g7-like gate; live
  18-run arms within noise (SE about 3%).

Plain Q4_K_M, whole stack (this PR + PR 2 + other PRs) vs upstream's engine at 2993f47, `-d 7` build vs `-d 4`
[notes/main/km/RESULTS.md, one session]: writing code 169.8 vs 133.8 tok/s (+27%), 30K tokens of code in context 110.2 vs
87.4 (+26%), short chat 142.6 vs 118.0 (+21%), reasoning mode 116.5 vs 89.1 (+31%). Not attributable to this PR alone.

Not measured: anything on Blackwell, Turing (sm_75), Ada (sm_89); the gate on plain Q4_K_M with an A-B-A; the sampled gate
at depth 7 on Q4_K_M.

## Correctness

Reference for KLD: llama.cpp logits on the same text (`~/kld/ud.ref` "mixed" set, `ud32k.ref` 32k set, `q4km.ref` for
Q4_K_M); noise level about 0.0001.
- 535e80c (PR 2): greedy ids identical at `-d 0..4` on 4 bench prompts (`-n 512`) and prompts 1 / 3 at `-n 1600`; KLD mixed
  98.752% top-1 / 0.00114 (base 98.782% / 0.00115).
- Gate + sampled drafts: greedy ids identical at `-d 0` and `-d 4` (UD prompts 0-3, Q4_K_M prompts 1, 3); KLD byte-identical
  to 535e80c (main path unchanged); default policy off-switch reproduces the old texts token for token.
- Sampled-draft exactness: chi-square test over 2000 + 2000 independent seeds x first 5 tokens: min p 0.472 over 7 tests
  (point-mass control 0.471); randomized-PIT of 17,393 committed tokens of sampled thinking: top-1 rate +0.02 pp (z +0.10),
  mid-PIT z -0.54, 0 tokens outside support (control 17,213 tokens: z +0.30 / +0.73). Power: the chi-square test rules out
  gross errors only (80% power for TV about 0.03 moved to a rare token); the PIT check detects a 0.5 pp top-1 bias at about
  2.5 sigma.
- MTP cuts (17e775f, c0a1783, f58e21e, d487e7e): greedy ids, accepted counts, KLD mixed identical; sampled tokens identical
  for fixed seeds (f58e21e); 17e775f changes only drafts.
- Wide verify (MK_WV=1, MAXT 5): tinv d0 == d4 on 4 prompts; KLD mixed 98.654% / 0.00114 (MK_WV=0: 98.692% / 0.00113),
  32k 99.145% / 0.00223 (99.096% / 0.00219); prefill-path KLD byte-identical to the MK_WV=0 build. Plain Q4_K_M: ids d0 ==
  d4, KLD 98.887% / 0.00127 vs 98.789% / 0.00134. Not byte-identical to the dp4a path, by design.
- MAXT 8: ids identical at `-d 0 / 4 / 7`; KLD mixed 0.00116 (within noise), 32k 0.00232 (+0.00013, at the edge of noise).
- MK_WV_Q6F: tinv q8 prompts 0-3 d0 == d4 == d7 (4 / 4); KLD mixed 98.737% / 0.00119, 32k 99.121% / 0.00219, prefill-path
  KLD byte-identical.
- W48 one-pass: ids identical to the two-pass build at short ctx, 38k and 101.6k (q8); 805ce7e: q8 tinv d0/4/7 12/12, KLD mixed byte-identical.
- MK_MTP_KVC: tinv d0/4/7 12/12 identical, drafts bit-identical.
- The default build (`MK_WV=0`, MAXT 5) of the wv commits: ids / KLD / prefill byte-identical to the previous deployed refs.

## Commits

Squashed into this PR (hash subject), in commit order; full list with stats in commits.txt:

```
b9349ce megakernel: draft confidence gate, runtime draft policy, sampled MTP drafts (H1), MK_SPEC_LOG logging
943c1bc megakernel: default draft policy = confidence gate + sampled MTP drafts (H1)
1e23ba6 megakernel: sampled-draft residual fallback never returns the rejected draft
d487e7e megakernel: the MTP draft head reads lm_head rows through draft_ids instead of a gathered copy
17e775f megakernel: re-quantize the MTP draft block to Q4_K at load (MK_MTP_Q4, default on)
f58e21e megakernel: cheaper sampled lm_head epilogues (bitonic top-32 merges, shared-memory sampling inputs)   [moved in from PR 2]
c0a1783 megakernel: spread the embedding prologue and the MTP M0 phase over more blocks
38521b7 bench: wvgemm_bench, int8 mma.m16n8k32 verify GEMM ...
5e653f7 bench: wvgemm2 - load diagnostics, v2 kernel, tile-major / fragment-ordered weight layout ...
820b77b megakernel: MK_WV M0 - wv tensor-core verify GEMM ...
15c8003 megakernel: MK_WV M1 WIP - every main-pass matrix re-laid-out in place at load ...
dc91306 megakernel: MK_WV build memory + M2 WIP (MK_MAXT 6..8) + MK_WV_FMK
2292859 megakernel: MK_WV M1 final - wv tensor-core verify GEMM in every main-pass phase
a675751 megakernel: MK_WV prefill reads the FM layout whole (FMT stage layout) + MK_MAXT > 5 depth cap policy key dctx
447a87c megakernel: T > 5 decode attention in one pass over all 48 query rows (attn_core_w48, MK_W48=1)
805ce7e megakernel: W48 spill fix: per-item thread indices + the call's scalars in shared memory (MK_W48S)
22bda66 megakernel: MK_WV_OUT (opt-in, default 0): out / ssm_out offset types on wv (measured slower; kept for reference)
9fed197 megakernel: MK_WV_Q6F (opt-in, default 0): every main-layer Q6_K fragment-ordered on wv (M3)
624803c megakernel (WIP, MAXT > 5 only): MK_MTP_KVC K/V-only MTP catch-up rows, policy blk=1 block verification, chain=2
a5631b7 megakernel (MAXT > 5): depth > 5 default gates (policy g7), S6 without local arrays
```

Moved: 535e80c (T-invariant attention partition) goes to PR 2, because 7996cfa (`ATT_BUCKET` routing) and the whole-tile
chunks of `attn_core_q8` need it, and PR 1's wide attention (dc91306, 447a87c) needs `attn_core_q8`; without the move the two
PRs depend on each other. f58e21e (sampled lm_head epilogues) comes in from PR 2 because it rewrites the staging of the
H1 `dq` rows and the draft head's LSE sums, i.e. code that exists only after b9349ce. Main should reflect both in INDEX.md.
The original dc91306 / 15c8003 subjects are long WIP messages; the squashed PR needs a rewritten commit message.

## Notes for the reviewer

- Default on: the gate/policy, sampled drafts at temp > 0 (already default via #21), T-invariant partition (PR 2), MTP Q4
  re-quantisation (`MK_MTP_Q4=1`), K/V-only catch-up (MAXT > 5 builds only), embed/M0 spread, draft head via `draft_ids`.
  Opt-in at compile time: `-DMK_WV=1`, `-DMK_MAXT=6..8`, `-DMK_WV_Q6F=1`, `-DMK_WV_OUT=1`, `-DMK_WV_FMK=0`,
  `-DMK_SPEC_LOG=1`. Opt-in at run time: `blk=1`, `chain=2`, `dctx`.
- MK_WV changes the bits of the output (not byte-identical to the dp4a path, T-invariant, KLD within noise). The MK_WV=0 /
  MAXT 5 build is byte-identical to the previous behaviour for the wv commits.
- Resource costs of MK_WV: +10.8 s model load (+0.6 s more with Q6F), ptxas peak of 25 GB or more for the merged tree
  (the mixed-quant engine alone needs about 11 GiB), larger binary. Needs the host to have the RAM; the Python build step
  must pass `-DMK_WV=1 -DMK_MAXT=8` only when it is asked to.
- Depth > 4 needs the q8 KV cache. Low-bit `MK_KV` presets at `-d 5..7` are part of the kv-cache PR.
- Turing (sm_75): `mma.m16n8k32` int8, the 99 KB shared-memory budget of `attn_core_w48` and the cp.async pipelines are
  sm_80+. MK_WV / MAXT > 5 must stay off for sm_75 (the plain engine is the only one with a Turing path). The gate, policy,
  `MK_MTP_Q4`, the draft-head change and the epilogue changes are architecture-neutral. Not tested on Turing.
- Gate thresholds were tuned on UD-Q4_K_XL logs; Q4_K_M and Blackwell not re-tuned.
- The rs == 0 fallback fix (1e23ba6) should go into master regardless of the rest of this PR.
- Open follow-ups from the notes: block verification re-measure at depth 7 (live arms are within noise), a faster T = 2
  main pass (see PR 2), FM layouts for Q8_0 (`resid_sout Q8_0` 58 us vs Q5_K 40) and IQ3_S ffn_down (156 us vs 86), and the
  q6 fill branch side effect (about 50 us per cycle) [LEDGER "From agent_spill2"].
