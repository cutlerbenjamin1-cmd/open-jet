# PORT.md for PR 1 (spec-verify): map onto origin/master 4d4db90

Our code: one `megakernel/qwen38_mk.cu` (8,499 lines at fabce52) that serves Q4_K_M and UD-Q4_K_XL. Upstream: `qwen38_mk.cu`
(3,163 lines, plain Q4_K_M: types Q4_K / Q5_K / Q6_K / Q8_0 only, `enum { T_F32, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14 }`) and
`qwen38_mixed.cu` (3,436 lines, adds `T_IQ4X`, `gemv_pairs_any`, `resid_any`, `ffn_gu_any`, `rows16`). The two engines are
full copies of the whole file (device code and the `MKEngine::Impl` host code), with `prefill_gemm.cuh` (plain) and
`prefill_gemm_mixed.cuh` (mixed) as their GEMM headers. #21 patched both files with identical hunks; every change below is
therefore "apply twice" unless marked. Line numbers are origin/master's.

Order of landing (see PR.md "Commits"): PR 2 first (it carries 535e80c, the T-invariant partition, and `attn_core_q8`), then
this PR. The wv prefill readers (`LAY=1`) must follow PR 3's `HalfW` restructuring of `mmh` (bd8a22e is later in our history
than a675751, so our wv reader diff was written against the pre-S4 `mmh`).

Effort scale: S = under a day incl. gates, M = 1-3 days, L = a week or more (compile and gate cycles dominate: a full
`qwen38_mixed.cu` build is about 11 GiB of RAM and minutes; MK_WV builds are about 25 GB).

## Map

| # | change (our commits) | lands in | engines | collides with | sm_75 | effort / risk |
|---|---|---|---|---|---|---|
| 1 | Sampled drafts: reconcile with #21 (b9349ce H1 part, 1e23ba6) | `qwen38_mk.cu` + `qwen38_mixed.cu`: `Launch::draft_prob` (L170 / L172), `residual_draw` (L1773 / L1898), `sample_targets` (L1804-1835 / L1929-1956), `mtp_pass` (L2014 / L2137), `Impl::draft_prob` | both | #21 af416bd / 6f11649 (same scheme, see PR.md table) | none (device math only) | S, low. Port only the delta: fallback excludes d; keep `MK_DRAFT_SAMPLING` |
| 2 | Confidence gate + runtime policy + logging (b9349ce, 943c1bc) | `Ctl` (`pad[8]` -> `dconf[8]`, `dchain`, `nd_prod`, `nd_stop`; with `MAXT` 5 `dconf[4]` is enough), `Bufs` (`blk_lse`, `dlog`), `Launch` (`gate_drop[]`, `gate_stop[]`, `gate_chain`), `lm_head` (`LSE` template flag, `LM_LSE/LM_BEST/LM_WLSE` misc slots), `mtp_pass`, `mk_cycle`'s draft loop (`!vc->nd_stop`), `Impl::Policy`, `launch()`, `step()` (L3144 / L3417), `prefill()` (`nd_avail` from `c.nd_prod`), `MKEngine::set_policy`, `capi.cpp` (+19: `mk_set_policy`, `mk_last`), `qwen38_mk.h` (+4: `set_policy`, `MKStats::last_*`), optionally `server/mk_server.py` (pass `MK_POLICY`, log `mk_last`) | both | #21 touched the same lines: `Launch` tail, `lm_head`'s extra `float2 * out` argument (ours: `float2 * cand`, same position, same `(out ? out : cB.cand)[...]` store), `mtp_pass(const Launch & P, ...)` (identical new first parameter), `hb.cand` allocation (`2 * MAXT * TOPK_MAX`), `launch()` initialiser. Textual conflicts, no semantic ones. Upstream's `__syncwarp()` before the `__shfl_sync` of the mapped token stays | none (host + `__expf`) | M, medium. Re-gate: greedy ids at `-d 0/4` unchanged, KLD byte-identical, mk_cycle stack frame and `mtp_pass<R>` spill counts (ours 24-36 B st; #21's message says extra local memory pushed the 144 MiB DeltaNet checkpoint past free memory at 149K, so check `MK_MEMLOG` free at 149K) |
| 3 | Depth rule: `depth=1` default, `chain=1`, `stop {0,.4,.4,.4}`, `drop4=.4` | host `Impl::Policy` defaults in each engine | both | #21 leaves `step()`'s `4 -> 3` rule in place | none | S. Needs #2 and PR 2's T-invariance; without PR 2 the gate makes greedy text depend on the gate's decisions |
| 4 | f58e21e sampled lm_head epilogues (bitonic merges, SM_CAND / SM_DQ staging) (moved in from PR 2) | `lm_head` TOPK epilogue, `sample_targets`, `draft_sample` replacement | both | #21's `dcand` in global memory and `residual_draw` reading `qd[]` from global memory: our version stages `q` in shared memory (SM_DQ). Rewrite for #21's data layout, keep `tk_merge` / `tk_tree16` / `tk_grid_chain` | `SM_DQ` / `SM_CAND` need shared memory; Turing's 64 KiB `SM_TOTAL` assert (`SM_TOTAL + 16 <= 64 * 1024`) must still hold; likely fine, check | M, medium. Exactness test: sampled tokens identical for fixed seeds vs the unoptimised build |
| 5 | MTP block re-quantised to Q4_K (17e775f, `MK_MTP_Q4`) | each engine's loader `want()` / re-encode path (upstream already calls `ggml_quantize_chunk`, L2274-2281 in `qwen38_mk.cu`, for its own re-quantisations) | both (plain: 3 matrices on Q4_K_M; mixed: 8 on UD) | upstream plain engine's `Impl` MTP weights type switch (`resid32<R, 10, T_Q8_0>` for `eh`, `fd.type == T_Q6K` branch at L2007): after the change `eh` and `fd` are Q4_K, so the Q8_0 / Q6_K instantiations used only by the MTP block become dead in the MTP path (still used by the trunk? check: `eh` is the only Q8_0 resid in the plain engine -> its instantiation can be dropped, saving compile time) | none | S-M, low. Policy key `mtpq` only if `MK_MTP_Q4=ab` is kept (can be dropped to save code) |
| 6 | Draft head through `draft_ids` (d487e7e) | `lm_head` prefetch + row pointer, `dv_add` (L2200-2215 in plain), `head_d` setup (L2509-2540), delete `gather_rows` | both | #21's `cW.draft_ids[id]` mapping of the TOPK result: compatible, d487e7e keeps `r` as draft-row index and maps only the output | none | S, low. Exact (output bit-identical) |
| 7 | Embedding prologue / MTP M0 over 10 blocks per row (c0a1783) | `embed_prologue`, MTP `M0` phase | both | upstream pinned-embedding change (#20, `embd_host` staged rows): check the staged-row path still lets block s stage "Q4_K super-blocks 2s, 2s+1" | none | S, low |
| 8 | wv tensor-core verify GEMM (820b77b ... 2292859): new `wv.cuh` (1,132 lines at fabce52), `wv_host.cuh` (238 lines), loader re-layout, per-phase wv tiles, `PairOut` | new headers; `qwen38_mixed.cu` first (all types), then `qwen38_mk.cu` with the Q4_K / Q5_K / Q6_K / Q8_0 subset | both, mixed first | `gemv_pairs` / `resid32` call sites (`resid32<T,6,T_Q4K>(W.o...)` at L1910 in plain) get wv alternatives under `#if MK_WV`; mixed has `gemv_pairs_any` / `resid_any` / `ffn_gu_any` dispatchers where the switch goes. Upstream mixed converts Q3_K / IQ3_S to Q6_K and IQ4_NL to Q8_0 at load (README "The GGUF"), so wv needs no native Q3_K / IQ3_S path until PR 6 lands native types | **sm_80+ only** (`mma.m16n8k32` s8). Plain engine on Turing must compile with `MK_WV=0`; guard `#if MK_WV && !QWEN_MK_SM75` and make the Makefile / `megakernel_build.py` refuse the combination | L, high. Biggest item: build memory (ptxas about 25 GB merged), 1.5M PTX lines; keep `PairOut`. Gates: tinv d0 == d4, KLD mixed + 32k, prefill-path KLD byte-identical, A-B-A |
| 9 | Prefill reads the wv layout (15c8003, dc91306, a675751 `prefill_gemm.cuh` parts; 9fed197 `FM6` stage reader) | `prefill_gemm.cuh` (plain) and `prefill_gemm_mixed.cuh` (mixed): `mmh<..., LAY=1>` | both | PR 3's pipelined `HalfW<TYPE, LAY, ROWS, ...>` (bd8a22e) which already has a `LAY` parameter; our `LAY=1` reader diff must be re-expressed on top of it. Also upstream's `prefill_gemm_mixed.cuh` is a separate 451-line file | sm_80+ (cp.async 16-B FMT stage) | M-L, high; exactness: prefill-path KLD byte-identical to a `MK_WV=0` build |
| 10 | `MK_MAXT` 6..8 (dc91306, a675751 `dctx`, a5631b7 g7, 624803c) | constants (`MAXT`, `CONV_RING` 16, `SM_CAND`, gate arrays `[MAXT - 1]`), `run_wide`, `lm_head` misc slots at `TL = 8`, `dn_core` `NE 15`, `Policy::{drop_last, dctx, g7}`, host `step()` cap | both | upstream `constexpr int MAXT = 5` (L52 / L55) and all `[MAXT]` arrays; `#if MK_MAXT > 5` blocks | sm_80+ | L, high. Only after #8. Needs `attn_core_q8` (PR 2) and the all-q8 KV cache; refuse `-d > 4` otherwise |
| 11 | `attn_core_w48` one-pass 48-row attention (447a87c, 805ce7e) | after PR 2's `attn_core_q8`, in each engine | both | PR 4's `MIX` (low-bit) template parameter: our `attn_core_w48<T, MIX>` has a MIX form that needs the kv-cache PR; port the `MIX=false` form first | **sm_80+**: 99,072 B shared memory (upstream Turing `SM_TOTAL <= 64 KiB`), `TP == 32` | M-L, high (register/spill sensitivity: `MK_W48S` is a spill fix; check ptxas per function after porting) |
| 12 | `MK_WV_FMK`, `MK_WV_OUT`, `MK_WV_Q6F` (2292859, 22bda66, 9fed197) | `wv.cuh`, loader | both (Q6F: plain engine has Q6_K ffn_down and, on Q4_K_M, Q6_K elsewhere) | none | sm_80+ | M. `MK_WV_OUT` is measured slower; leave it out of the upstream port (saves a build variant) |
| 13 | `MK_MTP_KVC`, `blk=1`, `chain=2` (624803c, a5631b7) | `mtp_kv_pass<1..3>`, `sample_targets_blk` | both | MAXT > 5 builds only | none | M, medium. `blk=1` and `chain=2` are opt-in, default off and within noise live: consider dropping them from the first upstream version (smaller diff, fewer instantiations; `sample_targets_blk` raised mk_cycle's stack to 1360 B before a5631b7) |
| 14 | `MK_SPEC_LOG` + `MK_SPEC_LOG_FILE` | `MKEngine::Impl::slog_*`, `dlog` | both | none | none | S. Compile switch only |
| 15 | bench: `bench/wvgemm_bench.cu` (2,296 lines at fabce52), `wv_ref.cuh`, `wv_extract.py` (38521b7, 5e653f7, 9fed197 part) | `megakernel/bench/` | n/a | upstream has no wv code to compare against | sm_80+ | S to include, or leave out: it is a 2.3k-line microbenchmark whose purpose is the go / no-go for #8. Suggest leaving it in this tree's history, or put it in the bench-suite PR |

## Specific things the port must check

1. **#21 vs H1** (items 1-2): after dropping H1, `Ctl::dsmp` is not needed to hold q, but should be kept (or replaced by a
   stored per-launch flag) so a launch verifies drafts exactly as they were made. In #21 the verifier branches on
   `P.draft_prob` of the current launch and `dcand` holds whatever the last drafting launch wrote. If `set_sampling` can
   change temperature between the launch that made drafts and the launch that verifies them (checkpoint restore, exact
   reuse in PR 5, `--batch`), a stale `dcand` is read. `qx == 0` makes a stale row reject safely, but a stale non-zero q
   is not exact. Our `dsmp` flag closes this. Not measured; found by reading.
2. **Dedupe vs sum**: if #21's dedupe is kept, the gate's `LSE` still sums both copies of a padded token's logit (the GEMV
   loop runs over rows). The effect on `conf` is negligible (one duplicated row in at most 32,768+) and was present in our
   gate runs; do not change the LSE code to match the dedupe.
3. **Policy default flip** changes behaviour for every user at once (depth=1 + gate). Re-measure on the Q4_K_M plain engine
   with an A-B-A before posting (our greedy gate gain on Q4_K_M is open; sampled A/B on Q4_K_M not done).
4. **Ctl layout**: `Ctl` is copied to the host (`hctl`) and read by `capi.cpp` / `main.cu`; adding fields moves `pad`. Both
   engines must keep identical `Ctl`/`Bufs` layouts since `capi.cpp` and `qwen38_mk.h` are shared by both libraries.
5. **Python side**: `src/megakernel_build.py` / `src/llama_server.py` pick the library by tensor roles/types. `-DMK_WV`,
   `-DMK_MAXT` need a build-variant knob. Proposal: `make MK_DEFS="-DMK_WV=1 -DMK_MAXT=8"` (a one-line Makefile change:
   `NVFLAGS += $(MK_DEFS)`) and a documented opt-in in `openjet setup`; do not enable by default on machines with less
   than about 32 GB of RAM. The server's `--draft` default (4) and max must follow `MAXT - 1`; `mk_new` already throws for
   `n_draft > MAXT - 1`.
6. **Upstream tests**: `tests/test_megakernel_variants.py` checks the engine routing; a build with `MK_WV`/`MAXT` variants
   needs either a third library name (`libqwen38mk_wv.so`) or must be treated as a compile option of the two existing ones.
   The former keeps Q4_K_M users' default untouched.
7. **Turing**: items 8-12 must be compiled out for `QWEN_MK_SM75`; items 1-7, 13, 14 are architecture-neutral. The plain
   engine's Turing build has `ATP = 16` (tile size), so the T-invariant partition (PR 2) uses `ATT_BUCKET = 20 * ATP = 320`
   there; the host cap `att_cap_T` is generic.
8. **Exactness gates to reuse** (scripts in notes/spec, notes/wvplumb, notes/dattn2/tinv_q8.sh): greedy ids at `-d 0/4/7`
   on 4 prompts (use the q8 cache, 8k-ctx fp16 runs do not exercise the q8 attention), KLD mixed / 32k vs llama.cpp
   reference, sampled exactness (`exact_test.py`, `pit_test.py`), default-build byte-identity.

## Suggested split for the actual upstream submissions
Because items 8-12 are the bulk and the only ones that need a new build variant, a safer upstream sequence is:
(a) items 1-7 + 14 as the "draft policy" PR (default-on, small, no new build variants), after PR 2;
(b) items 8-9, 12 as "wide verify, `MK_WV=1`" (opt-in, `MAXT` 5);
(c) items 10-11, 13 as "seven drafts, `MK_MAXT=8`".
This is a suggestion to main; the PR text in PR.md describes the whole set.
