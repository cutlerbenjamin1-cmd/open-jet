# README draft for PR 1 (megakernel/README.md)

Written against origin/master 4d4db90's megakernel/README.md. Five edits, each with the heading it goes after. Text that
is already in upstream's README is not repeated. Our README at fabce52 documents none of the policy / wide-verify items
(only the T-invariant partition sentence and `MK_MTP_Q4`, which PR 2 and this PR carry), so everything here is new text.
Numbers are RTX 3090 / UD-Q4_K_XL, sources in PR.md.

---

## Edit 1: "How it gets its speed" - replace the bullet "**Sampling** keeps the 32 best logits ..." and add two bullets after it

(Position: replaces the existing Sampling bullet; the two new bullets follow it. Upstream's `MK_DRAFT_SAMPLING` text in the
CLI-flags paragraph stays as it is.)

```markdown
- **Sampling** keeps the 32 best logits of each verified row. At temperature > 0 each MTP draft is sampled from the draft
  head's own distribution (the same top-k / top-p / min-p / temperature transform as the target) and verified by rejection
  sampling: accepted with `min(1, p/q)`, otherwise the token is drawn from `max(0, p - q)`, which keeps the sampled
  distribution exact (`MK_DRAFT_SAMPLING=greedy`: argmax drafts accepted with `p(draft)`). If the residual is empty (only
  through float rounding) the fallback draw excludes the rejected draft.
- **Draft depth by confidence**: the draft head also computes the log-sum-exp of its logits, so the device knows the
  MTP's own top-1 probability for each draft. A draft is kept while the running product of those probabilities stays
  above a threshold: after draft k >= 2 drafting stops when the product is below 0.4, and the 4th draft (the T = 5 verify
  row) is dropped when it is below 0.4. The decision is made on the device, inside the cycle, so there is no extra round
  trip. This replaces the fixed "4 drafts, 3 after a rejection" rule (+2.9% greedy, +11.7..13.5% sampled thinking on an
  RTX 3090). Policy keys are listed under [Draft policy](#draft-policy).
- **Greedy output does not depend on the draft depth**: the attention chunk boundaries depend on the context length only
  (whole 32-position tiles; a launch never straddles a multiple of 640 positions), so the float summation order of a
  row's attention does not depend on how many tokens the cycle verifies. This is what lets the gate change T freely
  without changing the text.
```

## Edit 2: "Build / run" - extend the CLI-flags paragraph

(Position: the paragraph that starts "CLI flags: `-d` drafts per cycle (0-4; ...". Change the `-d` clause and append the
env vars.)

```markdown
`-d` drafts per cycle (0-4, or up to 7 in a `MK_MAXT=8` build; with the default policy a cycle makes fewer drafts when the
draft head is not confident), ...
```

and append to the end of the flags paragraph:

```markdown
`MK_POLICY` (draft policy, see below), `MK_MTP_Q4=0` (keep the MTP block in its stored types, see
[The GGUF](#the-gguf)), `MK_MTP_KVC=0` (MAXT > 5 builds: catch-up rows before the last five run the full MTP pass instead
of K/V only), `MK_PROFILE_SEQ=1` (with `MK_PROFILE`: min time of every barrier position, i.e. per-layer phase costs).
```

## Edit 3: new subsection "Draft policy" (position: directly after the CLI-flags paragraph of "Build / run", before "### The GGUF")

```markdown
### Draft policy

`MK_POLICY` is a comma-separated list of `key=value`; the empty string is the default. It can also be set from C
(`mk_set_policy`) and read back with `mk_last` (T, drafts made and accepted by the last step).

| key | default | meaning |
|---|---|---|
| `depth` | 1 | 1: always up to `-d` drafts and let the gate decide; 0: `-d` minus one after a rejection (at least 3); 2: 1 + the number accepted last cycle (clamped to `dmin`..`-d`) |
| `dmin` | 1 | lower clamp of `depth=2` |
| `chain` | 1 | gate on the running product of the drafts' own top-1 probabilities (1), on each draft's own (0), or on the product of the drafted tokens' probabilities under the sampling distribution (2, sampled drafts only; no better offline) |
| `stop`, `stopK` | 0, 0.4, 0.4, 0.4 for drafts 1-4 | keep draft K and stop drafting when the gate value is below this (`stop` sets all K, `stopK` one) |
| `drop`, `dropK` | 0, 0, 0, 0.4 | drop draft K and stop when below this |
| `h1` | 1 | sample the drafts at temperature > 0 (see `MK_DRAFT_SAMPLING`) |
| `droplast`, `dctx`, `g7`, `blk` | see below | `MK_MAXT > 5` builds only |

The previous behaviour (4 drafts, 3 after a rejection, argmax drafts) is
`MK_POLICY="depth=0,drop=0,stop=0,chain=0,h1=0"`. The thresholds were fitted on logged cycles (a build with
`-DMK_SPEC_LOG=1` and `MK_SPEC_LOG_FILE=out.jsonl` writes one JSON line per cycle: the drafts, the MTP's top-32 and
log-sum-exp, the verify targets, and the main model's top-32 rows when sampling) and checked with live A-B-A runs on
UD-Q4_K_XL; they have not been re-tuned for Q4_K_M or other GPUs.
```

## Edit 4: "### The GGUF" - add after the table / the paragraph about `token_embd`

```markdown
The MTP draft block (`blk.64.*`, `nextn.eh_proj`) is the one lossy exception: its matrices stored wider than Q4_K
(Q5_K, Q6_K, Q8_0: all of them in UD-Q4_K_XL; `eh_proj` and `attn_v` / `ffn_down` in Q4_K_M) are re-quantised to Q4_K at
load (`ggml_quantize_chunk`, 1-3 s), so every MTP pass streams about 0.1 GB less (UD: -107 MiB of VRAM, greedy +1.5%).
Only the drafts change: the unchanged main model verifies them, so greedy output and the sampled output distribution stay
exact; acceptance moved by under a point. `MK_MTP_Q4=0` keeps the stored types (`MK_MTP_Q4=ab` loads both and the policy
key `mtpq=0|1` switches between them, for A/B tests). The draft head reads its rows of `lm_head` through the
draft-vocabulary id list instead of a gathered copy (-158 MiB).
```

## Edit 5: new section "## Wide verify and seven drafts (opt-in build)" (position: after "## Prefill", before "## Turing (RTX 8000)")

```markdown
## Wide verify and seven drafts (opt-in build)

A verify pass of T tokens reads every weight once; with the dp4a GEMV its cost grows with T (about 0.75 ms per row up to
T = 5, 1.6-2.1 ms per row beyond, where the loop becomes ALU-bound), which caps the useful draft depth at 4. Two
compile-time switches change that:

- `-DMK_WV=1` runs every main-pass projection of a cycle (DeltaNet in/out projections, attention projections, gate/up with
  the SiLU epilogue, `out`, `ffn_down`) as an int8 tensor-core GEMM (`mma.m16n8k32`, `wv.cuh`) over a weight layout that is
  rebuilt in place at load: Q4_K / Q5_K / IQ4_XS are fragment-ordered so every lane load is one 16-byte load, IQ4_NL /
  Q8_0 and the offset types tile-major. Time per verify row is nearly flat in T. It serves every T, `-d 0` included, adds
  about 11 s to model load and no VRAM, and prefill reads the same layout. The arithmetic order differs from the dp4a path
  (it is fixed and T-invariant, so greedy output is still independent of the draft depth), so output is not bit-identical
  to a build without it; KLD against llama.cpp is within noise. `-DMK_WV_Q6F=1` additionally puts every Q6_K matrix on it.
- `-DMK_MAXT=8` (needs `MK_WV=1`) lets `-d` go up to 7. T = 6..8 run as one padded 8-row pass; decode attention runs
  all 48 query rows (6 heads x 8 tokens) in one pass over the K/V tiles. Depths above 4 need the q8 KV cache (`--kv q8`).
  For depths above 5 the default gate thresholds are stricter (`g7`: stop below 0.25 after drafts 2-4, drop draft 5 below
  0.3, stop below 0.3 after drafts 5-6, never drop the deepest draft), and the MTP's catch-up rows before the last five
  only update their K/V (`MK_MTP_KVC`).

| RTX 3090, UD-Q4_K_XL, 120K ctx, q8 KV | `-d 4`, default build | `MK_WV=1`, `-d 4` | `MK_WV=1 MK_MAXT=8`, `-d 7` |
|---|---|---|---|
| greedy decode, short prompts, tok/s | 117.9 | 127.3 (+7.9%) | 138.7 (+17.6%) |
| sampled thinking, tok/s | 105.7 | 117.3 (+11.0%) | 126.3 (+19.5%) |
| decode at 38K, tok/s | 97.0 | 103.7 (+6.9%) | 102.7 (+5.8%) |
| decode at 101.6K, tok/s | 96.8 | 100.3 (+3.7%) | 105.7 (+9.2%) |

One session, fan at 100%. The first column is the previously deployed build (9f25c2f); the other two were built from the
next tree (2ea89b6), so they also carry whatever else changed between the two. Compiling an `MK_WV` build needs
considerably more host memory than the default build (ptxas peaked near 25 GB for the merged tree); `MK_WV` is not for
sm_75 (int8 `m16n8k32` and the 99 KB attention pipelines are sm_80+).
```

## Edit 6: "## Limitations" (optional one-liner)

```markdown
- Drafts above 4 per cycle need a `-DMK_WV=1 -DMK_MAXT=8` build and the q8 KV cache.
```
