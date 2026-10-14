# Upstream state (fetched 2026-10-14; origin/master = 4d4db90)

Since our fork point 2993f47 upstream merged our #19 / #20 and much more (git log
2993f47..origin/master; diff with `git diff 2993f47 origin/master -- <path>`):

- **210eda0 "Merge pull requests #19 and #20 with model-specific mixed-quant builds"**: upstream did NOT keep one engine.
  megakernel/qwen38_mk.cu (3163 lines) stays the plain Q4_K_M engine; a new megakernel/qwen38_mixed.cu (3436 lines,
  derived from our #19) is "the heavier mixed-quant engine"; prefill_gemm_mixed.cuh (451 lines); the Python side routes by
  GGUF tensor roles / types to the right library variant (src/megakernel_build.py, src/llama_server.py, tests
  test_megakernel_variants.py). Pinned embeddings + MK_PF_CAP (#20) were applied to both engines.
- **#21 (iiLaurens, af416bd / bf10f5e + 6f11649 by the maintainer): "sample MTP drafts and verify by rejection"**: sample
  the draft from q, accept with min(1, p/q), else draw from max(0, p - q); default probabilistic at temp > 0
  (MK_DRAFT_SAMPLING=greedy restores point-mass); draft top-32 via lm_head's TOPK path; dedupe draft candidates.
  OVERLAPS our b9349ce / 943c1bc / 1e23ba6 (H1 sampled MTP drafts + residual fallback, 10-11) - our PR 1 must say how
  ours relates (same scheme? differences: confidence gate, runtime policy, depth policy) and propose reconciling, not
  duplicating.
- **#18 Turing sm75 support (Alex Gorevski: 45031ed, d71c3a1, fca6328, 5397a05 cuda_compat.h)**: our kernels (mma
  m16n8k32 int8, cp.async, etc.) are sm_86-only in places; each PR notes whether it needs an sm75 fallback / guard.
- Packaging and naming: megakernel/pyproject.toml, src/agent_setup.py, src/cli.py, engine naming (dc458b3), docs, tests
  (test_prefill_bench.py, test_think_bench.py: upstream now has tests for its bench scripts).
- megakernel/README.md: +186 lines upstream (two engines, Turing, benchmarks). Our README at fabce52 also grew; README
  drafts must start from upstream's current README (git show origin/master:megakernel/README.md) and add our sections.

Consequence: our tree (one 8,499-line qwen38_mk.cu serving Q4_K_M AND UD) cannot be stacked onto origin/master
as-is. Porting the code into upstream's two-engine layout needs compiling and GPU gates, which this drafting phase does
did NOT do (no compiling). This phase produces, per PR: the description, the README text, the
commit list, and a porting map (which upstream file / function each change lands in, what conflicts, what overlaps
upstream already has). The bench-suite PR (pure Python / shell, new files) is staged now as a real branch on origin/master.
