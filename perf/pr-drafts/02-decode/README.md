# README draft for PR 2 (megakernel/README.md)

Written against origin/master 4d4db90's megakernel/README.md. Edits, each with the heading it goes after. The attention and
weights text is taken from our README at fabce52 where it is right, adapted to upstream's two-engine wording. Numbers are
RTX 3090, sources in PR.md.

---

## Edit 1: "How it gets its speed" - replace the first bullet's Q6_K clause and the "Attention" bullet

First bullet, replace "Q6_K and Q8_0 are re-laid-out at load time (values unchanged) so every lane does aligned 16/8-byte
loads; Q6_K blocks carry their scales and `d` in one 8-byte word." with:

```markdown
Q6_K and Q8_0 are re-laid-out at load time (values unchanged) so every lane does aligned 16/8-byte loads; a Q6_K row is two
streams, 192 B of values per super-block and an 18 B per block stream of scales and `d` (210 B per block, the GGUF size).
```

Attention bullet, replace the whole bullet with:

```markdown
- **Attention**: split-K flash decoding with tensor-core MMAs (Q.K^T and P.V in fp16 with fp32 accumulation, online softmax)
  over 32-position K/V tiles. The chunk boundaries depend on the context length only (chunks are whole tiles of
  `32 * ceil(L / 640)` positions, and a launch never straddles a multiple of 640 positions), so a row's float summation
  order does not depend on how many tokens the cycle verifies and greedy output is the same at every draft depth. With the
  q8 cache (`attn_core_q8`) each block runs a three-deep software pipeline with one barrier per tile (S of tile j, softmax of
  tile j - 1 and P.V of tile j - 2 on specialised warps); the raw q8 K and V tiles stream through `cp.async` rings and both MMA
  operands are converted from int8 in registers. Same arithmetic as the simple loop (bit-identical output), about 2x faster
  at 100K context: 650-750 GB/s of KV per call; decode +18.7% at 101.6K depth, +7.3% at 38K (`bench/dattn_bench.cu`).
  Chunks of a single tile (context <= 640) keep the simple loop, which has no pipeline to fill. The pipeline needs 92 KB of
  shared memory and the q8 cache; the fp16 cache and sm_75 use the simple loop.
```

## Edit 2: "Build / run" - CLI/env paragraph

Add to the end of the flags paragraph (after `MK_DRAFT_SAMPLING`):

```markdown
`MK_DATTN=0` (q8 cache: the previous, unpipelined decode attention; same output), `MK_STACK_LIMIT=<bytes>` (CUDA stack-size
limit per thread; default 0, since the kernels need no dynamic stack, which saves about 220 MiB of reserved local memory),
`MK_AB_F32=1` (mixed engine: keep `ssm_alpha` / `ssm_beta` as fp32 rows, see The GGUF).
```

## Edit 3: "### The GGUF" - mixed-engine note (after the sentence "OpenJet selects the mixed engine ...")

```markdown
In the mixed engine UD-Q4_K_XL's Q8_0 `ssm_alpha` / `ssm_beta` are kept as int8 rows with fp16 scales (the kernels form
`(q * d) * attn_norm`, which is exactly the folded fp32 value the loader used to build; 66 MiB less VRAM, bit-identical).
Files that store them as F32 (the plain Q4_K_M) are unchanged.
```

## Edit 4: "## Results" / benchmarks (only if upstream wants a table; otherwise skip)

Suggested text for the engine-level numbers, all UD-Q4_K_XL on an RTX 3090, 120K ctx, q8 KV, `-d 4`, A-B-A:

| decode tok/s | before | after |
|---|---|---|
| 101.6K tokens of context | 104.25 / 103.67 | 123.40 (+18.7%) |
| 38K tokens of context | 114.21 / 113.95 | 122.38 (+7.3%) |
| short context | unchanged (+0.5% with the one-tile routing) | |

Measured with UD-Q4_K_XL (the mixed engine). The plain Q4_K_M engine gets the same attention kernel; it was checked for exactness only, not re-measured with the q8
cache at long context.

## Edit 5: "## Files"

Add: `bench/dattn_bench.cu` - decode attention microbenchmark (`attn_core_t` against `attn_core_q8` on a random q8 cache:
bitwise comparison of the partials, T-invariance check, time per call).
