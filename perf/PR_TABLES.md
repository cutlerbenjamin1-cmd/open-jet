# Tables for PRs / the upstream README (same numbers as REPORT.md)

RTX 3090 (24 GB), Qwen3.8-27B. llama.cpp = tuned b11548 (`--spec-type draft-mtp --spec-draft-n-max 4`, q8_0 KV, `-fa on`,
`-b 2048 -ub 1024`), best of three sessions. megakernel = fabce52 built with `-DMK_WV=1 -DMK_MAXT=8 -DMK_WV_Q6F=1`,
`-d 7 --kv q8 --exact-reuse`, `MK_PF_CAP=1024`.

Decode and prefill, UD-Q4_K_XL, 120k context:

| | llama.cpp | megakernel | |
|---|---|---|---|
| greedy decode, short prompts | 82.7 tok/s | 144.7 tok/s | 1.75x |
| sampled thinking decode | 65.8 tok/s | 129.8 tok/s | 1.97x |
| decode at 38k / 101.6k tokens | 62.3 / 53.4 tok/s | 109.9 / 102.0 tok/s | 1.76x / 1.91x |
| prefill 38k / 101.6k tokens | 29.6 / 101.0 s | 23.0 / 79.0 s | 1.29x / 1.28x |
| time to first token, short prompt | 124 ms | 71 ms | |
| re-answer after editing an earlier message (11.7k chat) | 0.33 s | 0.08 s | |
| VRAM, whole card | 22.8 GB | 21.1 GB | -1.7 GB |

The author's `bench/code_bench.py` / `bench/think_bench.py`, plain Q4_K_M:

| tok/s | upstream 2993f47, `-d 4` | this branch, `-d 7 --kv q8` | |
|---|---|---|---|
| writing code | 133.8 | 173.3 | +29% |
| 30K tokens of code in context | 87.4 | 117.5 | +34% |
| short prompts (mean of 4) | 118.0 | 149.2 | +26% |
| reasoning mode | 89.1 | 125.4 | +41% |
| prefill, 29.6K tokens | 1,527 | 1,811 | +19% |

Long context with `MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048` (preset D'), `-c 262144`:

| | UD-Q4_K_XL | plain Q4_K_M |
|---|---|---|
| prefill 260,896 tokens | 337 s | 333 s |
| decode at 261k depth | 62.1 tok/s | 61.8 tok/s |
| follow-up turn at 261k (prompt reused) | 2.2 s | 2.2 s |
| VRAM, whole card | 23.6 GB | 23.3 GB |
| needle in a haystack at 190k / 261k | 4 / 4 | 4 / 4 |

Quality, 282 questions (GSM8K, MMLU-Pro, ARC-C, MATH-500, HumanEval, NIAH to 100k, LongBench v2), greedy, same scorer:
llama.cpp 229, megakernel UD 230 (D' 230), plain Q4_K_M 231 (D' 231). D' matched q8 on every item.
