# Benchmark suite

Scripts for comparing the megakernel with llama.cpp (or two megakernel builds) through the same requests: speed,
time to first token, prefill, decode at depth, prompt reuse, VRAM, answer quality, long context. They complement the
single-purpose scripts one level up (`bench.py`, `code_bench.py`, `think_bench.py`, `prefill_bench.py`), which this
suite does not duplicate; `cli_bench.py` runs the megakernel halves of `code_bench.py` / `think_bench.py` on several
builds in one session.

Nothing here knows a machine path. Models, binaries, servers and data are arguments, config entries or environment
variables. No dataset or prompt file is committed; the fetch / prepare scripts download or build them.

Requirements: Python 3.10+ (standard library only), `nvidia-smi` for the VRAM / clock records. `qa_fetch.py` uses
`pandas` + `pyarrow` for HumanEval when present (otherwise the Hugging Face rows API). `qa_prep_long.py` needs
llama.cpp's `llama-tokenize`. Servers are `megakernel/server/mk_server.py` and `llama-server`, anything speaking
`/v1/chat/completions` with `/health` works.

## Layout and data

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_DATA` | `./bench_data` | item files and downloads (`<set>/items.json`, ...); keep it outside the repository |
| `BENCH_PROMPT_DIR` | `./bench_data/prompts` | `pf38k.txt`, `pf64k.txt`, `pf101k.txt` |
| `BENCH_GPU` | unset | `nvidia-smi -i` index for the VRAM / clock records |
| `BENCH_GGUF`, `LLAMA_TOKENIZE` | unset | tokenizer model and binary for `qa_prep_long.py` |
| `BENCH_GPU_KEEP` | unset | `gpu_keep` binary for `cli_bench.py` |
| `BENCH_WRAP` | unset | command prefix for each timed block of `state_aba.sh` (a GPU lock, for instance) |

`bench_data/` and `benchmark_results/` are git-ignored. All scripts append JSON lines (one record per request) and
every record carries the `label` of the system, so one file can hold many systems and sessions.

## 1. Head-to-head: `h2h_client.py`, `run_suite.py`, `h2h_summ.py`

```
python3 make_prompts.py --out bench_data/prompts            # once: pf38k / pf64k / pf101k from the repo sources
python3 h2h_client.py 8096 mk h2h.jsonl all                  # against a server that is already running
python3 run_suite.py my_config.json                          # or: start / stop the servers from a config
python3 h2h_summ.py --system mk=h2h.jsonl:mk --system llama=h2h.jsonl:llama
```

`h2h_client.py` sends the same requests to any server and records the server's `timings` plus wall time:

| Test names | Request | Measures |
|---|---|---|
| `warmup`, `warmup_clocks` | "Say hi." then a 512-token story | not reported; the second brings the clocks up |
| `ttft_short_*` | 3 one-line questions, `max_tokens` 1, thinking off | time to first token (`prompt_ms`) |
| `greedy_nothink_p*` | 4 chat prompts, 512 tokens, temperature 0 | greedy decode tok/s |
| `sampled_think_p*_s*` | the same prompts, thinking on, temperature 1.0 / top_p 0.95 / top_k 20, seeds 11 and 22 | sampled decode tok/s, draft acceptance if reported |
| `followup_turn1/2` | 30k-character prompt, then a follow-up turn on the same chat | prefill at ~11.6k tokens; prompt time of a follow-up (prompt reuse) |
| `long_pf38k`, `long_pf101k` | the 38k and 101k-token prompts | prefill time, decode at 38k / 101k depth, VRAM after each |
| `long_pf101k_fresh` | the 101k prompt again, `cache_prompt: false`, 16 tokens | 101k from scratch, when the first run reused the 38k prefix |
| `edit_*` | edit an earlier user message (11.7k-token chat), edit the end of a 9.1k-token system prompt; each followed by the same prompt with `cache_prompt: false` | prompt reuse when the change is behind the last turn |
| `vram_*`, `gpu_*` | `nvidia-smi` | whole-card VRAM idle / after long prompts; GPU temperature, clocks, power at start and end |

The mode `edit` runs only the warm-up and the `edit_*` scenarios.

`run_suite.py CONFIG [label ...]` starts one server per run (the command, environment and clients come from the
config; see `example_config.json`), waits for `/health`, runs the clients, stops the server and waits `settle_s`.
It refuses to start when something already listens on the port. `--dry-run` prints the commands, `--list` the labels.
To hold the GPU against other users, wrap the runner in your lock command.

`h2h_summ.py` builds the comparison table: `--system NAME=FILE:LABEL[+FILE:LABEL...]` (several pairs pool sessions
and take the best value per metric; the JSON keeps every value), `--maxctx FILE:LABEL` summarises a max-context run,
`--json` writes the rows.

Reading the numbers: the card state moves whole sessions by several percent (temperature, clocks); the clients log it,
and the repeat of one system at the start and the end of a session shows the drift. Compare systems within one session.

## 2. Quality set: `qa_fetch.py`, `qa_client.py`, `qa_score.py`

```
python3 -I qa_fetch.py                                       # downloads into $BENCH_DATA
python3 -I qa_client.py 8096 mk qa.jsonl                     # all sets (or name some: gsm8k arc_c ...)
python3 -I qa_score.py qa.jsonl --pair mk llama
```

Greedy, zero-shot, the same prompts for every engine, thinking off except `math500_think` (8192 tokens). Seeded
samples of public benchmarks:

| Set | Items | Source (license as stated on the dataset card when written; check before redistributing) |
|---|---|---|
| `gsm8k` | 50 | openai/gsm8k test (MIT) |
| `mmlu_pro` | 70 (5 per category) | TIGER-Lab/MMLU-Pro (MIT) |
| `arc_c` | 50 | allenai/ai2_arc, ARC-Challenge test (CC BY-SA 4.0) |
| `math500` / `math500_think` | 40 (8 per level) / 10 (level 5) | HuggingFaceH4/MATH-500 (subset of MATH, MIT) |
| `humaneval` | 41 (every 4th) | openai/openai_humaneval (MIT) |
| `longbench_v2` | 12, 16k-108k tokens | THUDM/LongBench-v2 (see its card) |
| `niah` | 9 (3 each at ~32k / 64k / 100k) | haystack: War and Peace, Project Gutenberg #2600 (public domain in the US) |

The scorer is offline and the same for every engine; it prints correct / n per label and, with `--pair`, the paired
agreement and the timing table. HumanEval completions are executed (model output): each in an isolated `python -I` in a
scratch directory under `$BENCH_DATA/humaneval`, 15 s timeout. Run the fetch, client and scorer with `python -I`.
At n = 40-70 per set a difference of a few items is within noise; read the paired columns, not only the percentages.

## 3. Long context: `qa_prep_long.py`, `maxctx_client.py`, `ctx_probe.sh`

```
export BENCH_GGUF=model.gguf LLAMA_TOKENIZE=llama-tokenize
python3 -I qa_prep_long.py long                              # longbench_v2/items.json, niah/items.json (for ~120k ctx)
python3 -I qa_prep_long.py max-niah 190000 248000 261000     # niah/items_max.json
python3 -I qa_prep_long.py max-lb                            # longbench_v2/items_max.json (125k-258k tokens)
ctx_probe.sh -b ./qwen38_mk -o probe -m model.gguf           # which context loads
python3 -I maxctx_client.py 8096 max_mk maxctx.jsonl 262144
```

* NIAH is RULER-style multi-key: four needles ("One of the special magic numbers for `<key>` is: `<7 digits>`.") at
  spread depths in the haystack, the question asks for one; prompts are sized with the model's own tokenizer.
* LongBench v2 uses its own no-chain-of-thought prompt; items are picked by token count with domains spread.
* `ctx_probe.sh` loads the CLI at each context and reports the KV line and the whole-card VRAM peak.
* `maxctx_client.py` prefills every item from scratch (a short unrelated request first) and records: prefill time at
  max context, decode speed at max depth, a follow-up turn on the same chat (prompt reuse at max context), VRAM, and the
  NIAH / LongBench answers (`qa_score.py` scores them as `niah_max` / `longbench_max`). Items that do not fit the
  context are skipped and counted in the first output line.

## 4. CLI benchmarks: `cli_bench.py`, `gpu_keep.cu`

`cli_bench.py --config engines.json` runs `code_bench.py`'s decode / context sweep and `think_bench.py`'s thinking
preset on each engine entry (binary, model, draft depth, `--kv`, environment) in one session and prints the table; see
its docstring for the file format. `--freeze-dir` keeps the generated prompt files so later sessions use identical
inputs. `gpu_keep.cu` is a spin kernel that keeps the GPU out of its idle clock state while the CLI loads its model
(`--gpu-keep ./gpu_keep`); build with `nvcc -O2 -arch=sm_XX -o gpu_keep gpu_keep.cu` for your GPU. It is killed when the
CLI prints its `[mk] load` line, before anything is timed.

## 5. Correctness tools

| Script | Use |
|---|---|
| `kld_compare.sh` | the megakernel (`-w mk -p prefill\|decode\|mixed`) or llama.cpp's batched path (`-w llama`) against a `tools/llama_ref` reference file: KL, perplexity, top-1 |
| `ref_ppl.py A.ref B.ref` | two quantizations of one model from their two reference files: perplexity of each, paired NLL difference with its standard error, top-1 agreement, KL(A to B) |
| `tinv_batch.sh BIN OUT` | greedy output at draft depths 0 and 4 (any list) must be identical; one process per binary when the CLI has `--batch` |
| `decode_state.sh`, `state_aba.sh` | decode speed at 38k / 64k / 101k context with the prefill restored from a post-prefill state file; A-B-A across two builds. Need a binary with `--state-dir`; the `dp` form needs the low-bit KV cache |

## What is not covered

* No GPU, build or server was started to prepare this directory: the tests (`tests/test_bench_suite.py`) are pure
  Python and cover the runner's command construction, the summary, the scorers, the prompt builders and the CLI wrapper.
* Quality items are small samples. They detect regressions of a few points, not differences of a point.
* VRAM is whole-card `nvidia-smi` memory, so other processes on the card count.
