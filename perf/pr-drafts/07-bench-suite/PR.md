# bench: head-to-head, quality and long-context benchmark suite (megakernel/bench/suite)

Branch: `bench-suite` on origin/master 4d4db90 (8 commits, 21 files, +2404 lines, nothing outside `megakernel/bench/suite/`, `tests/test_bench_suite.py` and one `.gitignore` line).

## Summary

Adds a benchmark suite that sends the same requests to the megakernel server and to llama-server (or to two megakernel builds) and compares speed, time to first token, prefill, decode at depth, prompt reuse, VRAM, answer quality and long-context retrieval. It also adds a config-driven runner that starts and stops the servers one at a time, scorers for the quality sets, a max-context probe and client, a CLI benchmark that reuses `code_bench.py` and `think_bench.py` on several builds, and the small correctness tools (KLD against a llama.cpp reference, T-invariance, state-file decode A-B-A). No dataset file is committed; the fetch and prepare scripts download them and name the sources. Nothing contains a machine path: models, binaries, data and servers are arguments, config entries or environment variables. It does not touch engine code and does not duplicate `bench.py`, `code_bench.py`, `think_bench.py` or `prefill_bench.py`.

## What each script measures

| Script | Measures / does |
|---|---|
| `h2h_client.py` | Against any OpenAI-compatible server: TTFT (3 one-line questions, 1 token), greedy decode (4 prompts, 512 tokens), sampled thinking decode (temperature 1.0, top_p 0.95, top_k 20, seeds 11 and 22), prefill of 11.6k / 38k / 101k-token prompts, decode at 11.6k / 38k / 101k depth, follow-up turn (prompt reuse), edit of an earlier user message and of the end of a system prompt (each with a `cache_prompt: false` rerun), whole-card VRAM idle and after long prompts, GPU clocks / temperature / power at start and end. Uses the server's own `timings`. |
| `make_prompts.py` | Builds `pf38k.txt`, `pf64k.txt`, `pf101k.txt` from source files (default: the repo's megakernel sources); pf101k starts with pf38k so the shared-prefix test is meaningful. |
| `run_suite.py`, `example_config.json` | Starts one server per run from a JSON config, waits for `/health`, runs `h2h` / `qa` / `maxctx` clients, stops it. Refuses to start when the port is busy; `--dry-run`, `--list`. |
| `h2h_summ.py` | One comparison table from the JSONL records; several sessions of one system can be pooled (best value per metric, all values kept in the JSON); max-context summary. |
| `qa_fetch.py`, `qa_client.py`, `qa_score.py` | Quality set: GSM8K 50, MMLU-Pro 70, ARC-Challenge 50, MATH-500 40 (+10 level-5 in thinking mode), HumanEval 41; greedy, same prompts for every engine; offline scorer with per-set correct / n and paired agreement. |
| `qa_prep_long.py` | Long-context items sized with the model's tokenizer: NIAH multi-key (War and Peace haystack, 4 needles, asks one; ~32k / 64k / 100k and up to 261k tokens) and LongBench v2 (12 items to 108k tokens; 125k-258k candidates). |
| `maxctx_client.py`, `ctx_probe.sh` | Probe which context loads (KV line, whole-card VRAM peak); at the largest context: prefill from scratch, decode at max depth, a follow-up turn, VRAM, NIAH and LongBench answers. |
| `cli_bench.py`, `gpu_keep.cu` | `code_bench.py` / `think_bench.py` megakernel halves over a list of builds / settings in one session; optional clock keeper during the CLI model load (source only, built by the user). |
| `kld_compare.sh`, `ref_ppl.py`, `tinv_batch.sh` | KL / perplexity / top-1 against a `tools/llama_ref` reference; two quantizations from two reference files; greedy output identical across draft depths. |
| `decode_state.sh`, `state_aba.sh` | Decode speed at 38k / 64k / 101k context from a post-prefill state file; A-B-A across two builds. These need `--state-dir` (PR 6); the `dp` form needs the low-bit KV cache (PR 4). |

## How to run

```
cd megakernel/bench/suite
python3 make_prompts.py --out bench_data/prompts
python3 -I qa_fetch.py                                   # downloads into $BENCH_DATA (default ./bench_data)
cp example_config.json my.json                           # set vars.model, vars.llama_server, ...
python3 run_suite.py my.json --dry-run                   # check the commands
python3 run_suite.py my.json                             # h2h + qa for each run
python3 h2h_summ.py --system mk=benchmark_results/suite/h2h.jsonl:mk --system llama=benchmark_results/suite/h2h.jsonl:llama
python3 -I qa_score.py benchmark_results/suite/qa.jsonl --pair mk llama
```

README.md in the directory has the full tables, the environment variables and the long-context / CLI sections.

## Tests

`tests/test_bench_suite.py` (17 tests, pure Python, no GPU / network / server, modelled on `tests/test_think_bench.py`): command construction and placeholder substitution of the runner, busy-port and fail-fast health handling, the example config, request records of the client (mocked HTTP), summary pooling (max for tok/s, min for times), the scorers, the paired table, HumanEval execution in an isolated interpreter, resume logic, prompt builders (byte-exact sizes, pf101k starts with pf38k), NIAH prompt structure, CLI-wrapper behaviour. `python -m unittest tests.test_bench_suite`: 17 tests OK. All files pass `py_compile`. The shell scripts pass `bash -n`; `gpu_keep.cu` is not compiled in this branch.

## Results

See RESULTS.md on the `qwen38-perf` branch (perf/RESULTS.md).

## Commits

See commits.txt (8 commits on 4d4db90). Lineage: the scripts are generalised copies of the ones used for the project's own head-to-head runs (h2h2 / sweep3 / qa / km / final in the notes); the measurement logic (requests, prompts, seeds, scorers, selection of items) is unchanged, only paths, process control and the argument handling changed. fb1639b (listed with this PR in INDEX.md) is the launcher-side commit and is not part of this branch; the branch contains only new files.

## Notes for the reviewer

* New files only; zero engine changes. Safe to merge independently of PRs 1-6.
* Left out on purpose: the machine-specific parts of the original runs (WSL wrappers, a GPU lock script, Windows llama-server paths), the dashboards, and every build / deploy script. Holding the GPU is left to the caller (`BENCH_WRAP`, or wrap the runner).
* Original scripts hard-coded two systems per runner; the config replaces `h2h2_run.py`, `sweep3_run.py`, `qa_run.py` and `final_run.py` (their system lists become `runs`).
* Item selection in `qa_fetch.py` reproduces the original lists only when the default set order is fetched together (one seeded generator).
* `qa_fetch.py` license column is from the dataset cards as remembered when written; check them before redistributing item files (none are committed).
* Run times per system are not stated here. Nothing was run on a GPU for this PR.
* Follow-ups: a Turing (sm75) variant of `gpu_keep.cu` is only a `-arch` flag; `maxctx_client.py` item fitting assumes `max_tokens` 64 / 1024.
