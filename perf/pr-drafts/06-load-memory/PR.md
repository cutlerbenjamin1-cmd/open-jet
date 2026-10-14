# megakernel: fast model load, CLI --batch and state files, native Q3_K / IQ3_S / IQ4_NL, presence / frequency penalty, watchdog and CUDA error handling, memory accounting

## Summary

This PR makes the engine cheaper to start and safer to run. The model loads in about 8 s warm / 12 s cold instead of 32 s / 60 s (UD-Q4_K_XL, launcher flags) because the GGUF is read ahead by several threads and the weight upload and re-layout overlap the reading. The CLI gets `--batch` (many generation jobs in one process, one load) and `--state-dir` (the post-prefill state of a long prompt is saved once and restored later), which turns long-context benchmark runs from minutes into seconds. Three weight types stop being re-encoded at load (Q3_K, IQ3_S, IQ4_NL) and are read natively, bit-identical and smaller, and device buffers come from the CUDA memory pool, which together free about 1 GB on UD-Q4_K_XL. The server answers presence_penalty / frequency_penalty. A watchdog turns a hung GPU into a message and an exit code, and CUDA errors inside the library are returned as an error (HTTP 500) instead of killing the server. `MK_MEMLOG=1` prints a per-stage device-memory ledger.

## What changed

**1. Fast model load (d96f7d6, 2e604fd, 21523be).** Host side: `loader.cuh` (new) `Prefetch`: 8 threads read the GGUF ahead of the loader in its order, at most 3 GiB ahead (one thread's page faults through the mmap ran at ~0.2 GB/s on a cold file; 8 threads ~3 GB/s); host buffers reused without zero-fill; an upload worker thread copies while the next tensor is prepared; `wv_relayout` works pointer-based, straight from the mapping; the MTP Q4_K requant and the pinning of `token_embd` run on background threads; a load-phase line `[mk] load ...`. Round 2: `parallel_rows` on a persistent pool (`RowPool`), the `wv` weight layout made on the GPU from the staged bytes (`MK_LOAD_GPURL=1`, default; the gather tables come from running the host relayout on a self-describing probe), restore GPU warm-up after a state restore. The byte producers are unchanged; `MK_WDIGEST=1` prints a digest of every weight buffer and `token_embd` for A/B checks. CLI: `MK_CK_SLOTS` defaults to 1 in the CLI (8 pinned slots cost ~2 s per process; the server keeps 8, PR 5) and the tokenizer loads on a thread next to the engine.

**2. CLI `--batch jobs.txt` (d96f7d6).** One model load; `MKEngine::reset` between jobs; `set_n_draft` per job. One job per line, arguments as on the command line (shell-like quoting), `-o <prefix>` required per job (writes `<prefix>.out` / `.err`), parsed before the load so a bad line fails early.

**3. Post-prefill state files `--state-dir DIR` (d96f7d6, 2e604fd).** `MKEngine::prefill_state`: the batched prefill's state (K/V rows, window rings, DeltaNet, conv, h ring; saved before the last prompt token so one file serves every draft depth) is stored once per key (binary hash, model, ctx, KV format, every `MK_*` setting that changes numerics, prompt) and restored instead of re-prefilling. `--digest` prints a hash of the sequence state after the first prefill (tests). `MK_STATE_WARM_MS` (default 1000) spins the GPU for that long after a restore so the first decode does not run at idle clocks.

**4. Native weight types (89e9754, 9c4b08c).** IQ4_NL: 144-byte blocks (8 fp16 d + indices), same arithmetic as the Q8_0 re-encode it replaces (`MK_IQ4NL_Q8=1` = old). IQ4_XS in 272-byte block pairs without padding. Q3_K and IQ3_S: re-laid-out at load into a value stream in the GEMV's unit order plus a scale stream (Q3_K 114 B per 256 weights; IQ3_S 96 B + 16 B meta with the grid read through L1, 112 B) instead of 210-B Q6_K blocks; the decode GEMV (`rows_q3k` / `rows_iq3s`) and the prefill GEMM (`TQ3K` / `TIQ3S`) feed exactly the integers and scales of the re-encoded path, so results are bit-identical (`MK_REENC3=1` restores the re-encode); every row is checked against ggml's `to_float` at load. Unroll 2 for these types at IT > 8 (unroll 4 spilled up to 252 B).

**5. Presence / frequency penalty (9801b98).** `logit -= frequency * c + presence * (c > 0)`, c = occurrences of the token among the tokens generated in this response (device counts `Bufs::pcnt`, reset by every prefill; the prompt does not count), vLLM semantics. Applied to the main lm_head rows and the MTP draft head before argmax / top-k, so before temperature and the samplers. A verify row adds the drafts before it to the committed counts, so greedy output does not depend on the draft depth and the sampled distribution stays exact; correctly-rounded arithmetic (`__fmul_rn` / `__fsub_rn`). Penalty 0 runs the previous lm_head instantiations (`PEN` template flag): bit-identical, `mk_cycle` stack unchanged. API: `MKEngine::set_penalty`, `mk_set_penalty`, CLI `--presence-penalty` / `--frequency-penalty`; `mk_server.py` accepts `presence_penalty` / `frequency_penalty` in [-2, 2] (was ignored with a warning).

**6. Watchdog and CUDA error handling (5f69600, 78873a0, fa27118, 9297c43).** `wait_device()` records an event and polls it with a deadline after `mk_cycle` launches, prefill chunks and checkpoint copies: `MK_WATCHDOG` seconds per decode launch (default 60; prefill chunks and checkpoint copies get at least 120 s; 0 = the runtime's plain blocking waits), on expiry a diagnostic and exit code 3. It exists because a driver fault the WSL2 guest was not told about (nvlddmkm event 153) left the host spinning for 33 minutes. `MK_WATCHDOG_SPIN` = seconds of busy polling before sleeping between polls. `CK` throws `MKCudaError` instead of `exit(1)`; the C API returns -1 with the message (`mk_last_error`) and resets the engine, `mk_server.py` answers 500 and keeps serving; sticky errors (illegal address, launch failure / trap, ECC, or an error that survives clearing) print `[mk] the CUDA error is sticky ...` and `_exit(4)`. The CLI catches engine exceptions (a prompt over the context used to `std::terminate`) and exits 1 with `[mk] error: ...`. `MK_FAULT=cuda:<n>|sticky:<n>` is a test hook (the n-th prefill fails half-way). `hb.logits` (5 MB, scoring only) is allocated on the first scoring call.

**7. Memory accounting and pool (a730f00, 8585087).** `MK_MEMLOG=1`: bytes `cudaMalloc`'d vs the drop in free memory per load stage, `mk_cycle` local memory and limits. Device buffers come from the default memory pool (`cudaMallocAsync`) instead of one `cudaMalloc` each, so no 2 MiB rounding per weight tensor (`MK_CUDAMALLOC=1` = old). a730f00 also contains the `MK_KV_EMU` accuracy-experiment build flag; see PORT.md (it belongs with PR 4, or can be dropped).

**8. `bench/bench.py --dump` (21523be).** The regex `IDS:(.*)` matched inside `IN_IDS:` and dumped the prompt ids; now `^IDS:` with `re.M`.

| Variable / flag | Default | Effect |
|---|---|---|
| `MK_LOAD_IO` | 8 | read-ahead threads (0 = off) |
| `MK_LOAD_ASYNC`, `MK_LOAD_RQBG`, `MK_LOAD_PINBG`, `MK_LOAD_POOL` | on | 0 = upload worker / background MTP requant / background token_embd pinning / persistent row pool off (A/B switches) |
| `MK_LOAD_GPURL` | 1 | wv layout made on the GPU (0 = host) |
| `MK_LOADLOG` | off | 1 = print every load phase on its own line |
| `MK_WDIGEST` | off | 1 = print a weight digest |
| `--batch FILE` | | batch of generation jobs, one model load |
| `--state-dir DIR`, `--digest` | | post-prefill state files; state hash |
| `MK_STATE_WARM_MS` | 1000 | GPU warm-up after a restore (ms) |
| `MK_REENC3`, `MK_IQ4NL_Q8` | 0 | 1 = old Q6_K / Q8_0 re-encode (A/B) |
| `--presence-penalty`, `--frequency-penalty` | 0 | CLI; server: request fields in [-2, 2] |
| `MK_WATCHDOG` | 60 | seconds per decode launch (prefill >= 120); 0 = off |
| `MK_FAULT` | off | `cuda:<n>` / `sticky:<n>` test hook |
| `MK_MEMLOG` | off | 1 = device memory ledger |
| `MK_CUDAMALLOC` | 0 | 1 = one `cudaMalloc` per buffer |

## Results

RTX 3090, WSL2. "Launcher config" = `-DMK_WV=1 -DMK_MAXT=8` build (PR 1/2 features), UD-Q4_K_XL, `-c 120000 -d 7 --kv q8`.

| Measurement | Before | After | Source and conditions |
|---|---|---|---|
| CLI process wall (`-p hi -n 8`), launcher config, cold / warm file cache | 59.7 s / 32.4 s | 11.9 s / 7.4-8.3 s (round 1 alone: 9.8 s warm) | LEDGER "From agent_tool" tool2 part; `--timing` hold, 005c902 vs 21523be |
| same, default build (no MK_WV flags) | 43.1 s / 17.3 s | 10.2 s / 7.7 s | same |
| `mk_server` time to ready (`-c 120000 -d 7 --kv q8`) | 59.7 s cold / 30.1 s warm | 12.2 s cold / 8.5 s warm | same |
| load inside the engine, round 1 | ~31 s warm | 7.9 s warm | LEDGER "From agent_tool" |
| Remaining load costs (round 2) | | upload stalls ~2.4 s (worker at ~4.8 GB/s), server checkpoint-pool pinning 1.65 s, repack ~1.6 s; RowPool: no measurable gain | same |
| tinv q8 p0-3 d0/4/7 | 204 s as 12 separate runs | 86 s as one `--batch` | agent_tool (round 1) |
| 101.6k run, q8 | 91.5 s | 14.4 s with a state file | agent_tool G1; files 1.59 GB (38k q8) / 1.31 GB (38k D') / 3.94 GB (101.6k q8); save 2-5 s once, restore 0.25-0.55 s warm |
| 38k run | 36 s | 13 s | same |
| Decode right after a 38k restore | 117 / 166 tok/s without the warm-up | 168 / 168 tok/s with `MK_STATE_WARM_MS=1000` (plain prefill 166 / 167) | agent_tool tool2 |
| Weights, UD-Q4_K_XL, native Q3_K / IQ3_S | | -128 MiB at 120k | 9c4b08c message |
| Weights, UD-Q4_K_M, native Q3_K / IQ3_S | | -353 MiB at 120k | same |
| Native IQ4_NL; IQ4_XS pairs | | -238 and -175 MiB of UD weights (the VRAM track's table lists -246 / -175 at 120k) | 89e9754 message; agent_vram.md "FINAL RESULTS" |
| Memory pool instead of cudaMalloc per tensor | | ~-470 MiB on UD, speed 0, KLD identical | 8585087 message; agent_vram.md |
| Decode with U1 (native Q3_K / IQ3_S), A-B-A | 116.85 / 117.43 tok/s | 116.73 tok/s (-0.35%, accepted counts identical) | agent_unsloth.md "Results", bench.py -d 4, 4 prompts x 2, one `--timing` hold |
| `hb.logits` lazy | | -4.86 MiB allocated; 0 MiB measured at the server's free memory (pool granularity) | agent_ttft.md |

The VRAM track's whole stack (this PR's items 4 and 7 plus items in PRs 1, 2, 3, 5) took the server at ctx 120000, cap 1024, q8 KV, -d 4 from 23,451 to 21,681 MiB nvidia-smi peak (-1,770 MiB) with decode +1.4%: agent_vram.md "FINAL RESULTS" (the stack includes commits that belong to other PRs, so this is not this PR's share).

## Correctness

* Loader (round 1, d96f7d6): tinv q8 p0-3 d0/4/7 as one `--batch` == 12 separate runs == the reference outputs of the build before (ids and accepted counts); KLD mixed byte-identical to the build before; state restore == fresh prefill bitwise at 38k q8, 38k D' and 101.6k q8 (ids, accepted, state digest; `-d 4` restores the `-d 7` file).
* Loader round 2 (2e604fd, gated in 21523be): `MK_WDIGEST` equals the previous loader's for UD with the launcher flags (also with `MK_LOAD_GPURL=0` and `MK_LOAD_ASYNC=0`; 397 matrices re-laid out on the device, digest 3cbc3200857b9ea0), UD default (c7bbc323b1b19b80) and Q4_K_M with the launcher flags (66d07cb0ad4b7bb6; 400 on the device); ptxas: 0 changed functions; tinv `--batch` 12/12 ids == reference; 38k q8 restore d7 / d4 == plain. A unit test `test_wvdev.cu` compared the device relayout with the host: 16/16 identical (notes/tool/).
* Native Q3_K / IQ3_S (9c4b08c): UD-Q4_K_XL KLD mixed byte-identical to the build before, greedy ids identical at d0..d4 (4 prompts x 512 tokens), Q4_K_M ids identical, UD-Q4_K_M native == `MK_REENC3`; every row checked against ggml at load. ptxas identical between the gated and the committed source (0 changed functions, 832 B cumulative stack).
* Presence / frequency penalty: penalty 0 gives ids identical to the build before at d0 and d4 (prompts 0-3, accepted counts identical); presence 1.5: d0 == d4 on prompts 0 and 1 and different from penalty 0. Server: no crash; a "yes x40" prompt: presence 1.5 unchanged, frequency 1.0 stops after 11 words; sampled (temp 0.7, seed 42): presence 1.5 gives 35 words vs 38; penalty 0 again identical to the first run (counts reset per request); presence 3.0 -> HTTP 400. NOT done: the chi-square test of the sampled distribution under speculation (scipy was missing).
* Watchdog / errors: `MK_FAULT=cuda:2` over the C API: `mk_prefill` returns -1 with the message, engine reset, next prefill gives the same 48 ids as a clean run; `sticky:2`: message + exit 4; server: request 1 HTTP 200, request 2 HTTP 500 with the error text, request 3 HTTP 200 (non-sticky), connection dropped and exit 4 (sticky); CLI prompt over the context: `[mk] error: prompt exceeds context`, exit 1 (was `std::terminate`). Watchdog: ptxas 0 changed functions, output bit-identical (5f69600 message); `MK_WATCHDOG=0` ids identical (agent_ttft.md). The watchdog's expiry path itself (exit 3) was not exercised in the notes I read: not measured.
* Memory pool / MEMLOG: KLD byte-identical, greedy ids identical, UD and Q4_K_M (agent_vram.md batch 1-3 exactness).

## Commits

Squashed into this PR (12), oldest first: a730f00 MK_MEMLOG (+ MK_KV_EMU); 8585087 memory pool; 5f69600 watchdog; fa27118 watchdog message names the awaited work; 78873a0 MKCudaError; 9297c43 lazy `hb.logits`; 89e9754 native IQ4_NL + IQ4_XS pairs; 9c4b08c native Q3_K / IQ3_S; 9801b98 presence / frequency penalty; d96f7d6 fast load, `--batch`, state files; 2e604fd load round 2 (WIP at commit time, gated by 21523be); 21523be round 2 gated, `bench.py --dump` fix.
Moved in / out: none. Note for the squash: a730f00's `MK_KV_EMU` part is KV-format experiment code (PR 4's subject); b655092 (checkpoint to pinned host memory, the biggest single VRAM item of the same track, -159 MiB) is in PR 5.

## Notes for the reviewer

* Defaults changed: device buffers come from the memory pool (`MK_CUDAMALLOC=1` returns to one cudaMalloc each), the loader reads ahead with 8 threads (`MK_LOAD_IO=0` off), CLI `MK_CK_SLOTS` defaults to 1, `MK_WATCHDOG` is on (60 s). Everything else is opt-in (`--batch`, `--state-dir`, `MK_MEMLOG`, `MK_LOADLOG`, `MK_WDIGEST`, `MK_FAULT`) or off at value 0 (penalties).
* Read-ahead uses a 3 GiB page-cache window; the load is disk-bound at ~3 GB/s cold. The gain depends on the file being on a fast disk; on a warm cache the remaining time is host CPU work.
* State files are large (1.3-3.9 GB each) and keyed on the binary hash: every rebuild invalidates them. They are a benchmarking / development aid; nothing in the server writes them.
* An on-disk cache of re-encoded weights was considered and rejected (reads 16.7 GB as slowly as the GGUF cold; staleness risk; 16.7 GB per layout).
* Known limits: the sampled-distribution chi-square test for penalties is not done; the MTP Q4_K requant and token_embd pinning threads add CPU load during the load; round 2's `RowPool` gave no measurable gain (kept, harmless).
* Turing (sm_75): all additions are host code, plain CUDA runtime calls, or small kernels; `cudaMallocAsync` needs CUDA 11.2+ and a driver with memory-pool support. The Q3_K / IQ3_S / IQ4_NL kernels follow the existing GEMV style and have not been compiled for sm_75. No Turing run was made.
