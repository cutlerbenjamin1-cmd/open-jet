# README draft for PR 6 (load, memory, robustness)

Against upstream's current `megakernel/README.md` (origin/master 4d4db90). Six edits.

## Edit 1: `## Build / run`, the sentence "The mixed loader accepts Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS projections, and converts Q3_K / IQ3_S to Q6_K and IQ4_NL to Q8_0."

Replace with (mixed engine only; the plain engine still refuses these types):

> The mixed loader accepts Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, Q3_K, IQ3_S and IQ4_NL projections. Q3_K and
> IQ3_S are re-laid-out at load into a value stream in the GEMV's unit order plus a scale stream (114 and 112 bytes
> per 256 weights, instead of 210-byte Q6_K blocks), IQ4_NL is read as 144-byte blocks (eight fp16 scales plus the
> indices, instead of a Q8_0 re-encode) and IQ4_XS as 272-byte block pairs. The kernels feed the same integers and
> scales as the re-encodes they replace, so results are bit-identical; the loader checks every re-laid-out row against
> ggml's dequantization. `MK_REENC3=1` (Q3_K / IQ3_S to Q6_K) and `MK_IQ4NL_Q8=1` (IQ4_NL to Q8_0) restore the
> re-encodes for A/B tests.

## Edit 2: new subsection after the CLI flags paragraph of `## Build / run` (the one ending "... the default accepts more drafts)."), before `### The GGUF`

> ### Load time, batch runs and state files
>
> The 17 GB GGUF is often cold in the page cache (under WSL2 it is dropped when idle), and one thread's page faults
> through the mapping read it at about 0.2 GB/s. The loader therefore reads the file ahead on `MK_LOAD_IO` threads
> (default 8, at most 3 GiB ahead of the loader, in the loader's order), reuses host buffers without zero-filling them,
> copies to the GPU on a worker thread while the next tensor is prepared, makes the wide-verify weight layout on the GPU
> (`MK_LOAD_GPURL=0`: on the host), and requantizes the MTP block and pins `token_embd` on background threads. On an
> RTX 3090 with a UD-Q4_K_XL file the CLI takes about 8 s warm / 12 s cold instead of 32 s / 60 s. The loader prints a
> one-line phase summary (`[mk] load ...`); `MK_LOADLOG=1` prints every phase, `MK_WDIGEST=1` a digest of every weight
> buffer (for checking that a loader change did not change a byte). The CLI keeps one checkpoint slot
> (`MK_CK_SLOTS=1`) to save pinned-memory allocation time; the server keeps the pool.
>
> - `--batch jobs.txt`: one model load, many generation jobs. One job per line with the usual command-line options
>   (quote with `'...'` or `"..."`), each with `-o <prefix>` (writes `<prefix>.out` and `<prefix>.err`); options before
>   `--batch` are the defaults of every line. The engine is reset between jobs and the draft depth can differ per job,
>   so a depth sweep (`-d 0`, `-d 4`, ...) costs one load. Outputs equal those of separate runs.
> - `--state-dir DIR`: the state after a long prompt's batched prefill (K/V rows, window rings, DeltaNet, conv and h
>   rings) is saved once per key (binary, model, context, KV format, the `MK_*` settings that change numerics, the
>   prompt) and restored instead of re-prefilling it. The state is saved before the last prompt token, so one file serves
>   every draft depth. Files are large (1.3-1.6 GB at 38k tokens, 3.9 GB at 101.6k with q8 KV; saving takes 2-5 s once,
>   a warm restore 0.25-0.55 s) and a rebuild invalidates them. After a restore the GPU is kept busy for
>   `MK_STATE_WARM_MS` (default 1000) so the first decode does not run at idle clocks. `--digest` prints a hash of the
>   sequence state (restore == fresh checks).
> - `MK_MEMLOG=1` prints a device-memory ledger per load stage (bytes allocated vs the drop in free memory) and
>   `mk_cycle`'s local-memory reservation. Device buffers come from the CUDA memory pool (`cudaMallocAsync`), which avoids
>   the 2 MiB rounding of one `cudaMalloc` per tensor (about 470 MiB on UD-Q4_K_XL); `MK_CUDAMALLOC=1` allocates each
>   buffer with `cudaMalloc` again.

## Edit 3: `## Server`, the bullet "Sampling: temperature, top_p, top_k (<= 32), min_p, seed; presence / repetition penalties are ignored (with a warning). One request at a time."

Replace with:

> - Sampling: temperature, top_p, top_k (<= 32), min_p, seed, presence_penalty / frequency_penalty (-2..2, vLLM's
>   semantics: the tokens generated in this response; the prompt does not count); repetition_penalty is ignored
>   (with a warning). One request at a time.

## Edit 4: new bullet list after the Server bullets (or a short `### Failures and the watchdog` under `## Server`)

> ### Failures and the watchdog
>
> - A CUDA error inside the library is returned as an error: the C API returns -1 with the message
>   (`mk_last_error`), the engine is reset, and the server answers HTTP 500 and keeps serving; the next prompt is
>   prefilled in full. An error that leaves the CUDA context unusable (illegal address, launch failure or trap, ECC,
>   or one that survives clearing) prints `[mk] the CUDA error is sticky ...` and exits with code 4. The CLI prints
>   `[mk] error: ...` and exits 1 (also for a prompt that exceeds the context).
> - Every wait for the GPU polls an event with a deadline: `MK_WATCHDOG` seconds per decode launch (default 60;
>   prefill chunks and checkpoint copies get at least 120 s; 0 = plain blocking waits). On expiry the process prints
>   diagnostics and exits with code 3. This exists for a driver fault that the WSL2 guest was not told about and that
>   otherwise leaves the host thread spinning forever. `MK_WATCHDOG_SPIN` is the number of seconds of busy polling before
>   the poll sleeps between checks.
> - `MK_FAULT=cuda:<n>` / `sticky:<n>` makes the n-th prefill fail half-way (a rejected CUDA call, or a trapping
>   kernel): for tests of the above.

## Edit 5: CLI flags paragraph of `## Build / run`

Add to the list of CLI flags: `--presence-penalty p` / `--frequency-penalty p` (generation penalties, default 0; applied
before temperature and top-k, also in greedy mode; under speculative decoding a verify row counts the drafts before it,
so greedy output is the same at every draft depth), `--batch jobs.txt`, `--state-dir dir`, `--digest`.

## Edit 6: `## Limitations`, replace the sampling bullet

Replace "Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order); no presence / repetition penalties yet (open-jet's non-thinking preset uses a presence penalty of 1.5)." with:

> - Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order), and presence / frequency
>   penalties on the generated tokens (applied first; greedy too; exact under speculative verification: a verify
>   row also counts the drafts before it); no repetition penalty. (Qwen recommends a presence penalty of 1.5 in
>   non-thinking mode against loops.)

Placement note: Edits 1-2 and 5 assume the README keeps one `## Build / run` section; if PR 1 (draft policy) rewrites the CLI flags paragraph, apply Edit 5 to the rewritten list.
