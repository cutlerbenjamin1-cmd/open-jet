# Build, run and check

Requirements: an Ampere or newer NVIDIA GPU with 24 GB (sm_86 tested; the new kernels use sm_80+ features: int8
`mma.m16n8k32`, `cp.async`, up to 99 KB of shared memory per block, so Turing is not supported on this branch), CUDA 13.2 (the version we build with; older 12.x is untested),
and llama.cpp built at `../llama.cpp` as in upstream's README (the engine links libllama for the tokenizer
and the reference tools).

## Build

The configuration we run (wide tensor-core verify, up to 7 drafts, fragment-ordered Q6_K):

```
cd megakernel
make -j2 qwen38_mk libqwen38mk.so NVFLAGS="-O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -dlcm=cg -Xptxas -v -Xcompiler -fPIC -DMK_WV=1 -DMK_MAXT=8 -DMK_WV_Q6F=1"
```

ptxas peaks at about 25 GB of RAM for this configuration (a plain `make` with the default flags needs less). The build
takes 20-30 minutes; `ptxas.log` lists registers and spills per function.

## Run

Server (OpenAI-compatible), the settings of our launcher:

```
MK_PF_CAP=1024 python server/mk_server.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 8091 \
    -c 120000 --kv q8 -d 7 --exact-reuse
```

- `-m Qwen3.8-27B-Q4_K_M.gguf` runs the plain file with the same engine.
- `MK_PF_CAP=1024` (prefill chunk) is what we use above 96k context: 0.34 GB less VRAM, prefill ~4.5% slower.
- `--exact-reuse` (MK_EXACT=1): a reused prompt answers bit for bit like the same prompt prefilled from scratch.
- Low-bit KV cache, preset D' (about 1.1 GB less than q8 at 120k, near-lossless; up to 262,144 tokens fit on 24 GB):
  `MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048` in the environment.

CLI:

```
./qwen38_mk -m Qwen3.8-27B-UD-Q4_K_XL.gguf -p "Write a C++ function that parses a GGUF header." -n 512 -d 7 --kv q8 \
    --draft-vocab draft_vocab_32k.bin
```

Note on timing short runs: after a CPU-bound model load an idle RTX 3090 drops to P8 and needs ~1.8 s to reach full
memory clock, so a 2-second decode right after a cold CLI start reads 30-40% low. Run it twice, or use the server
(our benchmark client sends a warm-up request first; the suite on `bench-suite` has a small clock keeper for CLI runs).

## Check

- The Reddit / README benchmarks: `bench/code_bench.py` and `bench/think_bench.py` run unchanged on this branch
  (`-d 7` needs `--kv q8`, which the scripts do not pass: add it to their command line); `bench/bench.py` takes `MK_BIN`.
- Accuracy against llama.cpp: `tools/llama_ref` writes a reference file, `qwen38_mk --kld <ref>` scores it (README,
  "Correctness").
- Greedy output does not depend on the draft depth: the same prompt at `-d 0`, `-d 4` and `-d 7` gives identical ids
  (`--ids`).
