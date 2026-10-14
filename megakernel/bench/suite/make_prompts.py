#!/usr/bin/env python3
"""Build the long prompt files: pf38k.txt (~38k tokens of source code) and pf101k.txt (~101k) for h2h_client.py, and
pf64k.txt (~64k) for decode_state.sh.

  python3 make_prompts.py [--out DIR] [--source FILE ...] [--extra FILE ...]

pf38k.txt = the first --bytes-38k bytes of the concatenated --source files (default: this repo's megakernel sources).
pf101k.txt = pf38k.txt followed by the --extra files (default: the rest of the megakernel sources, then any extra
file given), cut at --bytes-101k.  pf101k therefore starts with pf38k, which is what the prompt-reuse tests need
(the 101k prompt shares a 38k-token prefix with the earlier request).  pf64k.txt is the same text cut at --bytes-64k.  The reference numbers in the notes used
100,000 bytes = 38,049 tokens and 325,100 bytes = 101,586 tokens of the Qwen3.8 tokenizer; other text gives other
token counts, so count them with the target model's tokenizer (llama-tokenize) when absolute lengths matter.
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MKDIR = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SOURCES = ["qwen38_mk.cu"]
DEFAULT_EXTRA = ["qwen38_mixed.cu", "prefill.cuh", "prefill_gemm.cuh", "prefill_gemm_mixed.cuh", "main.cu"]


def read_all(paths):
    parts = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            parts.append(f.read())
    return "".join(parts)


def cut_bytes(text, n):
    """Cut at n UTF-8 bytes without splitting a character."""
    return text.encode("utf-8")[:n].decode("utf-8", errors="ignore")


def build(sources, extra, n38, n101, n64=207_975):
    base = read_all(sources)
    if len(base.encode("utf-8")) < n38:
        raise SystemExit(f"--source files hold {len(base.encode('utf-8'))} bytes, need {n38}")
    p38 = cut_bytes(base, n38)
    full = base + read_all(extra)
    p101 = cut_bytes(full, n101)
    if not p101.startswith(p38):
        raise SystemExit("internal error: pf101k does not start with pf38k")
    if len(p101.encode("utf-8")) < n101:
        raise SystemExit(f"sources hold only {len(p101.encode('utf-8'))} bytes, need {n101}: add --extra files")
    return p38, p101, cut_bytes(full, n64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.environ.get("BENCH_PROMPT_DIR", os.path.join("bench_data", "prompts")))
    ap.add_argument("--source", nargs="+", default=[os.path.join(MKDIR, f) for f in DEFAULT_SOURCES])
    ap.add_argument("--extra", nargs="*", default=[os.path.join(MKDIR, f) for f in DEFAULT_EXTRA])
    ap.add_argument("--bytes-38k", type=int, default=100_000)
    ap.add_argument("--bytes-64k", type=int, default=207_975)
    ap.add_argument("--bytes-101k", type=int, default=325_100)
    a = ap.parse_args(argv)
    p38, p101, p64 = build(a.source, a.extra, a.bytes_38k, a.bytes_101k, a.bytes_64k)
    os.makedirs(a.out, exist_ok=True)
    for name, text in (("pf38k.txt", p38), ("pf64k.txt", p64), ("pf101k.txt", p101)):
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"{name}: {len(text.encode('utf-8'))} bytes -> {os.path.join(a.out, name)}")


if __name__ == "__main__":
    main()
