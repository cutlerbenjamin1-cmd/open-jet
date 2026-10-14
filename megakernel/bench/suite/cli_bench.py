#!/usr/bin/env python3
"""Run the megakernel halves of code_bench.py and think_bench.py on several engine builds / settings in one session.

  python3 cli_bench.py --config engines.json [label ...] [--reps 2] [--sizes 0,30000] [--out cli_bench.json]
                       [--freeze-dir DIR] [--gpu-keep BINARY] [--think-source FILE]

engines.json: {"engines": [{"label": "up", "binary": "megakernel/qwen38_mk", "model": "~/models/x.gguf",
                            "depth": 4, "kv": "q8", "env": {"MK_PF_CAP": "1024"}}, ...]}
`kv` is passed to the CLI as --kv (omit it to keep the CLI default); `env` is set for that engine only.  For each engine:
  short   the 4 prompts of prompts.txt, 512 tokens, best of --reps          -> decode tok/s, tok/cycle
  ctx     a pasted source file of N tokens + a coding task, 256 tokens      -> prompt and decode tok/s
  think   the xhigh thinking preset, 2 seeds x 4 prompts, 1024 tokens       -> decode tok/s, draft acceptance
The benchmark functions are code_bench.py's and think_bench.py's own (same prompts, best-of-N, greedy code runs,
seeds 1000 / 1001), so the numbers are comparable with theirs.  Results are written after every engine; an engine already
marked done in --out is skipped (resume).  The table of decode speeds is printed at the end.
--freeze-dir keeps the generated N-token prompt files (ctx_N.txt): a second session with the same directory feeds
every engine the identical inputs even if the repository's sources changed; --think-source does the same for the
source file think_bench pastes (default: its own prefill.cuh, first 8000 characters).
--gpu-keep BINARY runs gpu_keep (gpu_keep.cu) while each CLI loads its model and kills it when the CLI prints its
"[mk] load" line, before any timed work: the CLI's load is CPU-bound, the GPU idles to a low clock state, and the first
second or two of a short timed decode would otherwise run at ramping clocks.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)


def load_engines(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    engines = cfg["engines"] if isinstance(cfg, dict) else cfg
    seen = set()
    for e in engines:
        for k in ("label", "binary", "model"):
            if k not in e:
                raise SystemExit(f"engine {e}: missing '{k}'")
        if e["label"] in seen:
            raise SystemExit(f"duplicate engine label {e['label']}")
        seen.add(e["label"])
        e.setdefault("depth", 4)
        e.setdefault("env", {})
        e["model"] = os.path.expanduser(os.path.expandvars(e["model"]))
        e["binary"] = os.path.expanduser(os.path.expandvars(e["binary"]))
    return engines


def make_run_kept(real_run, mk_binary, keep_binary):
    """subprocess.run replacement: for the CLI, run keep_binary during the model load (killed at '[mk] load')."""
    def run_kept(cmd, *a, **k):
        if not (isinstance(cmd, list) and cmd and cmd[0] == mk_binary):
            return real_run(cmd, *a, **k)
        keep = subprocess.Popen([keep_binary], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=k.get("env"))
        so, se = [], []
        t = threading.Thread(target=lambda: so.append(p.stdout.read()))
        t.start()
        for line in p.stderr:
            se.append(line)
            if keep.poll() is None and line.startswith("[mk] load"):
                keep.kill()
        p.wait()
        t.join()
        if keep.poll() is None:
            keep.kill()
        keep.wait()
        res = subprocess.CompletedProcess(cmd, p.returncode, "".join(so), "".join(se))
        if k.get("check") and p.returncode:
            raise subprocess.CalledProcessError(p.returncode, cmd, res.stdout, res.stderr)
        return res
    return run_kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sizes", default="0,30000", help="context sizes in tokens for the ctx part")
    ap.add_argument("--out", default="cli_bench.json")
    ap.add_argument("--freeze-dir")
    ap.add_argument("--gpu-keep", default=os.environ.get("BENCH_GPU_KEEP"))
    ap.add_argument("--think-source")
    ap.add_argument("--bench-dir", default=BENCH, help="directory holding code_bench.py / think_bench.py / prompts.txt")
    a = ap.parse_args(argv)

    sys.path.insert(0, a.bench_dir)
    import code_bench as cb  # noqa: E402
    import think_bench as tb  # noqa: E402

    engines = [e for e in load_engines(a.config) if not a.labels or e["label"] in a.labels]
    sizes = [int(x) for x in a.sizes.split(",")]
    cb.DV = tb.DV = os.path.join(os.path.dirname(a.bench_dir), "draft_vocab_32k.bin")
    if a.think_source:
        with open(a.think_source) as f:
            src = f.read()[:8000]
        tb.file_prompt = lambda: "Here is a CUDA source file:\n\n" + src + "\n\nFind one possible bug or fragile spot in this file and suggest a fix."
    os.makedirs(cb.TMP, exist_ok=True)
    os.makedirs(tb.TMP, exist_ok=True)
    fdir = a.freeze_dir or cb.TMP
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(a.bench_dir, "prompts.txt")) as f:
        prompts = [l.strip() for l in f if l.strip()]
    files = {}
    for n in sizes:
        files[n] = os.path.join(fdir, f"ctx_{n}.txt")
        if not os.path.exists(files[n]):
            with open(files[n], "w") as f:
                f.write(cb.ctx_prompt(n))

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    real_run = subprocess.run
    for e in engines:
        lab, d = e["label"], e["depth"]
        if res.get(lab, {}).get("done"):
            continue
        cb.MK = tb.MK = e["binary"]
        cb.MODEL = tb.MODEL = e["model"]
        if a.gpu_keep:
            subprocess.run = make_run_kept(real_run, e["binary"], a.gpu_keep)
        for k in {k for x in engines for k in x["env"]}:  # the CLI inherits os.environ: only this engine's settings
            os.environ.pop(k, None)
        os.environ.update(e["env"])
        kv = ["--kv", e["kv"]] if e.get("kv") else []
        r = {"binary": e["binary"], "model": e["model"], "depth": d, "kv": e.get("kv") or "cli default", "env": e["env"],
             "short": [], "ctx": {}}
        t0 = time.time()
        print(f"\n######## {lab} ({e['binary']}, -d {d}) {time.strftime('%H:%M:%S')}", flush=True)
        for i, p in enumerate(prompts):
            runs = [cb.mk_run(["-p", p, "-n", "512", "-d", str(d)] + kv)[0] for _ in range(a.reps)]
            b = cb.best(runs, "dtps")
            r["short"].append(b["dtps"])
            print(f"[{lab}] prompt{i}: {b['dtps']:.1f} tok/s ({b['tpc']:.2f} tok/cycle); runs {[x['dtps'] for x in runs]}", flush=True)
        for n in sizes:
            runs = [cb.mk_run(["-f", files[n], "-n", "256", "-d", str(d)] + kv)[0] for _ in range(a.reps)]
            x = dict(pn=runs[0]["pn"], ptps=cb.best(runs, "ptps")["ptps"], dtps=cb.best(runs, "dtps")["dtps"],
                     tpc=cb.best(runs, "dtps")["tpc"], runs=[(y["ptps"], y["dtps"]) for y in runs])
            r["ctx"][str(n)] = x
            print(f"[{lab}] ctx {x['pn']} tok: prompt {x['ptps']:.0f} tok/s, decode {x['dtps']:.1f} tok/s ({x['tpc']:.2f} tok/cycle)",
                  flush=True)
        th = tb.run_mk(tb.PROMPTS, 2, 1024, d, model=e["model"], megakernel=e["binary"], kv=e.get("kv") or "auto")
        r["think"] = {f"{i}_{s}": v for (i, s), v in th.items()}
        r["think_mean"] = round(statistics.mean(v["tps"] for v in th.values()), 2)
        acc = sum(v["acc"][0] for v in th.values()) / max(1, sum(v["acc"][1] for v in th.values()))
        r["think_acc"] = round(acc, 4)
        r["short_mean"] = round(statistics.mean(r["short"]), 2)
        r["done"], r["minutes"] = True, round((time.time() - t0) / 60, 1)
        res[lab] = r
        subprocess.run = real_run
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{lab}] short mean {r['short_mean']}, ctx {sizes[0]} {r['ctx'][str(sizes[0])]['dtps']}, ctx {sizes[-1]} "
              f"{r['ctx'][str(sizes[-1])]['dtps']}, reasoning {r['think_mean']} tok/s (draft acceptance {acc:.1%}), "
              f"{r['minutes']} min", flush=True)

    print("\n| engine | " + " | ".join(f"decode ctx {n}" for n in sizes) + " | short prompts (mean of 4) | reasoning mode | "
          f"prefill ctx {sizes[-1]} |")
    print("|---|" + "---|" * (len(sizes) + 3))
    for e in engines:
        r = res.get(e["label"])
        if r:
            print(f"| {e['label']} (-d {r['depth']}) | " + " | ".join(f"{r['ctx'][str(n)]['dtps']:.1f}" for n in sizes) +
                  f" | {r['short_mean']:.1f} | {r['think_mean']:.1f} | {r['ctx'][str(sizes[-1])]['ptps']:.0f} |")
    print("CLI_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
