#!/usr/bin/env python3
"""Summarise h2h_client.py / maxctx_client.py records into one comparison table (markdown on stdout, JSON optional).

  python3 h2h_summ.py --system mk=h2h.jsonl:mk --system llama=h2h.jsonl:llama [--json out.json]
  python3 h2h_summ.py --system llama=a.jsonl:llama+b.jsonl:llama   # best value per metric over several sessions
  python3 h2h_summ.py --maxctx maxctx.jsonl:max_mk --maxctx maxctx.jsonl:max_llama

--system NAME=FILE:LABEL[+FILE:LABEL...]: a column named NAME from the records of LABEL in FILE.  Giving several
FILE:LABEL pairs takes, per metric, the best value over them (max for tok/s, min for times and VRAM) and says so in
the JSON ("sessions" holds every value), so a re-run of the same system can be pooled openly.
Means are over the matching requests (names starting with the prefix, `*_fresh` reruns excluded); one-shot metrics
use the single named request.  A metric with no record prints "-".
"""
import argparse
import json
import statistics as st


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def recs(src, lb, test, exact=True):
    return [r for r in src if r["label"] == lb and (r["test"] == test if exact else r["test"].startswith(test))]


def mean(src, lb, prefix, key, scale=1.0):
    v = [r[key] * scale for r in recs(src, lb, prefix, False) if r.get(key) is not None and not r["test"].endswith("_fresh")]
    return round(st.mean(v), 2) if v else None


def one(src, lb, test, key, scale=1.0):
    rs = recs(src, lb, test)
    return round(rs[0][key] * scale, 3) if rs and rs[0].get(key) is not None else None


def pf_scratch(src, lb):
    """101k prompt from scratch: an explicit no-cache rerun, else an uncached long_pf101k."""
    rs = recs(src, lb, "long_pf101k_nocache") or recs(src, lb, "long_pf101k_fresh") or \
        [r for r in recs(src, lb, "long_pf101k") if not r.get("cached")]
    return round(rs[0]["prompt_ms"] / 1e3, 2) if rs and rs[0].get("prompt_ms") else None


def pf_shared(src, lb):
    """101k prompt right after the 38k prompt it starts with: whatever the engine reuses."""
    rs = recs(src, lb, "long_pf101k")
    return round(rs[0]["prompt_ms"] / 1e3, 2) if rs and rs[0].get("prompt_ms") else None


def gb(src, lb, test):
    rs = recs(src, lb, test)
    return round(rs[0]["mib"] / 1024, 2) if rs else None


METRICS = [  # key, label, unit, better, fn(records, label)
    ("greedy", "Greedy decode, short prompts (mean of 4)", "tok/s", "up", lambda s, l: mean(s, l, "greedy_nothink", "decode_tps")),
    ("sampled", "Sampled thinking decode (mean of 8)", "tok/s", "up", lambda s, l: mean(s, l, "sampled_think", "decode_tps")),
    ("dec11k", "Decode at 11.6k depth", "tok/s", "up", lambda s, l: mean(s, l, "followup_turn1", "decode_tps")),
    ("dec38k", "Decode at 38k depth", "tok/s", "up", lambda s, l: mean(s, l, "long_pf38k", "decode_tps")),
    ("dec101k", "Decode at 101.6k depth", "tok/s", "up", lambda s, l: mean(s, l, "long_pf101k", "decode_tps")),
    ("ttft", "Time to first token, short prompt", "ms", "down", lambda s, l: mean(s, l, "ttft_short", "prompt_ms")),
    ("turn2", "Follow-up turn, prompt time (11.8k chat)", "ms", "down", lambda s, l: one(s, l, "followup_turn2", "prompt_ms")),
    ("pf11k", "Prefill 11.6k tokens", "s", "down", lambda s, l: one(s, l, "followup_turn1", "prompt_ms", 1e-3)),
    ("pf38k", "Prefill 38k tokens", "s", "down", lambda s, l: one(s, l, "long_pf38k", "prompt_ms", 1e-3)),
    ("pf101k", "Prefill 101.6k tokens, from scratch", "s", "down", pf_scratch),
    ("pf101k_shared", "101.6k prompt after the 38k prompt it starts with", "s", "down", pf_shared),
    ("edit_msg", "Edit an earlier message (11.7k-token chat)", "s", "down", lambda s, l: one(s, l, "edit_turn2_edited", "prompt_ms", 1e-3)),
    ("edit_sys", "Edit the end of a 9.1k-token system prompt", "s", "down", lambda s, l: one(s, l, "edit_system_edited", "prompt_ms", 1e-3)),
    ("vram_idle", "VRAM, server idle (whole card)", "GB", "down", lambda s, l: gb(s, l, "vram_idle")),
    ("vram_101k", "VRAM after the 101.6k prompt (whole card)", "GB", "down", lambda s, l: gb(s, l, "vram_after_pf101k")),
]


def best(vals, better):
    v = [x for x in vals if x is not None]
    return (max(v) if better == "up" else min(v)) if v else None


def parse_system(spec):
    name, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"--system {spec!r}: expected NAME=FILE:LABEL[+FILE:LABEL...]")
    pairs = []
    for part in rest.split("+"):
        path, _, label = part.rpartition(":")
        if not path or not label:
            raise SystemExit(f"--system {spec!r}: expected FILE:LABEL, got {part!r}")
        pairs.append((path, label))
    return name, pairs


def sweep(systems, cache=None):
    """systems: [(name, [(path, label), ...])] -> [row per metric]."""
    cache = {} if cache is None else cache
    rows = []
    for key, label, unit, better, fn in METRICS:
        row = {"key": key, "metric": label, "unit": unit, "better": better, "sessions": {}}
        for name, pairs in systems:
            vals = []
            for path, lb in pairs:
                if path not in cache:
                    cache[path] = load(path)
                vals.append(fn(cache[path], lb))
            row["sessions"][name] = vals
            row[name] = best(vals, better)
        rows.append(row)
    return rows


def maxctx_summary(path, label):
    """Max-context records (maxctx_client.py): speed at the largest context, VRAM, and the per-item records."""
    rs = [r for r in load(path) if r["label"] == label]
    if not rs:
        return None

    def g(t):
        return next((r for r in rs if r["test"] == t), None)

    t1, t2 = g("speed_max_turn1"), g("speed_max_turn2")
    mib = lambda r: r and round(r["mib"] / 1024, 2)  # noqa: E731
    return {"ctx": rs[0]["ctx"],
            "prefill_tokens": t1 and t1.get("prompt_tokens"),
            "prefill_s": t1 and t1.get("prompt_ms") and round(t1["prompt_ms"] / 1e3, 1),
            "prefill_tok_s": t1 and t1.get("prompt_ms") and round(t1["prompt_tokens"] / (t1["prompt_ms"] / 1e3)),
            "decode_tok_s": t1 and t1.get("decode_tps"),
            "turn2_prompt_ms": t2 and t2.get("prompt_ms"), "turn2_cached": t2 and t2.get("cached"),
            "turn2_decode_tok_s": t2 and t2.get("decode_tps"),
            "vram_idle_gb": mib(g("vram_idle")), "vram_after_max_gb": mib(g("vram_after_max")),
            "items": [{k: r.get(k) for k in ("set", "id", "tokens_est", "prompt_tokens", "prompt_ms", "decode_tps",
                                             "finish", "domain", "difficulty")} for r in rs if r.get("set")]}


def fmt(x):
    return "-" if x is None else str(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", action="append", default=[], help="NAME=FILE:LABEL[+FILE:LABEL...]")
    ap.add_argument("--maxctx", action="append", default=[], help="FILE:LABEL (maxctx_client.py records)")
    ap.add_argument("--json", help="write the rows (and the max-context summaries) here")
    a = ap.parse_args(argv)
    systems = [parse_system(s) for s in a.system]
    out = {}
    if systems:
        rows = sweep(systems)
        out["sweep"] = rows
        names = [n for n, _ in systems]
        print("| metric | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for r in rows:
            print(f"| {r['metric']} ({r['unit']}) | " + " | ".join(fmt(r[n]) for n in names) + " |")
    for spec in a.maxctx:
        path, _, label = spec.rpartition(":")
        m = maxctx_summary(path, label)
        out.setdefault("maxctx", {})[label] = m
        print(label, {k: v for k, v in (m or {}).items() if k != "items"})
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    print("SUMM_DONE")


if __name__ == "__main__":
    main()
