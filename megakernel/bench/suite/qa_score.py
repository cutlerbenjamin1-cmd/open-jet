#!/usr/bin/env python3
"""Score qa_client.py / maxctx_client.py records offline; the same scorer for every engine.

  python3 -I qa_score.py FILE.jsonl [FILE.jsonl ...] [--data-dir DIR] [--pair A B] [--out scores.json]

Prints, per set, the correct / n of every label found in the files; with --pair A B also the paired agreement
(both right / only A / only B / both wrong) and the timing table (wall, prefill tok/s, decode tok/s, truncated
answers) of those two labels.  Scores: GSM8K number after "Answer:", MMLU-Pro / ARC letter, MATH-500 \\boxed{} with a
light normalisation, HumanEval by running the unit tests, NIAH exact substring of the number, LongBench v2 letter.
Max-context records (maxctx_client.py: sets niah_max, longbench_max) are scored with the NIAH / LongBench rules.
HumanEval completions are model output and are executed: each in an isolated `python -I` in a scratch directory
under <data-dir>/humaneval (outside any repository), 15 s timeout per task.  Run this script with `python -I` too.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

SETS = ["gsm8k", "mmlu_pro", "arc_c", "math500", "math500_think", "humaneval", "niah", "longbench_v2", "niah_max",
        "longbench_max"]
ITEM_FILES = {"niah_max": os.path.join("niah", "items_max.json"), "longbench_max": os.path.join("longbench_v2", "items_max.json")}


def num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def s_gsm8k(it, out):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", out)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", out)
    v, g = (num(m[-1]) if m else None), num(it["answer"])
    return v is not None and g is not None and abs(v - g) < 1e-6


def s_mmlu_pro(it, out):
    m = re.findall(r"answer is \(?([A-J])\)?", out) or re.findall(r"\(([A-J])\)", out)
    return bool(m) and m[-1] == it["answer"]


def s_arc_c(it, out):
    m = re.search(r"[A-E1-5]", out.strip())
    return bool(m) and m.group(0) == it["answer"]


def boxed(out):
    i = out.rfind("\\boxed")
    if i < 0:
        return None
    j = out.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(out):
        if out[k] == "{":
            depth += 1
        elif out[k] == "}":
            depth -= 1
            if depth == 0:
                return out[j + 1:k]
        k += 1
    return None


def norm(a):
    a = a.strip().strip("$").strip()
    for x in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "^\\circ", "^{\\circ}", "\\%", "%", "\\$", "dfrac", "tfrac"):
        a = a.replace(x, "frac" if x in ("dfrac", "tfrac") else "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mbox\{([^}]*)\}", r"\1", a)
    a = a.replace(" ", "").rstrip(".")
    return a


def s_math(it, out):
    b = boxed(out)
    if b is None:
        return False
    x, g = norm(b), norm(it["answer"])
    if x == g:
        return True
    nx, ng = num(x), num(g)
    return nx is not None and ng is not None and abs(nx - ng) < 1e-6


def s_humaneval(it, out, scratch):
    m = re.findall(r"```(?:python)?\n(.*?)```", out, re.S)
    code = m[0] if m else out
    prog = (code if f"def {it['entry_point']}" in code else it["prompt"] + code) + "\n\n" + it["test"] + \
        f"\n\ncheck({it['entry_point']})\n"
    p = os.path.join(scratch, re.sub(r"\W", "_", it["id"]) + ".py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, "-I", p], cwd=scratch, capture_output=True, timeout=15)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def s_niah(it, out):
    return it["answer"] in out


def s_long(it, out):
    m = re.findall(r"correct answer is \(?([A-D])\)?", out) or re.findall(r"\(([A-D])\)", out)
    return bool(m) and m[0] == it["answer"]


SC = {"gsm8k": s_gsm8k, "mmlu_pro": s_mmlu_pro, "arc_c": s_arc_c, "math500": s_math, "math500_think": s_math,
      "niah": s_niah, "longbench_v2": s_long, "niah_max": s_niah, "longbench_max": s_long}


def load_items(data_dir):
    items = {}
    for s in SETS:
        p = os.path.join(data_dir, ITEM_FILES.get(s, os.path.join(s, "items.json")))
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for it in json.load(f):
                    items[(s, it["id"])] = it
    return items


def load_records(paths):
    recs = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("ok") and r.get("set") in SC:
                    recs[(r["label"], r["set"], r["id"])] = r
    return recs


def score(recs, items, scratch):
    res = {}
    for (lb, s, i), r in recs.items():
        it = items.get((s, i))
        if it is None:
            continue  # an item file that is not here (e.g. a record of another data set)
        res[(lb, s, i)] = s_humaneval(it, r["content"], scratch) if s == "humaneval" else SC[s](it, r["content"])
    return res


def timing(recs, label, s):
    rs = [r for (l, ss, _), r in recs.items() if l == label and ss == s]
    pt, pms = sum(r.get("prompt_tokens") or 0 for r in rs), sum(r.get("prompt_ms") or 0 for r in rs)
    pn, dms = sum(r.get("pred_n") or 0 for r in rs), sum(r.get("pred_ms") or 0 for r in rs)
    return {"wall_s": round(sum(r["wall_s"] for r in rs), 1), "prefill_tok_s": round(pt / (pms / 1e3), 1) if pms else None,
            "decode_tok_s": round(pn / (dms / 1e3), 1) if dms else None,
            "truncated": sum(1 for r in rs if r.get("finish") == "length")}


def pair_table(res, la, lb):
    summary = {}
    for s in SETS:
        ids = sorted({i for (_, ss, i) in res if ss == s and (la, s, i) in res and (lb, s, i) in res})
        if not ids:
            continue
        a, b = [res[(la, s, i)] for i in ids], [res[(lb, s, i)] for i in ids]
        summary[s] = {"n": len(ids), la: sum(a), lb: sum(b), "both": sum(x and y for x, y in zip(a, b)),
                      "only_" + la: sum(x and not y for x, y in zip(a, b)),
                      "only_" + lb: sum(y and not x for x, y in zip(a, b)),
                      "neither": sum(not x and not y for x, y in zip(a, b))}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data-dir", default=os.environ.get("BENCH_DATA", "bench_data"))
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--out", help="JSON output (default: qa_scores.json next to the first input)")
    a = ap.parse_args(argv)
    items = load_items(a.data_dir)
    recs = load_records(a.files)
    he = os.path.join(a.data_dir, "humaneval")
    scratch = tempfile.mkdtemp(prefix="he_", dir=he if os.path.isdir(he) else None)
    res = score(recs, items, scratch)
    labels = sorted({lb for lb, _, _ in res})
    out = {"labels": labels, "sets": {}, "timing": {}}
    for s in SETS:
        for lb in labels:
            ks = [k for k in res if k[0] == lb and k[1] == s]
            if ks:
                out["sets"].setdefault(s, {})[lb] = {"n": len(ks), "correct": sum(res[k] for k in ks)}
                out["timing"].setdefault(s, {})[lb] = timing(recs, lb, s)
    out["items"] = [{"label": l, "set": s, "id": i, "correct": ok} for (l, s, i), ok in sorted(res.items())]
    print("| set | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for s in SETS:
        if s in out["sets"]:
            print(f"| {s} | " + " | ".join(f"{out['sets'][s][lb]['correct']}/{out['sets'][s][lb]['n']}" if lb in out["sets"][s]
                                          else "-" for lb in labels) + " |")
    if a.pair:
        la, lb = a.pair
        out["pair"] = pair_table(res, la, lb)
        print(f"\n| set | n | {la} | {lb} | both right | only {la} | only {lb} | both wrong |")
        print("|---|---|---|---|---|---|---|---|")
        for s, v in out["pair"].items():
            print(f"| {s} | {v['n']} | {v[la]} ({100 * v[la] / v['n']:.0f}%) | {v[lb]} ({100 * v[lb] / v['n']:.0f}%) | "
                  f"{v['both']} | {v['only_' + la]} | {v['only_' + lb]} | {v['neither']} |")
        print("\n| set | " + " | ".join(f"{x} wall s / prefill tok/s / decode tok/s / truncated" for x in (la, lb)) + " |")
        for s in out["pair"]:
            t = out["timing"][s]
            print(f"| {s} | " + " | ".join(f"{t[x]['wall_s']} / {t[x]['prefill_tok_s']} / {t[x]['decode_tok_s']} / {t[x]['truncated']}"
                                          for x in (la, lb)) + " |")
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.files[0])), "qa_scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("QA_SCORE_DONE")


if __name__ == "__main__":
    main()
