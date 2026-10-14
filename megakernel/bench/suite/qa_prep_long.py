#!/usr/bin/env python3
"""Build the long-context item sets, sized with the model's own tokenizer (llama-tokenize on the GGUF, vocab only).

  python3 -I qa_prep_long.py long                         items for a ~120k context
  python3 -I qa_prep_long.py max-lb                       LongBench v2 candidates of 125k-258k tokens
  python3 -I qa_prep_long.py max-niah LEN [LEN ...]       2 NIAH items at each length (tokens), e.g. 190000 248000
      [--data-dir DIR] [--llama-tokenize PATH] [--gguf FILE]

Needs qa_fetch.py's longbench_v2 and niah downloads and a llama.cpp `llama-tokenize` (--llama-tokenize, else
$LLAMA_TOKENIZE, else `llama-tokenize` on PATH) plus the GGUF whose tokenizer sizes the prompts (--gguf, else $BENCH_GGUF).

long    -> longbench_v2/items.json: 12 items whose prompt fits 120k ctx (<= 108k tokens): 4 at 16-40k, 4 at 40-75k,
           4 at 75-108k tokens, domains spread, seeded.
           niah/items.json: RULER-style multi-key: War and Peace as the haystack, 4 needles "One of the special magic
           numbers for <key> is: <7 digits>." at spread depths, the question asks for one key; 3 items each at
           ~32k / ~64k / ~100k tokens (an early, a middle and a late needle asked).
max-lb  -> longbench_v2/items_max_cand.json: candidates of 125-258k tokens in bins 125-160k / 160-200k / 200-258k,
           up to 4 each, seeded, domains spread; and items_max.json (the file maxctx_client.py reads): 2 per bin, a
           second domain preferred.  maxctx_client.py then keeps the ones that fit the server's context.
max-niah-> niah/items_max.json (the file maxctx_client.py reads): same construction, 2 items per length (an early and
           a late needle), the length corrected up to 4 times until the prompt is within 600 tokens of the target.
Inputs are untrusted downloads; nothing in them is executed.  Run with `python -I`.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile

KEYS = ["amber-falcon", "silver-orchard", "quiet-harbor", "crimson-meadow", "hollow-pine", "violet-canyon",
        "golden-thicket", "frozen-delta", "ivory-summit", "copper-lagoon", "scarlet-ridge", "misty-vale"]
HEAD = ("Some special magic numbers are hidden within the following text. Make sure to memorize them. I will quiz you "
        "about the numbers afterwards.\n\n")
DEPTHS = [0.08, 0.2, 0.33, 0.47, 0.6, 0.72, 0.86, 0.95]


class Tokenizer:
    def __init__(self, exe, gguf, tmp, timeout=600):
        self.exe, self.gguf, self.tmp, self.timeout = exe, gguf, tmp, timeout
        os.makedirs(tmp, exist_ok=True)

    def count(self, text):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", dir=self.tmp, delete=False) as f:
            f.write(text)
            p = f.name
        try:
            r = subprocess.run([self.exe, "-m", self.gguf, "-f", p, "--show-count", "--log-disable"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
            m = re.search(r"Total number of tokens:\s*(\d+)", r.stdout + r.stderr)
            return int(m.group(1)) if m else None
        finally:
            os.remove(p)


def haystack(data_dir):
    with open(os.path.join(data_dir, "niah", "pg2600.txt"), encoding="utf-8") as f:
        raw = f.read()
    a, b = raw.find("CHAPTER I"), raw.rfind("*** END OF THE PROJECT GUTENBERG")
    return re.sub(r"\r\n?", "\n", raw[a:b if b > 0 else len(raw)])


def lb_item(d, n):
    return {"id": d["_id"], "domain": d["domain"], "sub_domain": d["sub_domain"], "difficulty": d["difficulty"],
            "length": d["length"], "approx_tokens": n, "question": d["question"], "choice_A": d["choice_A"],
            "choice_B": d["choice_B"], "choice_C": d["choice_C"], "choice_D": d["choice_D"], "answer": d["answer"],
            "context": d["context"]}


def niah_prompt(hay, length, ratio, keys, vals, depths, target):
    body = hay[: int((length - 150) * ratio)]
    cut = body.rfind("\n")
    body = body[:cut] if cut > 0 else body
    parts, last = [], 0
    for d, key, v in zip(depths, keys, vals):
        pos = body.find("\n", int(len(body) * d))
        pos = pos if pos > 0 else int(len(body) * d)
        parts += [body[last:pos], f"\nOne of the special magic numbers for {key} is: {v}.\n"]
        last = pos
    parts.append(body[last:])
    q = f"\n\nWhat is the special magic number for {target} mentioned in the provided text? Answer with the number only."
    return HEAD + "".join(parts) + q


def prep_long(tok, data_dir):
    rng = random.Random(20261013)
    with open(os.path.join(data_dir, "longbench_v2", "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    print(f"LongBench v2: {len(data)} items", flush=True)
    cand = [d for d in data if 50_000 <= len(d["context"]) <= 480_000]
    rng.shuffle(cand)
    bins = {0: [], 1: [], 2: []}
    edges = [(16_000, 40_000), (40_000, 75_000), (75_000, 108_000)]
    seen_dom = {0: set(), 1: set(), 2: set()}
    for d in cand:
        if all(len(bins[b]) >= 4 for b in bins):
            break
        prompt_extra = len(d["question"]) + sum(len(d[f"choice_{c}"]) for c in "ABCD") + 400
        n = tok.count(d["context"])
        if n is None:
            continue
        n += prompt_extra // 3
        for b, (lo, hi) in enumerate(edges):
            if lo <= n < hi and len(bins[b]) < 4 and (d["domain"] not in seen_dom[b] or len(seen_dom[b]) >= 3):
                bins[b].append(lb_item(d, n))
                seen_dom[b].add(d["domain"])
                print(f"  longbench pick bin {b}: {d['_id']} {d['domain']} / {d['difficulty']} ~{n} tok", flush=True)
                break
    lb = bins[0] + bins[1] + bins[2]
    with open(os.path.join(data_dir, "longbench_v2", "items.json"), "w", encoding="utf-8") as f:
        json.dump(lb, f, indent=1)
    print(f"longbench_v2: {len(lb)} items, tokens {[x['approx_tokens'] for x in lb]}", flush=True)

    hay = haystack(data_dir)
    sample = hay[:400_000]
    ratio = len(sample) / tok.count(sample)  # chars per token for this text
    print(f"haystack: {len(hay)} chars, {ratio:.3f} chars / token", flush=True)
    items = []
    for length in (32_000, 64_000, 100_000):
        for k in range(3):
            keys = rng.sample(KEYS, 4)
            vals = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
            depths = sorted(rng.sample(DEPTHS, 4))
            target = keys[[0, 2, 3][k]]  # ask an early, a middle and a late needle
            prompt = niah_prompt(hay, length, ratio, keys, vals, depths, target)
            n = tok.count(prompt)
            it = {"id": f"niah_{length // 1000}k_{k}", "length": length, "tokens": n, "key": target,
                  "answer": vals[keys.index(target)], "depth": depths[keys.index(target)], "prompt": prompt}
            items.append(it)
            print(f"  niah {it['id']}: {n} tokens, asks {target} at depth {it['depth']}", flush=True)
    with open(os.path.join(data_dir, "niah", "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)


def est_tokens(t):  # rough tokens: ~1.4 chars / token for CJK-heavy text, ~3.6 otherwise
    cjk = sum(1 for ch in t[:20000] if ord(ch) > 0x2E80) / max(1, min(len(t), 20000))
    return len(t) / (1.4 if cjk > 0.3 else 3.6)


def prep_max_lb(tok, data_dir):
    rng = random.Random(20261014)
    with open(os.path.join(data_dir, "longbench_v2", "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(data_dir, "longbench_v2", "items.json"), encoding="utf-8") as f:
        used = {x["id"] for x in json.load(f)}
    print(f"LongBench v2: {len(data)} items", flush=True)
    cand = [d for d in data if d["_id"] not in used and 100_000 <= est_tokens(d["context"]) <= 300_000]
    rng.shuffle(cand)
    edges = [(125_000, 160_000), (160_000, 200_000), (200_000, 258_000)]
    bins = {b: [] for b in range(len(edges))}
    seen = {b: set() for b in bins}
    tried = 0
    for d in cand:
        if all(len(v) >= 4 for v in bins.values()):
            break
        tried += 1
        extra = f"{d['question']} {d['choice_A']} {d['choice_B']} {d['choice_C']} {d['choice_D']}"
        n = tok.count(d["context"])
        if n is None:
            continue
        n += tok.count(extra) + 120  # template + instructions
        for b, (lo, hi) in enumerate(edges):
            if lo <= n < hi and len(bins[b]) < 4 and (d["domain"] not in seen[b] or len(seen[b]) >= 3):
                bins[b].append(lb_item(d, n))
                seen[b].add(d["domain"])
                print(f"  pick bin {b}: {d['_id']} {d['domain']} / {d['difficulty']} {n} tok", flush=True)
                break
    out = [x for b in bins for x in bins[b]]
    with open(os.path.join(data_dir, "longbench_v2", "items_max_cand.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"tried {tried}; candidates {[(x['id'][:8], x['approx_tokens']) for x in out]}", flush=True)
    # the max-context run uses 2 per length bin, domains spread (picked by hand in the reference runs; here: the first
    # candidate of each bin, then the first one of another domain, else the next one)
    pick = []
    for b in bins:
        first = bins[b][:1]
        rest = sorted(bins[b][1:], key=lambda x: x["domain"] == first[0]["domain"]) if first else []
        pick += first + rest[:1]
    with open(os.path.join(data_dir, "longbench_v2", "items_max.json"), "w", encoding="utf-8") as f:
        json.dump(pick, f, indent=1)
    print(f"items_max.json: {len(pick)} items (edit the file to choose differently)", flush=True)


def prep_max_niah(tok, data_dir, lengths):
    rng = random.Random(20261014)
    hay = haystack(data_dir)
    sample = hay[:400_000]
    ratio = len(sample) / tok.count(sample)
    print(f"haystack: {len(hay)} chars, {ratio:.3f} chars / token, ~{len(hay) / ratio:.0f} tokens", flush=True)
    items = []
    for length in lengths:
        for k in range(2):
            keys = rng.sample(KEYS, 4)
            vals = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
            depths = sorted(rng.sample(DEPTHS, 4))
            target = keys[[0, 3][k]]  # an early and a late needle
            r = ratio  # chars / token for this length; corrected below (it drifts over the book)
            for _ in range(4):
                prompt = niah_prompt(hay, length, r, keys, vals, depths, target)
                n = tok.count(prompt)
                if abs(n - length) <= 600:
                    break
                r *= length / n  # too few tokens -> more chars per target token
            it = {"id": f"niahmax_{length // 1000}k_{k}", "length": length, "tokens": n, "key": target,
                  "answer": vals[keys.index(target)], "depth": depths[keys.index(target)], "prompt": prompt}
            items.append(it)
            print(f"  {it['id']}: {n} tokens, asks {target} at depth {it['depth']}", flush=True)
    with open(os.path.join(data_dir, "niah", "items_max.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["long", "max-lb", "max-niah"])
    ap.add_argument("lengths", nargs="*", type=int, help="max-niah: prompt lengths in tokens")
    ap.add_argument("--data-dir", default=os.environ.get("BENCH_DATA", "bench_data"))
    ap.add_argument("--llama-tokenize", default=os.environ.get("LLAMA_TOKENIZE") or shutil.which("llama-tokenize"))
    ap.add_argument("--gguf", default=os.environ.get("BENCH_GGUF"))
    a = ap.parse_args(argv)
    if not a.llama_tokenize or not a.gguf:
        raise SystemExit("need llama-tokenize (--llama-tokenize / $LLAMA_TOKENIZE / PATH) and a GGUF (--gguf / $BENCH_GGUF)")
    if a.what == "max-niah" and not a.lengths:
        raise SystemExit("max-niah needs at least one length")
    tok = Tokenizer(a.llama_tokenize, a.gguf, os.path.join(a.data_dir, "_tok"), 900 if a.what != "long" else 600)
    {"long": lambda: prep_long(tok, a.data_dir), "max-lb": lambda: prep_max_lb(tok, a.data_dir),
     "max-niah": lambda: prep_max_niah(tok, a.data_dir, a.lengths)}[a.what]()
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
