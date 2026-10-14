#!/usr/bin/env python3
"""Fetch the benchmark questions for the quality set.  Nothing is committed to the repository; this downloads them.

  python3 -I qa_fetch.py [--data-dir DIR] [set ...]     (sets: gsm8k mmlu_pro arc_c math500 humaneval longbench_v2 niah)

Writes <data-dir>/<set>/items.json (default $BENCH_DATA, else ./bench_data; keep it outside the repository).  Sampling is
deterministic (seeded), so every engine gets the same items:
  gsm8k          50 of the 1319 test questions            openai/gsm8k, config main            (license on the dataset card: MIT)
  mmlu_pro       5 per category x 14 = 70 questions       TIGER-Lab/MMLU-Pro, test              (MIT)
  arc_c          50 of the 1172 test questions            allenai/ai2_arc ARC-Challenge, test   (CC BY-SA 4.0)
  math500        40 (8 per difficulty level) + math500_think: 10 more level-5   HuggingFaceH4/MATH-500 (a subset of the MATH
                 problems from the PRM800K split; MATH is MIT)
  humaneval      every 4th of the 164 problems = 41       openai/openai_humaneval, test         (MIT)
  longbench_v2   the whole 465 MB data.json (503 items); prep_long selects by token count   THUDM/LongBench-v2
  niah           the haystack text: Project Gutenberg #2600 (War and Peace), public domain in the US; see the Project
                 Gutenberg license notice in the file header
One seeded generator draws the samples in the order gsm8k, mmlu_pro, arc_c, math500, so the item lists of the
reference runs are reproduced when those sets are fetched together (the default); a subset gives other items.
The license names are what the dataset cards stated when this was written: check each card before redistributing any
item file.  Small sets use the Hugging Face datasets-server rows API; HumanEval uses its parquet file when pandas and
pyarrow are installed (the rows API rate-limits it).  Downloads are untrusted data: they go to the data directory
only and nothing in them is executed.  Run with `python -I` so no module next to the data is imported.
"""
import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "open-jet-bench-suite/1.0"}
LB_URL = "https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json"
HE_URL = "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet"
HAY_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
SEED = 20261013
ALL = ["gsm8k", "mmlu_pro", "arc_c", "math500", "humaneval", "longbench_v2", "niah"]


def get(url, path=None, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                if path is None:
                    return json.loads(r.read())
                with open(path, "wb") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                return path
        except Exception as e:
            print(f"  retry {i + 1}: {url[:100]}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def rows(dataset, config, split, offset, length=100):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    d = get(f"https://datasets-server.huggingface.co/rows?{q}")
    return [r["row"] for r in d["rows"]]


class Fetcher:
    def __init__(self, data_dir):
        self.d = data_dir
        self.rng = random.Random(SEED)

    def save(self, name, items):
        os.makedirs(os.path.join(self.d, name), exist_ok=True)
        p = os.path.join(self.d, name, "items.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=1)
        print(f"{name}: {len(items)} items -> {p}", flush=True)

    def gsm8k(self):  # 50 of 1319
        g = rows("openai/gsm8k", "main", "test", 0) + rows("openai/gsm8k", "main", "test", 600) + \
            rows("openai/gsm8k", "main", "test", 1200)
        self.save("gsm8k", [{"id": f"gsm8k_{i}", "question": r["question"],
                             "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                            for i, r in enumerate(self.rng.sample(g, 50))])

    def mmlu_pro(self):  # 5 per category (14 categories)
        pool = []
        for off in range(0, 12032, 1500):
            pool += rows("TIGER-Lab/MMLU-Pro", "default", "test", off)
            pool += rows("TIGER-Lab/MMLU-Pro", "default", "test", off + 750)
        bycat = {}
        for r in pool:
            bycat.setdefault(r["category"], []).append(r)
        mp = []
        for cat in sorted(bycat):
            for r in self.rng.sample(bycat[cat], min(5, len(bycat[cat]))):
                mp.append({"id": f"mmlupro_{r['question_id']}", "category": cat, "question": r["question"],
                           "options": r["options"], "answer": r["answer"]})
        self.save("mmlu_pro", mp)

    def arc_c(self):  # 50 of 1172
        a = rows("allenai/ai2_arc", "ARC-Challenge", "test", 0) + rows("allenai/ai2_arc", "ARC-Challenge", "test", 500) + \
            rows("allenai/ai2_arc", "ARC-Challenge", "test", 1000)
        self.save("arc_c", [{"id": r["id"], "question": r["question"], "labels": r["choices"]["label"],
                             "texts": r["choices"]["text"], "answer": r["answerKey"]} for r in self.rng.sample(a, 50)])

    def math500(self):  # 40 for non-thinking (8 per level), 10 other level-5 for thinking mode
        m = []
        for off in range(0, 500, 100):
            m += rows("HuggingFaceH4/MATH-500", "default", "test", off)
        bylvl = {}
        for r in m:
            bylvl.setdefault(int(r["level"]), []).append(r)
        nt, used = [], set()
        for lvl in sorted(bylvl):
            for r in self.rng.sample(bylvl[lvl], 8):
                nt.append(r)
                used.add(r["unique_id"])
        th = self.rng.sample([r for r in bylvl[5] if r["unique_id"] not in used], 10)

        def cv(r):
            return {"id": r["unique_id"], "level": r["level"], "subject": r["subject"], "problem": r["problem"],
                    "answer": r["answer"]}
        self.save("math500", [cv(r) for r in nt])
        self.save("math500_think", [cv(r) for r in th])

    def humaneval(self):  # every 4th of 164 = 41
        try:
            import pandas as pd
            os.makedirs(os.path.join(self.d, "humaneval"), exist_ok=True)
            pq = os.path.join(self.d, "humaneval", "test.parquet")
            if not os.path.exists(pq):
                get(HE_URL, pq, tries=5)
            h = [{"task_id": r.task_id, "prompt": r.prompt, "test": r.test, "entry_point": r.entry_point}
                 for r in pd.read_parquet(pq).itertuples()]
        except ImportError:
            h = rows("openai/openai_humaneval", "openai_humaneval", "test", 0) + \
                rows("openai/openai_humaneval", "openai_humaneval", "test", 100)
        self.save("humaneval", [{"id": r["task_id"], "prompt": r["prompt"], "test": r["test"],
                                 "entry_point": r["entry_point"]} for r in h[::4]])

    def longbench_v2(self):  # the whole file; selection by token count happens in qa_prep_long.py
        os.makedirs(os.path.join(self.d, "longbench_v2"), exist_ok=True)
        lb = os.path.join(self.d, "longbench_v2", "data.json")
        if not os.path.exists(lb) or os.path.getsize(lb) < 400_000_000:
            print("downloading LongBench v2 data.json (465 MB) ...", flush=True)
            get(LB_URL, lb, tries=5)
        print(f"longbench_v2: {os.path.getsize(lb)} bytes", flush=True)

    def niah(self):  # haystack: War and Peace
        os.makedirs(os.path.join(self.d, "niah"), exist_ok=True)
        hs = os.path.join(self.d, "niah", "pg2600.txt")
        if not os.path.exists(hs):
            get(HAY_URL, hs)
        print(f"niah haystack: {os.path.getsize(hs)} bytes", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sets", nargs="*", help=f"default: {' '.join(ALL)}")
    ap.add_argument("--data-dir", default=os.environ.get("BENCH_DATA", "bench_data"))
    a = ap.parse_args(argv)
    bad = set(a.sets) - set(ALL)
    if bad:
        raise SystemExit(f"unknown sets: {sorted(bad)}")
    f = Fetcher(a.data_dir)
    for s in a.sets or ALL:
        getattr(f, s)()
    print("FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
