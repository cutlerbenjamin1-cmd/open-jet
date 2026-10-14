#!/usr/bin/env python3
"""Benchmark-question client for a quality A/B against an OpenAI-compatible server (mk_server.py or llama-server).

  python3 -I qa_client.py <port> <label> <out.jsonl> [set ...] [--host H] [--data-dir DIR]

Sets (default: all, in this order): gsm8k, mmlu_pro, arc_c, math500, humaneval, niah, longbench_v2, math500_think.
Greedy (temperature 0, top_k 1), the same zero-shot prompts for every engine; thinking off except math500_think.
Records the answer text, token counts and server timings per item; scoring is offline (qa_score.py).
Resumes: items this label already answered ("ok") in <out.jsonl> are skipped.
Item files come from qa_fetch.py / qa_prep_long.py (untrusted downloads): their text only goes into prompts and
nothing in them is executed here.  --data-dir defaults to $BENCH_DATA, else ./bench_data.
"""
import argparse
import json
import os
import time
import urllib.request

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
GREEDY = {"temperature": 0.0, "top_k": 1}
LET = "ABCDEFGHIJ"


def p_gsm8k(it):
    return (f"Solve the following math problem. Show your work briefly, then give the final answer on the last line as "
            f"'Answer: <number>'.\n\n{it['question']}"), 768, False


def p_mmlu_pro(it):
    opts = "\n".join(f"({LET[i]}) {o}" for i, o in enumerate(it["options"]))
    return (f"The following is a multiple choice question about {it['category']}. Think step by step, then finish with "
            f"'The answer is (X)' where X is the letter of the correct option.\n\nQuestion: {it['question']}\nOptions:\n{opts}"), 1024, False


def p_arc_c(it):
    opts = "\n".join(f"({l}) {t}" for l, t in zip(it["labels"], it["texts"]))
    return f"{it['question']}\n{opts}\n\nAnswer with the letter of the correct option only.", 8, False


def p_math(it):
    return f"Solve the following problem. Put the final answer in \\boxed{{}}.\n\n{it['problem']}", 1536, False


def p_math_think(it):
    return f"Solve the following problem. Put the final answer in \\boxed{{}}.\n\n{it['problem']}", 8192, True


def p_humaneval(it):
    return ("Complete the following Python function. Reply with the complete function (including the signature and any "
            f"imports it needs) in one ```python code block.\n\n```python\n{it['prompt']}```"), 768, False


def p_long(it, max_tokens=128):  # LongBench v2's own prompt (no chain of thought)
    return (f"Please read the following text and answer the question below.\n\n<text>\n{it['context']}\n</text>\n\n"
            f"What is the correct answer to this question: {it['question']}\nChoices:\n(A) {it['choice_A']}\n"
            f"(B) {it['choice_B']}\n(C) {it['choice_C']}\n(D) {it['choice_D']}\n\n"
            "Format your response as follows: \"The correct answer is (insert answer here)\"."), max_tokens, False


def p_niah(it):
    return it["prompt"], 64, False


SETS = [("gsm8k", p_gsm8k), ("mmlu_pro", p_mmlu_pro), ("arc_c", p_arc_c), ("math500", p_math), ("humaneval", p_humaneval),
        ("niah", p_niah), ("longbench_v2", p_long), ("math500_think", p_math_think)]


def item_file(data_dir, name):
    return os.path.join(data_dir, name, "items.json")


def answered(out_path, label):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("label") == label and r.get("ok"):
                        done.add((r["set"], r["id"]))
                except Exception:
                    pass
    return done


def ask(url, prompt, max_tokens, think):
    body = {"messages": [{"role": "user", "content": prompt}], "stream": False, "max_tokens": max_tokens, **GREEDY}
    if not think:
        body.update(NO_THINK)
    r = urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=3600)
    return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("port", type=int)
    ap.add_argument("label")
    ap.add_argument("out")
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data-dir", default=os.environ.get("BENCH_DATA", "bench_data"))
    a = ap.parse_args(argv)
    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    want = a.sets or [s for s, _ in SETS]
    done = answered(a.out, a.label)
    out = open(a.out, "a", encoding="utf-8")
    for name, fn in SETS:
        if name not in want:
            continue
        with open(item_file(a.data_dir, name), encoding="utf-8") as f:
            items = json.load(f)
        print(f"== {a.label} {name}: {len(items)} items {time.strftime('%H:%M:%S')}", flush=True)
        for it in items:
            if (name, it["id"]) in done:
                continue
            prompt, max_tokens, think = fn(it)
            rec = {"label": a.label, "set": name, "id": it["id"]}
            t0 = time.perf_counter()
            try:
                d = ask(url, prompt, max_tokens, think)
                msg = d["choices"][0]["message"]
                t, u = d.get("timings", {}), d.get("usage", {})
                rec.update(ok=True, wall_s=round(time.perf_counter() - t0, 3), content=msg.get("content") or "",
                           reasoning_len=len(msg.get("reasoning_content") or ""), finish=d["choices"][0].get("finish_reason"),
                           prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           prompt_ms=t.get("prompt_ms"), pred_n=t.get("predicted_n"), pred_ms=t.get("predicted_ms"))
            except Exception as e:
                rec.update(ok=False, wall_s=round(time.perf_counter() - t0, 3), error=str(e)[:300])
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{a.label:6s} {name:13s} {it['id'][:28]:28s} {'ok ' if rec['ok'] else 'ERR'} {rec.get('prompt_tokens')} -> "
                  f"{rec.get('completion_tokens')} tok, {rec['wall_s']} s", flush=True)
    print("QA_CLIENT_DONE", flush=True)


if __name__ == "__main__":
    main()
