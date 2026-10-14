#!/usr/bin/env python3
"""Max-context set: one server started at the largest context that fits; every long item is prefilled from scratch
(a short unrelated request first), greedy, thinking off.

  python3 -I maxctx_client.py <port> <label> <out.jsonl> <ctx> [--host H] [--data-dir DIR]

  speed : the longest NIAH haystack prompt that fits, asked for a 512-token summary: prefill from scratch at ~max
          ctx, decode at max depth; then a follow-up turn on the same chat (prompt reuse at max ctx); VRAM after
  niah  : <data-dir>/niah/items_max.json (RULER-style multi-key, 4 needles, asks one; max_tokens 64): the 190k pair
          and the pair at the largest length that fits (qa_prep_long.py max-niah builds them)
  lb    : <data-dir>/longbench_v2/items_max.json, the items that fit with 1024 answer tokens: LongBench v2's own
          no-CoT prompt but max_tokens 1024 (128 cut most answers short at this length)
Records go to <out.jsonl> (test, set, id, timings, VRAM); qa_score.py scores the niah_max / longbench_max sets and
h2h_summ.py --maxctx summarises the speed records.  Item files are untrusted downloads: their text only goes into
prompts, nothing is executed here.  Run with `python -I`.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
GREEDY = {"temperature": 0.0, "top_k": 1}


class Run:
    def __init__(self, host, port, label, out_path, ctx, data_dir, gpu=None):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.label, self.ctx, self.d, self.gpu = label, ctx, data_dir, gpu
        self.out = open(out_path, "a", encoding="utf-8")

    def vram(self):
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.gpu is not None:
            cmd += ["-i", str(self.gpu)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return int(r.stdout.strip().splitlines()[0])

    def log(self, rec):
        self.out.write(json.dumps({"label": self.label, "ctx": self.ctx, **rec}) + "\n")
        self.out.flush()

    def req(self, test, messages, max_tokens, extra=None, **kw):
        body = {"messages": messages, "stream": False, "max_tokens": max_tokens, **GREEDY, **NO_THINK, **kw}
        rec = {"test": test, **(extra or {})}
        t0 = time.perf_counter()
        try:
            r = urllib.request.urlopen(urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                                              headers={"Content-Type": "application/json"}), timeout=3600)
            d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            t, u = d.get("timings", {}), d.get("usage", {})
            pn, pms = t.get("predicted_n") or u.get("completion_tokens"), t.get("predicted_ms")
            rec.update(ok=True, wall_s=round(time.perf_counter() - t0, 3), content=msg.get("content") or "",
                       finish=d["choices"][0].get("finish_reason"), prompt_tokens=u.get("prompt_tokens"),
                       cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), prompt_ms=t.get("prompt_ms"),
                       pred_n=pn, pred_ms=pms, decode_tps=round(pn / (pms / 1e3), 2) if pn and pms else None)
        except Exception as e:
            msg = {}
            rec.update(ok=False, wall_s=round(time.perf_counter() - t0, 3), error=str(e)[:300])
        self.log(rec)
        pt, pms = rec.get("prompt_tokens"), rec.get("prompt_ms")
        print(f"{self.label:8s} {test:28s} {'ok ' if rec['ok'] else 'ERR'} prompt {pt} (cached {rec.get('cached')}) "
              f"{pms and round(pms)} ms{f' = {pt / (pms / 1e3):.0f} tok/s' if pt and pms else ''} | {rec.get('pred_n')} tok "
              f"{rec.get('decode_tps')} tok/s | wall {rec['wall_s']} s {rec.get('error', '')}", flush=True)
        return msg, rec

    def other(self):  # short unrelated request: the next prompt shares no prefix with the engine's cache
        self.req("reset_short", [{"role": "user", "content": "Say hi."}], 4)

    def items(self, rel):
        with open(os.path.join(self.d, rel), encoding="utf-8") as f:
            return json.load(f)

    def run(self):
        ctx = self.ctx
        self.log({"test": "vram_idle", "mib": self.vram()})
        self.req("warmup", [{"role": "user", "content": "Say hi."}], 8)
        self.req("warmup_clocks", [{"role": "user", "content": "Write a short story (about 300 words) about a lighthouse keeper."}], 512)
        niah = self.items(os.path.join("niah", "items_max.json"))
        lb = self.items(os.path.join("longbench_v2", "items_max.json"))
        fit_n = [it for it in niah if it["tokens"] + 64 + 64 <= ctx]
        top_len = max(it["length"] for it in fit_n)  # the 190k pair + the pair at the largest length that fits
        fit_n = [it for it in fit_n if it["length"] in (190_000, top_len)]
        fit_lb = [it for it in lb if it["approx_tokens"] + 1024 + 64 <= ctx]
        print(f"== {self.label} ctx {ctx}: niah {len(fit_n)} / {len(niah)} fit, longbench {len(fit_lb)} / {len(lb)} fit "
              f"{time.strftime('%H:%M:%S')}", flush=True)

        # speed at max context: the longest NIAH haystack, a long answer, then a follow-up turn
        big = max(fit_n, key=lambda it: it["tokens"])
        text = big["prompt"].rsplit("\n\nWhat is the special magic number", 1)[0]
        m = [{"role": "user", "content": text + "\n\nSummarize the story told in the text above in detail."}]
        self.other()
        a, _ = self.req("speed_max_turn1", m, 512, {"src": big["id"]})
        self.log({"test": "vram_after_max", "mib": self.vram()})
        m += [{"role": "assistant", "content": a.get("content") or ""},
              {"role": "user", "content": "Now name five characters from the text and one sentence about each."}]
        self.req("speed_max_turn2", m, 256)

        for it in fit_n:
            self.other()
            _, r = self.req(f"niah_{it['id']}", [{"role": "user", "content": it["prompt"]}], 64,
                            {"set": "niah_max", "id": it["id"], "answer": it["answer"], "tokens_est": it["tokens"]})
            print(f"   niah {it['id']}: answer {it['answer']} -> {'HIT' if it['answer'] in (r.get('content') or '') else 'MISS'}",
                  flush=True)

        for it in fit_lb:
            prompt = (f"Please read the following text and answer the question below.\n\n<text>\n{it['context']}\n</text>\n\n"
                      f"What is the correct answer to this question: {it['question']}\nChoices:\n(A) {it['choice_A']}\n"
                      f"(B) {it['choice_B']}\n(C) {it['choice_C']}\n(D) {it['choice_D']}\n\n"
                      "Format your response as follows: \"The correct answer is (insert answer here)\".")
            self.other()
            self.req(f"lb_{it['id'][:12]}", [{"role": "user", "content": prompt}], 1024,
                     {"set": "longbench_max", "id": it["id"], "answer": it["answer"], "domain": it["domain"],
                      "difficulty": it["difficulty"], "tokens_est": it["approx_tokens"]})
        self.log({"test": "vram_end", "mib": self.vram()})
        print("MAXCTX_CLIENT_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("port", type=int)
    ap.add_argument("label")
    ap.add_argument("out")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data-dir", default=os.environ.get("BENCH_DATA", "bench_data"))
    ap.add_argument("--gpu", default=os.environ.get("BENCH_GPU"))
    a = ap.parse_args(argv)
    Run(a.host, a.port, a.label, a.out, a.ctx, a.data_dir, a.gpu).run()


if __name__ == "__main__":
    main()
