#!/usr/bin/env python3
"""Head-to-head client for an OpenAI-compatible server (mk_server.py or llama-server): speed, time to first token,
prefill, decode at depth, prompt reuse (follow-up turn, edits behind the last turn), VRAM.

  python3 h2h_client.py <port> <label> <out.jsonl> [all|edit] [--prompt-dir DIR] [--host H] [--gpu N]

  all : every test, then (if the 101k prompt reused a cached prefix) a from-scratch rerun, then the edit scenarios
  edit: warm-up + the edit scenarios only

Uses the server-reported `timings` (prompt_ms / predicted_n / predicted_ms) plus the client wall time and appends one
JSON record per request to <out.jsonl> (label = the system under test).  The long prompts pf38k.txt / pf101k.txt come
from --prompt-dir (default $BENCH_PROMPT_DIR, else ./bench_data/prompts); make_prompts.py builds them.
The same request body goes to every system; greedy runs use temperature 0 / top_k 1 with thinking off; the
"sampled_think" runs use temperature 1.0, top_p 0.95, top_k 20 and fixed seeds.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

PROMPTS = [
    "Write a Python function that checks whether a number is prime, with a docstring and a few doctests.",
    "Explain how a hash map works internally, including collision handling and resizing, for a junior developer.",
    "Write a short story (about 300 words) about a lighthouse keeper who discovers a message in a bottle.",
    "Give me a step-by-step plan for migrating a Flask REST API to FastAPI, including testing and deployment.",
]
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
GREEDY = {"temperature": 0.0, "top_k": 1}


class Client:
    def __init__(self, host, port, label, out_path, prompt_dir, gpu=None):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.label = label
        self.prompt_dir = prompt_dir
        self.gpu = gpu
        self.out = open(out_path, "a", encoding="utf-8")

    def prompt_file(self, name):
        with open(os.path.join(self.prompt_dir, name), encoding="utf-8") as f:
            return f.read()

    def smi(self, fields):
        cmd = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        if self.gpu is not None:
            cmd += ["-i", str(self.gpu)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]

    def vram(self):
        return int(self.smi("memory.used")[0])

    def log_vram(self, test):
        self.out.write(json.dumps({"label": self.label, "test": test, "mib": self.vram()}) + "\n")
        self.out.flush()

    def gpu_state(self, tag):  # cooling moves whole sessions by ~7-11%: record it with every run
        t, sm, mem, fan, pw = self.smi("temperature.gpu,clocks.sm,clocks.mem,fan.speed,power.draw")
        self.out.write(json.dumps({"label": self.label, "test": f"gpu_{tag}", "temp_c": t, "sm_mhz": sm, "mem_mhz": mem,
                                   "fan_pct": fan, "power_w": pw}) + "\n")
        print(f"== {self.label}: GPU {tag}: {t} C, SM {sm} MHz, mem {mem} MHz, fan {fan}%, {pw} W", flush=True)

    def req(self, test, messages, **kw):
        body = {"messages": messages, "stream": False, **kw}
        t0 = time.perf_counter()
        r = urllib.request.urlopen(urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                                          headers={"Content-Type": "application/json"}), timeout=1800)
        d = json.loads(r.read())
        wall = time.perf_counter() - t0
        t = d.get("timings", {})
        u = d.get("usage", {})
        msg = d["choices"][0]["message"]
        pn, pms = t.get("predicted_n") or u.get("completion_tokens"), t.get("predicted_ms")
        rec = {"label": self.label, "test": test, "wall_s": round(wall, 3),
               "prompt_tokens": u.get("prompt_tokens"),
               "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "prompt_ms": t.get("prompt_ms"), "pred_n": pn, "pred_ms": pms,
               "decode_tps": round(pn / (pms / 1e3), 2) if pn and pms else None,
               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
               "finish": d["choices"][0].get("finish_reason")}
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()
        print(f"{self.label:9s} {test:22s} prompt {rec['prompt_tokens']} (cached {rec['cached']}) "
              f"{rec['prompt_ms'] and round(rec['prompt_ms'])} ms | {pn} tok {rec['decode_tps']} tok/s | wall {rec['wall_s']} s",
              flush=True)
        return msg, rec

    def other(self):  # short unrelated request: the next prompt shares no prefix with the engine's cache
        self.req("reset_short", [{"role": "user", "content": "Say hi."}], max_tokens=4, **GREEDY, **NO_THINK)

    def edit(self):
        """Prompt reuse when the conversation changes behind the last turn."""
        doc = self.prompt_file("pf38k.txt")[:30000]
        kw = dict(max_tokens=96, **GREEDY, **NO_THINK)
        self.other()
        m = [{"role": "user", "content": doc + "\n\nSummarize the text above in three bullet points."}]
        a, _ = self.req("edit_turn1", m, **kw)
        m += [{"role": "assistant", "content": a.get("content") or ""},
              {"role": "user", "content": "List three risks the text mentions."}]
        a, _ = self.req("edit_turn2", m, **kw)
        m += [{"role": "assistant", "content": a.get("content") or ""},
              {"role": "user", "content": "Which one is the worst, and why?"}]
        self.req("edit_turn3", m, **kw)
        m2 = m[:3] + [{"role": "user", "content": "List three benefits the text mentions."}]  # turn 2's user message edited
        self.req("edit_turn2_edited", m2, **kw)
        self.other()
        self.req("edit_turn2_edited_fresh", m2, cache_prompt=False, **kw)  # the same prompt without reuse
        sysmsg = doc[:24000] + "\n\nYou are a helpful assistant. Today is Monday."
        ms = [{"role": "system", "content": sysmsg},
              {"role": "user", "content": "What day is it, and what is the text about?"}]
        self.req("edit_system", ms, **kw)
        ms2 = [{"role": "system", "content": sysmsg.replace("Monday", "Tuesday")}, ms[1]]
        self.req("edit_system_edited", ms2, **kw)
        self.other()
        self.req("edit_system_edited_fresh", ms2, cache_prompt=False, **kw)

    def run(self, mode):
        print(f"== {self.label}: idle VRAM {self.vram()} MiB", flush=True)
        self.log_vram("vram_idle")
        self.req("warmup", [{"role": "user", "content": "Say hi."}], max_tokens=8, **GREEDY, **NO_THINK)
        # ~3 s of decode first so the timed requests start at full GPU clocks (a card that idled during the server
        # load would otherwise run the first requests at ramping clocks)
        self.req("warmup_clocks", [{"role": "user", "content": PROMPTS[2]}], max_tokens=512, **GREEDY, **NO_THINK)
        self.gpu_state("start")
        if mode == "all":
            for i, q in enumerate(["What is the capital of Australia? One word.",
                                   "Name a prime number above 100. Just the number.",
                                   "Translate 'good morning' into Spanish. Only the translation."]):
                self.req(f"ttft_short_{i}", [{"role": "user", "content": q}], max_tokens=1, **GREEDY, **NO_THINK)
            for i, p in enumerate(PROMPTS):
                self.req(f"greedy_nothink_p{i}", [{"role": "user", "content": p}], max_tokens=512, **GREEDY, **NO_THINK)
            for s in (11, 22):
                for i, p in enumerate(PROMPTS):
                    self.req(f"sampled_think_p{i}_s{s}", [{"role": "user", "content": p}], max_tokens=768,
                             temperature=1.0, top_p=0.95, top_k=20, seed=s)
            doc = self.prompt_file("pf38k.txt")
            m = [{"role": "user", "content": doc[:30000] + "\n\nSummarize the text above in three bullet points."}]
            r1, _ = self.req("followup_turn1", m, max_tokens=200, **GREEDY)
            m += [{"role": "assistant", "content": r1.get("content") or "",
                   "reasoning_content": r1.get("reasoning_content") or ""},
                  {"role": "user", "content": "Now name two risks the text mentions."}]
            self.req("followup_turn2", m, max_tokens=200, **GREEDY)
            q101, rec = None, {}
            for name in ("pf38k", "pf101k"):
                doc = self.prompt_file(f"{name}.txt")
                q101 = [{"role": "user", "content": doc + "\n\nWhat is the main purpose of the code above? Answer in detail."}]
                _, rec = self.req(f"long_{name}", q101, max_tokens=256, **GREEDY, **NO_THINK)
                self.log_vram(f"vram_after_{name}")
            if rec.get("cached"):  # pf101k reused pf38k's prefix: also time it from scratch
                self.other()
                self.req("long_pf101k_fresh", q101, max_tokens=16, cache_prompt=False, **GREEDY, **NO_THINK)
            print(f"== {self.label}: VRAM after long prompts {self.vram()} MiB", flush=True)
        self.edit()
        self.gpu_state("end")
        self.out.flush()
        print("CLIENT_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("port", type=int)
    ap.add_argument("label")
    ap.add_argument("out")
    ap.add_argument("mode", nargs="?", default="all", choices=["all", "edit"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--prompt-dir", default=os.environ.get("BENCH_PROMPT_DIR", os.path.join("bench_data", "prompts")))
    ap.add_argument("--gpu", default=os.environ.get("BENCH_GPU"), help="nvidia-smi GPU index for VRAM / clock records")
    a = ap.parse_args(argv)
    Client(a.host, a.port, a.label, a.out, a.prompt_dir, a.gpu).run(a.mode)


if __name__ == "__main__":
    main()
