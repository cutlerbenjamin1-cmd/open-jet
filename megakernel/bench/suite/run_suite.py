#!/usr/bin/env python3
"""Start / stop servers from a JSON config and run the suite's clients against each one, one server at a time.

  python3 run_suite.py CONFIG.json [label ...] [--dry-run] [--list]

For every run in the config (or only the labels given): start the server command, wait for GET /health, run the
listed clients (h2h_client.py, qa_client.py, maxctx_client.py) with the run's label, stop the server, wait
`settle_s`, go on.  Output files land in `out_dir`, server logs in out_dir/srv_<label>.log.  The runner refuses to
start while something already listens on the port, so it never measures a server it did not launch.

Config (see example_config.json):
  port, host, out_dir, data_dir, prompt_dir, ready_timeout (s), settle_s (s), vars {name: value}
  systems {name: {"cmd": [argv...], "env": {K: V}}}        how to launch a server
  runs [{"label", "system", "env": {K: V}, "clients": [{"kind": "h2h"|"qa"|"maxctx", "out": file, ...}]}]
Placeholders {port} {host} {out_dir} {data_dir} {prompt_dir} {label} and every key of `vars` are substituted in
commands, env values and client arguments; ${NAME} expands environment variables.  Nothing machine-specific is
built in: model, server and binary paths all come from the config.
Client options: h2h {"mode": "all"|"edit"}, qa {"sets": [..]}, maxctx {"ctx": N} (required).
To hold the GPU for a whole run (a lock, a clock keeper), wrap the runner itself: `gpu-lock python3 run_suite.py ...`.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {"port": 8096, "host": "127.0.0.1", "out_dir": "benchmark_results/suite", "data_dir": "bench_data",
            "prompt_dir": "", "ready_timeout": 600, "settle_s": 15, "vars": {}}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    if not cfg["prompt_dir"]:
        cfg["prompt_dir"] = os.path.join(cfg["data_dir"], "prompts")
    for need in ("systems", "runs"):
        if need not in cfg:
            raise SystemExit(f"config has no '{need}'")
    return cfg


def subst(value, table):
    """Replace {name} for the names in `table` only (JSON braces in arguments stay), then expand ${ENV}."""
    if not isinstance(value, str):
        return value
    out = re.sub(r"\{(\w+)\}", lambda m: str(table[m.group(1)]) if m.group(1) in table else m.group(0), value)
    return os.path.expandvars(out)


def table_for(cfg, run):
    t = {k: cfg[k] for k in ("port", "host", "out_dir", "data_dir", "prompt_dir")}
    t.update(cfg["vars"])
    t["label"] = run["label"]
    return t


def server_command(cfg, run):
    sysd = cfg["systems"].get(run["system"])
    if sysd is None:
        raise SystemExit(f"run {run['label']}: unknown system '{run['system']}'")
    t = table_for(cfg, run)
    cmd = [subst(a, t) for a in sysd["cmd"]]
    env = {k: subst(v, t) for k, v in {**sysd.get("env", {}), **run.get("env", {})}.items()}
    return cmd, env


def client_commands(cfg, run):
    t = table_for(cfg, run)
    port, label = str(cfg["port"]), run["label"]
    cmds = []
    for c in run.get("clients", []):
        out = os.path.join(cfg["out_dir"], subst(c["out"], t))
        kind = c["kind"]
        if kind == "h2h":
            cmds.append([sys.executable, os.path.join(HERE, "h2h_client.py"), port, label, out, c.get("mode", "all"),
                         "--host", cfg["host"], "--prompt-dir", cfg["prompt_dir"]])
        elif kind == "qa":
            cmds.append([sys.executable, "-I", os.path.join(HERE, "qa_client.py"), port, label, out, *c.get("sets", []),
                         "--host", cfg["host"], "--data-dir", cfg["data_dir"]])
        elif kind == "maxctx":
            if "ctx" not in c:
                raise SystemExit(f"run {label}: a maxctx client needs 'ctx'")
            cmds.append([sys.executable, "-I", os.path.join(HERE, "maxctx_client.py"), port, label, out, str(c["ctx"]),
                         "--host", cfg["host"], "--data-dir", cfg["data_dir"]])
        else:
            raise SystemExit(f"run {label}: unknown client kind '{kind}'")
    return cmds


def port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def healthy(proc, host, port, timeout):
    """True once /health answers 200; False as soon as the server process is gone (fail fast) or on timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print(f"server process exited (rc {proc.returncode}) before it was ready", flush=True)
            return False
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_one(cfg, run):
    label = run["label"]
    cmd, env = server_command(cfg, run)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"\n######## {label} start {time.strftime('%H:%M:%S')}\n  {' '.join(cmd)}", flush=True)
    with open(os.path.join(cfg["out_dir"], f"srv_{label}.log"), "w") as log:
        proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            ok = healthy(proc, cfg["host"], cfg["port"], cfg["ready_timeout"])
            print(f"{label}: healthy={ok} after {time.time() - t0:.0f} s", flush=True)
            if ok:
                for c in client_commands(cfg, run):
                    subprocess.run(c)
        finally:
            stop(proc)
    time.sleep(cfg["settle_s"])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("labels", nargs="*", help="run only these labels (default: all, in config order)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, start nothing")
    ap.add_argument("--list", action="store_true", help="list the labels and exit")
    a = ap.parse_intermixed_args(argv)  # options may sit between the labels (plain parse_args rejects that on <= 3.12)
    cfg = load_config(a.config)
    runs = [r for r in cfg["runs"] if not a.labels or r["label"] in a.labels]
    unknown = set(a.labels) - {r["label"] for r in cfg["runs"]}
    if unknown:
        raise SystemExit(f"unknown labels: {sorted(unknown)}")
    if a.list:
        for r in cfg["runs"]:
            print(r["label"])
        return 0
    if a.dry_run:
        for r in runs:
            cmd, env = server_command(cfg, r)
            print(f"[{r['label']}] env {env}\n  server: {' '.join(cmd)}")
            for c in client_commands(cfg, r):
                print("  client:", " ".join(c))
        return 0
    if port_in_use(cfg["host"], cfg["port"]):
        sys.exit(f"something already listens on {cfg['host']}:{cfg['port']}: not starting (stop it or change `port`)")
    failed = [r["label"] for r in runs if not run_one(cfg, r)]
    print("SUITE_DONE" + (f" (not healthy: {failed})" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
