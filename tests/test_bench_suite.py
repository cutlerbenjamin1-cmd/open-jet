import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock
import uuid


SUITE = Path(__file__).resolve().parents[1] / "megakernel" / "bench" / "suite"


def load(name):
    spec = importlib.util.spec_from_file_location(f"suite_{name}", SUITE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_suite = load("run_suite")
h2h_client = load("h2h_client")
h2h_summ = load("h2h_summ")
make_prompts = load("make_prompts")
qa_client = load("qa_client")
qa_score = load("qa_score")
qa_prep_long = load("qa_prep_long")
cli_bench = load("cli_bench")
for mod in (run_suite, h2h_client, h2h_summ, make_prompts, qa_client, qa_score, qa_prep_long, cli_bench):
    assert mod.main  # every script is importable without side effects


class Work(unittest.TestCase):
    def setUp(self):
        self.work = Path(__file__).parent / (".bench-suite-test-" + uuid.uuid4().hex)
        self.work.mkdir()
        self.addCleanup(shutil.rmtree, self.work)


class RunnerTests(Work):
    CFG = {
        "port": 9001, "out_dir": "OUT", "vars": {"model": "m.gguf"},
        "systems": {
            "mk": {"cmd": ["srv", "-m", "{model}", "--port", "{port}"], "env": {"A": "{label}"}},
            "llama": {"cmd": ["ll", "--chat-template-kwargs", "{\"preserve_thinking\": true}"]},
        },
        "runs": [
            {"label": "x", "system": "mk", "env": {"B": "1"},
             "clients": [{"kind": "h2h", "mode": "edit", "out": "{label}.jsonl"}, {"kind": "qa", "out": "qa.jsonl", "sets": ["gsm8k"]},
                         {"kind": "maxctx", "out": "m.jsonl", "ctx": 1000}]},
            {"label": "y", "system": "llama"},
        ],
    }

    def cfg(self):
        p = self.work / "c.json"
        p.write_text(json.dumps(self.CFG))
        return run_suite.load_config(str(p))

    def test_substitution_keeps_json_braces_and_expands_env(self):
        t = {"port": 1}
        self.assertEqual(run_suite.subst("{\"a\": {port}}", t), "{\"a\": 1}")
        with mock.patch.dict("os.environ", {"BENCH_T": "v"}):
            self.assertEqual(run_suite.subst("${BENCH_T}/x", t), "v/x")

    def test_server_and_client_commands(self):
        cfg = self.cfg()
        cmd, env = run_suite.server_command(cfg, cfg["runs"][0])
        self.assertEqual(cmd, ["srv", "-m", "m.gguf", "--port", "9001"])
        self.assertEqual(env, {"A": "x", "B": "1"})
        cmd, _ = run_suite.server_command(cfg, cfg["runs"][1])
        self.assertEqual(cmd[-1], "{\"preserve_thinking\": true}")
        h2h, qa, mx = run_suite.client_commands(cfg, cfg["runs"][0])
        self.assertEqual(h2h[2:5], ["9001", "x", str(Path("OUT") / "x.jsonl")])
        self.assertEqual(h2h[5], "edit")
        self.assertIn("-I", qa)
        self.assertEqual(qa[qa.index("gsm8k") - 1], str(Path("OUT") / "qa.jsonl"))
        self.assertEqual(mx[mx.index("1000") - 1], str(Path("OUT") / "m.jsonl"))

    def test_unknown_system_and_client_kind_fail(self):
        cfg = self.cfg()
        cfg["runs"][1]["system"] = "nope"
        with self.assertRaises(SystemExit):
            run_suite.server_command(cfg, cfg["runs"][1])
        cfg["runs"][0]["clients"] = [{"kind": "zzz", "out": "o"}]
        with self.assertRaises(SystemExit):
            run_suite.client_commands(cfg, cfg["runs"][0])

    def test_dry_run_starts_nothing(self):
        p = self.work / "c.json"
        p.write_text(json.dumps(self.CFG))
        with mock.patch.object(run_suite.subprocess, "Popen") as popen, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(run_suite.main([str(p), "--dry-run", "x"]), 0)
        popen.assert_not_called()
        self.assertIn("srv -m m.gguf --port 9001", out.getvalue())
        self.assertNotIn("[y]", out.getvalue())

    def test_refuses_busy_port_and_unknown_label(self):
        p = self.work / "c.json"
        p.write_text(json.dumps(self.CFG))
        with mock.patch.object(run_suite, "port_in_use", return_value=True):
            with self.assertRaises(SystemExit):
                run_suite.main([str(p)])
        with self.assertRaises(SystemExit):
            run_suite.main([str(p), "--dry-run", "nope"])

    def test_health_fails_fast_when_server_exits(self):
        proc = mock.Mock()
        proc.poll.return_value = 3
        proc.returncode = 3
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(run_suite.healthy(proc, "127.0.0.1", 9, 5))

    def test_example_config_loads(self):
        cfg = run_suite.load_config(str(SUITE / "example_config.json"))
        for r in cfg["runs"]:
            run_suite.server_command(cfg, r)
            run_suite.client_commands(cfg, r)


class H2HTests(Work):
    def test_request_record_and_cached_tokens(self):
        c = h2h_client.Client("127.0.0.1", 1, "lab", str(self.work / "o.jsonl"), str(self.work))
        self.addCleanup(c.out.close)
        body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}},
                "timings": {"prompt_ms": 20.0, "predicted_n": 8, "predicted_ms": 2000.0}}
        resp = mock.MagicMock()
        resp.read.return_value = json.dumps(body).encode()
        with mock.patch.object(h2h_client.urllib.request, "urlopen", return_value=resp) as uo, \
                contextlib.redirect_stdout(io.StringIO()):
            msg, rec = c.req("t", [{"role": "user", "content": "x"}], max_tokens=1, cache_prompt=False)
        sent = json.loads(uo.call_args.args[0].data)
        self.assertFalse(sent["cache_prompt"])
        self.assertFalse(sent["stream"])
        self.assertEqual((rec["cached"], rec["decode_tps"], rec["prompt_ms"]), (4, 4.0, 20.0))
        self.assertEqual(msg["content"], "hi")
        c.out.flush()
        self.assertEqual(json.loads((self.work / "o.jsonl").read_text())["test"], "t")

    def test_summary_pools_sessions_and_marks_missing(self):
        def f(name, recs):
            p = self.work / name
            p.write_text("\n".join(json.dumps(r) for r in recs))
            return str(p)
        a = f("a.jsonl", [{"label": "l", "test": "greedy_nothink_p0", "decode_tps": 100.0},
                          {"label": "l", "test": "greedy_nothink_p1", "decode_tps": 110.0},
                          {"label": "l", "test": "ttft_short_0", "prompt_ms": 50.0},
                          {"label": "l", "test": "long_pf101k", "prompt_ms": 20000.0, "cached": 0},
                          {"label": "l", "test": "vram_idle", "mib": 2048}])
        b = f("b.jsonl", [{"label": "l", "test": "greedy_nothink_p0", "decode_tps": 120.0},
                          {"label": "l", "test": "ttft_short_0", "prompt_ms": 40.0}])
        rows = {r["key"]: r for r in h2h_summ.sweep([("one", [(a, "l")]), ("pool", [(a, "l"), (b, "l")])])}
        self.assertEqual(rows["greedy"]["one"], 105.0)
        self.assertEqual(rows["greedy"]["pool"], 120.0)  # best over sessions for "up" metrics
        self.assertEqual(rows["ttft"]["pool"], 40.0)  # min for "down" metrics
        self.assertEqual(rows["greedy"]["sessions"]["pool"], [105.0, 120.0])
        self.assertEqual(rows["pf101k"]["one"], 20.0)
        self.assertEqual(rows["vram_idle"]["one"], 2.0)
        self.assertIsNone(rows["dec38k"]["one"])
        self.assertEqual(h2h_summ.parse_system("n=C:\\x.jsonl:lab+y.jsonl:l2")[1], [("C:\\x.jsonl", "lab"), ("y.jsonl", "l2")])
        with self.assertRaises(SystemExit):
            h2h_summ.parse_system("broken")


class PromptTests(Work):
    def test_pf101k_starts_with_pf38k_and_sizes_are_exact(self):
        src = self.work / "a.txt"
        src.write_text("é" * 80 + "x" * 400, encoding="utf-8")
        ex = self.work / "b.txt"
        ex.write_text("y" * 400, encoding="utf-8")
        p38, p101, p64 = make_prompts.build([str(src)], [str(ex)], 300, 700, 500)
        self.assertEqual(len(p38.encode()), 300)
        self.assertEqual(len(p101.encode()), 700)
        self.assertTrue(p101.startswith(p38) and p101.startswith(p64))
        with self.assertRaises(SystemExit):
            make_prompts.build([str(src)], [str(ex)], 300, 5000)


class QualityTests(Work):
    def test_scorers(self):
        self.assertTrue(qa_score.s_gsm8k({"answer": "1200"}, "so ... Answer: $1,200."))
        self.assertFalse(qa_score.s_gsm8k({"answer": "7"}, "Answer: 8"))
        self.assertTrue(qa_score.s_mmlu_pro({"answer": "C"}, "thus The answer is (C)"))
        self.assertTrue(qa_score.s_arc_c({"answer": "B"}, " B"))
        self.assertTrue(qa_score.s_math({"answer": "\\frac{1}{2}"}, "so \\boxed{\\dfrac{1}{2}}"))
        self.assertTrue(qa_score.s_math({"answer": "0.5"}, "\\boxed{.5}"))  # numeric fallback
        self.assertEqual(qa_score.boxed("\\boxed{a{b}c}"), "a{b}c")
        self.assertTrue(qa_score.s_niah({"answer": "1234567"}, "It is 1234567."))
        self.assertTrue(qa_score.s_long({"answer": "D"}, "The correct answer is (D)"))
        self.assertFalse(qa_score.s_long({"answer": "D"}, "no idea"))

    def test_humaneval_runs_tests_in_isolated_python(self):
        it = {"id": "HumanEval/0", "prompt": "def f(x):\n", "entry_point": "f",
              "test": "def check(c):\n    assert c(2) == 4\n"}
        self.assertTrue(qa_score.s_humaneval(it, "```python\ndef f(x):\n    return x * 2\n```", str(self.work)))
        self.assertFalse(qa_score.s_humaneval(it, "```python\ndef f(x):\n    return 0\n```", str(self.work)))

    def test_score_files_and_pairing(self):
        d = self.work / "data"
        (d / "gsm8k").mkdir(parents=True)
        (d / "gsm8k" / "items.json").write_text(json.dumps([{"id": "g0", "question": "q", "answer": "4"},
                                                          {"id": "g1", "question": "q", "answer": "5"}]))
        recs = self.work / "r.jsonl"
        rows = [{"label": lb, "set": "gsm8k", "id": i, "ok": True, "content": f"Answer: {v}", "wall_s": 1.0}
                for lb, vals in (("A", (4, 5)), ("B", (4, 0))) for i, v in zip(("g0", "g1"), vals)]
        rows.append({"label": "A", "set": "gsm8k", "id": "g1", "ok": False})  # failed requests are not scored
        recs.write_text("\n".join(json.dumps(r) for r in rows))
        items = qa_score.load_items(str(d))
        res = qa_score.score(qa_score.load_records([str(recs)]), items, str(self.work))
        pair = qa_score.pair_table(res, "A", "B")["gsm8k"]
        self.assertEqual((pair["n"], pair["A"], pair["B"], pair["both"], pair["only_A"]), (2, 2, 1, 1, 1))

    def test_client_prompts_and_resume(self):
        p, n, think = qa_client.p_arc_c({"question": "Q", "labels": ["A", "B"], "texts": ["x", "y"]})
        self.assertIn("(B) y", p)
        self.assertEqual((n, think), (8, False))
        self.assertTrue(qa_client.p_math_think({"problem": "p"})[2])
        out = self.work / "o.jsonl"
        out.write_text(json.dumps({"label": "L", "set": "gsm8k", "id": "a", "ok": True}) + "\n" +
                       json.dumps({"label": "L", "set": "gsm8k", "id": "b", "ok": False}) + "\nnot json\n")
        self.assertEqual(qa_client.answered(str(out), "L"), {("gsm8k", "a")})

    def test_niah_prompt_has_all_needles_and_question(self):
        hay = "\n".join(f"line {i} of the book" for i in range(2000))
        keys, vals, depths = qa_prep_long.KEYS[:4], ["1111111", "2222222", "3333333", "4444444"], [0.1, 0.3, 0.5, 0.8]
        prompt = qa_prep_long.niah_prompt(hay, 3000, 3.0, keys, vals, depths, keys[2])
        for k, v in zip(keys, vals):
            self.assertIn(f"One of the special magic numbers for {k} is: {v}.", prompt)
        self.assertTrue(prompt.startswith(qa_prep_long.HEAD))
        self.assertTrue(prompt.endswith(f"for {keys[2]} mentioned in the provided text? Answer with the number only."))


class CliBenchTests(Work):
    def test_engine_config(self):
        p = self.work / "e.json"
        p.write_text(json.dumps({"engines": [{"label": "a", "binary": "b", "model": "~/m.gguf"}]}))
        e = cli_bench.load_engines(str(p))[0]
        self.assertEqual((e["depth"], e["env"]), (4, {}))
        self.assertNotIn("~", e["model"])
        p.write_text(json.dumps([{"label": "a", "binary": "b", "model": "m"}, {"label": "a", "binary": "b", "model": "m"}]))
        with self.assertRaises(SystemExit):
            cli_bench.load_engines(str(p))

    def test_run_kept_only_wraps_the_cli_and_stops_keeper_at_load(self):
        real = mock.Mock(return_value="real")
        wrapped = cli_bench.make_run_kept(real, "mk", "keep")
        self.assertEqual(wrapped(["other"]), "real")
        keep, cli = mock.Mock(), mock.Mock()
        keep.poll.return_value = None
        cli.stderr = iter(["[mk] load 1.0 s\n", "[mk] decode ...\n"])
        cli.stdout.read.return_value = "out"
        cli.returncode = 0
        with mock.patch.object(cli_bench.subprocess, "Popen", side_effect=[keep, cli]):
            res = wrapped(["mk", "-m", "x"], env={})
        keep.kill.assert_called()
        self.assertEqual(res.stdout, "out")
        self.assertIn("[mk] decode", res.stderr)


if __name__ == "__main__":
    unittest.main()
