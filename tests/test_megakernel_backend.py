import asyncio
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import llama_server
from src.llama_server import LlamaServerClient, megakernel_unavailable_reason
from src.runtime_registry import create_runtime_client

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/models/Qwen3.8-27B-Q4_K_M.gguf"


def _load_mk_server():
    path = ROOT / "megakernel" / "server" / "mk_server.py"
    spec = importlib.util.spec_from_file_location("mk_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MegakernelSelectionTests(unittest.TestCase):
    def test_other_models_are_not_served_by_the_megakernel(self) -> None:
        for model in ("/m/Qwen3.6-27B-UD-IQ3_XXS.gguf", "/m/Qwen3.8-27B-Q6_K.gguf", "/m/qwen3.8-27b-q4_k_m.bin",
                      "/m/Qwen3.8-27B-UD-Q4_K_M.gguf", "/m/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"):
            self.assertIn("only supports", megakernel_unavailable_reason(model))

    def test_ud_q4_k_xl_is_served_by_the_megakernel(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value="0"):
            self.assertIsNone(megakernel_unavailable_reason("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf"))

    def test_setup_selector_names_the_runtime(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_gpu", return_value="0"):
            self.assertEqual(llama_server.model_runtime_label(MODEL), "OpenJet megakernel, 1.5x faster")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf"),
                             "OpenJet megakernel, 1.5x faster")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.8-27B-UD-Q4_K_M.gguf"), "llama.cpp")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.5-9B-Q4_K_M.gguf"), "llama.cpp")
            self.assertEqual(llama_server.model_runtime_label("/m/mmproj-Qwen3.8-27B-BF16.gguf"), "llama.cpp")
        with mock.patch.object(llama_server, "_megakernel_gpu", return_value=None):
            self.assertEqual(llama_server.model_runtime_label(MODEL), "llama.cpp")

    def test_missing_build_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=None):
            self.assertIn("not built", megakernel_unavailable_reason(MODEL))

    def test_unsupported_gpu_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value=None):
            self.assertIn("RTX 30/40/50", megakernel_unavailable_reason(MODEL))

    def test_supported_setup_has_no_reason(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value="0"):
            self.assertIsNone(megakernel_unavailable_reason(MODEL))

    def test_backend_setting_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            LlamaServerClient(model=MODEL, backend="vllm")

    def test_registry_passes_the_backend_setting(self) -> None:
        cfg = {"runtime": "llama_cpp", "llama_model": MODEL, "llama_backend": "llama"}
        client = create_runtime_client(cfg)
        self.assertEqual(client.backend, "llama")
        self.assertEqual(create_runtime_client({"runtime": "llama_cpp", "llama_model": MODEL}).backend, "auto")


def _write_gguf(path: Path, types: dict[str, int]) -> None:
    """A GGUF header (no metadata, no data) declaring 1-D tensors of the given ggml types."""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(types), 0))
        for name, kind in types.items():
            raw = name.encode()
            f.write(struct.pack("<Q", len(raw)) + raw + struct.pack("<IQ", 1, 256) + struct.pack("<IQ", kind, 0))


class MegakernelQuantMixTests(unittest.TestCase):
    PROBES = ("blk.0.ssm_out.weight", "blk.0.ssm_alpha.weight", "blk.64.nextn.eh_proj.weight")

    def _mix(self, probe_types: tuple[int, int, int], other: int = 12) -> str | None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "Qwen3.8-27B-Q4_K_M.gguf"
            types = dict(zip(self.PROBES, probe_types))
            types["blk.0.ffn_down.weight"] = other
            _write_gguf(path, types)
            return llama_server._megakernel_quant_mix(str(path))

    def test_benchmarked_mixes_are_native(self) -> None:
        self.assertEqual(self._mix((13, 0, 8)), "native")  # unsloth Q4_K_M
        self.assertEqual(self._mix((8, 8, 14), other=23), "native")  # unsloth UD-Q4_K_XL

    def test_other_readable_mixes_are_semi(self) -> None:
        self.assertEqual(self._mix((12, 12, 12)), "semi")  # plain llama-quantize Q4_K_M
        self.assertEqual(self._mix((14, 0, 8), other=11), "semi")

    def test_types_the_megakernel_cannot_read_fall_back_to_llama(self) -> None:
        self.assertIsNone(self._mix((13, 0, 8), other=2))  # Q4_0
        self.assertIsNone(self._mix((13, 0, 8), other=16))  # IQ2_XXS


def _nvidia_smi(*lines: str):
    return mock.patch.object(
        llama_server.subprocess, "run",
        return_value=subprocess.CompletedProcess([], 0, stdout="\n".join(lines) + "\n", stderr=""),
    )


class MegakernelGpuDetectionTests(unittest.TestCase):
    def test_a_qualifying_gpu_after_the_first_is_found(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 7.5, 8192", "1, GPU-bbb, 8.6, 24576"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertEqual(llama_server._megakernel_gpu(), "1")

    def test_cuda_visible_devices_limits_the_choice(self) -> None:
        lines = ("0, GPU-aaa, 8.6, 24576", "1, GPU-bbb, 7.5, 8192")
        with _nvidia_smi(*lines), mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertIsNone(llama_server._megakernel_gpu())
        with _nvidia_smi(*lines), mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,1"}):
            self.assertEqual(llama_server._megakernel_gpu(), "0")

    def test_small_or_other_arch_gpus_do_not_qualify(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 8.6, 12288", "1, GPU-bbb, 7.5, 24576", "2, GPU-ccc, 9.0, 81559"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertIsNone(llama_server._megakernel_gpu())

    def test_ada_and_blackwell_24gb_gpus_qualify(self) -> None:
        for cap, mem in (("8.9", "24564"), ("12.0", "32607")):
            with _nvidia_smi(f"0, GPU-aaa, {cap}, {mem}"), \
                    mock.patch.dict(llama_server.os.environ, {}, clear=True):
                self.assertEqual(llama_server._megakernel_gpu(), "0")


class MegakernelVisionArgsTests(unittest.TestCase):
    def test_mmproj_is_passed_when_set_and_the_encoder_is_built(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root, mmproj = Path(d), Path(d) / "mmproj.gguf"
            mmproj.write_bytes(b"GGUF")
            with mock.patch.dict(llama_server.os.environ, {"OPENJET_MEGAKERNEL_MMPROJ": str(mmproj)}):
                self.assertEqual(llama_server._megakernel_vision_args(root), [])  # (no libqwen38vision.so)
                (root / "libqwen38vision.so").write_bytes(b"")
                self.assertEqual(llama_server._megakernel_vision_args(root), ["--mmproj", str(mmproj)])
            with mock.patch.dict(llama_server.os.environ, {"OPENJET_MEGAKERNEL_MMPROJ": ""}):
                self.assertEqual(llama_server._megakernel_vision_args(root), [])
            with mock.patch.dict(llama_server.os.environ, {"OPENJET_MEGAKERNEL_MMPROJ": str(root / "missing.gguf")}):
                self.assertEqual(llama_server._megakernel_vision_args(root), [])


class MegakernelStartTests(unittest.TestCase):
    def _client(self, backend: str) -> LlamaServerClient:
        client = LlamaServerClient(model=MODEL, binary="/opt/llama-server", backend=backend)
        client._cleanup_stale_inference_processes = mock.AsyncMock()
        client._resolve_device = mock.Mock(return_value="vulkan")
        return client

    def _run(self, coro):
        return asyncio.run(coro)

    def test_required_megakernel_fails_on_a_non_cuda_device(self) -> None:
        client = self._client("megakernel")
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            with self.assertRaisesRegex(RuntimeError, "desktop CUDA GPU"):
                self._run(client.start())
        client._start_once.assert_not_called()

    def test_required_megakernel_fails_on_jetson(self) -> None:
        client = self._client("megakernel")
        client._resolve_device.return_value = "cuda"
        client._is_jetson_platform = mock.Mock(return_value=True)
        client._ensure_jetson_clocks_sudoers = mock.Mock()
        client._maximize_gpu_clocks = mock.Mock()
        client._prepare_memory_for_launch = mock.AsyncMock(return_value=None)
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            with self.assertRaisesRegex(RuntimeError, "jetson"):
                self._run(client.start())
        client._start_once.assert_not_called()

    def test_fallback_to_llama_server_resets_the_active_backend(self) -> None:
        client = self._client("auto")
        client.active_backend = "megakernel"
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            self._run(client.start())
        client._start_once.assert_called_once()
        self.assertEqual(client.active_backend, "llama.cpp")

    def test_stopping_the_server_stops_the_stderr_reader(self) -> None:
        async def scenario() -> None:
            client = LlamaServerClient(model=MODEL)
            stderr = asyncio.StreamReader()
            client._proc = mock.Mock(returncode=0, stderr=stderr)
            client._stderr_task = asyncio.create_task(client._capture_startup_stderr(llama_server.deque()))
            await asyncio.sleep(0)
            task = client._stderr_task
            await client._stop_server()
            self.assertIsNone(client._proc)
            stderr.feed_data(b"late line\n")
            with self.assertRaises(asyncio.CancelledError):
                await task
            await client._http.aclose()

        self._run(scenario())


class MegakernelServerParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _feed_all(self, parser, chunks):
        out = []
        for c in chunks:
            out += parser.feed(c)
        out += parser.feed("", final=True)
        merged = {"reasoning": "", "content": ""}
        for kind, text in out:
            merged[kind] += text
        return merged

    def test_reasoning_and_content_are_split_across_chunk_boundaries(self) -> None:
        parser = self.mk.OutputParser(thinking=True, tools=[])
        text = "Think about it.\n</think>\n\nThe answer is 4."
        merged = self._feed_all(parser, [text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(merged["reasoning"], "Think about it.\n")
        self.assertEqual(merged["content"], "The answer is 4.")

    def test_tool_call_markup_is_held_back_and_parsed(self) -> None:
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}}}}]
        parser = self.mk.OutputParser(thinking=False, tools=tools)
        text = ("Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n"
                "<parameter=max_lines>\n10\n</parameter>\n</function>\n</tool_call>")
        merged = self._feed_all(parser, [text[i:i + 5] for i in range(0, len(text), 5)])
        self.assertEqual(merged["content"], "Let me look.")
        calls = self.mk.parse_tool_calls(parser.tool_text, tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname", "max_lines": 10})

    def test_string_parameters_are_not_json_decoded(self) -> None:
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "properties": {"cmd": {"type": "string"}}}}}]
        calls = self.mk.parse_tool_calls("<tool_call>\n<function=run>\n<parameter=cmd>\n123\n</parameter>\n</function>\n</tool_call>", tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "123"})

    def test_assistant_tool_call_arguments_are_decoded_for_the_template(self) -> None:
        msgs = self.mk.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}])
        self.assertEqual(msgs[0]["content"], "")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})

    def test_images_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self.mk.normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])



class _FakeEngine:
    """MKEngine's sequence / checkpoint semantics, without a GPU"""

    exact = False  # (an MK_EXACT engine never appends; see Server.load_prompt)

    def __init__(self):
        self.seq, self.dn_c, self.cks, self.calls = [], 0, [], []

    def sequence(self):
        return list(self.seq)

    def committed(self):
        return self.dn_c

    def checkpoints(self):
        return sorted(self.cks)

    def truncate(self, n):
        self.seq = self.seq[:n]

    def reset(self):
        self.seq, self.dn_c, self.cks = [], 0, []
        self.calls.append(("reset",))

    def rewind_to(self, pos):
        if pos not in self.cks:
            return False
        self.seq, self.dn_c = self.seq[:pos], pos
        self.cks = [c for c in self.cks if c <= pos]
        self.calls.append(("rewind", pos))
        return True

    def prefill(self, toks, ck_at=-1, cks=()):
        n0 = len(self.seq)
        self.seq += toks
        self.cks += [c for c in [*cks, ck_at] if n0 < c < len(self.seq)]
        self.dn_c = len(self.seq)
        self.calls.append(("prefill", n0, len(toks), ck_at, tuple(cks)))
        self.seq.append(7)  # the first generated token
        return 7


class MegakernelPromptReuseTests(unittest.TestCase):
    IM = 1  # <|im_start|>

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _server(self, every=8):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeEngine(), self.IM
        srv.args = mock.Mock(ck_every=every, ck_msgs=1)
        return srv

    def test_checkpoints_at_message_starts_the_generation_prompt_and_every_n_tokens(self) -> None:
        srv = self._server()
        toks = [self.IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, self.IM, 5, 5, self.IM, 9, 9]
        self.assertEqual(srv.load_prompt(toks), (7, 0))
        self.assertEqual(srv.eng.calls, [("reset",), ("prefill", 0, len(toks), 15, (8, 12, 15, 16))])

    def test_a_divergence_rewinds_to_the_last_checkpoint_before_it(self) -> None:
        srv = self._server()
        first = [self.IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, self.IM, 5, 5, self.IM, 9, 9]
        srv.load_prompt(first)
        srv.eng.dn_c = len(srv.eng.seq) + 5  # decoding went on
        second = first[:14] + [6, 6, self.IM, 9, 9]  # diverges at 14: checkpoints 8, 12 are before it
        self.assertEqual(srv.load_prompt(second), (7, 12))
        self.assertEqual(srv.eng.calls[-2:], [("rewind", 12), ("prefill", 12, len(second) - 12, 16, (16,))])

    def test_an_extension_of_the_sequence_is_appended_without_a_rewind(self) -> None:
        srv = self._server(every=0)
        first = [self.IM, 5, 5, self.IM, 9]
        srv.load_prompt(first)
        second = srv.eng.sequence() + [self.IM, 6, self.IM, 9]
        self.assertEqual(srv.load_prompt(second), (7, len(first) + 1))
        self.assertEqual(srv.eng.calls[-1], ("prefill", len(first) + 1, 4, len(second) - 2, (len(second) - 2,)))

    def test_no_checkpoint_before_the_divergence_means_a_full_prefill(self) -> None:
        srv = self._server(every=0)
        srv.load_prompt([self.IM, 5, 5, self.IM, 9])
        srv.eng.dn_c += 3
        self.assertEqual(srv.load_prompt([self.IM, 6, 6, self.IM, 9]), (7, 0))
        self.assertEqual(srv.eng.calls[-2][0], "reset")


class _FakeVisionEngine(_FakeEngine):
    """_FakeEngine with mk_prefill_img's images argument and a character tokenizer"""

    def tokenize(self, text):
        return [ord(c) for c in text]

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        self.calls.append(("images", tuple((p, nx, ny) for p, nx, ny, _ in images)))
        return super().prefill(toks, ck_at, cks)


class _FakeImage:
    def __init__(self, key, nx, ny):
        self.key, self.nx, self.ny, self.n = key, nx, ny, nx * ny
        self.encoded = 0

    def rows(self):
        self.encoded += 1
        return 0  # (an address)


class MegakernelVisionTests(unittest.TestCase):
    VS, VE, PAD, IM = 1001, 1002, 1003, 1

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _media(self, content):
        media = []
        msgs = self.mk.normalize_messages([{"role": "user", "content": content}], media)
        return msgs[0]["content"], media

    def test_image_parts_are_flattened_as_llama_server_does(self) -> None:
        url = {"url": "data:image/png;base64,iVBORw0KGgo="}
        img = {"type": "image_url", "image_url": url}
        m = self.mk.MEDIA_MARKER
        self.assertEqual(self._media([{"type": "text", "text": "a"}, img, {"type": "text", "text": "b"}])[0], f"a{m}b")
        self.assertEqual(self._media([img, {"type": "text", "text": "b"}])[0], f"{m}b")
        self.assertEqual(self._media([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, img])[0], f"a\nb{m}")
        text, media = self._media([img, img])
        self.assertEqual((text, len(media)), (m + m, 2))
        self.assertEqual(media[0][:4], bytes([0x89, 0x50, 0x4E, 0x47]))  # (the decoded data: URL)
        # a message without images keeps its parts, as before
        self.assertIsInstance(self._media([{"type": "text", "text": "a"}])[0], list)

    def test_remote_image_urls_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self._media([{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}])

    def _server(self, images):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeVisionEngine(), self.IM
        srv.args = mock.Mock(ck_every=0, ck_msgs=1)
        srv.vis_start, srv.vis_end, srv.img_pad = self.VS, self.VE, self.PAD
        srv.vision = mock.Mock()
        srv.vision.image.side_effect = lambda data: images[data]
        return srv

    def test_each_image_becomes_its_rows_between_vision_start_and_end(self) -> None:
        im = _FakeImage("k", 3, 2)
        srv = self._server({b"png": im})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"ab{m}c", [b"png"])
        self.assertEqual(toks, [ord("a"), ord("b"), self.VS] + [self.PAD] * 6 + [self.VE, ord("c")])
        self.assertEqual(images, [(3, im)])

    def test_a_different_image_at_the_same_place_is_not_reused(self) -> None:
        a, b = _FakeImage("a", 2, 2), _FakeImage("b", 2, 2)
        srv = self._server({b"A": a, b"B": b})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        self.assertEqual(srv.eng.calls[-2], ("images", ((2, 2, 2),)))
        toks2, images2 = srv.tokenize_prompt(f"x{m}yz", [b"B"])
        self.assertEqual(toks2, toks)  # (the placeholders are alike)
        _, reused = srv.load_prompt(toks2, images2)
        self.assertEqual(reused, 0)  # (no checkpoint before the image: a full prefill)
        self.assertEqual(b.encoded, 1)

    def test_the_same_image_is_reused_and_a_rewind_into_it_feeds_it_again(self) -> None:
        a = _FakeImage("a", 4, 4)
        srv = self._server({b"A": a})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        srv.eng.cks.append(10)  # a checkpoint inside the image (positions 2 .. 17)
        srv.eng.dn_c = len(srv.eng.seq) + 3
        toks2, images2 = srv.tokenize_prompt(f"x{m}yw", [b"A"])
        _, reused = srv.load_prompt(toks2, images2)
        self.assertEqual(reused, 10)
        self.assertEqual(srv.eng.calls[-3:-1], [("rewind", 10), ("images", ((2, 4, 4),))])


if __name__ == "__main__":
    unittest.main()
