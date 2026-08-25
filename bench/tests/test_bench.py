"""Unit tests for engine bench (mock chat_fn / CLI; no external engines)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, str(ROOT))

from bench.backends import BackendConfig, ChatResult, parse_backend_arg  # noqa: E402
from bench.cli import main as cli_main  # noqa: E402
from bench.families import FAMILIES, select_families  # noqa: E402
from bench.metrics import exact_match, token_overlap  # noqa: E402
from bench.report import render_markdown, write_reports  # noqa: E402
from bench.runner import pick_ref_backend, run_bench  # noqa: E402


class TestFamilies(unittest.TestCase):
    def test_registry_count_and_kinds(self):
        self.assertEqual(len(FAMILIES), 28)
        kinds = {f.kind for f in FAMILIES}
        self.assertEqual(kinds, {"text", "vl", "vla"})
        self.assertEqual(FAMILIES[0].path, "qwen/qwen3-0.6b")
        self.assertEqual(FAMILIES[-1].base_model, "robbyant/lingbot-vla-v2-6b")

    def test_select_filter(self):
        xs = select_families(["lfm/lfm2-350m", "openvla/openvla-7b"])
        self.assertEqual([x.path for x in xs], ["lfm/lfm2-350m", "openvla/openvla-7b"])
        with self.assertRaises(ValueError):
            select_families(["no/such"])


class TestMetrics(unittest.TestCase):
    def test_overlap_and_exact(self):
        self.assertEqual(token_overlap("a b c", "a b d"), 0.5)
        self.assertTrue(exact_match(" hi ", "hi"))
        self.assertFalse(exact_match("a", "b"))


class TestBackends(unittest.TestCase):
    def test_parse_backend(self):
        b = parse_backend_arg("aria=http://127.0.0.1:8080/")
        self.assertEqual(b.id, "aria")
        self.assertEqual(b.base_url, "http://127.0.0.1:8080")
        with self.assertRaises(ValueError):
            parse_backend_arg("bad")


class TestRunnerReport(unittest.TestCase):
    def test_pick_ref(self):
        cfgs = [
            BackendConfig("aria", "http://a"),
            BackendConfig("ollama", "http://o"),
            BackendConfig("llamacpp", "http://l"),
        ]
        self.assertEqual(pick_ref_backend(cfgs, None), "llamacpp")
        self.assertEqual(pick_ref_backend(cfgs, "ollama"), "ollama")

    def test_run_bench_mock_chat(self):
        def fake_chat(cfg, *, model, prompt, max_tokens, temperature, stream):
            content = f"{cfg.id}-reply-to-{prompt.split()[0]}"
            return ChatResult(
                status="ok",
                content=content,
                latency_ms=10.0 + len(cfg.id),
                completion_tokens=4,
            )

        backends = [
            BackendConfig("aria", "http://127.0.0.1:1"),
            BackendConfig("llamacpp", "http://127.0.0.1:2"),
            BackendConfig("ollama", "http://127.0.0.1:3"),
            BackendConfig("vllm", "http://127.0.0.1:4"),
        ]
        with mock.patch("bench.runner.probe", return_value=(True, "ok")):
            report = run_bench(
                backends=backends,
                family_paths=["qwen/qwen3.5-2b", "openvla/openvla-7b"],
                prompts=["Hello world"],
                max_tokens=8,
                warmup=0,
                runs=1,
                measure_ttft=False,
                chat_fn=fake_chat,
            )
        self.assertEqual(report["mode"], "engine_bench")
        self.assertFalse(report["ci_fail"])
        self.assertEqual(len(report["families"]), 2)
        self.assertEqual(len(report["backends"]), 4)
        self.assertEqual(len(report["results"]), 8)
        self.assertTrue(all(r["status"] in ("ok", "ok_partial") for r in report["results"]))
        aria_q = next(
            r
            for r in report["results"]
            if r["family"] == "qwen/qwen3.5-2b" and r["backend"] == "aria"
        )
        self.assertEqual(aria_q["quality"]["status"], "ok")
        self.assertEqual(aria_q["quality"]["ref_backend"], "llamacpp")
        self.assertIn("token_overlap_mean", aria_q["quality"])

        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "bench_report.json"
            jp2, mp = write_reports(report, jp)
            self.assertTrue(jp2.is_file())
            self.assertTrue(mp.is_file())
            loaded = json.loads(jp2.read_text(encoding="utf-8"))
            self.assertEqual(loaded["summary"]["backends_configured"], 4)
            md = mp.read_text(encoding="utf-8")
            self.assertIn("Aria Engine Bench Report", md)
            self.assertIn("qwen/qwen3.5-2b", md)

    def test_skip_unreachable(self):
        def boom(*_a, **_k):
            raise AssertionError("chat should not run when probe fails")

        with mock.patch("bench.runner.probe", return_value=(False, "down")):
            report = run_bench(
                backends=[BackendConfig("aria", "http://127.0.0.1:9")],
                family_paths=["lfm/lfm2-350m"],
                prompts=["hi"],
                warmup=0,
                runs=1,
                chat_fn=boom,
            )
        self.assertEqual(report["results"][0]["status"], "skipped")
        self.assertEqual(report["summary"]["results_skipped"], 1)

    def test_cli_list_and_run(self):
        self.assertEqual(cli_main(["list-families"]), 0)
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "out.json"
            report = {
                "mode": "engine_bench",
                "ci_fail": False,
                "summary": {
                    "families_total": 1,
                    "backends_configured": 2,
                    "results_ok": 0,
                    "results_skipped": 0,
                    "results_error": 0,
                },
                "backends": [],
                "results": [],
                "families": [],
                "meta": {},
            }
            with mock.patch("bench.cli.run_bench", return_value=report):
                rc = cli_main(
                    [
                        "run",
                        "--backend",
                        "aria=http://127.0.0.1:1",
                        "--backend",
                        "llamacpp=http://127.0.0.1:2",
                        "--report",
                        str(jp),
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(jp.is_file())
            self.assertTrue(jp.with_suffix(".md").is_file())

    def test_cli_requires_backend(self):
        self.assertEqual(cli_main(["run"]), 2)

    def test_render_md_smoke(self):
        md = render_markdown(
            {
                "mode": "engine_bench",
                "ci_fail": False,
                "meta": {
                    "generated_at": "t",
                    "max_tokens": 1,
                    "warmup": 0,
                    "runs": 1,
                    "ref_backend": None,
                },
                "summary": {"families_total": 1},
                "backends": [
                    {
                        "id": "aria",
                        "base_url": "http://x",
                        "probe_ok": True,
                        "probe_detail": "ok",
                    }
                ],
                "results": [],
            }
        )
        self.assertIn("Summary", md)


if __name__ == "__main__":
    unittest.main()
