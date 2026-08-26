"""Unit tests for aria_engine model-name parsing and auto-download (no network)."""
import json
import os
import tempfile
import unittest

import aria_engine
from aria_engine import (
    _aria_home,
    _hub_bearer,
    _hub_file_urls,
    _is_valid_bundle,
    _parse_bundle_name,
    _preferred_public_hub,
    download_model,
)


class ParseTests(unittest.TestCase):
    def test_q4_suffix(self):
        self.assertEqual(_parse_bundle_name("gemma-4-e2b-it_q4"), ("gemma-4-e2b-it", "int4"))

    def test_q8_suffix(self):
        self.assertEqual(_parse_bundle_name("foo_q8"), ("foo", "int8"))

    def test_q326_suffix(self):
        self.assertEqual(_parse_bundle_name("foo_q326"), ("foo", "int326"))

    def test_q326_channel_suffix(self):
        self.assertEqual(_parse_bundle_name("foo_q326_channel"), ("foo", "int326"))

    def test_q3dot26_suffix(self):
        self.assertEqual(_parse_bundle_name("foo_q3.26"), ("foo", "int326"))

    def test_no_suffix_defaults_int4(self):
        self.assertEqual(_parse_bundle_name("foo"), ("foo", "int4"))

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _parse_bundle_name("foo/bar")
        with self.assertRaises(ValueError):
            _parse_bundle_name("foo_q9")
        with self.assertRaises(ValueError):
            _parse_bundle_name("")


class ValidBundleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def _write_bundle(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "weight.bin"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": "aria-quant-bundle"}))

    def test_valid(self):
        self._write_bundle(os.path.join(self.root, "m"))
        self.assertTrue(_is_valid_bundle(os.path.join(self.root, "m")))

    def test_missing_weight(self):
        d = os.path.join(self.root, "m")
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": "aria-quant-bundle"}))
        self.assertFalse(_is_valid_bundle(d))

    def test_wrong_format(self):
        d = os.path.join(self.root, "m")
        os.makedirs(d)
        with open(os.path.join(d, "weight.bin"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": "other"}))
        self.assertFalse(_is_valid_bundle(d))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = self._tmp.name
        self._env = {"ARIA_COMPUTE_HOME": self.home}
        self._orig = dict(os.environ)
        os.environ.update(self._env)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig)

    def test_token_optional_when_cached(self):
        cache = os.path.join(_aria_home(), "models", "foo_q4")
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, "weight.bin"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(cache, "config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": "aria-quant-bundle"}))
        self.assertEqual(download_model("foo_q4", None), cache)

    def test_cached_bundle_skips_download(self):
        cache = os.path.join(_aria_home(), "models", "foo_q4")
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, "weight.bin"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(cache, "config.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"format": "aria-quant-bundle"}))
        # should not raise / hit network
        self.assertEqual(download_model("foo_q4", "tok"), cache)

    def test_preferred_hub_follows_site_tld(self):
        self.assertEqual(_preferred_public_hub("https://ariacompute.com"), "huggingface")
        self.assertEqual(_preferred_public_hub("https://ariacompute.cn"), "modelscope")
        self.assertEqual(_preferred_public_hub(None), "huggingface")

    def test_dashboard_token_not_sent_to_hub(self):
        self.assertIsNone(_hub_bearer("sk-bf-95076ed1-8c1a-4efa-b33c-f52c1d7f9f24"))
        self.assertIsNone(_hub_bearer("bfvk-test"))
        self.assertEqual(_hub_bearer("hf_abc"), "hf_abc")

    def test_hub_urls_follow_upload_layout(self):
        hf = _hub_file_urls("huggingface", "gemma-4-e2b-it_q4", "config.json")
        self.assertTrue(
            any(
                "/ariacompute/gemma-4-e2b-it_q4/resolve/main/v1.0/gemma-4-e2b-it_q4/config.json"
                in u
                for u in hf
            )
        )
        ms = _hub_file_urls("modelscope", "gemma-4-e2b-it_q4", "weight.bin")
        self.assertTrue(any("/v1.0/gemma-4-e2b-it_q4/weight.bin" in u for u in ms))
        self.assertFalse(any("/api/dashboard/" in u for u in hf + ms))


if __name__ == "__main__":
    unittest.main()
