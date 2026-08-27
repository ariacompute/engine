"""Instance-level Engine.auth (in-memory; does not write config.yml)."""
import os
import tempfile
import unittest
from unittest import mock

from aria_engine import (
    CN_CLOUD,
    CN_SITE,
    CN_UPGRADE,
    Engine,
    apply_auth,
    default_auth_config,
    fill_auth_urls,
)


class ApplyAuthTests(unittest.TestCase):
    def test_defaults(self):
        cfg = default_auth_config()
        self.assertEqual(cfg["hybrid_mode"], "balance")
        self.assertEqual(cfg["hybrid_execution"], "hybrid")
        self.assertTrue(cfg["hybrid_semantic"])
        self.assertEqual(cfg["hybrid_semantic_timeout_ms"], 800)
        self.assertEqual(cfg["hybrid_semantic_cache_size"], 512)
        self.assertEqual(cfg["compute"], "auto")

    def test_invalid_enum_raises(self):
        with self.assertRaises(ValueError):
            apply_auth(default_auth_config(), {"hybrid_mode": "fast"})
        with self.assertRaises(ValueError):
            apply_auth(default_auth_config(), {"hybrid_execution": "local"})
        with self.assertRaises(ValueError):
            apply_auth(default_auth_config(), {"compute": "gpu"})

    def test_fill_urls_from_cn_site(self):
        got = fill_auth_urls({"site_url": CN_SITE, "cloud_url": "", "upgrade_url": ""})
        self.assertEqual(got["cloud_url"], CN_CLOUD)
        self.assertEqual(got["upgrade_url"], CN_UPGRADE)
        self.assertEqual(got["site_url"], CN_SITE)


class EngineAuthTests(unittest.TestCase):
    def test_all_fields_roundtrip(self):
        eng = Engine()
        eng.auth(
            cloud_api_key="sk-test",
            cloud_url=CN_CLOUD,
            site_url=CN_SITE,
            upgrade_url=CN_UPGRADE,
            hybrid_mode="cost",
            hybrid_execution="device",
            hybrid_semantic=False,
            hybrid_semantic_timeout_ms=250,
            hybrid_semantic_cache_size=16,
            compute="cpu",
            hf_token="hf_abc",
            modelscope_api_token="ms_xyz",
        )
        st = eng.auth_status()
        self.assertEqual(st["cloud_api_key"], "sk-test")
        self.assertEqual(st["hybrid_mode"], "cost")
        self.assertEqual(st["hybrid_execution"], "device")
        self.assertFalse(st["hybrid_semantic"])
        self.assertEqual(st["hybrid_semantic_timeout_ms"], 250)
        self.assertEqual(st["hybrid_semantic_cache_size"], 16)
        self.assertEqual(st["compute"], "cpu")
        self.assertEqual(st["hf_token"], "hf_abc")
        self.assertEqual(st["modelscope_api_token"], "ms_xyz")
        self.assertEqual(st["site_url"], CN_SITE)

    def test_partial_merge(self):
        eng = Engine()
        eng.auth(hf_token="hf_one", hybrid_mode="intelligence")
        eng.auth(compute="cuda")
        st = eng.auth_status()
        self.assertEqual(st["hf_token"], "hf_one")
        self.assertEqual(st["hybrid_mode"], "intelligence")
        self.assertEqual(st["compute"], "cuda")

    def test_invalid_enum_leaves_state(self):
        eng = Engine()
        eng.auth(hybrid_mode="cost")
        with self.assertRaises(ValueError):
            eng.auth(hybrid_mode="nope")
        self.assertEqual(eng.auth_status()["hybrid_mode"], "cost")

    def test_clear_resets_instance(self):
        eng = Engine()
        eng.auth(hf_token="hf_x", hybrid_mode="cost")
        eng.auth_clear()
        st = eng.auth_status()
        self.assertEqual(st["hf_token"], "")
        self.assertEqual(st["hybrid_mode"], "balance")

    def test_fills_urls_from_site_tld(self):
        eng = Engine()
        eng.auth(site_url="https://ariacompute.cn")
        st = eng.auth_status()
        self.assertEqual(st["cloud_url"], CN_CLOUD)
        self.assertEqual(st["upgrade_url"], CN_UPGRADE)

    def test_does_not_write_config_yml(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        with mock.patch.dict(os.environ, {"ARIA_COMPUTE_HOME": home.name}, clear=False):
            eng = Engine()
            eng.auth(
                cloud_api_key="sk-test",
                site_url="https://ariacompute.com",
                hf_token="hf_x",
            )
            self.assertFalse(os.path.isfile(os.path.join(home.name, "config.yml")))

    def test_detect_urls_from_key_mocked(self):
        eng = Engine()
        with mock.patch("aria_engine._probe_dashboard", side_effect=lambda site, key: "ariacompute.cn" in site):
            eng.auth(cloud_api_key="sk-region")
        st = eng.auth_status()
        self.assertEqual(st["site_url"], CN_SITE)
        self.assertEqual(st["cloud_url"], CN_CLOUD)
        self.assertEqual(st["upgrade_url"], CN_UPGRADE)


if __name__ == "__main__":
    unittest.main()
