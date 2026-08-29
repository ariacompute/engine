"""Instance-level Engine.auth (in-memory; does not write config.yml)."""
import os
import tempfile
import unittest
from unittest import mock

from aria_engine import (
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
        self.assertEqual(cfg["compute"], "auto")
        self.assertEqual(cfg["router"], "")

    def test_invalid_enum_raises(self):
        with self.assertRaises(ValueError):
            apply_auth(default_auth_config(), {"compute": "gpu"})

    def test_fill_urls_from_cn_site(self):
        got = fill_auth_urls({"site_url": CN_SITE, "upgrade_url": ""})
        self.assertEqual(got["upgrade_url"], CN_UPGRADE)
        self.assertEqual(got["site_url"], CN_SITE)


class EngineAuthTests(unittest.TestCase):
    def test_all_fields_roundtrip(self):
        eng = Engine()
        eng.auth(
            router="http://127.0.0.1:8080",
            site_url=CN_SITE,
            upgrade_url=CN_UPGRADE,
            compute="cpu",
            hf_token="hf_abc",
            modelscope_api_token="ms_xyz",
        )
        st = eng.auth_status()
        self.assertEqual(st["router"], "http://127.0.0.1:8080")
        self.assertEqual(st["compute"], "cpu")
        self.assertEqual(st["hf_token"], "hf_abc")
        self.assertEqual(st["modelscope_api_token"], "ms_xyz")
        self.assertEqual(st["site_url"], CN_SITE)

    def test_partial_merge(self):
        eng = Engine()
        eng.auth(hf_token="hf_one", router="http://127.0.0.1:1")
        eng.auth(compute="cuda")
        st = eng.auth_status()
        self.assertEqual(st["hf_token"], "hf_one")
        self.assertEqual(st["router"], "http://127.0.0.1:1")
        self.assertEqual(st["compute"], "cuda")

    def test_invalid_enum_leaves_state(self):
        eng = Engine()
        eng.auth(compute="cpu")
        with self.assertRaises(ValueError):
            eng.auth(compute="gpu")
        self.assertEqual(eng.auth_status()["compute"], "cpu")

    def test_clear_resets_instance(self):
        eng = Engine()
        eng.auth(hf_token="hf_x", compute="cpu")
        eng.auth_clear()
        st = eng.auth_status()
        self.assertEqual(st["hf_token"], "")
        self.assertEqual(st["compute"], "auto")

    def test_fills_urls_from_site_tld(self):
        eng = Engine()
        eng.auth(site_url="https://ariacompute.cn")
        st = eng.auth_status()
        self.assertEqual(st["upgrade_url"], CN_UPGRADE)

    def test_does_not_write_config_yml(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        with mock.patch.dict(os.environ, {"ARIA_COMPUTE_HOME": home.name}, clear=False):
            eng = Engine()
            eng.auth(
                router="http://127.0.0.1:8080",
                site_url="https://ariacompute.com",
                hf_token="hf_x",
            )
            self.assertFalse(os.path.isfile(os.path.join(home.name, "config.yml")))


if __name__ == "__main__":
    unittest.main()
