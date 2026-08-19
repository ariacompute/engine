"""Unit tests for aria_engine lib resolution (env override, bundled lib, errors)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import aria_engine
from aria_engine import _default_lib_path, _load_lib


class LibPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pkg_dir = self._tmp.name

    def _touch(self, rel: str):
        p = os.path.join(self.pkg_dir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb"):
            pass
        return p

    def test_linux_default_name(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIsNone(_default_lib_path(self.pkg_dir))
            self._touch("lib/libaria_ffi.so")
            self.assertEqual(
                _default_lib_path(self.pkg_dir),
                os.path.join(self.pkg_dir, "lib", "libaria_ffi.so"),
            )

    def test_darwin_default_name(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self._touch("lib/libaria_ffi.dylib")
            self.assertEqual(
                _default_lib_path(self.pkg_dir),
                os.path.join(self.pkg_dir, "lib", "libaria_ffi.dylib"),
            )

    def test_windows_default_name(self):
        with mock.patch.object(sys, "platform", "win32"):
            self._touch("lib/aria_ffi.dll")
            self.assertEqual(
                _default_lib_path(self.pkg_dir),
                os.path.join(self.pkg_dir, "lib", "aria_ffi.dll"),
            )

    def test_missing_bundled_returns_none(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIsNone(_default_lib_path(self.pkg_dir))

    def test_wrong_platform_lib_ignored(self):
        # .dll present but platform is linux -> still None
        with mock.patch.object(sys, "platform", "linux"):
            self._touch("lib/aria_ffi.dll")
            self.assertIsNone(_default_lib_path(self.pkg_dir))


class LoadLibTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pkg_dir = self._tmp.name

    def test_resolution_prefers_env(self):
        # resolve via patched CDLL capture
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        env = {"ARIA_FFI_LIB": "/custom/libaria_ffi.so"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                _load_lib()
        self.assertEqual(captured["path"], "/custom/libaria_ffi.so")

    def test_resolution_falls_back_to_bundled(self):
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        bundled = os.path.join(self.pkg_dir, "lib", "libaria_ffi.so")
        os.makedirs(os.path.dirname(bundled), exist_ok=True)
        with open(bundled, "wb"):
            pass
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("aria_engine._default_lib_path", return_value=bundled):
                    with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                        _load_lib()
        self.assertEqual(captured["path"], bundled)

    def test_explicit_path_wins(self):
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                _load_lib("/direct/libaria_ffi.so")
        self.assertEqual(captured["path"], "/direct/libaria_ffi.so")

    def test_missing_lib_raises(self):
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("aria_engine._default_lib_path", return_value=None):
                    with self.assertRaises(RuntimeError) as ctx:
                        _load_lib()
        self.assertIn("Cannot locate libaria_ffi", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
