"""Unit tests for ariaengine lib resolution (env override, bundled lib, errors)."""
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from ariaengine import (
    _cached_ffi_path,
    _default_lib_path,
    _extract_ffi_archive,
    _ffi_asset_os,
    _load_lib,
    _select_latest_stable,
    _upgrade_org,
    ensure_ffi_lib,
)


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
            self._touch("lib/libariaengine_ffi.so")
            self.assertEqual(
                _default_lib_path(self.pkg_dir),
                os.path.join(self.pkg_dir, "lib", "libariaengine_ffi.so"),
            )

    def test_darwin_default_name(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self._touch("lib/libariaengine_ffi.dylib")
            self.assertEqual(
                _default_lib_path(self.pkg_dir),
                os.path.join(self.pkg_dir, "lib", "libariaengine_ffi.dylib"),
            )

    def test_windows_default_name(self):
        with mock.patch.object(sys, "platform", "win32"):
            self._touch("lib/ariaengine_ffi.dll")
            self.assertEqual(
                _default_lib_path(self.pkg_dir),
                os.path.join(self.pkg_dir, "lib", "ariaengine_ffi.dll"),
            )

    def test_missing_bundled_returns_none(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIsNone(_default_lib_path(self.pkg_dir))

    def test_wrong_platform_lib_ignored(self):
        # .dll present but platform is linux -> still None
        with mock.patch.object(sys, "platform", "linux"):
            self._touch("lib/ariaengine_ffi.dll")
            self.assertIsNone(_default_lib_path(self.pkg_dir))


class LoadLibTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pkg_dir = self._tmp.name

    def test_resolution_prefers_env(self):
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        env = {"ARIAENGINE_FFI_LIB": "/custom/libariaengine_ffi.so"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                _load_lib()
        self.assertEqual(captured["path"], "/custom/libariaengine_ffi.so")

    def test_resolution_falls_back_to_bundled(self):
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        bundled = os.path.join(self.pkg_dir, "lib", "libariaengine_ffi.so")
        os.makedirs(os.path.dirname(bundled), exist_ok=True)
        with open(bundled, "wb"):
            pass
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch.dict(os.environ, {"ARIA_COMPUTE_HOME": home.name}, clear=True):
                with mock.patch("ariaengine._default_lib_path", return_value=bundled):
                    with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                        _load_lib()
        self.assertEqual(captured["path"], bundled)

    def test_resolution_falls_back_to_home_lib(self):
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        cached = os.path.join(home.name, "lib", "libariaengine_ffi.so")
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        with open(cached, "wb"):
            pass
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch.dict(os.environ, {"ARIA_COMPUTE_HOME": home.name}, clear=True):
                with mock.patch("ariaengine._default_lib_path", return_value=None):
                    with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                        _load_lib()
        self.assertEqual(captured["path"], cached)

    def test_explicit_path_wins(self):
        captured = {}

        def fake_cdll(path):
            captured["path"] = path
            return object()

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("ctypes.CDLL", side_effect=fake_cdll):
                _load_lib("/direct/libariaengine_ffi.so")
        self.assertEqual(captured["path"], "/direct/libariaengine_ffi.so")

    def test_missing_lib_raises(self):
        with mock.patch.object(sys, "platform", "linux"):
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("ariaengine._default_lib_path", return_value=None):
                    with mock.patch("ariaengine._cached_ffi_path", return_value=None):
                        with mock.patch(
                            "ariaengine.ensure_ffi_lib",
                            side_effect=RuntimeError("Cannot locate libariaengine_ffi"),
                        ):
                            with self.assertRaises(RuntimeError) as ctx:
                                _load_lib()
        self.assertIn("Cannot locate libariaengine_ffi", str(ctx.exception))


class FfiReleaseTests(unittest.TestCase):
    def test_asset_os(self):
        self.assertEqual(_ffi_asset_os("Linux", "x86_64"), "linux_x86_64")
        self.assertEqual(_ffi_asset_os("Linux", "aarch64"), "linux_arm64")
        self.assertEqual(_ffi_asset_os("Darwin", "arm64"), "macos")
        self.assertEqual(_ffi_asset_os("Windows", "AMD64"), "windows_x86_64")
        with self.assertRaises(RuntimeError):
            _ffi_asset_os("Linux", "ppc64le")

    def test_select_latest_stable(self):
        releases = [
            {"tag_name": "v0.7.1", "draft": False, "prerelease": False},
            {"tag_name": "v0.8.0-rc1", "draft": False, "prerelease": True},
            {"tag_name": "v0.7.2", "draft": False, "prerelease": False},
            {"tag_name": "v0.9.0", "draft": True, "prerelease": False},
        ]
        self.assertEqual(_select_latest_stable(releases), "0.7.2")

    def test_upgrade_org_from_site(self):
        with mock.patch("ariaengine._config_yml_scalar", return_value=None):
            self.assertEqual(_upgrade_org("https://ariacompute.com"), "https://github.com/ariacompute")
            self.assertEqual(_upgrade_org("https://ariacompute.cn"), "https://gitee.com/ariacompute")

    def test_extract_and_cached_skip(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        archive = os.path.join(home.name, "libariaengine_ffi_0.1.0_linux_x86_64.tar.gz")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("libariaengine_ffi.so")
            data = b"dummy-ffi"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        with open(archive, "wb") as f:
            f.write(buf.getvalue())
        dest_dir = os.path.join(home.name, "lib")
        with mock.patch.object(sys, "platform", "linux"):
            got = _extract_ffi_archive(archive, dest_dir, "libariaengine_ffi.so")
        self.assertEqual(os.path.basename(got), "libariaengine_ffi.so")
        with open(got, "rb") as f:
            self.assertEqual(f.read(), b"dummy-ffi")
        with mock.patch.dict(os.environ, {"ARIA_COMPUTE_HOME": home.name}, clear=True):
            with mock.patch.object(sys, "platform", "linux"):
                with mock.patch("ariaengine._default_lib_path", return_value=None):
                    with mock.patch("ariaengine._http_get_bytes") as http:
                        self.assertEqual(ensure_ffi_lib(), got)
                        http.assert_not_called()
                        self.assertEqual(_cached_ffi_path(), got)

    def test_ensure_downloads_latest_stable(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("libariaengine_ffi.so")
            data = b"from-release"
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        archive_bytes = buf.getvalue()
        releases = [
            {
                "tag_name": "v0.7.1",
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "libariaengine_ffi_0.7.1_linux_x86_64.tar.gz",
                        "browser_download_url": "https://example.invalid/lib.tar.gz",
                    }
                ],
            }
        ]

        def fake_http(url, dest=None):
            if "api.github.com" in url or "gitee.com" in url:
                return json.dumps(releases).encode()
            if dest:
                os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(archive_bytes)
                return b""
            return archive_bytes

        with mock.patch.dict(os.environ, {"ARIA_COMPUTE_HOME": home.name}, clear=True):
            with mock.patch.object(sys, "platform", "linux"):
                with mock.patch("ariaengine.platform.system", return_value="Linux"):
                    with mock.patch("ariaengine.platform.machine", return_value="x86_64"):
                        with mock.patch("ariaengine._default_lib_path", return_value=None):
                            with mock.patch("ariaengine._http_get_bytes", side_effect=fake_http):
                                got = ensure_ffi_lib()
        self.assertEqual(os.path.basename(got), "libariaengine_ffi.so")
        with open(got, "rb") as f:
            self.assertEqual(f.read(), b"from-release")


if __name__ == "__main__":
    unittest.main()
