import json
import os
import unittest

from ariaengine import Engine


@unittest.skipUnless(os.environ.get("ARIAENGINE_FFI_LIB") and os.environ.get("ARIA_BUNDLE"), "need ARIAENGINE_FFI_LIB and ARIA_BUNDLE")
class BindingTests(unittest.TestCase):
    def setUp(self):
        self.eng = Engine(os.environ["ARIA_BUNDLE"])

    def tearDown(self):
        self.eng.close()

    def test_complete_ok(self):
        out = self.eng.complete([{"role": "user", "content": "hi"}], {"max_tokens": 2})
        self.assertTrue(out.get("success"))
        self.assertTrue(out.get("response"))

    def test_complete_tools_ok(self):
        out = self.eng.complete(
            [{"role": "user", "content": "hi"}],
            {"max_tokens": 2},
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
        self.assertIn("function_calls", out)

    def test_embed_ok(self):
        out = self.eng.embed("hello")
        emb = out["data"][0]["embedding"]
        self.assertGreater(len(emb), 0)

    def test_transcribe_ok(self):
        out = self.eng.transcribe(bytes([0, 1, 2, 3, 4, 5]))
        self.assertIn("text", out)

    def test_init_missing_path(self):
        with self.assertRaises(RuntimeError):
            Engine("/no/such/path")

    def test_complete_bad_json(self):
        # Engine.complete always json.dumps; exercise FFI bad path via ctypes in separate test if needed
        pass

    def test_embed_bad_input(self):
        with self.assertRaises(RuntimeError):
            self.eng.embed("")

    def test_transcribe_bad_audio(self):
        with self.assertRaises(RuntimeError):
            self.eng.transcribe(b"")


if __name__ == "__main__":
    unittest.main()
