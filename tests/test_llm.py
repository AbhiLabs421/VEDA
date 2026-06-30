"""Tests for the stdlib LLM client (Ollama and OpenAI stream parsing).

The HTTP layer is mocked: urllib is asked to open one of our fake
file-like objects that play back a recorded server response, so the
test never touches the network.
"""

import io
import json
import unittest
from unittest.mock import patch

from vedax.llm import llm_settings_from_env, stream_chat


def fake_urlopen(body):
    captured = {}

    def opener(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return io.BytesIO(body)

    return opener, captured


class TestOllamaStream(unittest.TestCase):
    def test_streams_message_content_and_stops_on_done(self):
        body = (
            json.dumps({"message": {"content": "Hel"}, "done": False}).encode()
            + b"\n"
            + json.dumps({"message": {"content": "lo "}, "done": False}).encode()
            + b"\n"
            + json.dumps({"message": {"content": "world."},
                          "done": True}).encode()
            + b"\n"
        )
        opener, captured = fake_urlopen(body)
        with patch("urllib.request.urlopen", opener):
            chunks = list(stream_chat(
                "https://example.test", "gpt-oss:20b",
                [{"role": "user", "content": "hi"}], api="ollama",
                token="secret"))
        self.assertEqual("".join(chunks), "Hello world.")
        self.assertEqual(captured["url"], "https://example.test/api/chat")
        self.assertEqual(captured["body"]["model"], "gpt-oss:20b")
        self.assertTrue(captured["body"]["stream"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")


class TestOpenAIStream(unittest.TestCase):
    def test_streams_sse_deltas(self):
        def event(delta_content):
            return b"data: " + json.dumps({
                "choices": [{"delta": {"content": delta_content}}]
            }).encode() + b"\n"

        body = event("Beat") + event("ing ") + event("RAG.") + b"data: [DONE]\n"
        opener, captured = fake_urlopen(body)
        with patch("urllib.request.urlopen", opener):
            chunks = list(stream_chat(
                "https://example.test/", "gpt-4o-mini",
                [{"role": "user", "content": "go"}], api="openai"))
        self.assertEqual("".join(chunks), "Beating RAG.")
        self.assertEqual(captured["url"],
                         "https://example.test/v1/chat/completions")


class TestSettings(unittest.TestCase):
    def test_env_fallback(self):
        with patch.dict("os.environ",
                        {"VEDAX_LLM_URL": "https://env.test",
                         "VEDAX_LLM_MODEL": "qwen"}):
            s = llm_settings_from_env()
            self.assertEqual(s["url"], "https://env.test")
            self.assertEqual(s["model"], "qwen")
            self.assertEqual(s["api"], "ollama")


if __name__ == "__main__":
    unittest.main()
