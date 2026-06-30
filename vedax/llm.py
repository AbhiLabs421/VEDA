"""Tiny Ollama / OpenAI-compatible client over stdlib urllib.

No external SDKs. Streams tokens as they arrive so the answer appears
live, like a real chat.
"""

import json
import os
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 180


def _post_stream(url, payload, headers, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(url, model, messages, api="ollama", token=None,
                timeout=DEFAULT_TIMEOUT):
    """Yield content chunks (str) as the model produces them.

    ``api='ollama'``  uses POST {url}/api/chat with NDJSON stream.
    ``api='openai'``  uses POST {url}/v1/chat/completions with SSE.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base = url.rstrip("/")

    if api == "ollama":
        body = {"model": model, "messages": messages, "stream": True}
        resp = _post_stream(base + "/api/chat", body, headers, timeout)
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("message", {}).get("content")
            if chunk:
                yield chunk
            if obj.get("done"):
                return
        return

    if api == "openai":
        body = {"model": model, "messages": messages, "stream": True}
        resp = _post_stream(base + "/v1/chat/completions", body,
                            headers, timeout)
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or [{}]
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            chunk = delta.get("content")
            if chunk:
                yield chunk
        return

    raise ValueError(f"unknown llm api: {api!r}")


def llm_settings_from_env(args=None):
    """Pull URL/model/api/token from CLI args or environment."""
    def pick(attr, env_key, default=None):
        return (getattr(args, attr, None) if args else None) \
               or os.environ.get(env_key, default)

    return {
        "url": pick("llm_url", "VEDAX_LLM_URL"),
        "model": pick("llm_model", "VEDAX_LLM_MODEL", "gpt-oss:20b"),
        "api": pick("llm_api", "VEDAX_LLM_API", "ollama"),
        "token": pick("llm_token", "VEDAX_LLM_TOKEN"),
    }
