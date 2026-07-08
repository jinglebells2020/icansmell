"""Tests for llm.py (Milestone 3) — OpenRouterClient. NO network is touched.

Every test injects a fake ``transport`` callable or exercises the missing-key
path, so the real urllib POST is never invoked. Security invariant asserted
throughout: the API key never leaks into an ``LLMError`` message.
"""
from __future__ import annotations

import pytest

from sniffsniff.llm import BASE_URL, DEFAULT_MODEL, LLMError, OpenRouterClient


def test_complete_returns_content():
    calls = {}

    def transport(url, headers, payload):
        calls["url"] = url
        calls["headers"] = headers
        calls["payload"] = payload
        return {"choices": [{"message": {"content": "hi"}}]}

    client = OpenRouterClient(api_key="testkey", transport=transport)
    out = client.complete([{"role": "user", "content": "yo"}])

    assert out == "hi"
    assert calls["url"] == BASE_URL


def test_complete_falls_back_to_reasoning_when_content_empty():
    def transport(url, headers, payload):
        return {"choices": [{"message": {"content": "", "reasoning": "deep thought"}}]}

    client = OpenRouterClient(api_key="testkey", transport=transport)
    assert client.complete([{"role": "user", "content": "yo"}]) == "deep thought"


def test_payload_and_headers_are_correct():
    captured = {}

    def transport(url, headers, payload):
        captured["headers"] = headers
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    msgs = [{"role": "user", "content": "hello"}]
    client = OpenRouterClient(api_key="testkey", transport=transport)
    client.complete(msgs, temperature=0.3)

    # Authorization carries the bearer key.
    assert captured["headers"]["Authorization"] == "Bearer testkey"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert "HTTP-Referer" in captured["headers"]
    assert "X-Title" in captured["headers"]

    # Payload has the right model, messages and temperature.
    assert captured["payload"]["model"] == DEFAULT_MODEL
    assert captured["payload"]["messages"] == msgs
    assert captured["payload"]["temperature"] == 0.3
    # max_tokens omitted when not requested.
    assert "max_tokens" not in captured["payload"]


def test_max_tokens_included_when_set():
    captured = {}

    def transport(url, headers, payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    client = OpenRouterClient(api_key="testkey", transport=transport)
    client.complete([{"role": "user", "content": "x"}], max_tokens=64)
    assert captured["payload"]["max_tokens"] == 64


def test_transport_error_becomes_llmerror_without_leaking_key():
    def transport(url, headers, payload):
        raise RuntimeError("boom")

    client = OpenRouterClient(api_key="supersecret", transport=transport)
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])

    msg = str(excinfo.value)
    assert "OpenRouter request failed" in msg
    assert "supersecret" not in msg


def test_missing_key_raises_llmerror_mentioning_env_var(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient(api_key=None)
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])

    msg = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in msg
    # No spurious key value in the message.
    assert "Bearer" not in msg


def test_api_key_defaults_to_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "envkey")
    captured = {}

    def transport(url, headers, payload):
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "ok"}}]}

    client = OpenRouterClient(transport=transport)
    client.complete([{"role": "user", "content": "x"}])
    assert captured["headers"]["Authorization"] == "Bearer envkey"
