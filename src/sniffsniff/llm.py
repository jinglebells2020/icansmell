"""Milestone 3 — a thin, dependency-free OpenRouter chat client.

The reasoner talks to OpenRouter's OpenAI-compatible chat endpoint using only
the standard library (:mod:`urllib`). The API key is read **only** from the
``OPENROUTER_API_KEY`` environment variable (or passed explicitly) and is placed
**only** in the ``Authorization`` header — it is never returned, logged, or
embedded in any exception text.

Tests inject a fake ``transport`` callable so no network is touched.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

__all__ = ["DEFAULT_MODEL", "BASE_URL", "LLMError", "OpenRouterClient"]

DEFAULT_MODEL = "moonshotai/kimi-k2-thinking"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(Exception):
    """Raised on a missing key or any OpenRouter request/parse failure.

    Its message never contains the API key or the request headers.
    """


class OpenRouterClient:
    """A minimal OpenRouter chat-completions client.

    Parameters
    ----------
    model:
        Chat model id (defaults to :data:`DEFAULT_MODEL`).
    api_key:
        Bearer key; defaults to ``os.environ.get("OPENROUTER_API_KEY")``.
    base_url, timeout, referer, title:
        Endpoint and the ``HTTP-Referer`` / ``X-Title`` attribution headers.
    transport:
        Optional ``callable(url, headers, payload) -> dict`` returning the parsed
        JSON response. Defaults to a real urllib POST; tests inject a fake.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 120.0,
        referer: str = "https://github.com/sniffsniff",
        title: str = "sniffsniff",
        transport=None,
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.referer = referer
        self.title = title
        self._transport = transport if transport is not None else self._urllib_transport

    def complete(self, messages: list[dict], *, temperature: float = 0.3, max_tokens: int | None = None) -> str:
        """Send ``messages`` and return the assistant text.

        Falls back to ``message.reasoning`` when ``message.content`` is falsy
        (the thinking model may return its answer there). Raises
        :class:`LLMError` on a missing key or any request/parse failure — never
        leaking the key.
        """
        if not self.api_key:
            raise LLMError(
                "the reasoner needs an OpenRouter key — set OPENROUTER_API_KEY "
                "(get one at openrouter.ai)"
            )

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }

        try:
            data = self._transport(self.base_url, headers, payload)
            message = data["choices"][0]["message"]
            content = message.get("content")
            if not content:
                content = message.get("reasoning")
            if not content:
                raise ValueError("empty completion (no content or reasoning)")
            return content
        except LLMError:
            raise
        except Exception as exc:  # transport / urllib / parse — sanitize the message.
            raise LLMError(f"OpenRouter request failed: {_short_reason(exc)}") from None

    def _urllib_transport(self, url: str, headers: dict, payload: dict) -> dict:
        """Default transport: a real urllib POST returning parsed JSON.

        HTTPError/URLError are wrapped into :class:`LLMError` without leaking the
        key or headers.
        """
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise LLMError(f"OpenRouter request failed: {_short_reason(exc)}") from None


def _short_reason(exc: Exception) -> str:
    """A compact, key-free description of a failure.

    Uses the exception's class name plus a best-effort short reason. Never
    includes headers or the payload, so the API key cannot leak here.
    """
    reason = getattr(exc, "reason", None)
    if reason is not None:
        return f"{type(exc).__name__}: {reason}"
    code = getattr(exc, "code", None)
    if code is not None:
        return f"{type(exc).__name__}: HTTP {code}"
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
