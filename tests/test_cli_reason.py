"""Tests for the ``reason`` CLI subcommand (no network).

The subcommand captures/sim one sniff, builds the geometry, and asks an LLM to
narrate it. Tests monkeypatch :class:`sniffsniff.llm.OpenRouterClient` so no
network is ever touched, and separately verify that a missing key degrades to a
friendly hint (return 1, no traceback, no key leak).
"""
from __future__ import annotations

import dataclasses

import pytest

from sniffsniff import cli
from sniffsniff.config import default_config


def _fast_config():
    return dataclasses.replace(
        default_config(),
        baseline_s=1,
        exposure_s=2,
        purge_s=1,
        plateau_s=0.5,
    )


@pytest.fixture
def fitted_model(tmp_path, monkeypatch):
    """Fit a real SmellModel from the simulator and return its path."""
    # Keep captures fast by patching default_config used inside the CLI.
    monkeypatch.setattr(cli, "default_config", _fast_config)
    model_path = tmp_path / "model.joblib"
    rc = cli.main(
        [
            "fit",
            "--sim",
            "--reps",
            "4",
            "--out",
            str(model_path),
        ]
    )
    assert rc == 0
    assert model_path.exists()
    return model_path


class _FakeClient:
    """Stand-in for OpenRouterClient: returns a fixed narrative, no network."""

    NARRATIVE = "Most likely coffee.\nWithin its cluster; not novel.\nSuggest: commit."

    def __init__(self, *args, **kwargs):
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return self.NARRATIVE


def test_reason_prints_narrative_and_returns_zero(fitted_model, monkeypatch, capsys):
    monkeypatch.setattr("sniffsniff.llm.OpenRouterClient", _FakeClient)

    rc = cli.main(
        [
            "reason",
            "--model",
            str(fitted_model),
            "--sim",
            "--odor",
            "coffee",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    # The fake narrative is printed (possibly multi-line).
    assert "Most likely coffee." in out
    assert "Suggest: commit." in out
    # The quick verdict line (predicted / novelty) is printed too.
    assert "predicted" in out.lower()
    assert "novelty" in out.lower()


def test_reason_missing_model_returns_one(tmp_path, capsys):
    missing = tmp_path / "nope.joblib"
    rc = cli.main(["reason", "--model", str(missing), "--sim", "--odor", "coffee"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "model" in out.lower()


def test_reason_missing_key_prints_hint_no_traceback(fitted_model, monkeypatch, capsys):
    # Real client + no key → LLMError with a friendly hint, caught by the handler.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    rc = cli.main(
        [
            "reason",
            "--model",
            str(fitted_model),
            "--sim",
            "--odor",
            "coffee",
        ]
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "OPENROUTER_API_KEY" in out
    # No traceback leaked to stdout.
    assert "Traceback" not in out
