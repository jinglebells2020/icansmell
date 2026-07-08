"""Lightweight tests for the ``tui`` CLI subcommand.

Running the real app needs a TTY, so instead of calling ``.run()`` we verify:

* ``_cmd_tui`` builds a :class:`SniffController` from the parsed args and forwards
  it to ``run_tui`` (which we stub out); and
* when ``textual`` cannot be imported, ``_cmd_tui`` prints the install hint and
  returns ``1``.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("textual")

from sniffsniff import cli
from sniffsniff.tui.controller import SniffController


def test_cmd_tui_builds_controller_and_runs(monkeypatch, tmp_path):
    captured = {}

    def fake_run_tui(controller, *, reps=8, label=None):
        captured["controller"] = controller
        captured["reps"] = reps
        captured["label"] = label

    # run_tui is imported inside _cmd_tui from sniffsniff.tui.app.
    monkeypatch.setattr("sniffsniff.tui.app.run_tui", fake_run_tui)

    rc = cli.main(
        [
            "tui",
            "--sim",
            "--out",
            str(tmp_path),
            "--model",
            str(tmp_path / "model.joblib"),
            "--reps",
            "3",
            "--label",
            "vinegar",
        ]
    )

    assert rc == 0
    ctrl = captured["controller"]
    assert isinstance(ctrl, SniffController)
    assert ctrl.use_sim is True
    assert ctrl.out_dir == str(tmp_path)
    assert captured["reps"] == 3
    assert captured["label"] == "vinegar"


def test_cmd_tui_missing_textual_returns_hint(monkeypatch, capsys, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sniffsniff.tui.app" or name.endswith("tui.app"):
            raise ImportError("no textual")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rc = cli.main(["tui", "--sim", "--out", str(tmp_path)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "pip install sniffsniff[tui]" in out
