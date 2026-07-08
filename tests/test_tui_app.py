"""Headless Pilot tests for the SniffApp Textual UI (all --sim).

The whole file is skipped when ``textual`` is not installed. Timing is kept short
via a fast config and ``reps=1`` so the sim sessions record quickly.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

pytest.importorskip("textual")

from sniffsniff.config import default_config
from sniffsniff.tui.app import SniffApp
from sniffsniff.tui.controller import SniffController
from sniffsniff.tui.nose import NoseWidget
from sniffsniff.tui.widgets import LogPanel, SensorBars, WorkflowPanel


def _fast_config():
    return dataclasses.replace(
        default_config(),
        baseline_s=1,
        exposure_s=2,
        purge_s=1,
        plateau_s=0.5,
    )


def _controller(tmp_path):
    return SniffController(
        _fast_config(),
        out_dir=tmp_path,
        use_sim=True,
        seed=0,
        model_path=str(tmp_path / "model.joblib"),
    )


def test_app_mounts_and_has_widgets(tmp_path):
    app = SniffApp(_controller(tmp_path), reps=1)

    async def scenario():
        async with app.run_test() as pilot:
            assert app.query_one(NoseWidget) is not None
            assert app.query_one(SensorBars) is not None
            assert app.query_one(WorkflowPanel) is not None
            assert app.query_one(LogPanel) is not None
            # Queryable by id too, so widget-specific tests can target them.
            assert app.query_one("#nose", NoseWidget) is not None
            await pilot.pause()

    asyncio.run(scenario())


def test_record_action_saves_a_sniff(tmp_path):
    ctrl = _controller(tmp_path)
    app = SniffApp(ctrl, reps=1, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

    asyncio.run(scenario())

    counts = ctrl.dataset_counts()
    assert counts.get("coffee", 0) >= 1


def test_quit_exits_cleanly(tmp_path):
    app = SniffApp(_controller(tmp_path), reps=1)

    async def scenario():
        async with app.run_test() as pilot:
            assert app.query_one(NoseWidget) is not None
            await app.action_quit()
            await pilot.pause()

    asyncio.run(scenario())
    assert app.is_running is False


def test_record_one_directly(tmp_path):
    """Deterministic non-Pilot assertion that a sniff is captured + saved."""
    ctrl = _controller(tmp_path)
    path = ctrl.record_one("coffee")
    assert path.exists()
    assert ctrl.dataset_counts().get("coffee", 0) >= 1
