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


# --- v2 guided-training app bindings -----------------------------------------

from sniffsniff.tui.controller import CLASSIFIERS


def test_next_prev_label_changes_current(tmp_path):
    app = SniffApp(_controller(tmp_path), reps=1, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            start = app.label
            await pilot.press("n")
            await pilot.pause()
            after_n = app.label
            assert after_n != start
            await pilot.press("p")
            await pilot.pause()
            assert app.label == start

    asyncio.run(scenario())


def test_plus_minus_changes_reps(tmp_path):
    app = SniffApp(_controller(tmp_path), reps=2)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("plus")
            await pilot.pause()
            assert app.reps == 3
            await pilot.press("minus")
            await pilot.press("minus")
            await pilot.pause()
            assert app.reps == 1
            # never drops below 1
            await pilot.press("minus")
            await pilot.pause()
            assert app.reps == 1

    asyncio.run(scenario())


def test_c_cycles_classifier(tmp_path):
    ctrl = _controller(tmp_path)
    app = SniffApp(ctrl, reps=1)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert ctrl.classifier == "knn"
            await pilot.press("c")
            await pilot.pause()
            assert ctrl.classifier == CLASSIFIERS[1]

    asyncio.run(scenario())


def test_x_deletes_last_sniff(tmp_path):
    ctrl = _controller(tmp_path)
    app = SniffApp(ctrl, reps=1, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 1
            await pilot.press("x")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 0

    asyncio.run(scenario())


def test_clear_needs_confirm(tmp_path):
    ctrl = _controller(tmp_path)
    app = SniffApp(ctrl, reps=1, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 1
            # First X arms but does not clear.
            await pilot.press("X")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 1
            assert app._clear_armed is True
            # Second X confirms and clears.
            await pilot.press("X")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts() == {}
            assert app._clear_armed is False

    asyncio.run(scenario())


def test_coach_updates_after_record(tmp_path):
    ctrl = _controller(tmp_path)
    app = SniffApp(ctrl, reps=1, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            from sniffsniff.tui.widgets import CoachPanel

            coach = app.query_one("#coach", CoachPanel)
            # Record label A, switch to B, record B — crossing into "≥2 classes"
            # must change the coach guidance.
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            before = str(coach.render())
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            after = str(coach.render())
            assert before != after

    asyncio.run(scenario())


def test_guided_flow_end_to_end(tmp_path):
    """The full guided workflow: record two labels, fit, identify, delete."""
    ctrl = _controller(tmp_path)
    app = SniffApp(ctrl, reps=1, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()

            # Record label A (coffee).
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 1

            # Switch to label B and record it.
            await pilot.press("n")
            await pilot.pause()
            label_b = app.label
            assert label_b != "coffee"
            await pilot.press("r")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get(label_b, 0) == 1

            # Two classes → ready to fit; coach nudges to train.
            assert ctrl.ready_to_fit() is True

            # Fit.
            await pilot.press("f")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.has_model() is True
            assert "trained" in ctrl.next_step().lower()

            # Identify.
            await pilot.press("i")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            # Delete a sniff of the current label; count drops.
            before = ctrl.dataset_counts().get(label_b, 0)
            app.label = label_b
            await pilot.press("x")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            after = ctrl.dataset_counts().get(label_b, 0)
            assert after == before - 1

    asyncio.run(scenario())
