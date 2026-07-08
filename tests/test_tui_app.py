"""Headless Pilot tests for the SniffApp Textual UI (all --sim).

With the v3 architecture the app runs ONE persistent monitor worker (infinite),
so captures are no longer "a worker that completes". These tests inject a *silent*
frame source (so the background worker does nothing) and drive a capture
deterministically through the same UI handler the worker uses
(``_on_engine_event``), which exercises the engine + UI wiring together.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

pytest.importorskip("textual")

from sniffsniff.config import default_config
from sniffsniff.monitor import ContinuousSim
from sniffsniff.tui.app import SniffApp
from sniffsniff.tui.controller import CLASSIFIERS, SniffController
from sniffsniff.tui.nose import NoseWidget
from sniffsniff.tui.widgets import CoachPanel, LabelList, LogPanel, SensorBars, WorkflowPanel


def _fast_config():
    return dataclasses.replace(
        default_config(), baseline_s=1, exposure_s=1, purge_s=1, plateau_s=0.5
    )


class _Silent:
    """A frame source that yields nothing — the monitor worker starts and finishes
    immediately, leaving app._engine free for deterministic manual driving."""

    def frames(self):
        return iter(())

    def close(self):
        pass

    def begin_odor(self, label, seed=None):
        pass


def _controller(tmp_path):
    return SniffController(
        _fast_config(), out_dir=tmp_path, use_sim=True, seed=0,
        model_path=str(tmp_path / "model.joblib"),
    )


def _app(tmp_path, **kw):
    ctrl = kw.pop("controller", None) or _controller(tmp_path)
    return ctrl, SniffApp(
        ctrl, source_factory=lambda c: _Silent(), paced=False, **kw
    )


def _drive_capture(app, label, *, save=True, seed=0, odor=None):
    """Run one capture (SETTLE → baseline → exposure → purge) through app._engine +
    the UI handler; return the sim source (so callers can keep stepping idle frames)."""
    names = app.controller.config.sensor_names()
    app._engine.settle_hold_s = 0.25       # fast, deterministic settle for tests
    app._engine.settle_max_wait_s = 2.0
    if not save:
        app._identify_pending = True
    app._engine.arm_capture(label, save=save)
    app._active_label = label
    sim = ContinuousSim(app.controller.config, seed=seed, noise_counts=0.0)
    # 1) settle on clean frames until the engine begins the timed capture
    for _ in range(2000):
        app._on_engine_event(app._engine.step(sim.read()), names)
        if app._engine.capturing:
            break
    # 2) present the odor and window the capture
    sim.begin_odor(odor or (label if label != "?" else "coffee"), seed=seed)
    for _ in range(app._engine.n):
        app._on_engine_event(app._engine.step(sim.read()), names)
    return sim


def _settle_app(app, sim, names):
    """Feed clean frames through the app until the engine leaves SETTLE."""
    app._engine.settle_hold_s = 0.25
    app._engine.settle_max_wait_s = 2.0
    for _ in range(2000):
        app._on_engine_event(app._engine.step(sim.read()), names)
        if app._engine.capturing:
            return
    raise AssertionError("engine never settled")


# --- mount / widgets ---------------------------------------------------------

def test_app_mounts_and_has_widgets(tmp_path):
    _, app = _app(tmp_path, reps=1)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            for w in (NoseWidget, SensorBars, LabelList, CoachPanel, LogPanel):
                assert app.query_one(w) is not None
            assert app.query_one("#status") is not None
            assert app._engine is not None  # monitor engine wired on mount

    asyncio.run(scenario())


def test_quit_exits_cleanly(tmp_path):
    _, app = _app(tmp_path, reps=1)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.action_quit()
            await pilot.pause()

    asyncio.run(scenario())
    assert app.is_running is False


# --- continuous view + recovery (the v3 features) ----------------------------

def test_bars_update_continuously_when_idle(tmp_path):
    _, app = _app(tmp_path)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            names = app.controller.config.sensor_names()
            sim = ContinuousSim(app.controller.config, seed=0)
            # feed idle frames (no capture armed) — bars must reflect live Rs
            for _ in range(3):
                ev = app._engine.step(sim.read())
                assert ev["phase"] == "monitor"
                app._on_engine_event(ev, names)
            text = str(app.query_one("#sensors", SensorBars).render())
            assert "MQ3" in text  # live sensor row rendered

    asyncio.run(scenario())


def test_recovery_status_after_a_sniff(tmp_path):
    _, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            names = app.controller.config.sensor_names()
            sim = _drive_capture(app, "coffee")  # one sniff -> arms recovery
            # subsequent idle frames drive the recovery teller
            for _ in range(30):
                app._on_engine_event(app._engine.step(sim.read()), names)
            status = str(app.query_one("#status").render()).lower()
            assert "recover" in status  # "recovering …" or "recovered"

    asyncio.run(scenario())


def test_record_captures_and_saves(tmp_path):
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            _drive_capture(app, "coffee")
            assert ctrl.dataset_counts().get("coffee", 0) == 1

    asyncio.run(scenario())


def test_action_record_arms_the_engine(tmp_path):
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_record()  # should arm a capture on the engine
            assert app._active_label == "coffee"
            assert app._engine.arm_capture("coffee") is False  # already armed

    asyncio.run(scenario())


# --- v2 guided-training bindings (no capture) --------------------------------

def test_next_prev_label_changes_current(tmp_path):
    _, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            start = app.label
            await pilot.press("n")
            await pilot.pause()
            assert app.label != start
            await pilot.press("p")
            await pilot.pause()
            assert app.label == start

    asyncio.run(scenario())


def test_plus_minus_changes_reps(tmp_path):
    _, app = _app(tmp_path, reps=2)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("plus")
            await pilot.pause()
            assert app.reps == 3
            await pilot.press("minus")
            await pilot.press("minus")
            await pilot.press("minus")
            await pilot.pause()
            assert app.reps == 1  # floored at 1

    asyncio.run(scenario())


def test_c_cycles_classifier(tmp_path):
    ctrl, app = _app(tmp_path, reps=1)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            assert ctrl.classifier == "knn"
            await pilot.press("c")
            await pilot.pause()
            assert ctrl.classifier == CLASSIFIERS[1]

    asyncio.run(scenario())


def test_x_deletes_last_sniff(tmp_path):
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            _drive_capture(app, "coffee")
            assert ctrl.dataset_counts().get("coffee", 0) == 1
            await pilot.press("x")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 0

    asyncio.run(scenario())


def test_clear_needs_confirm(tmp_path):
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            _drive_capture(app, "coffee")
            assert ctrl.dataset_counts().get("coffee", 0) == 1
            await pilot.press("X")  # arms
            await pilot.pause()
            assert ctrl.dataset_counts().get("coffee", 0) == 1
            assert app._clear_armed is True
            await pilot.press("X")  # confirms
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts() == {}

    asyncio.run(scenario())


# --- fit / identify / think --------------------------------------------------

def _spy_log(app):
    messages: list[str] = []
    original = app._log

    def logger(msg):
        messages.append(msg)
        original(msg)

    app._log = logger
    return messages


async def _record_two_and_fit(pilot, app):
    _drive_capture(app, "coffee", seed=0)
    app.label = app.controller.known_labels()[1]  # vinegar
    _drive_capture(app, app.label, seed=1)
    await pilot.press("f")
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


def test_guided_flow_record_fit_identify_delete(tmp_path):
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _record_two_and_fit(pilot, app)
            assert ctrl.has_model() is True
            assert "trained" in ctrl.next_step().lower()

            # identify one sniff via the engine (save=False) -> stashes geometry
            _drive_capture(app, "?", save=False, odor="coffee")
            assert app._last_geometry is not None

            # delete a sniff of the current label; count drops
            label_b = app.label
            before = ctrl.dataset_counts().get(label_b, 0)
            await pilot.press("x")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ctrl.dataset_counts().get(label_b, 0) == before - 1

    asyncio.run(scenario())


def test_think_before_identify_logs_hint(tmp_path):
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            messages = _spy_log(app)
            await pilot.press("t")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any("identify" in m.lower() for m in messages)

    asyncio.run(scenario())


def test_think_logs_narrative_after_identify(tmp_path, monkeypatch):
    ctrl, app = _app(tmp_path, label="coffee")
    narrative = "Most likely coffee.\nWithin its cluster; not novel."
    monkeypatch.setattr("sniffsniff.reason.reason", lambda geometry, client, **k: narrative)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            await _record_two_and_fit(pilot, app)
            _drive_capture(app, "?", save=False, odor="coffee")
            assert app._last_geometry is not None

            messages = _spy_log(app)
            await pilot.press("t")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            joined = "\n".join(messages)
            assert "Most likely coffee." in joined
            assert "Within its cluster; not novel." in joined

    asyncio.run(scenario())


def test_record_one_directly(tmp_path):
    """Controller-level capture still works independent of the TUI."""
    ctrl = _controller(tmp_path)
    path = ctrl.record_one("coffee")
    assert path.exists()
    assert ctrl.dataset_counts().get("coffee", 0) >= 1


class _FakeRealSource:
    """A write-capable, non-streaming source standing in for a real serial link."""

    def __init__(self):
        self.cmds = []

    def frames(self):
        return iter(())

    def close(self):
        pass

    def write_command(self, text):
        self.cmds.append(text.strip())
        return True


def test_servo_driven_on_phase_transitions(tmp_path):
    cfg = _fast_config()  # servo_enabled True via default_config (0 / 105)
    ctrl = SniffController(
        cfg, out_dir=tmp_path, use_sim=False, port="x",
        model_path=str(tmp_path / "m.joblib"),
    )
    src = _FakeRealSource()
    app = SniffApp(ctrl, source_factory=lambda c: src, paced=False)

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            names = cfg.sensor_names()
            app._engine.arm_capture("coffee", save=True)
            app._active_label = app._rep_label = "coffee"
            app._rep_total = app._reps_remaining = 1
            sim = ContinuousSim(cfg, seed=0, noise_counts=0.0)
            _settle_app(app, sim, names)   # servo holds fresh air during settle
            sim.begin_odor("coffee")
            for _ in range(app._engine.n):
                app._on_engine_event(app._engine.step(sim.read()), names)
            assert src.cmds[0] == "S0"     # settle / baseline -> fresh air (0°)
            assert "S105" in src.cmds      # exposure -> sample (105°)
            assert src.cmds[-1] == "S0"    # purge -> fresh air (0°)

    asyncio.run(scenario())


def test_sim_does_not_drive_a_servo(tmp_path):
    # in sim mode there is no serial write; servo commands must not be attempted
    ctrl, app = _app(tmp_path, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            names = app.controller.config.sensor_names()
            # a silent source has no write_command; driving frames must not error
            sim = ContinuousSim(app.controller.config, seed=0, noise_counts=0.0)
            app._engine.arm_capture("coffee")
            _settle_app(app, sim, names)
            sim.begin_odor("coffee")
            for _ in range(app._engine.n):
                app._on_engine_event(app._engine.step(sim.read()), names)
            # nothing to assert about a servo; the point is no crash + a sniff saved
            assert app.controller.dataset_counts().get("coffee", 0) == 1

    asyncio.run(scenario())


def test_auto_reps_settle_gated(tmp_path):
    cfg = _fast_config()
    ctrl, app = _app(tmp_path, reps=2, label="coffee")

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            names = cfg.sensor_names()
            # start a 2-rep sequence (mimic action_record without source coupling)
            app._rep_label = "coffee"
            app._rep_total = app._reps_remaining = 2
            app._engine.arm_capture("coffee", save=True)
            app._active_label = "coffee"
            sim = ContinuousSim(cfg, seed=0, noise_counts=0.0)
            # rep 1: settle then capture
            _settle_app(app, sim, names)
            sim.begin_odor("coffee")
            for _ in range(app._engine.n):
                app._on_engine_event(app._engine.step(sim.read()), names)
            assert app._reps_remaining == 1
            assert app._engine.busy is True   # rep 2 was re-armed and is settling
            # rep 2: settle (gates on return-to-rest) then capture
            for _ in range(2000):
                app._on_engine_event(app._engine.step(sim.read()), names)
                if app._engine.capturing:
                    break
            sim.begin_odor("coffee")
            for _ in range(app._engine.n):
                app._on_engine_event(app._engine.step(sim.read()), names)
            assert app._reps_remaining == 0
            assert ctrl.dataset_counts().get("coffee", 0) == 2  # both reps saved

    asyncio.run(scenario())
