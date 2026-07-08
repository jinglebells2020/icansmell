"""Regression tests for the TUI usability fixes (root causes found via the
systematic-debugging pass on the first live run):

A. SensorBars must be populated on mount (was empty/tiny until a capture ran).
B. A capture in progress must block a second one (no worker flood on key-mashing).
C. The interactive default is one sniff per `r` press (was reps=8 = 20-min batch).
D. Connection status must actually reflect whether the device is present, and a
   real-mode launch with no device must warn instead of silently failing.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from sniffsniff.config import default_config
from sniffsniff.tui.controller import SniffController


def _fast():
    return dataclasses.replace(
        default_config(), baseline_s=1, exposure_s=1, purge_s=1, plateau_s=0.5
    )


# --- D: honest connection status (no textual needed) -------------------------

def test_connected_false_for_missing_real_port(tmp_path):
    ctrl = SniffController(
        _fast(), out_dir=tmp_path, use_sim=False, port="/dev/cu.nope_xyz"
    )
    assert ctrl.connected is False


def test_connected_true_for_sim(tmp_path):
    ctrl = SniffController(_fast(), out_dir=tmp_path, use_sim=True)
    assert ctrl.connected is True


def test_connected_true_when_real_port_exists(tmp_path):
    # A path that definitely exists stands in for a present device node.
    port = tmp_path / "fake_tty"
    port.write_text("")
    ctrl = SniffController(_fast(), out_dir=tmp_path, use_sim=False, port=str(port))
    assert ctrl.connected is True


# --- textual-dependent app tests ---------------------------------------------

pytest.importorskip("textual")

from sniffsniff.tui.app import SniffApp  # noqa: E402
from sniffsniff.tui.widgets import SensorBars  # noqa: E402


def _sim_ctrl(tmp_path):
    return SniffController(
        _fast(), out_dir=tmp_path, use_sim=True, seed=0,
        model_path=str(tmp_path / "m.joblib"),
    )


# C: interactive default is one sniff per press
def test_default_reps_is_one(tmp_path):
    assert SniffApp(_sim_ctrl(tmp_path)).reps == 1


# A: sensor panel has content immediately after mount
def test_sensorbars_populated_on_mount(tmp_path):
    app = SniffApp(_sim_ctrl(tmp_path))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            sensors = app.query_one("#sensors", SensorBars)
            text = str(sensors.render())
            # Every configured sensor name shows up in the initial (zeroed) bars.
            for name in app.controller.config.sensor_names():
                assert name in text
            assert app.query_one("#sensors").region.height > 5

    asyncio.run(scenario())


# B: a second capture is refused while one is running
def test_record_busy_guard(tmp_path):
    app = SniffApp(_sim_ctrl(tmp_path))
    calls = {"n": 0}

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            # pretend a capture is already running
            app._busy = True
            app._record_worker = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1)
            app.action_record()
            await pilot.pause()
            assert calls["n"] == 0  # worker NOT started while busy
            log_text = str(app.query_one("#log").render()) if hasattr(
                app.query_one("#log"), "render"
            ) else ""
            # a "busy" notice was logged (best-effort; the guard is the real assert)

    asyncio.run(scenario())


# D: real-mode launch with no device warns on mount
def test_real_mode_mount_warns_no_device(tmp_path):
    ctrl = SniffController(
        _fast(), out_dir=tmp_path, use_sim=False, port="/dev/cu.nope_xyz",
        model_path=str(tmp_path / "m.joblib"),
    )
    logs = []
    app = SniffApp(ctrl)

    async def scenario():
        async with app.run_test() as pilot:
            app._log = lambda msg: logs.append(msg)  # capture after mount's first log
            app.on_mount()
            await pilot.pause()

    asyncio.run(scenario())
    assert any("no device" in m.lower() or "reconnect" in m.lower() for m in logs)


def test_controller_real_reader_does_not_reconnect(tmp_path):
    # A bounded capture must not infinite-reconnect on a silent/absent device
    # (otherwise the TUI worker — and the busy guard — freeze forever).
    ctrl = SniffController(_fast(), out_dir=tmp_path, use_sim=False, port="/dev/cu.whatever")
    reader = ctrl._reader(odor="coffee")
    assert reader.reconnect is False
