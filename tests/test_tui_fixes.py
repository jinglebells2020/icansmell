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


class _Silent:
    """Frame source that yields nothing — the monitor worker starts + finishes at
    once, and (real mode) never opens a serial port."""

    def frames(self):
        return iter(())

    def close(self):
        pass

    def begin_odor(self, label, seed=None):
        pass


def _mk(ctrl):
    return SniffApp(ctrl, source_factory=lambda c: _Silent(), paced=False)


# C: interactive default is one sniff per press
def test_default_reps_is_one(tmp_path):
    assert _mk(_sim_ctrl(tmp_path)).reps == 1


# A: sensor panel has content immediately after mount
def test_sensorbars_populated_on_mount(tmp_path):
    app = _mk(_sim_ctrl(tmp_path))

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


# B: a second capture is refused while one is already being captured
def test_record_busy_guard(tmp_path):
    from sniffsniff.monitor import ContinuousSim

    app = _mk(_sim_ctrl(tmp_path))

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()
            # get the engine into an active capture (arm + one step promotes it)
            app._engine.arm_capture("coffee")
            app._engine.step(ContinuousSim(app.controller.config).read())
            assert app._engine.capturing is True
            logs = []
            app._log = logs.append
            app.action_record()  # must refuse while a sniff is being captured
            assert app._engine.arm_capture("coffee") is False
            assert any("busy" in m.lower() for m in logs)

    asyncio.run(scenario())


# D: real-mode launch with no device warns on mount
def test_real_mode_mount_warns_no_device(tmp_path):
    ctrl = SniffController(
        _fast(), out_dir=tmp_path, use_sim=False, port="/dev/cu.nope_xyz",
        model_path=str(tmp_path / "m.joblib"),
    )
    app = _mk(ctrl)  # silent source -> mount doesn't try to open the missing port
    logs = []
    app._log = logs.append  # spy set BEFORE mount so on_mount's warning is captured

    async def scenario():
        async with app.run_test() as pilot:
            await pilot.pause()

    asyncio.run(scenario())
    assert any("no device" in m.lower() or "reconnect" in m.lower() for m in logs)


def test_controller_real_reader_does_not_reconnect(tmp_path):
    # A bounded capture must not infinite-reconnect on a silent/absent device
    # (otherwise the TUI worker — and the busy guard — freeze forever).
    ctrl = SniffController(_fast(), out_dir=tmp_path, use_sim=False, port="/dev/cu.whatever")
    reader = ctrl._reader(odor="coffee")
    assert reader.reconnect is False
