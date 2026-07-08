"""Tests for sniffsniff.tui.widgets — pure render helpers (no running App)."""
import numpy as np
import pytest

from sniffsniff.tui.widgets import (
    bar_row,
    render_sensor_bars,
    render_workflow,
)

_FULL = "█"
_EMPTY = "░"


# --- bar_row ------------------------------------------------------------------


def test_bar_row_full_at_vmax():
    row = bar_row("MQ3", 100.0, 100.0, width=14)
    assert _FULL * 14 in row
    assert _EMPTY not in row


def test_bar_row_empty_at_zero():
    row = bar_row("MQ3", 0.0, 100.0, width=14)
    assert _EMPTY * 14 in row
    assert _FULL not in row


def test_bar_row_half():
    row = bar_row("MQ3", 50.0, 100.0, width=10)
    assert _FULL * 5 in row
    assert row.count(_FULL) == 5
    assert row.count(_EMPTY) == 5


def test_bar_row_clamps_over_vmax():
    row = bar_row("MQ3", 999.0, 100.0, width=14)
    assert row.count(_FULL) == 14
    assert _EMPTY not in row


def test_bar_row_noisy_appends_warn():
    clean = bar_row("MQ3", 40000.0, 100000.0, noisy=False)
    dirty = bar_row("MQ3", 40000.0, 100000.0, noisy=True)
    assert "⚠" in dirty
    assert "⚠" not in clean


def test_bar_row_length_stable_across_values():
    lengths = {
        len(bar_row("MQ3", v, 100.0, width=14))
        for v in (0.0, 25.0, 50.0, 99.9, 100.0)
    }
    assert len(lengths) == 1, "bar length must not depend on value"


def test_bar_row_vmax_zero_is_empty_not_crash():
    row = bar_row("MQ3", 5.0, 0.0, width=8)
    assert row.count(_EMPTY) == 8
    assert _FULL not in row


def test_bar_row_shows_name_and_value():
    row = bar_row("MQ135", 41200.0, 100000.0)
    assert "MQ135" in row
    assert "41.2k" in row


# --- render_sensor_bars -------------------------------------------------------


def test_render_sensor_bars_one_line_per_sensor_plus_header():
    names = ["MQ3", "MQ135", "MQ2"]
    values = np.array([10.0, 20.0, 30.0])
    text = render_sensor_bars(names, values, vmax=30.0, phase="exposure", elapsed=1.5)
    lines = text.split("\n")
    assert len(lines) == 1 + len(names)
    assert "exposure" in lines[0]
    for name in names:
        assert any(name in line for line in lines[1:])


def test_render_sensor_bars_honors_noisy_flags():
    names = ["A", "B"]
    values = np.array([1.0, 2.0])
    text = render_sensor_bars(names, values, vmax=2.0, noisy=[False, True])
    lines = text.split("\n")[1:]
    assert "⚠" not in lines[0]
    assert "⚠" in lines[1]


# --- render_workflow ----------------------------------------------------------


def test_render_workflow_disconnected_no_model():
    text = render_workflow(False, {}, False)
    assert "✗ connected" in text
    assert "model: ✗" in text


def test_render_workflow_connected_with_counts_and_model():
    text = render_workflow(True, {"coffee": 3, "vinegar": 2}, True)
    assert "✓ connected" in text
    assert "coffee" in text and "3" in text
    assert "vinegar" in text
    assert "model: ✓" in text


# --- widget construction (needs textual) --------------------------------------


def test_widgets_construct():
    pytest.importorskip("textual")
    from sniffsniff.tui.widgets import LogPanel, SensorBars, WorkflowPanel

    SensorBars()
    WorkflowPanel()
    LogPanel()


def test_nose_widget_constructs():
    pytest.importorskip("textual")
    from sniffsniff.tui.nose import NoseWidget

    w = NoseWidget()
    assert w.animation.state == "idle"
