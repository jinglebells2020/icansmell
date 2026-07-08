"""Tests for sniffsniff.tui.widgets — pure render helpers (no running App)."""
import numpy as np
import pytest

from sniffsniff.tui.widgets import (
    bar_row,
    render_capture,
    render_coach,
    render_header,
    render_label_list,
    render_sensors,
    sparkline,
    trend,
)

_FULL = "█"
_EMPTY = "░"


# --- bar_row (stable rendering primitive) -------------------------------------


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


# --- sparkline ----------------------------------------------------------------


def test_sparkline_empty_is_empty():
    assert sparkline([]) == ""


def test_sparkline_length_matches_window():
    assert len(sparkline([1, 2, 3, 4, 5])) == 5
    assert len(sparkline([1, 2, 3, 4, 5], width=3)) == 3


def test_sparkline_flat_is_even_and_visible():
    out = sparkline([7.0, 7.0, 7.0])
    assert len(set(out)) == 1  # a flat window renders as one repeated glyph
    assert out.strip() != ""   # ... and it is visible (not blank)


def test_sparkline_rising_is_monotonic_nondecreasing():
    out = sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    # low sample maps to a lower block than a high sample
    assert out[0] != out[-1]
    assert out[-1] == "█"


# --- trend --------------------------------------------------------------------


def test_trend_flat_for_short_window():
    assert trend([1, 2]) == "flat"


def test_trend_rising_and_falling():
    assert trend([1, 1, 2, 5, 9, 12]) == "rising"
    assert trend([12, 9, 5, 2, 1, 1]) == "falling"


def test_trend_deadband_reads_flat():
    # a change smaller than the deadband is flat
    assert trend([10.0, 10.0, 10.01, 10.02], deadband=1.0) == "flat"


# --- render_sensors -----------------------------------------------------------


def test_render_sensors_one_line_per_sensor():
    names = ["MQ3", "MQ135", "MQ2"]
    values = np.array([10.0, 20.0, 30.0])
    histories = [[10.0], [20.0], [30.0]]
    text = render_sensors(names, values, histories, vmax=30.0)
    lines = text.split("\n")
    assert len(lines) == len(names)
    for name in names:
        assert any(name in line for line in lines)


def test_render_sensors_honors_noisy_flags():
    names = ["A", "B"]
    values = np.array([1.0, 2.0])
    histories = [[1.0], [2.0]]
    text = render_sensors(names, values, histories, vmax=2.0, noisy=[False, True])
    lines = text.split("\n")
    assert "⚠" not in lines[0]
    assert "⚠" in lines[1]


def test_render_sensors_missing_history_does_not_crash():
    names = ["A", "B"]
    values = np.array([1.0, 2.0])
    text = render_sensors(names, values, histories=[], vmax=2.0)
    assert len(text.split("\n")) == 2


def test_render_sensors_non_finite_does_not_crash():
    # a dead/open channel yields inf/nan Rs; the live view must not crash on int(nan)
    names = ["A", "B", "C"]
    values = np.array([np.nan, np.inf, 5.0])
    histories = [[np.nan, 1.0], [np.inf, 2.0], [3.0, 5.0]]
    text = render_sensors(names, values, histories, vmax=5.0)
    assert len(text.split("\n")) == 3


def test_sparkline_and_trend_tolerate_non_finite():
    assert len(sparkline([float("nan"), 1.0, float("inf"), 3.0])) == 4
    assert trend([float("nan"), float("inf"), 1.0, 2.0, 5.0, 9.0]) in {
        "rising", "flat", "falling"
    }


# --- render_header ------------------------------------------------------------


def test_render_header_sim_and_model():
    text = render_header("sim", connected=True, has_model=True, phase_label="monitoring")
    assert "sniffsniff" in text
    assert "sim" in text
    assert "model ✓" in text
    assert "monitoring" in text


def test_render_header_live_no_model():
    text = render_header("real", connected=True, has_model=False, phase_label="recording")
    assert "live" in text
    assert "no model" in text
    assert "recording" in text


# --- render_capture -----------------------------------------------------------


def test_render_capture_stepper_shows_all_steps():
    text = render_capture("exposure", frac=0.5, detail="⏺ capturing")
    for step in ("settle", "baseline", "exposure", "purge"):
        assert step in text
    assert "50%" in text
    assert "⏺ capturing" in text


def test_render_capture_idle_has_no_progress_bar():
    text = render_capture("monitor", frac=None, detail="")
    assert "%" not in text
    assert "idle" in text


def test_render_capture_recover_mentions_recover():
    text = render_capture("recover", frac=0.8, detail="… recovering")
    assert "recover" in text.lower()
    assert "80%" in text


# --- render_label_list --------------------------------------------------------


def test_render_label_list_marks_current_and_shows_counts():
    labels = ["coffee", "vinegar", "alcohol"]
    counts = {"coffee": 3, "vinegar": 0}
    text = render_label_list(labels, counts, current="vinegar")
    lines = text.split("\n")
    assert len(lines) == 3
    current_line = next(l for l in lines if "vinegar" in l)
    coffee_line = next(l for l in lines if "coffee" in l)
    assert "▸" in current_line
    assert "▸" not in coffee_line
    assert "3" in coffee_line
    assert "0" in current_line
    alcohol_line = next(l for l in lines if "alcohol" in l)
    assert "0" in alcohol_line


def test_render_label_list_dot_meter_and_done_check():
    text = render_label_list(["coffee"], {"coffee": 3}, good_reps=3)
    assert "●●●" in text  # meter full at good_reps
    assert "✓" in text    # done marker


def test_render_label_list_partial_meter():
    text = render_label_list(["coffee"], {"coffee": 1}, good_reps=3)
    assert "●" in text and "○" in text  # one filled, two empty
    assert "✓" not in text


def test_render_label_list_current_not_in_list_marks_nothing():
    text = render_label_list(["coffee"], {}, current="zzz")
    assert "▸" not in text


# --- render_coach -------------------------------------------------------------


def test_render_coach_contains_next_step_text():
    text = render_coach(
        "Enough data — press f to train.",
        connected=True,
        label="coffee",
        reps=3,
        classifier="knn",
        has_model=False,
    )
    assert "Enough data" in text
    assert "NEXT" in text.upper()


def test_render_coach_header_reflects_state():
    text = render_coach(
        "Trained ✓ — press i to identify, m for the smell map.",
        connected=True,
        label="vinegar",
        reps=2,
        classifier="svm",
        has_model=True,
    )
    assert "vinegar" in text
    assert "svm" in text
    assert "2" in text


# --- widget construction (needs textual) --------------------------------------


def test_widgets_construct():
    pytest.importorskip("textual")
    from sniffsniff.tui.widgets import (
        CapturePanel,
        CoachPanel,
        HeaderBar,
        LabelList,
        LogPanel,
        SensorBars,
    )

    HeaderBar()
    SensorBars()
    CapturePanel()
    LabelList()
    CoachPanel()
    LogPanel()
