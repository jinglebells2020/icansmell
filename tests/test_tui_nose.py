"""Tests for sniffsniff.tui.nose — pure NoseAnimation logic (no running App)."""
import pytest

from sniffsniff.tui.nose import IDLE_FRAMES, SNIFF_FRAMES, NoseAnimation


# --- frame integrity ----------------------------------------------------------


@pytest.mark.parametrize("frames", [IDLE_FRAMES, SNIFF_FRAMES])
def test_frames_equal_line_count(frames):
    line_counts = {frame.count("\n") for frame in frames}
    assert len(line_counts) == 1, "every frame in a state must have equal line count"


@pytest.mark.parametrize("frames", [IDLE_FRAMES, SNIFF_FRAMES])
def test_frames_equal_width(frames):
    for frame in frames:
        widths = {len(line) for line in frame.split("\n")}
        assert len(widths) == 1, "every line of a frame must share one width"


def test_frame_counts_in_spec_range():
    assert 2 <= len(IDLE_FRAMES) <= 4
    assert 3 <= len(SNIFF_FRAMES) <= 6


# --- NoseAnimation ------------------------------------------------------------


def test_defaults_to_idle():
    nose = NoseAnimation()
    assert nose.state == "idle"
    assert nose.index == 0
    assert nose.frames() == IDLE_FRAMES


def test_advance_returns_current_then_increments():
    nose = NoseAnimation()
    assert nose.advance() == IDLE_FRAMES[0]
    assert nose.index == 1
    assert nose.advance() == IDLE_FRAMES[1]
    assert nose.index == 2


def test_advance_cycles_and_wraps():
    nose = NoseAnimation()
    seen = [nose.advance() for _ in range(len(IDLE_FRAMES))]
    assert seen == IDLE_FRAMES
    # after a full cycle it wraps back to the first frame
    assert nose.index == 0
    assert nose.advance() == IDLE_FRAMES[0]


def test_set_state_switches_list_and_resets_index():
    nose = NoseAnimation()
    nose.advance()
    nose.advance()
    assert nose.index == 2
    nose.set_state("sniffing")
    assert nose.state == "sniffing"
    assert nose.index == 0
    assert nose.frames() == SNIFF_FRAMES
    assert nose.advance() == SNIFF_FRAMES[0]


def test_set_state_back_to_idle():
    nose = NoseAnimation()
    nose.set_state("sniffing")
    nose.set_state("idle")
    assert nose.state == "idle"
    assert nose.frames() == IDLE_FRAMES


def test_set_state_rejects_unknown():
    nose = NoseAnimation()
    with pytest.raises(ValueError):
        nose.set_state("chewing")
