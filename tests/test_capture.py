"""Tests for bounded, guided session capture.

The critical property: capturing from a *live* reader (which streams forever)
must stop after exactly one session's worth of frames — never hang.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from sniffsniff.capture import capture_session, phase_of, session_frame_count
from sniffsniff.config import default_config
from sniffsniff.record import phase_slices
from sniffsniff.simulator import SimulatedReader, Simulator


def _short():
    # (1 + 2 + 1) s * 20 Hz = 80 frames — cheap sessions for tests.
    return replace(
        default_config(), baseline_s=1.0, exposure_s=2.0, purge_s=1.0, plateau_s=0.5
    )


class _InfiniteReader:
    """A reader whose frames() never ends — models a live serial stream."""

    def __init__(self):
        self.closed = False

    def frames(self):
        k = 0
        while True:
            yield (k * 50, np.zeros(6, dtype=np.int64))
            k += 1

    def close(self):
        self.closed = True


def test_session_frame_count():
    assert session_frame_count(_short()) == 80


def test_capture_bounds_an_infinite_reader():
    # THE key test: a forever-streaming reader must be bounded, not hang.
    cfg = _short()
    reader = _InfiniteReader()
    frames = capture_session(reader, cfg)
    assert len(frames) == session_frame_count(cfg) == 80
    assert reader.closed  # capture closes the reader when done


def test_capture_reports_phases_in_order():
    cfg = _short()
    reader = SimulatedReader(Simulator(cfg, seed=0).sniff_frames("coffee"))
    seen = []
    capture_session(reader, cfg, on_phase=lambda ph, k, n: seen.append(ph))
    assert seen == ["baseline", "exposure", "purge"]


def test_capture_invokes_on_frame_for_every_frame():
    cfg = _short()
    reader = SimulatedReader(Simulator(cfg, seed=0).sniff_frames("coffee"))
    ticks = []
    capture_session(reader, cfg, on_frame=lambda k, n, ph, fr: ticks.append(k))
    assert ticks == list(range(session_frame_count(cfg)))


def test_capture_partial_when_stream_ends_early():
    cfg = _short()
    reader = SimulatedReader([(0, np.zeros(6, dtype=np.int64))])  # only 1 frame
    frames = capture_session(reader, cfg)
    assert len(frames) == 1  # returns what it got, does not pad or hang


def test_phase_of_boundaries():
    cfg = _short()
    n = session_frame_count(cfg)
    sl = phase_slices(n, cfg)
    assert phase_of(0, sl) == "baseline"
    assert phase_of(sl["baseline"][1] - 1, sl) == "baseline"
    assert phase_of(sl["exposure"][0], sl) == "exposure"
    assert phase_of(sl["purge"][0], sl) == "purge"
    assert phase_of(n - 1, sl) == "purge"
