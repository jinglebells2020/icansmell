"""Bounded, guided session capture — the real-hardware recording front-end.

A live :class:`~sniffsniff.serialio.SerialReader` streams forever, so capturing a
sniff means taking *exactly one session's worth* of frames (baseline + exposure +
purge) and then stopping. This module bounds that capture and reports phase
transitions so an operator can be cued when to present and remove the sample.

Without this, draining a live reader never returns — the CLI would hang.
"""
from __future__ import annotations

from typing import Callable, Optional

from .config import Config
from .record import phase_slices

__all__ = ["session_frame_count", "phase_of", "capture_session"]


def session_frame_count(config: Config) -> int:
    """Number of frames in one full baseline+exposure+purge session at ``scan_hz``."""
    total_s = config.baseline_s + config.exposure_s + config.purge_s
    return int(round(total_s * config.scan_hz))


def phase_of(k: int, slices: dict) -> str:
    """Which protocol phase frame index ``k`` falls in (``"baseline"``/``"exposure"``/``"purge"``)."""
    if k < slices["baseline"][1]:
        return "baseline"
    if k < slices["exposure"][1]:
        return "exposure"
    return "purge"


def capture_session(
    reader,
    config: Config,
    *,
    on_phase: Optional[Callable[[str, int, int], None]] = None,
    on_frame: Optional[Callable[[int, int, str, tuple], None]] = None,
) -> list:
    """Capture exactly one session's frames from ``reader``, then stop.

    Pulls up to :func:`session_frame_count` frames from ``reader.frames()`` and
    returns them as a list. This bounds a *live* reader that would otherwise stream
    forever. If the stream ends early (e.g. the simulator's finite session, or a
    disconnect), the frames captured so far are returned — never padded, never hung.

    ``on_phase(phase, k, n)`` fires once at each phase transition (baseline →
    exposure → purge); ``on_frame(k, n, phase, frame)`` fires for every captured
    frame (``frame`` is the ``(t_ms, raw)`` tuple). The reader is closed when done.
    """
    n = session_frame_count(config)
    slices = phase_slices(n, config)
    frames: list = []
    current_phase: Optional[str] = None

    iterator = reader.frames()
    try:
        for k in range(n):
            frame = next(iterator, None)
            if frame is None:
                break  # stream ended early — return what we have
            frames.append(frame)

            phase = phase_of(k, slices)
            if phase != current_phase:
                current_phase = phase
                if on_phase is not None:
                    on_phase(phase, k, n)
            if on_frame is not None:
                on_frame(k, n, phase, frame)
    finally:
        try:
            reader.close()
        except Exception:  # pragma: no cover - close() is best-effort
            pass
    return frames
