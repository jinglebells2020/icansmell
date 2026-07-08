"""Display panels for the sniffsniff TUI.

Each panel is split into a **pure render helper** (a function that turns state
into a string, unit-testable without a running App) and a thin
:class:`textual.widgets.Static` wrapper that calls the helper and pushes the
result via ``self.update(...)``.

* :func:`bar_row` / :class:`SensorBars` — a live per-sensor bar chart.
* :class:`WorkflowPanel` — a connected / captured / model checklist.
* :class:`LogPanel` — a scrolling event log.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

# Block characters for the bar fill / empty track.
_FULL = "█"
_EMPTY = "░"
_WARN = "⚠"


def _fmt_value(value: float) -> str:
    """Compact human value: ``41200`` -> ``41.2k``, ``950`` -> ``950``."""
    v = float(value)
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def bar_row(
    name: str,
    value: float,
    vmax: float,
    width: int = 14,
    noisy: bool = False,
) -> str:
    """Render one sensor row: ``"MQ3   ██████░░░░  41.2k [⚠]"``.

    The bar is ``width`` cells wide; ``value / vmax`` (clamped to ``[0, 1]``)
    of the cells are filled with :data:`_FULL`, the rest with :data:`_EMPTY`.
    A ``value`` of ``vmax`` fills the whole bar; ``0`` leaves it empty. When
    ``noisy`` is true a ``[⚠]`` warn mark is appended. The overall structure
    (and so its length for a given ``name``/``vmax``) is stable regardless of
    ``value``.
    """
    if vmax <= 0:
        frac = 0.0
    else:
        frac = value / vmax
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0

    filled = int(round(frac * width))
    if filled > width:
        filled = width
    bar = _FULL * filled + _EMPTY * (width - filled)

    warn = f" [{_WARN}]" if noisy else ""
    return f"{name:<6} {bar}  {_fmt_value(value):>6}{warn}"


def render_sensor_bars(
    names: list[str],
    values: np.ndarray,
    vmax: float,
    phase: Optional[str] = None,
    elapsed: Optional[float] = None,
    noisy: Optional[list[bool]] = None,
) -> str:
    """Pure body for :class:`SensorBars`: a header line + one bar per sensor."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if noisy is None:
        noisy = [False] * len(names)

    header_bits = []
    if phase is not None:
        header_bits.append(f"phase: {phase}")
    if elapsed is not None:
        header_bits.append(f"t={elapsed:5.1f}s")
    header = "  ".join(header_bits) if header_bits else "sensors"

    rows = [header]
    for i, name in enumerate(names):
        v = float(values[i]) if i < len(values) else 0.0
        flag = bool(noisy[i]) if i < len(noisy) else False
        rows.append(bar_row(name, v, vmax, noisy=flag))
    return "\n".join(rows)


def render_workflow(
    connected: bool,
    counts: dict[str, int],
    has_model: bool,
) -> str:
    """Pure body for :class:`WorkflowPanel`: a small status checklist."""
    ok, no = "✓", "✗"
    lines = [f"{ok if connected else no} connected"]

    total = sum(counts.values())
    lines.append(f"{ok if total else no} captured ({total})")
    for label in sorted(counts):
        lines.append(f"    {label:<14} {counts[label]}")

    lines.append(f"model: {ok if has_model else no}")
    return "\n".join(lines)


try:  # textual is an optional extra; render helpers above import without it.
    from textual.widgets import RichLog, Static

    class SensorBars(Static):
        """Live per-sensor bar chart. ``vmax`` auto-scales to the running max."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._running_max = 1.0

        def update_values(
            self,
            names: list[str],
            values: np.ndarray,
            phase: Optional[str] = None,
            noisy: Optional[list[bool]] = None,
            elapsed: Optional[float] = None,
        ) -> None:
            arr = np.asarray(values, dtype=float).reshape(-1)
            peak = float(arr.max()) if arr.size else 0.0
            if peak > self._running_max:
                self._running_max = peak
            self.update(
                render_sensor_bars(
                    names,
                    arr,
                    self._running_max,
                    phase=phase,
                    elapsed=elapsed,
                    noisy=noisy,
                )
            )

    class WorkflowPanel(Static):
        """Connected / captured / model checklist."""

        def update_state(
            self,
            connected: bool,
            counts: dict[str, int],
            has_model: bool,
        ) -> None:
            self.update(render_workflow(connected, counts, has_model))

    class LogPanel(RichLog):
        """A scrolling event log."""

        def __init__(self, *args, max_lines: int = 200, **kwargs) -> None:
            kwargs.setdefault("max_lines", max_lines)
            kwargs.setdefault("wrap", True)
            super().__init__(*args, **kwargs)

        def write_line(self, msg: str) -> None:
            self.write(msg)

except ImportError:  # pragma: no cover - exercised only when textual is absent
    SensorBars = None  # type: ignore[assignment,misc]
    WorkflowPanel = None  # type: ignore[assignment,misc]

    class LogPanel:  # type: ignore[no-redef]
        """Textual-free fallback: keep the last ``max_lines`` messages in a deque."""

        def __init__(self, max_lines: int = 200) -> None:
            self.lines: deque[str] = deque(maxlen=max_lines)

        def write_line(self, msg: str) -> None:
            self.lines.append(msg)
