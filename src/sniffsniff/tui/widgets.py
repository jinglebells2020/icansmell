"""Display panels for the sniffsniff TUI (modern-dashboard redesign).

Each panel is split into a **pure render helper** (a function that turns state
into a markup string, unit-testable without a running App) and a thin
:class:`textual.widgets.Static` wrapper that calls the helper and pushes the
result via ``self.update(...)``. Helpers embed Textual console markup
(``[green]…[/]``) so individual cells can be coloured; every visible token still
appears as plain text, so substring-based tests remain valid.

* :func:`render_header` / :class:`HeaderBar` — wordmark + source/model/phase pills.
* :func:`render_sensors` / :class:`SensorBars` — per-sensor sparkline + bar + trend.
* :func:`render_capture` / :class:`CapturePanel` — phase stepper + progress + detail.
* :func:`render_label_list` / :class:`LabelList` — dot-meter label counts.
* :func:`render_coach` / :class:`CoachPanel` — NEXT guidance + state line.
* :class:`LogPanel` — a scrolling event log.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

import numpy as np


def _finite(values: Sequence[float]) -> list[float]:
    """Coerce a sequence to finite floats (non-finite -> 0.0).

    Live ``Rs`` can be ``inf``/``nan`` for a dead or open channel (``counts_to_rs``
    divides by a near-zero ``V_RL``); the display must degrade to a flat 0, never
    crash on ``int(nan)``.
    """
    out = []
    for v in values:
        f = float(v)
        out.append(f if math.isfinite(f) else 0.0)
    return out

# Block characters for the bar fill / empty track and the sparkline ramp.
_FULL = "█"
_EMPTY = "░"
_WARN = "⚠"
_SPARK = " ▁▂▃▄▅▆▇█"  # index 0 == blank, 1..8 == rising eighths
_ACCENT = "#4ec9b0"  # teal accent used for progress fills (matches the nord-ish theme)

# Per-label target the dot-meter fills toward (mirrors controller.GOOD_REPS; kept as a
# module constant so widgets stay import-light and don't pull the controller graph).
DEFAULT_GOOD_REPS = 3

# Trend key -> (glyph, colour).
_TREND = {
    "rising": ("▲", "green"),
    "falling": ("▼", "red"),
    "flat": ("─", "grey50"),
}


def _esc(text: str) -> str:
    """Neutralise console-markup in untrusted text (e.g. a user-typed label).

    Only ``[`` needs escaping to stop a stray token opening a style tag; Textual
    already degrades gracefully on unknown tags, so this just keeps odd labels
    (``"gar[lic"``) from bleeding colour into the rest of the line.
    """
    return str(text).replace("[", r"\[")


def _fmt_value(value: float) -> str:
    """Compact human value: ``41200`` -> ``41.2k``, ``950`` -> ``950``."""
    v = float(value)
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def _bar(frac: float, width: int) -> str:
    """A ``width``-cell bar, ``frac`` (clamped to ``[0, 1]``) of it filled."""
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    filled = int(round(frac * width))
    if filled > width:
        filled = width
    return _FULL * filled + _EMPTY * (width - filled)


def _gauge(frac: float, width: int) -> str:
    """A colour-marked bar: accent-teal fill on a dim track (for the sensor rows)."""
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    filled = int(round(frac * width))
    if filled > width:
        filled = width
    return f"[{_ACCENT}]{_FULL * filled}[/][grey35]{_EMPTY * (width - filled)}[/]"


def sparkline(values: Sequence[float], width: Optional[int] = None) -> str:
    """Render ``values`` as a unicode sparkline, normalised over the shown window.

    The last ``width`` samples (all of them if ``width`` is None) are mapped onto
    the eight rising block glyphs by their position between the window's own min
    and max, so small live wiggles are visible regardless of absolute magnitude.
    A flat or single-sample window renders as a low, even line (steady signal).
    """
    vals = _finite(values)
    if width is not None and len(vals) > width:
        vals = vals[-width:]
    if not vals:
        return ""
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    ramp = _SPARK[1:]  # drop the blank cell — a steady signal should still be visible
    if span <= 0:
        return ramp[0] * len(vals)
    top = len(ramp) - 1
    out = []
    for v in vals:
        idx = int((v - lo) / span * top + 0.5)
        idx = 0 if idx < 0 else (top if idx > top else idx)
        out.append(ramp[idx])
    return "".join(out)


def trend(values: Sequence[float], deadband: float = 0.0) -> str:
    """Classify the recent slope of ``values`` as ``rising`` / ``falling`` / ``flat``.

    Compares the mean of the most recent third of the window against the third
    before it; a change smaller than ``deadband`` (in signal units) reads as flat.
    Too-short windows are flat.
    """
    vals = _finite(values)
    n = len(vals)
    if n < 4:
        return "flat"
    half = max(1, n // 3)
    recent = sum(vals[-half:]) / half
    older = sum(vals[-2 * half : -half]) / half
    delta = recent - older
    if delta > deadband:
        return "rising"
    if delta < -deadband:
        return "falling"
    return "flat"


def bar_row(
    name: str,
    value: float,
    vmax: float,
    width: int = 14,
    noisy: bool = False,
) -> str:
    """Render one plain sensor row: ``"MQ3   ██████░░░░  41.2k [⚠]"``.

    A tested rendering primitive kept stable across the redesign: the bar is
    ``width`` cells, ``value / vmax`` (clamped) of them filled, and its overall
    length does not depend on ``value``. :func:`render_sensors` uses the richer
    :func:`_sensor_row` instead.
    """
    frac = 0.0 if vmax <= 0 else value / vmax
    bar = _bar(frac, width)
    warn = f" [{_WARN}]" if noisy else ""
    return f"{name:<6} {bar}  {_fmt_value(value):>6}{warn}"


def _sensor_row(
    name: str,
    spark: str,
    bar: str,
    value: float,
    trend_key: str = "flat",
    noisy: bool = False,
) -> str:
    """One live sensor row with inline colour: name · sparkline · bar · value · trend."""
    glyph, colour = _TREND.get(trend_key, _TREND["flat"])
    warn = f" [yellow]{_WARN}[/]" if noisy else "  "
    return (
        f"[b]{name:<5}[/] "
        f"[cyan]{spark:<14}[/] "
        f"{bar} "
        f"{_fmt_value(value):>6} "
        f"[{colour}]{glyph}[/]{warn}"
    )


def render_sensors(
    names: list[str],
    values: np.ndarray,
    histories: Sequence[Sequence[float]],
    vmax: float,
    noisy: Optional[list[bool]] = None,
) -> str:
    """Pure body for :class:`SensorBars`: one live row per sensor (no header line).

    ``histories[i]`` is the rolling sample window for channel ``i`` (drives its
    sparkline and trend); ``values`` are the current magnitudes (drive the bar).
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if noisy is None:
        noisy = [False] * len(names)
    deadband = 0.01 * vmax if vmax > 0 else 0.0
    rows = []
    for i, name in enumerate(names):
        v = float(values[i]) if i < values.size else 0.0
        hist = histories[i] if i < len(histories) else []
        spark = sparkline(hist, width=14)
        frac = 0.0 if vmax <= 0 else v / vmax
        bar = _gauge(frac, 10)
        tkey = trend(hist, deadband)
        flag = bool(noisy[i]) if i < len(noisy) else False
        rows.append(_sensor_row(name, spark, bar, v, tkey, flag))
    return "\n".join(rows)


def render_header(
    source: str,
    connected: bool,
    has_model: bool,
    phase_label: str,
) -> str:
    """Pure body for :class:`HeaderBar`: wordmark + source / model / phase pills."""
    src = "sim" if source == "sim" else "live"
    src_colour = "yellow" if src == "sim" else ("green" if connected else "red")
    model_txt = "model ✓" if has_model else "no model"
    model_colour = "green" if has_model else "grey50"
    phase_colour = {
        "monitoring": "cyan",
        "settling": "yellow",
        "recording": "magenta",
        "recovering": "blue",
        "idle": "grey50",
    }.get(phase_label, "cyan")
    brand = "[b]sniffsniff[/][grey50] · e-nose[/]"
    pills = (
        f"[{src_colour}]● {src}[/]    "
        f"[{model_colour}]{model_txt}[/]    "
        f"[{phase_colour}]◉ {phase_label}[/]"
    )
    return f"{brand}     {pills}"


# Capture lifecycle steps shown in the stepper (settle precedes the recorded window).
_STEPS = ["settle", "baseline", "exposure", "purge"]


def render_capture(
    phase: str,
    frac: Optional[float],
    detail: str,
) -> str:
    """Pure body for :class:`CapturePanel`: phase stepper + progress bar + detail.

    ``phase`` is one of the :data:`_STEPS`, ``"recover"``, or ``"monitor"`` (idle).
    ``frac`` (0..1) draws a progress bar when known; ``detail`` is the status line.
    """
    if phase in _STEPS:
        active = _STEPS.index(phase)
        cells = []
        for i, step in enumerate(_STEPS):
            if i < active:
                cells.append(f"[green]✓ {step}[/]")
            elif i == active:
                cells.append(f"[b reverse] {step} [/]")
            else:
                cells.append(f"[grey50]○ {step}[/]")
        stepper = " [grey50]→[/] ".join(cells)
    elif phase == "recover":
        stepper = "[blue]↺ recovering[/][grey50] — sensors returning to baseline[/]"
    else:  # monitor / idle
        stepper = (
            "[grey50]◇ idle — press[/] [b]r[/] [grey50]to record ·[/] "
            "[b]i[/] [grey50]to identify[/]"
        )

    lines = [stepper]
    if frac is not None:
        pct = int(round(max(0.0, min(1.0, frac)) * 100))
        lines.append(f"[{_ACCENT}]{_bar(frac, 30)}[/] [b]{pct:>3}%[/]")
    if detail:
        lines.append(detail)
    return "\n".join(lines)


def render_label_list(
    labels: list[str],
    counts: dict[str, int],
    current: Optional[str] = None,
    good_reps: int = DEFAULT_GOOD_REPS,
) -> str:
    """Pure body for :class:`LabelList`: one dot-meter row per label, current marked.

    ``"▸ coffee   ●●● 3 ✓"`` for a current, fully-collected label; ``"  vinegar  ●●○ 2"``
    otherwise. The meter fills ``min(count, good_reps)`` dots toward ``good_reps``.
    """
    rows = []
    for label in labels:
        n = counts.get(label, 0)
        filled = min(max(n, 0), good_reps)
        dots = (
            "[green]" + "●" * filled + "[/]"
            + "[grey50]" + "○" * max(0, good_reps - filled) + "[/]"
        )
        text = f"{_esc(label):<12}"
        if label == current:
            mark = "[b cyan]▸[/]"
            name = f"[b]{text}[/]"
        else:
            mark = " "
            name = f"[grey62]{text}[/]"
        done = " [green]✓[/]" if n >= good_reps and n > 0 else ""
        rows.append(f"{mark} {name} {dots} [b]{n:>2}[/]{done}")
    return "\n".join(rows)


def render_coach(
    next_step: str,
    connected: bool,
    label: str,
    reps: int,
    classifier: str,
    has_model: bool,
) -> str:
    """Pure body for :class:`CoachPanel`: a NEXT guidance line + a dim state line."""
    src = "[green]connected[/]" if connected else "[red]offline[/]"
    model = "[green]model ✓[/]" if has_model else "[grey50]model ✗[/]"
    dot = " [grey50]·[/] "
    state = dot.join(
        [src, f"label [b]{_esc(label)}[/]", f"reps [b]{reps}[/]",
         f"clf [b]{classifier}[/]", model]
    )
    return f"[b]NEXT[/]  {next_step}\n\n[grey62]{state}[/]"


try:  # textual is an optional extra; render helpers above import without it.
    from textual.widgets import RichLog, Static

    class HeaderBar(Static):
        """Wordmark + live source / model / phase pills."""

        def update_header(
            self,
            source: str,
            connected: bool,
            has_model: bool,
            phase_label: str,
        ) -> None:
            self.update(render_header(source, connected, has_model, phase_label))

    class SensorBars(Static):
        """Live per-sensor sparkline + bar + trend. ``vmax`` auto-scales to the max."""

        def __init__(self, *args, history: int = 24, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._running_max = 1.0
            self._hist_len = history
            self._history: Optional[list[deque]] = None

        def update_values(
            self,
            names: list[str],
            values: np.ndarray,
            phase: Optional[str] = None,  # accepted for call-site compatibility
            noisy: Optional[list[bool]] = None,
            elapsed: Optional[float] = None,
        ) -> None:
            arr = np.asarray(values, dtype=float).reshape(-1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if self._history is None or len(self._history) != len(names):
                self._history = [deque(maxlen=self._hist_len) for _ in names]
            for i in range(len(names)):
                self._history[i].append(float(arr[i]) if i < arr.size else 0.0)
            peak = float(arr.max()) if arr.size else 0.0
            if peak > self._running_max:
                self._running_max = peak
            hists = [list(d) for d in self._history]
            self.update(render_sensors(names, arr, hists, self._running_max, noisy))

    class CapturePanel(Static):
        """Phase stepper + progress bar + a status detail line."""

        def update_capture(
            self,
            phase: str,
            frac: Optional[float],
            detail: str,
        ) -> None:
            self.update(render_capture(phase, frac, detail))

    class LabelList(Static):
        """One dot-meter row per known label; the current label marked."""

        def update_labels(
            self,
            labels: list[str],
            counts: dict[str, int],
            current: Optional[str] = None,
            good_reps: int = DEFAULT_GOOD_REPS,
        ) -> None:
            self.update(render_label_list(labels, counts, current, good_reps))

    class CoachPanel(Static):
        """A NEXT guidance line + a dim state line."""

        def update_coach(
            self,
            next_step: str,
            connected: bool,
            label: str,
            reps: int,
            classifier: str,
            has_model: bool,
        ) -> None:
            self.update(
                render_coach(
                    next_step, connected, label, reps, classifier, has_model
                )
            )

    class LogPanel(RichLog):
        """A scrolling event log."""

        # Never take keyboard focus: the arrow keys belong to label navigation,
        # and a focused RichLog would swallow them to scroll itself instead.
        can_focus = False

        def __init__(self, *args, max_lines: int = 200, **kwargs) -> None:
            kwargs.setdefault("max_lines", max_lines)
            kwargs.setdefault("wrap", True)
            # markup stays OFF: log lines carry verbatim content (paths, LLM
            # narratives) that must not be reinterpreted as console markup.
            super().__init__(*args, **kwargs)

        def write_line(self, msg: str) -> None:
            self.write(msg)

except ImportError:  # pragma: no cover - exercised only when textual is absent
    HeaderBar = None  # type: ignore[assignment,misc]
    SensorBars = None  # type: ignore[assignment,misc]
    CapturePanel = None  # type: ignore[assignment,misc]
    LabelList = None  # type: ignore[assignment,misc]
    CoachPanel = None  # type: ignore[assignment,misc]

    class LogPanel:  # type: ignore[no-redef]
        """Textual-free fallback: keep the last ``max_lines`` messages in a deque."""

        def __init__(self, max_lines: int = 200) -> None:
            self.lines: deque[str] = deque(maxlen=max_lines)

        def write_line(self, msg: str) -> None:
            self.lines.append(msg)
