"""An animated ASCII/Unicode nose for the sniffsniff TUI.

The nose is drawn as multi-line art with two moods:

* **idle** — a gentle "breathing" cycle (the bridge lifts and settles) across a
  few frames.
* **sniffing** — a quicker cycle where little intake marks (``~ ≈ /``) are drawn
  in toward the nostrils, as if the nose is pulling air.

Every frame inside a mood has the **same number of lines and the same width**,
so the animation never jumps or reflows when it advances. :class:`NoseAnimation`
is pure (no textual dependency) and fully unit-testable; :class:`NoseWidget`
wraps it in a :class:`textual.widgets.Static` with a self-adjusting interval.
"""
from __future__ import annotations

# --- art ---------------------------------------------------------------------
#
# Each frame is a single multi-line string. Within a state list every frame has
# the same line count and every line is the same width (padded below), so the
# widget can swap frames in place without the layout shifting.

# Idle: a calm nose that gently "breathes" — the bridge rises a touch, the
# nostrils flare a hair, then it settles back. 4 frames, 7 lines each.
IDLE_FRAMES: list[str] = [
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        "  | (o) |  \n"
        "   \\ ˘ /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        "  | (o) |  \n"
        "   \\ _ /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        "  | (O) |  \n"
        "   \\ ‿ /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        "  | (o) |  \n"
        "   \\ _ /   \n"
        "    '-'    "
    ),
]

# Sniffing: same nose, but faint intake marks travel in from the sides toward
# the nostrils and the nostrils widen. 5 frames, 7 lines each, same width.
SNIFF_FRAMES: list[str] = [
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        "~ | (o) | ~\n"
        "   \\ _ /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        " ≈| (O) |≈ \n"
        "   \\ o /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  | . . |  \n"
        "  /(O O)\\  \n"
        "   \\ O /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |  .  |  \n"
        "  | (O) |  \n"
        "   \\ o /   \n"
        "    '-'    "
    ),
    (
        "    .-.    \n"
        "   /   \\   \n"
        "  |     |  \n"
        "  |     |  \n"
        " /| (o) |\\ \n"
        "   \\ _ /   \n"
        "    '-'    "
    ),
]

# Interval speeds (seconds) used by NoseWidget.
IDLE_INTERVAL = 0.4
SNIFF_INTERVAL = 0.15


class NoseAnimation:
    """Pure frame cursor over the idle / sniffing art (no textual dependency).

    The animation holds a ``state`` (``"idle"`` or ``"sniffing"``) and an
    ``index`` into the matching frame list. :meth:`advance` returns the current
    frame and steps the cursor forward, wrapping at the end of the list.
    """

    def __init__(self) -> None:
        self.state: str = "idle"
        self.index: int = 0

    def frames(self) -> list[str]:
        """Return the frame list for the current state."""
        return SNIFF_FRAMES if self.state == "sniffing" else IDLE_FRAMES

    def set_state(self, state: str) -> None:
        """Switch to ``state`` and reset the cursor to the first frame."""
        if state not in ("idle", "sniffing"):
            raise ValueError(f"unknown nose state: {state!r}")
        self.state = state
        self.index = 0

    def advance(self) -> str:
        """Return the current frame, then step (wrapping) to the next one."""
        frames = self.frames()
        frame = frames[self.index % len(frames)]
        self.index = (self.index + 1) % len(frames)
        return frame


try:  # textual is an optional extra; the pure logic above must import without it.
    from textual.widgets import Static

    class NoseWidget(Static):
        """A :class:`Static` that renders and animates a :class:`NoseAnimation`.

        On mount it starts a repeating interval that advances the animation and
        pushes the new frame. :meth:`set_state` forwards to the animation and
        re-tunes the interval so sniffing runs faster than idle.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.animation = NoseAnimation()
            self._timer = None

        def on_mount(self) -> None:
            self.update(self.animation.advance())
            self._timer = self.set_interval(IDLE_INTERVAL, self._tick)

        def _tick(self) -> None:
            self.update(self.animation.advance())

        def set_state(self, state: str) -> None:
            """Switch the nose mood and re-tune the animation speed."""
            self.animation.set_state(state)
            interval = SNIFF_INTERVAL if state == "sniffing" else IDLE_INTERVAL
            if self._timer is not None:
                self._timer.stop()
            self._timer = self.set_interval(interval, self._tick)
            self.update(self.animation.advance())

except ImportError:  # pragma: no cover - exercised only when textual is absent
    NoseWidget = None  # type: ignore[assignment,misc]
