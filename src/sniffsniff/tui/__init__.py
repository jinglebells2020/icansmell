"""sniffsniff TUI package — orchestration layer + widgets (Milestone 3).

Exposes the non-UI :class:`SniffController`. Importing this package must not pull
in ``textual`` (the widget modules do), so the controller can run headlessly in
tests and scripts.
"""
from __future__ import annotations

from .controller import SniffController

__all__ = ["SniffController"]
