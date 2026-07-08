"""Airflow-servo control over the serial link.

The Uno firmware moves the airflow servo when it receives an ``S<angle>\\n``
command. :class:`ServoLink` opens the serial port and sends those commands. Used
for manual calibration (find the fresh-air-open and sample-open angles with
``sniffsniff servo``) and, later, to switch airflow automatically during a
recording's phases.

Opening the port resets the Uno (DTR), so :meth:`ServoLink.open` waits out the
boot before the first command — same reason the reader uses a startup delay.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

__all__ = ["ServoLink"]


class ServoLink:
    """Send angle commands (0..180°) to the airflow servo over serial.

    ``opener`` is an optional zero-arg callable returning an object with
    ``write(bytes)``, ``flush()`` and ``close()`` — injected by tests so no real
    hardware is needed. The default opens a real ``pyserial.Serial``.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        *,
        startup_delay_s: float = 2.5,
        opener: Optional[Callable[[], object]] = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.startup_delay_s = float(startup_delay_s)
        self._opener = opener if opener is not None else self._default_opener
        self._serial: Optional[object] = None  # opened lazily

    def _default_opener(self):
        import serial  # local import: no import-time hardware dependency

        return serial.Serial(self.port, self.baud, timeout=1)

    def open(self) -> "ServoLink":
        """Open the port (once) and wait for the Uno to boot after its reset."""
        if self._serial is None:
            self._serial = self._opener()
            if self.startup_delay_s:
                time.sleep(self.startup_delay_s)
        return self

    def move(self, angle) -> int:
        """Move the servo to ``angle`` (clamped to 0..180). Returns the sent angle."""
        a = max(0, min(180, int(angle)))
        self.open()
        self._serial.write(f"S{a}\n".encode("ascii"))
        try:
            self._serial.flush()
        except Exception:  # pragma: no cover - flush is best-effort
            pass
        return a

    def close(self) -> None:
        """Close the port if open; safe to call anytime, idempotent."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # pragma: no cover
                pass
            self._serial = None

    def __enter__(self) -> "ServoLink":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
