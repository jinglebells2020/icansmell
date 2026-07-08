"""Serial ingest for the sniffsniff e-nose: CSV line parsing + a robust reader.

The device streams one CSV line per full-array scan::

    <millis>,<c0>,<c1>,...,<c(N-1)>\\n

``parse_line`` turns such a line into a ``(t_ms, raw)`` frame, where ``raw`` is
an ``int64`` array of ADC counts in channel order. It is channel-count-
agnostic: the number of channels is a parameter, and the default of ``6`` is only
this build's array size — nothing here hard-codes 6 structurally.

``SerialReader`` wraps a byte-stream source (a real ``pyserial.Serial`` by
default) behind an injectable ``opener`` seam so it can be driven with a fake in
tests. The port is opened *lazily* in :meth:`SerialReader.frames`, never in
``__init__``, so constructing a reader has no hardware side effects. On a serial
error mid-stream it reconnects with exponential backoff (when ``reconnect`` is
set); malformed and short lines are skipped, never fatal.

``SimulatedReader`` (in ``simulator.py``) presents the identical ``frames()`` /
``close()`` interface, so the simulator is a drop-in for the real port.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

import numpy as np

__all__ = ["parse_line", "SerialReader"]

# Backoff schedule (seconds) for reconnect attempts: doubles each failure up to
# a cap, so a flapping device is retried promptly at first, then patiently.
_BACKOFF_INITIAL_S: float = 0.1
_BACKOFF_MAX_S: float = 5.0


def parse_line(line: str, n_channels: int = 6) -> Optional[tuple[int, np.ndarray]]:
    """Parse one CSV scan line into a ``(t_ms, raw)`` frame, or ``None``.

    The line must comma-split (after stripping) into exactly ``n_channels + 1``
    fields, each a base-10 integer: a ``millis`` timestamp followed by
    ``n_channels`` ADC counts. Any deviation — wrong field count, a non-integer
    field, an empty/blank line — yields ``None`` so the caller can skip it
    without crashing. ``raw`` is ``int64`` in channel order 0..N-1.

    Note: values are not range-checked against 0..1023; a hardware channel that
    rails is still a valid integer here and is flagged downstream, not dropped.
    """
    if line is None:
        return None
    stripped = line.strip()
    if not stripped:
        return None

    fields = stripped.split(",")
    if len(fields) != n_channels + 1:
        return None

    values = np.empty(n_channels + 1, dtype=np.int64)
    for i, field in enumerate(fields):
        token = field.strip()
        try:
            values[i] = int(token)
        except ValueError:
            return None

    t_ms = int(values[0])
    raw = values[1:].astype(np.int64, copy=True)
    return t_ms, raw


def _default_opener(port: str, baud: int) -> Callable[[], object]:
    """Build the default zero-arg opener that lazily constructs a real Serial.

    ``pyserial`` is imported inside the returned callable so that importing this
    module — and constructing a ``SerialReader`` — never requires pyserial or a
    real port. Only actually iterating ``frames()`` on a real device touches it.
    """

    def opener() -> object:
        import serial  # local import: no import-time hardware dependency

        return serial.Serial(port, baud, timeout=1)

    return opener


class SerialReader:
    """Reads ``(t_ms, raw)`` frames from a line-oriented byte stream.

    The stream source is produced on demand by ``opener`` — a zero-arg callable
    returning an object with ``readline() -> bytes`` and ``close()``. This is the
    seam that lets tests inject a fake and lets the real path stay lazy: nothing
    is opened until :meth:`frames` is first iterated.

    Parameters
    ----------
    port, baud:
        Passed to the default pyserial opener. Ignored when a custom ``opener``
        is supplied (the caller owns connection details in that case).
    n_channels:
        Expected ADC channel count; frames with a different field count are
        skipped. Not hard-coded to 6.
    reconnect:
        When true, a serial error or end-of-stream triggers a backoff + reopen;
        when false, end-of-stream ends iteration and errors propagate.
    opener:
        Optional injected factory (see above). Defaults to a real Serial opener.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        n_channels: int = 6,
        reconnect: bool = True,
        opener: Optional[Callable[[], object]] = None,
        startup_delay_s: float = 0.0,
    ) -> None:
        self.port = port
        self.baud = baud
        self.n_channels = n_channels
        self.reconnect = reconnect
        self.startup_delay_s = float(startup_delay_s)
        self._opener = opener if opener is not None else _default_opener(port, baud)
        self._serial: Optional[object] = None  # opened lazily in frames()

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield parsed frames forever (or until the stream ends).

        Opens the port lazily on first iteration. Decodes each line, skips any
        that fail to parse, and yields ``(t_ms, raw)`` for the rest. An empty
        ``readline()`` (``b""``) marks end-of-stream. On a serial error, or on
        end-of-stream when ``reconnect`` is set, the current handle is closed and
        a fresh one is opened after an exponential backoff; without ``reconnect``
        the iterator simply stops (EOF) or the error propagates.
        """
        backoff = _BACKOFF_INITIAL_S
        while True:
            if self._serial is None:
                self._serial = self._opener()
                # Opening the port toggles DTR, which RESETS an Arduino; it then
                # needs ~1-2 s to boot before it streams. Wait it out so the first
                # reads don't hit the silent boot gap (and give up).
                if self.startup_delay_s:
                    time.sleep(self.startup_delay_s)
            try:
                for line in self._iter_lines(self._serial):
                    frame = parse_line(line, self.n_channels)
                    if frame is not None:
                        yield frame
                        backoff = _BACKOFF_INITIAL_S  # progress: reset backoff
            except (OSError, ValueError) as exc:  # serial/read error
                self._close_serial()
                if not self.reconnect:
                    raise exc
                time.sleep(backoff)
                backoff = min(backoff * 2.0, _BACKOFF_MAX_S)
                continue
            # Reached end-of-stream (readline returned b"").
            self._close_serial()
            if not self.reconnect:
                return
            time.sleep(backoff)
            backoff = min(backoff * 2.0, _BACKOFF_MAX_S)

    def _iter_lines(self, ser: object) -> Iterator[str]:
        """Yield decoded lines from ``ser`` until ``readline()`` returns empty."""
        while True:
            data = ser.readline()
            if not data:  # b"" -> end of stream / timeout with nothing pending
                return
            if isinstance(data, bytes):
                yield data.decode("ascii", errors="replace")
            else:
                yield str(data)

    def _close_serial(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # pragma: no cover - close() best-effort
                pass
            self._serial = None

    def write_command(self, text: str) -> bool:
        """Write a command line to the open port (host→device, e.g. ``"S105"``).

        Best-effort and non-fatal: appends a newline if missing, returns True on a
        successful write, False if the port isn't open yet or the write hiccups —
        so a servo command never crashes the read loop. Call from the same thread
        that iterates :meth:`frames` (the two share one serial handle).
        """
        ser = self._serial
        if ser is None:
            return False
        line = text if text.endswith("\n") else text + "\n"
        try:
            ser.write(line.encode("ascii"))
            try:
                ser.flush()
            except Exception:  # pragma: no cover - flush best-effort
                pass
            return True
        except Exception:  # pragma: no cover - write hiccup, don't crash the loop
            return False

    def close(self) -> None:
        """Close the underlying stream if one was opened; safe to call anytime."""
        self._close_serial()
