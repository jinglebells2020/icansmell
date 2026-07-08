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

import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np

__all__ = ["parse_line", "SerialReader", "MergedReader", "merge_counts", "build_reader"]

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


def merge_counts(
    latest: list[Optional[np.ndarray]], widths: list[int]
) -> np.ndarray:
    """Concatenate per-board latest counts into one flat frame, in board order.

    A board that has not produced a frame yet (``None``) contributes ``widths[i]``
    zeros, so the combined frame always has the full width. Pure and side-effect
    free (the unit under test for the merge itself).
    """
    parts: list[np.ndarray] = []
    for counts, w in zip(latest, widths):
        if counts is None:
            parts.append(np.zeros(w, dtype=np.int64))
        else:
            parts.append(np.asarray(counts, dtype=np.int64).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


class MergedReader:
    """Merge several per-board readers into one flat frame stream (equal peers).

    Each board reader is pumped by its own daemon thread that keeps only that
    board's *latest* counts. A host-clocked emit loop yields the concatenated
    frame at ``scan_hz`` — the two Unos have unrelated ``millis()`` epochs, so the
    combined ``t_ms`` is generated here (``k * step_ms``), not taken from a board.

    The first frame waits until *every* board has produced at least one frame
    (bounded by ``start_timeout_s``); after that a stalled board simply holds its
    last values. Presents the same ``frames()`` / ``write_command()`` / ``close()``
    surface as :class:`SerialReader`, so the engine and controller are unchanged.
    """

    def __init__(
        self,
        readers: list,
        widths: list[int],
        *,
        servo_index: int = 0,
        scan_hz: int = 20,
        start_timeout_s: float = 10.0,
    ) -> None:
        if len(readers) != len(widths):
            raise ValueError("readers and widths must have equal length")
        self._readers = list(readers)
        self._widths = list(widths)
        self._servo_index = servo_index
        self._scan_hz = max(1, int(scan_hz))
        self._start_timeout_s = float(start_timeout_s)
        self._latest: list[Optional[np.ndarray]] = [None] * len(readers)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _pump(self, i: int) -> None:
        """Feed board ``i``'s latest counts from its reader until stopped."""
        try:
            for frame in self._readers[i].frames():
                if self._stop.is_set():
                    return
                with self._lock:
                    self._latest[i] = np.asarray(frame[1], dtype=np.int64)
        except Exception:  # pragma: no cover - a dead board holds its last value
            return

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield merged ``(t_ms, raw[N])`` frames at ``scan_hz`` (host-clocked)."""
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._pump, args=(i,), daemon=True)
            for i in range(len(self._readers))
        ]
        for t in self._threads:
            t.start()

        step = 1.0 / self._scan_hz
        # Wait for every board's first frame (bounded), else report no data.
        waited = 0.0
        poll = min(step, 0.05)
        while not self._stop.is_set():
            with self._lock:
                ready = all(c is not None for c in self._latest)
            if ready:
                break
            time.sleep(poll)
            waited += poll
            if waited >= self._start_timeout_s:
                self.close()
                raise RuntimeError(
                    "no data from one or more boards within the startup timeout"
                )

        step_ms = round(1000 / self._scan_hz)
        k = 0
        while not self._stop.is_set():
            with self._lock:
                counts = merge_counts(self._latest, self._widths)
            yield (k * step_ms, counts)
            k += 1
            time.sleep(step)

    def write_command(self, text: str) -> bool:
        """Route a host→device command (e.g. servo ``"S105"``) to the servo board."""
        reader = self._readers[self._servo_index]
        write = getattr(reader, "write_command", None)
        return bool(write(text)) if write is not None else False

    def close(self) -> None:
        """Stop the pump threads and close every board reader."""
        self._stop.set()
        for reader in self._readers:
            try:
                reader.close()
            except Exception:  # pragma: no cover - close best-effort
                pass


def build_reader(
    config,
    *,
    port: Optional[str] = None,
    reconnect: bool = True,
    startup_delay_s: float = 0.0,
    openers: Optional[list] = None,
):
    """Build the right reader for ``config``: a :class:`SerialReader` for a single
    board, or a :class:`MergedReader` for a multi-board rig.

    ``port`` supplies the device for any board whose config ``port`` is ``None``
    (legacy single-board rigs set the port on the CLI). ``openers`` (one per board)
    injects fake byte streams in tests.
    """
    boards = config.boards
    if len(boards) <= 1:
        board = boards[0] if boards else None
        effective = (board.port if (board and board.port) else None) or port
        opener = openers[0] if openers else None
        return SerialReader(
            effective,
            n_channels=config.n_channels,
            reconnect=reconnect,
            startup_delay_s=startup_delay_s,
            opener=opener,
        )

    readers = []
    for i, board in enumerate(boards):
        opener = openers[i] if openers else None
        readers.append(
            SerialReader(
                board.port or port,
                n_channels=board.n_channels,
                reconnect=reconnect,
                startup_delay_s=startup_delay_s,
                opener=opener,
            )
        )
    servo_index = next((i for i, b in enumerate(boards) if b.servo), 0)
    widths = [b.n_channels for b in boards]
    return MergedReader(
        readers, widths, servo_index=servo_index, scan_hz=config.scan_hz
    )
