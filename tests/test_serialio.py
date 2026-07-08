"""Tests for sniffsniff.serialio — CSV frame parsing + hardware-free SerialReader.

The serial layer is channel-count-agnostic: ``parse_line`` derives the expected
field count from ``n_channels`` and never hard-codes 6 (the default is 6 only
because that is this build's array size). ``SerialReader`` opens its port lazily
through an injectable ``opener`` seam, so these tests exercise the full frame
loop with a ``FakeSerial`` and touch no real hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from sniffsniff.serialio import SerialReader, parse_line


class FakeSerial:
    """Minimal stand-in for pyserial's Serial: hands out queued lines.

    ``readline()`` returns each queued bytes line in turn, then ``b""`` forever
    (pyserial's read-timeout convention for "no data"). Records whether it was
    closed so tests can assert lifecycle behaviour.
    """

    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False
        self.readline_calls = 0

    def readline(self) -> bytes:
        self.readline_calls += 1
        if self._lines:
            return self._lines.pop(0)
        return b""

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# parse_line
# --------------------------------------------------------------------------- #

def test_parse_line_valid_default_six_channels():
    result = parse_line("12,1,2,3,4,5,6")
    assert result is not None
    t_ms, raw = result
    assert t_ms == 12
    assert isinstance(t_ms, int)
    np.testing.assert_array_equal(raw, np.array([1, 2, 3, 4, 5, 6], dtype=np.int64))
    assert raw.dtype == np.int64
    assert raw.shape == (6,)


def test_parse_line_strips_whitespace_and_trailing_newline():
    result = parse_line("  100 , 10, 20 ,30,40, 50 ,60 \r\n")
    assert result is not None
    t_ms, raw = result
    assert t_ms == 100
    np.testing.assert_array_equal(raw, np.array([10, 20, 30, 40, 50, 60], dtype=np.int64))


def test_parse_line_negative_and_zero_values_are_valid_integers():
    # parse_line does not clamp to the 0..1023 ADC range; it only requires ints.
    result = parse_line("0,0,0,0,0,0,0")
    assert result is not None
    t_ms, raw = result
    assert t_ms == 0
    np.testing.assert_array_equal(raw, np.zeros(6, dtype=np.int64))


def test_parse_line_wrong_field_count_too_few():
    # 6 fields total => only 5 channels for the default n_channels=6 -> None
    assert parse_line("12,1,2,3,4,5") is None


def test_parse_line_wrong_field_count_too_many():
    # 8 fields total => 7 channels for the default n_channels=6 -> None
    assert parse_line("12,1,2,3,4,5,6,7") is None


def test_parse_line_non_integer_field():
    assert parse_line("12,1,2,x,4,5,6") is None


def test_parse_line_float_field_rejected():
    # A float string is not an integer field.
    assert parse_line("12,1,2,3.5,4,5,6") is None


def test_parse_line_empty_string():
    assert parse_line("") is None


def test_parse_line_blank_string():
    assert parse_line("   \r\n") is None


def test_parse_line_channel_count_agnostic_three_channels():
    # n_channels=3 => require exactly 4 integer fields.
    result = parse_line("7,100,200,300", n_channels=3)
    assert result is not None
    t_ms, raw = result
    assert t_ms == 7
    assert raw.shape == (3,)
    np.testing.assert_array_equal(raw, np.array([100, 200, 300], dtype=np.int64))
    # The default-6 field count must now be rejected.
    assert parse_line("12,1,2,3,4,5,6", n_channels=3) is None


def test_parse_line_channel_count_agnostic_eight_channels():
    line = "5," + ",".join(str(i) for i in range(8))
    result = parse_line(line, n_channels=8)
    assert result is not None
    t_ms, raw = result
    assert t_ms == 5
    assert raw.shape == (8,)
    np.testing.assert_array_equal(raw, np.arange(8, dtype=np.int64))


# --------------------------------------------------------------------------- #
# SerialReader (hardware-free via injected opener)
# --------------------------------------------------------------------------- #

def test_serial_reader_does_not_open_in_init():
    opened = {"count": 0}

    def opener():
        opened["count"] += 1
        return FakeSerial([b"1,1,2,3,4,5,6\n"])

    reader = SerialReader("PORTX", opener=opener)
    # Construction must be side-effect-free: the port opens lazily in frames().
    assert opened["count"] == 0
    reader.close()


def test_serial_reader_yields_parsed_frames_and_skips_malformed():
    lines = [
        b"1,10,20,30,40,50,60\n",
        b"garbage line\n",          # non-integer -> skipped
        b"2,11,21,31,41,51,61\n",
        b"3,1,2,3,4,5\n",           # wrong field count -> skipped
        b"\n",                       # blank -> skipped
        b"4,12,22,32,42,52,62\n",
    ]
    fake = FakeSerial(lines)
    reader = SerialReader("PORTX", reconnect=False, opener=lambda: fake)

    frames = list(reader.frames())
    reader.close()

    t_vals = [t for t, _ in frames]
    assert t_vals == [1, 2, 4]
    np.testing.assert_array_equal(frames[0][1], np.array([10, 20, 30, 40, 50, 60], dtype=np.int64))
    np.testing.assert_array_equal(frames[1][1], np.array([11, 21, 31, 41, 51, 61], dtype=np.int64))
    np.testing.assert_array_equal(frames[2][1], np.array([12, 22, 32, 42, 52, 62], dtype=np.int64))
    for _, raw in frames:
        assert raw.dtype == np.int64


def test_serial_reader_stops_on_empty_readline_without_reconnect():
    # b"" signals end-of-stream; with reconnect=False frames() must terminate.
    fake = FakeSerial([b"1,1,2,3,4,5,6\n"])
    reader = SerialReader("PORTX", reconnect=False, opener=lambda: fake)
    frames = list(reader.frames())
    assert len(frames) == 1
    reader.close()


def test_serial_reader_honours_n_channels():
    fake = FakeSerial([b"9,1,2,3\n", b"10,4,5,6,7,8,9\n"])  # second line has 6 chans
    reader = SerialReader("PORTX", n_channels=3, reconnect=False, opener=lambda: fake)
    frames = list(reader.frames())
    reader.close()
    assert len(frames) == 1
    t_ms, raw = frames[0]
    assert t_ms == 9
    assert raw.shape == (3,)
    np.testing.assert_array_equal(raw, np.array([1, 2, 3], dtype=np.int64))


def test_serial_reader_close_closes_underlying_serial():
    fake = FakeSerial([b"1,1,2,3,4,5,6\n"])
    reader = SerialReader("PORTX", reconnect=False, opener=lambda: fake)
    list(reader.frames())
    reader.close()
    assert fake.closed is True


def test_serial_reader_reconnect_backoff_on_serial_error(monkeypatch):
    """A serial error mid-stream triggers reconnect with backoff, then recovers.

    First opened device raises on its second readline; the reader should sleep
    (backoff) and open a fresh device that yields the remaining frame.
    """
    sleeps = []
    monkeypatch.setattr("sniffsniff.serialio.time.sleep", lambda s: sleeps.append(s))

    class FlakySerial:
        def __init__(self, lines, raise_after):
            self._lines = list(lines)
            self._raise_after = raise_after
            self._reads = 0
            self.closed = False

        def readline(self):
            self._reads += 1
            if self._reads > self._raise_after:
                raise OSError("device disconnected")
            if self._lines:
                return self._lines.pop(0)
            return b""

        def close(self):
            self.closed = True

    # Device 0 yields one frame then raises (disconnect). Device 1 yields the
    # recovered frame. Any *third* open means the reader is looping past the
    # data we care about; a sentinel opener proves the recovery frame arrived
    # via a genuine reconnect and lets us stop the (otherwise infinite,
    # reconnecting) generator deterministically.
    devices = [
        FlakySerial([b"1,1,2,3,4,5,6\n"], raise_after=1),  # yields 1 frame, then errors
        FlakySerial([b"2,7,8,9,10,11,12\n"], raise_after=10),  # yields 1 frame, then EOF
    ]
    made = []

    class _StopSignal(Exception):
        pass

    def opener():
        idx = len(made)
        if idx >= len(devices):
            raise _StopSignal()
        dev = devices[idx]
        made.append(dev)
        return dev

    reader = SerialReader("PORTX", reconnect=True, opener=opener)

    gen = reader.frames()
    first = next(gen)
    second = next(gen)
    # Draining further makes the reader reconnect on device-1 EOF and hit the
    # sentinel opener, which surfaces as our _StopSignal.
    with pytest.raises(_StopSignal):
        next(gen)
    reader.close()

    assert first[0] == 1
    assert second[0] == 2
    # Reconnect happened: two real devices were opened and backoff slept.
    assert len(made) == 2
    assert len(sleeps) >= 1
    assert devices[0].closed is True


def test_serial_reader_close_before_frames_is_safe():
    reader = SerialReader("PORTX", opener=lambda: FakeSerial([]))
    # No frames() call yet -> nothing opened; close() must not raise.
    reader.close()


def test_serial_reader_reconnects_after_eof_when_enabled(monkeypatch):
    """With reconnect=True, a clean EOF (b"") reopens the port and continues."""
    import itertools
    from sniffsniff import serialio

    monkeypatch.setattr(serialio.time, "sleep", lambda _s: None)  # no real backoff wait

    serials = [
        FakeSerial([b"1,1,2,3,4,5,6\n"]),
        FakeSerial([b"2,7,8,9,10,11,12\n"]),
        FakeSerial([b"3,13,14,15,16,17,18\n"]),
    ]
    opened = {"n": 0}

    def opener():
        s = serials[opened["n"]]
        opened["n"] += 1
        return s

    reader = SerialReader("ignored", n_channels=6, reconnect=True, opener=opener)
    frames = list(itertools.islice(reader.frames(), 3))
    reader.close()

    assert [t for t, _ in frames] == [1, 2, 3]      # frames span three reconnects
    assert opened["n"] == 3                          # reopened after each EOF
    assert serials[0].closed and serials[1].closed   # stale handles were closed


def test_serial_reader_startup_delay_waits_after_open(monkeypatch):
    """Opening resets the Uno; the reader must wait startup_delay_s before reading
    so the first reads don't land in the silent boot gap."""
    from sniffsniff import serialio

    slept = []
    monkeypatch.setattr(serialio.time, "sleep", lambda s: slept.append(s))
    reader = SerialReader(
        "x", n_channels=6, reconnect=False, startup_delay_s=2.5,
        opener=lambda: FakeSerial([b"1,1,2,3,4,5,6\n"]),
    )
    frames = list(reader.frames())
    assert 2.5 in slept                       # waited for the board to boot
    assert frames and frames[0][0] == 1       # then read normally


class _WSerial:
    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


def test_write_command_before_open_returns_false():
    r = SerialReader("x", opener=lambda: _WSerial())
    assert r.write_command("S90") is False  # port not opened yet


def test_write_command_writes_when_open():
    fake = _WSerial()
    r = SerialReader("x", opener=lambda: fake)
    r._serial = fake  # simulate an open handle (as during frames())
    assert r.write_command("S105") is True
    assert r.write_command("S0\n") is True  # already-newline'd not doubled
    assert fake.written == b"S105\nS0\n"


# --------------------------------------------------------------------------- #
# merge_counts + MergedReader + build_reader (dual-Uno ingest)
# --------------------------------------------------------------------------- #

import time  # noqa: E402

from sniffsniff.config import Board, Channel, Config, default_config, load_config  # noqa: E402
from sniffsniff.serialio import MergedReader, build_reader, merge_counts  # noqa: E402


class _FakeBoard:
    """A board reader stand-in: yields queued frames (optionally forever)."""

    def __init__(self, frames, loop=True, delay=0.001):
        self._frames = list(frames)
        self._loop = loop
        self._delay = delay
        self.cmds = []
        self.closed = False

    def frames(self):
        while True:
            for f in self._frames:
                yield f
                time.sleep(self._delay)
            if not self._loop:
                return

    def write_command(self, text):
        self.cmds.append(text)
        return True

    def close(self):
        self.closed = True


def _collect(reader, n, cap=500):
    out = []
    for f in reader.frames():
        out.append(f)
        if len(out) >= n:
            break
        if len(out) >= cap:
            break
    reader.close()
    return out


# --- merge_counts (pure) -----------------------------------------------------

def test_merge_counts_concatenates_in_order():
    merged = merge_counts([np.array([1, 2, 3]), np.array([7, 8])], [3, 2])
    np.testing.assert_array_equal(merged, [1, 2, 3, 7, 8])
    assert merged.dtype == np.int64


def test_merge_counts_zero_fills_a_missing_board():
    merged = merge_counts([np.array([1, 2, 3]), None], [3, 2])
    np.testing.assert_array_equal(merged, [1, 2, 3, 0, 0])


# --- MergedReader (threaded, fake boards) ------------------------------------

def test_merged_reader_combines_two_streams():
    a = _FakeBoard([(0, np.array([1, 2, 3], dtype=np.int64))])
    b = _FakeBoard([(0, np.array([7, 8], dtype=np.int64))])
    mr = MergedReader([a, b], [3, 2], scan_hz=200, start_timeout_s=2.0)
    frames = _collect(mr, 3)
    assert len(frames) == 3
    for _, counts in frames:
        assert counts.shape == (5,)
        np.testing.assert_array_equal(counts[:3], [1, 2, 3])
        np.testing.assert_array_equal(counts[3:], [7, 8])
    # host-clocked timestamps step by step_ms (1000/200 = 5)
    assert frames[0][0] == 0
    assert frames[1][0] == 5


def test_merged_reader_holds_last_when_a_board_stalls():
    a = _FakeBoard([(0, np.array([1, 1, 1], dtype=np.int64))])         # forever
    b = _FakeBoard([(0, np.array([9, 9], dtype=np.int64))], loop=False)  # once, then stops
    mr = MergedReader([a, b], [3, 2], scan_hz=200, start_timeout_s=2.0)
    frames = _collect(mr, 5)
    assert len(frames) == 5
    for _, counts in frames:
        np.testing.assert_array_equal(counts[3:], [9, 9])  # b's last value held


def test_merged_reader_startup_timeout_when_a_board_is_silent():
    a = _FakeBoard([(0, np.array([1, 1, 1], dtype=np.int64))])
    silent = _FakeBoard([], loop=False)  # never yields
    mr = MergedReader([a, silent], [3, 2], scan_hz=200, start_timeout_s=0.2)
    with pytest.raises(RuntimeError):
        for _ in mr.frames():
            break
    mr.close()


def test_merged_reader_write_command_routes_to_servo_board():
    a = _FakeBoard([(0, np.array([1], dtype=np.int64))])
    b = _FakeBoard([(0, np.array([2], dtype=np.int64))])
    mr = MergedReader([a, b], [1, 1], servo_index=1)
    assert mr.write_command("S105") is True
    assert b.cmds == ["S105"]
    assert a.cmds == []


def test_merged_reader_close_closes_all_boards():
    a = _FakeBoard([(0, np.array([1], dtype=np.int64))])
    b = _FakeBoard([(0, np.array([2], dtype=np.int64))])
    mr = MergedReader([a, b], [1, 1])
    mr.close()
    assert a.closed and b.closed


# --- build_reader factory ----------------------------------------------------

def test_build_reader_single_board_returns_serialreader():
    r = build_reader(default_config(), port="PORTX", reconnect=False)
    assert isinstance(r, SerialReader)
    assert r.n_channels == 6


def _dual_cfg():
    return Config(
        bits=10, vref=5.0, vcc=5.0,
        channels=(
            Channel(0, "MQ5", 1000.0, 0), Channel(1, "MQ3", 1000.0, 0),
            Channel(2, "MQ2", 1000.0, 1),
        ),
        boards=(Board("/dev/a", 2, True, 0), Board("/dev/b", 1, False, 2)),
        scan_hz=20, baseline_s=1, exposure_s=1, purge_s=1, plateau_s=1,
        ema_alphas=(0.1,), max_cv=0.05, recover_tol=0.02,
    )


def test_build_reader_multiboard_returns_mergedreader():
    r = build_reader(_dual_cfg(), reconnect=False)
    assert isinstance(r, MergedReader)
    assert len(r._readers) == 2
    assert r._widths == [2, 1]
    assert r._servo_index == 0  # board 0 has the servo


class _LoopingSerial:
    """A byte source that repeats its CSV lines forever (a live board that keeps
    streaming), so a SerialReader over it never hits end-of-stream."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0
        self.closed = False

    def readline(self):
        line = self._lines[self._i % len(self._lines)]
        self._i += 1
        return line

    def close(self):
        self.closed = True


def test_build_reader_multiboard_merges_two_real_serial_streams(tmp_path):
    """Full path: build_reader -> MergedReader -> two SerialReaders parsing CSV ->
    one merged 9-channel frame in board order."""
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "sniffsniff.toml")
    assert cfg.n_channels == 9
    a = _LoopingSerial([b"1,10,11,12,13,14,15\n"])   # Uno 1: 6 counts
    b = _LoopingSerial([b"2,70,71,72\n"])            # Uno 2: 3 counts
    reader = build_reader(cfg, openers=[lambda: a, lambda: b])
    assert isinstance(reader, MergedReader)

    frames = _collect(reader, 2)
    assert len(frames) == 2
    for _, counts in frames:
        assert counts.shape == (9,)
        np.testing.assert_array_equal(counts[:6], [10, 11, 12, 13, 14, 15])
        np.testing.assert_array_equal(counts[6:], [70, 71, 72])
    assert a.closed and b.closed
