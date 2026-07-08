"""Typed configuration for the sniffsniff e-nose.

Parses ``sniffsniff.toml`` into an immutable :class:`Config`. The channel table
defines the number of sensors ``N`` (``n_channels``); the rest of the pipeline
derives ``N`` from here rather than hard-coding it.

The array may be split across **multiple boards** (Arduino Unos). Each board owns
a contiguous slice of the flat channel vector; :class:`Config` still exposes a
single flat ``channels`` (concatenated in board order), so everything downstream
of ingest is unchanged — only the reader (see :mod:`sniffsniff.serialio`) needs
to know the board layout in order to merge the two serial streams.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Channel:
    """One channel: its index, the sensor wired to it, its load resistor, board."""

    ch: int
    sensor: str
    rl: float
    board: int = 0


@dataclass(frozen=True)
class Board:
    """One physical board (Uno) and where its channels sit in the flat vector.

    ``port`` is the serial device (``None`` for a legacy single-board config, where
    the port comes from the CLI ``--port``). ``start`` is the offset of this board's
    first channel in the flat ``Config.channels`` vector; the board owns
    ``channels[start : start + n_channels]``. ``servo`` marks the board that hosts
    the airflow servo.
    """

    port: str | None
    n_channels: int
    servo: bool
    start: int


@dataclass(frozen=True)
class Config:
    """Immutable board + array + timing + feature configuration.

    ``channels`` is stored ordered by ``ch`` 0..N-1, so anything ordered by
    channel index can iterate ``channels`` directly. ``boards`` describes how the
    array is split across physical boards (one entry for a single-board rig).
    """

    bits: int
    vref: float
    vcc: float
    channels: tuple[Channel, ...]
    boards: tuple[Board, ...]
    scan_hz: int
    baseline_s: float
    exposure_s: float
    purge_s: float
    plateau_s: float
    ema_alphas: tuple[float, ...]
    max_cv: float
    recover_tol: float
    # Airflow servo (optional): switches the fresh-air vs sample straw during a
    # capture. Angles are found via `sniffsniff servo`. Disabled by default so a
    # rig without a servo is unaffected.
    servo_enabled: bool = False
    servo_pin: int = 12
    servo_fresh_air_angle: int = 0
    servo_sample_angle: int = 105
    # Adaptive-capture tuning: wait for a stable baseline before R0, gate exposure
    # on a plateau, and EMA-smooth the live signal so per-frame noise doesn't fool
    # the settle/recovery detectors. Tolerance reuses ``recover_tol``.
    settle_hold_s: float = 3.0        # signal must be flat this long before baseline
    settle_max_wait_s: float = 30.0   # give up settling after this (proceed anyway)
    # Dynamic exposure ends when the aggregate response STOPS GROWING (stops setting
    # new highs for plateau_hold_s), never before min_exposure_s, capped by exposure_s.
    # Gating on growth (not flatness) keeps weak/slow odors — whose slope is buried in
    # sensor noise — from being cut short. Tuned on the real rig (fresh milk).
    min_exposure_s: float = 15.0      # exposure floor — never plateau-stop before this
    plateau_hold_s: float = 8.0       # end after the response sets no new high this long
    plateau_eps: float = 0.005        # min fractional growth counted as a new high (0.5pp)
    smooth_alpha: float = 0.2         # EMA factor for the detectors (0 = no smoothing)

    def rl_array(self) -> np.ndarray:
        """Per-channel load resistances, shape ``(N,)`` float64, ordered by ch."""
        return np.array([c.rl for c in self.channels], dtype=np.float64)

    def sensor_names(self) -> list[str]:
        """Sensor names, length ``N``, ordered by ch 0..N-1."""
        return [c.sensor for c in self.channels]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def multi_board(self) -> bool:
        """True when the array spans more than one physical board."""
        return len(self.boards) > 1


def _build_channels(raw_channels: list[dict], board: int = 0) -> tuple[Channel, ...]:
    """Validate a raw channel table and return channels ordered by ``ch``.

    Requires the ``ch`` values to be exactly the set ``{0..len-1}`` with no
    duplicates or gaps; raises :class:`ValueError` otherwise. ``ch`` is the local
    A-pin index on ``board``.
    """
    n = len(raw_channels)
    if n == 0:
        raise ValueError("config must define at least one channel")

    ch_values = [int(c["ch"]) for c in raw_channels]
    if sorted(ch_values) != list(range(n)):
        raise ValueError(
            f"channel 'ch' values must be exactly {{0..{n - 1}}} unique; got "
            f"{sorted(ch_values)}"
        )

    ordered = sorted(raw_channels, key=lambda c: int(c["ch"]))
    return tuple(
        Channel(ch=int(c["ch"]), sensor=str(c["sensor"]), rl=float(c["rl"]), board=board)
        for c in ordered
    )


def _build_multiboard(raw_boards: list[dict]) -> tuple[tuple[Channel, ...], tuple[Board, ...]]:
    """Flatten a list of per-board channel tables into one channel vector + boards.

    Each board's local ``ch`` values must be ``{0..nb-1}``; global channel indices
    are assigned by concatenating boards in order.
    """
    if not raw_boards:
        raise ValueError("array.board must list at least one board")
    flat: list[Channel] = []
    boards: list[Board] = []
    start = 0
    for b_idx, board in enumerate(raw_boards):
        local = _build_channels(list(board["channels"]), board=b_idx)
        for local_ch, c in enumerate(local):
            flat.append(Channel(ch=start + local_ch, sensor=c.sensor, rl=c.rl, board=b_idx))
        nb = len(local)
        boards.append(
            Board(
                port=(str(board["port"]) if board.get("port") is not None else None),
                n_channels=nb,
                servo=bool(board.get("servo", False)),
                start=start,
            )
        )
        start += nb
    return tuple(flat), tuple(boards)


def _config_from_dict(data: dict) -> Config:
    board = data["board"]
    array = data["array"]
    timing = data["timing"]
    features = data["features"]
    baseline = data["baseline"]

    servo = data.get("servo", {})
    capture = data.get("capture", {})

    if "board" in array:  # multi-board: [[array.board]] tables
        channels, boards = _build_multiboard(list(array["board"]))
        servo_enabled = any(b.servo for b in boards)
    else:                 # legacy single board: [array].channels
        channels = _build_channels(list(array["channels"]))
        servo_enabled = bool(servo.get("enabled", False))
        boards = (
            Board(port=None, n_channels=len(channels), servo=servo_enabled, start=0),
        )

    return Config(
        bits=int(board["bits"]),
        vref=float(board["vref"]),
        vcc=float(array["vcc"]),
        channels=channels,
        boards=boards,
        scan_hz=int(timing["scan_hz"]),
        baseline_s=float(timing["baseline_s"]),
        exposure_s=float(timing["exposure_s"]),
        purge_s=float(timing["purge_s"]),
        plateau_s=float(timing["plateau_s"]),
        ema_alphas=tuple(float(a) for a in features["ema_alphas"]),
        max_cv=float(baseline["max_cv"]),
        recover_tol=float(baseline["recover_tol"]),
        servo_enabled=servo_enabled,
        servo_pin=int(servo.get("pin", 12)),
        servo_fresh_air_angle=int(servo.get("fresh_air_angle", 0)),
        servo_sample_angle=int(servo.get("sample_angle", 105)),
        settle_hold_s=float(capture.get("settle_hold_s", 3.0)),
        settle_max_wait_s=float(capture.get("settle_max_wait_s", 30.0)),
        min_exposure_s=float(capture.get("min_exposure_s", 15.0)),
        plateau_hold_s=float(capture.get("plateau_hold_s", 8.0)),
        plateau_eps=float(capture.get("plateau_eps", 0.005)),
        smooth_alpha=float(capture.get("smooth_alpha", 0.2)),
    )


def load_config(path) -> Config:
    """Load and validate a :class:`Config` from a TOML file at ``path``."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return _config_from_dict(data)


def default_config() -> Config:
    """Return the built-in 6-sensor single-board defaults (no file read).

    This is the Foundation-milestone baseline the test-suite builds on; the shipped
    ``sniffsniff.toml`` describes the current (dual-board, 9-sensor) rig.
    """
    channels = (
        Channel(ch=0, sensor="MQ3", rl=1000.0),
        Channel(ch=1, sensor="MQ135", rl=1000.0),
        Channel(ch=2, sensor="MQ2", rl=1000.0),
        Channel(ch=3, sensor="MQ4", rl=1000.0),
        Channel(ch=4, sensor="MQ8", rl=1000.0),
        Channel(ch=5, sensor="MQ7", rl=1000.0),
    )
    return Config(
        bits=10,
        vref=5.0,
        vcc=5.0,
        channels=channels,
        boards=(Board(port=None, n_channels=6, servo=True, start=0),),
        scan_hz=20,
        baseline_s=15.0,
        exposure_s=45.0,
        purge_s=90.0,
        plateau_s=10.0,
        ema_alphas=(0.1, 0.01, 0.001),
        max_cv=0.05,
        recover_tol=0.02,
        servo_enabled=True,
        servo_pin=12,
        servo_fresh_air_angle=0,
        servo_sample_angle=105,
        settle_hold_s=3.0,
        settle_max_wait_s=30.0,
        min_exposure_s=15.0,
        plateau_hold_s=8.0,
        plateau_eps=0.005,
        smooth_alpha=0.2,
    )
