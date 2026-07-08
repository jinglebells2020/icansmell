"""Typed configuration for the sniffsniff e-nose.

Parses ``sniffsniff.toml`` into an immutable :class:`Config`. The channel table
defines the number of sensors ``N`` (``n_channels``); the rest of the pipeline
derives ``N`` from here rather than hard-coding it.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Channel:
    """One channel: its index, the sensor wired to it, and its load resistor."""

    ch: int
    sensor: str
    rl: float


@dataclass(frozen=True)
class Config:
    """Immutable board + array + timing + feature configuration.

    ``channels`` is stored ordered by ``ch`` 0..N-1, so anything ordered by
    channel index can iterate ``channels`` directly.
    """

    bits: int
    vref: float
    vcc: float
    channels: tuple[Channel, ...]
    scan_hz: int
    baseline_s: float
    exposure_s: float
    purge_s: float
    plateau_s: float
    ema_alphas: tuple[float, ...]
    max_cv: float
    recover_tol: float

    def rl_array(self) -> np.ndarray:
        """Per-channel load resistances, shape ``(N,)`` float64, ordered by ch."""
        return np.array([c.rl for c in self.channels], dtype=np.float64)

    def sensor_names(self) -> list[str]:
        """Sensor names, length ``N``, ordered by ch 0..N-1."""
        return [c.sensor for c in self.channels]

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def _build_channels(raw_channels: list[dict]) -> tuple[Channel, ...]:
    """Validate a raw channel table and return channels ordered by ``ch``.

    Requires the ``ch`` values to be exactly the set ``{0..len-1}`` with no
    duplicates or gaps; raises :class:`ValueError` otherwise.
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
        Channel(ch=int(c["ch"]), sensor=str(c["sensor"]), rl=float(c["rl"]))
        for c in ordered
    )


def _config_from_dict(data: dict) -> Config:
    board = data["board"]
    array = data["array"]
    timing = data["timing"]
    features = data["features"]
    baseline = data["baseline"]

    channels = _build_channels(list(array["channels"]))

    return Config(
        bits=int(board["bits"]),
        vref=float(board["vref"]),
        vcc=float(array["vcc"]),
        channels=channels,
        scan_hz=int(timing["scan_hz"]),
        baseline_s=float(timing["baseline_s"]),
        exposure_s=float(timing["exposure_s"]),
        purge_s=float(timing["purge_s"]),
        plateau_s=float(timing["plateau_s"]),
        ema_alphas=tuple(float(a) for a in features["ema_alphas"]),
        max_cv=float(baseline["max_cv"]),
        recover_tol=float(baseline["recover_tol"]),
    )


def load_config(path) -> Config:
    """Load and validate a :class:`Config` from a TOML file at ``path``."""
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return _config_from_dict(data)


def default_config() -> Config:
    """Return the built-in defaults matching ``sniffsniff.toml`` (no file read)."""
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
        scan_hz=20,
        baseline_s=15.0,
        exposure_s=45.0,
        purge_s=90.0,
        plateau_s=10.0,
        ema_alphas=(0.1, 0.01, 0.001),
        max_cv=0.05,
        recover_tol=0.02,
    )
