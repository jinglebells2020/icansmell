"""Three-phase sniff protocol, per-sniff R0 re-baselining, and dataset writer.

A "sniff" is one baseline -> exposure -> purge session. :class:`SniffRecorder`
turns a list of raw ``(t_ms, raw[N])`` frames into a fully calibrated, labeled
:class:`SniffResult` (Rs series, per-sniff R0, fractional curve, and the
``N*8``-D feature vector) and persists it as a compressed ``.npz`` plus a row in
``manifest.csv``.

Everything is channel-count-agnostic: ``N`` flows from ``config.n_channels`` and
the shapes of the input arrays; nothing hard-codes 6.
"""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import calibrate, features
from .config import Config

__all__ = [
    "compute_rs_series",
    "compute_r0",
    "baseline_cv",
    "noisy_channels",
    "phase_slices",
    "SniffResult",
    "SniffRecorder",
]

_MANIFEST_HEADER = ["id", "label", "path", "n_samples"]


def compute_rs_series(raw: np.ndarray, config: Config) -> np.ndarray:
    """Convert a raw counts series ``(T, N)`` to a sensor-resistance series.

    Applies the calibration chain counts -> V_RL -> Rs per element, using the
    config's per-channel load resistors and board ADC params. Returns a
    ``(T, N)`` float64 array (``+inf`` where a channel reads open/rail).
    """
    raw = np.asarray(raw)
    return calibrate.counts_to_rs(
        raw, config.rl_array(), config.vcc, config.vref, config.bits
    )


def compute_r0(rs_baseline: np.ndarray) -> np.ndarray:
    """Per-channel clean-air baseline resistance: mean Rs over the baseline frames.

    ``rs_baseline`` is ``(T_b, N)``; returns ``(N,)`` float64.
    """
    rs_baseline = np.asarray(rs_baseline, dtype=np.float64)
    return rs_baseline.mean(axis=0)


def baseline_cv(rs_baseline: np.ndarray) -> np.ndarray:
    """Per-channel coefficient of variation (std/mean) over the baseline window.

    ``rs_baseline`` is ``(T_b, N)``; returns ``(N,)`` float64 with population
    std (``ddof=0``). A channel with zero mean yields ``inf``/``nan`` naturally.
    """
    rs_baseline = np.asarray(rs_baseline, dtype=np.float64)
    mean = rs_baseline.mean(axis=0)
    std = rs_baseline.std(axis=0, ddof=0)
    return std / mean


def noisy_channels(rs_baseline: np.ndarray, max_cv: float) -> np.ndarray:
    """Boolean mask of channels whose baseline CV exceeds ``max_cv``.

    Only *finite* CVs are compared: a non-finite CV (from an open/rail channel
    with ``inf`` Rs) is a different fault and is left out of the noise mask.
    ``rs_baseline`` is ``(T_b, N)``; returns ``(N,)`` bool.
    """
    cv = baseline_cv(rs_baseline)
    return np.isfinite(cv) & (cv > float(max_cv))


def phase_slices(n_frames: int, config: Config) -> dict:
    """Compute half-open ``(start, end)`` frame slices for each protocol phase.

    Boundaries come from ``config`` timing * ``scan_hz`` and are clamped to
    ``n_frames`` so short/truncated sessions never index out of range:

    * ``baseline`` = ``[0, n_base)``
    * ``exposure`` = ``[n_base, n_base + n_exp)``
    * ``purge``    = ``[n_base + n_exp, n_frames)``
    * ``plateau``  = last ``plateau_s`` of the exposure window

    All returned as plain ``(int, int)`` tuples.
    """
    hz = config.scan_hz

    def clamp(x: int) -> int:
        return max(0, min(int(x), n_frames))

    n_base = round(config.baseline_s * hz)
    n_exp = round(config.exposure_s * hz)
    n_plateau = round(config.plateau_s * hz)

    b_start = 0
    b_end = clamp(n_base)
    e_start = b_end
    e_end = clamp(n_base + n_exp)
    p_start = e_end
    p_end = n_frames

    # plateau = trailing plateau_s of exposure, clamped inside [e_start, e_end].
    plat_start = max(e_start, e_end - n_plateau)
    plat_end = e_end

    return {
        "baseline": (b_start, b_end),
        "exposure": (e_start, e_end),
        "purge": (p_start, p_end),
        "plateau": (plat_start, plat_end),
    }


@dataclass
class SniffResult:
    """Everything computed for one sniff, ready to persist or hand downstream."""

    raw: np.ndarray
    t_ms: np.ndarray
    rs: np.ndarray
    r0: np.ndarray
    fractional: np.ndarray
    features: np.ndarray
    label: str
    slices: dict


class SniffRecorder:
    """Process raw frame lists into :class:`SniffResult` and write them to disk."""

    def __init__(self, config: Config, out_dir) -> None:
        self.config = config
        self.out_dir = Path(out_dir)

    # -- computation (no side effects) ------------------------------------
    def process(self, frames: list[tuple[int, np.ndarray]], label: str) -> SniffResult:
        """Compute the full calibrated result for a sniff (no disk writes).

        Splits the session via :func:`phase_slices`, computes the Rs series,
        derives R0 from the baseline window, forms the fractional curve
        ``rs/R0 - 1`` over the whole series, and extracts the ``N*8``-D feature
        vector using the phase slices.
        """
        if not frames:
            raise ValueError("cannot process an empty frame list")

        n = self.config.n_channels
        t_ms = np.array([int(t) for t, _ in frames], dtype=np.int64)
        raw = np.array([np.asarray(r, dtype=np.int64) for _, r in frames], dtype=np.int64)
        if raw.shape[1] != n:
            raise ValueError(
                f"frame width {raw.shape[1]} != config.n_channels {n}"
            )

        n_frames = raw.shape[0]
        slices = phase_slices(n_frames, self.config)

        rs = compute_rs_series(raw, self.config)

        b_lo, b_hi = slices["baseline"]
        rs_baseline = rs[b_lo:b_hi]
        r0 = compute_r0(rs_baseline)

        # Warn (don't fail) if the clean-air baseline is too noisy on any channel;
        # the operator can re-baseline. Mirrors the spec's baseline-rejection rule.
        noisy = noisy_channels(rs_baseline, self.config.max_cv)
        if np.any(noisy):
            cv = baseline_cv(rs_baseline)
            names = self.config.sensor_names()
            detail = ", ".join(
                f"{names[i]} CV={cv[i]:.3f}" for i in np.nonzero(noisy)[0]
            )
            warnings.warn(
                f"baseline too noisy on {int(noisy.sum())} channel(s) "
                f"(CV > max_cv={self.config.max_cv}): {detail}. "
                "Consider re-baselining in clean air.",
                stacklevel=2,
            )

        ratio = calibrate.rs_to_ratio(rs, r0)
        fractional = calibrate.ratio_to_fractional(ratio)

        feats = features.extract_features(
            fractional,
            exposure=slices["exposure"],
            purge=slices["purge"],
            plateau=slices["plateau"],
            ema_alphas=self.config.ema_alphas,
        )

        return SniffResult(
            raw=raw,
            t_ms=t_ms,
            rs=rs,
            r0=r0,
            fractional=fractional,
            features=feats,
            label=label,
            slices=slices,
        )

    # -- persistence ------------------------------------------------------
    def _next_id(self, label: str) -> str:
        """Deterministic id ``<label>_<k:04d>`` where ``k`` counts prior rows."""
        k = self._count_label_rows(label)
        return f"{label}_{k:04d}"

    def _count_label_rows(self, label: str) -> int:
        manifest = self.out_dir / "manifest.csv"
        if not manifest.exists():
            return 0
        count = 0
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("label") == label:
                    count += 1
        return count

    def save(self, result: SniffResult) -> Path:
        """Write ``<out>/<label>/<id>.npz`` and append a manifest row.

        The id is a per-label sequence number derived from the number of
        existing manifest rows with the same label (no wall-clock), so repeated
        records increment ``label_0000``, ``label_0001``, ...
        Returns the ``.npz`` path.
        """
        label = result.label
        sniff_id = self._next_id(label)

        label_dir = self.out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        npz_path = label_dir / f"{sniff_id}.npz"

        feature_col_names = features.feature_names(self.config.sensor_names())
        meta = {
            "label": label,
            "id": sniff_id,
            "n_samples": int(result.raw.shape[0]),
            "n_channels": self.config.n_channels,
            "slices": {k: list(v) for k, v in result.slices.items()},
            "feature_names": feature_col_names,
            "config": self._config_snapshot(),
        }

        np.savez_compressed(
            npz_path,
            raw=result.raw,
            t_ms=result.t_ms,
            rs=result.rs,
            r0=result.r0,
            fractional=result.fractional,
            features=result.features,
            meta=json.dumps(meta),
        )

        self._append_manifest(sniff_id, label, npz_path, int(result.raw.shape[0]))
        return npz_path

    def _append_manifest(self, sniff_id: str, label: str, path: Path, n_samples: int) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.out_dir / "manifest.csv"
        new = not manifest.exists()
        with manifest.open("a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(_MANIFEST_HEADER)
            writer.writerow([sniff_id, label, str(path), n_samples])

    def _config_snapshot(self) -> dict:
        cfg = self.config
        return {
            "bits": cfg.bits,
            "vref": cfg.vref,
            "vcc": cfg.vcc,
            "channels": [
                {"ch": c.ch, "sensor": c.sensor, "rl": c.rl} for c in cfg.channels
            ],
            "scan_hz": cfg.scan_hz,
            "baseline_s": cfg.baseline_s,
            "exposure_s": cfg.exposure_s,
            "purge_s": cfg.purge_s,
            "plateau_s": cfg.plateau_s,
            "ema_alphas": list(cfg.ema_alphas),
            "max_cv": cfg.max_cv,
            "recover_tol": cfg.recover_tol,
        }

    def record(self, frames, label) -> Path:
        """Convenience: :meth:`process` then :meth:`save`; returns the npz path."""
        result = self.process(frames, label)
        return self.save(result)
