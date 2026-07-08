"""Non-UI orchestration layer for the sniffsniff TUI.

:class:`SniffController` is the seam between the interactive TUI (Textual) and the
Milestone 1/2 pipeline. It owns *no* UI: it opens readers (simulated or live),
runs bounded capture sessions, records sniffs to disk, trains the smell model, and
identifies unknown sniffs — reporting progress through plain callbacks so a UI (or
a test) can drive it however it likes.

Intentionally free of any ``textual`` import: everything here is pure Python +
the reused pipeline modules, so it can be unit-tested headlessly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .. import calibrate
from ..capture import capture_session
from ..dataset import load_dataset
from ..geometry import serialize_geometry
from ..model import SmellModel, cross_val_accuracy
from ..record import SniffRecorder
from ..serialio import SerialReader
from ..simulator import SimulatedReader, Simulator

__all__ = ["SniffController"]


class SniffController:
    """Drive capture / record / train / identify over sim or live hardware."""

    def __init__(
        self,
        config,
        out_dir="data",
        *,
        use_sim=True,
        port="/dev/cu.usbmodem101",
        seed=0,
        model_path="model.joblib",
    ) -> None:
        self.config = config
        self.out_dir = out_dir
        self.use_sim = use_sim
        self.port = port
        self.seed = seed
        self.model_path = model_path

    # ---------------------------------------------------------------- status
    @property
    def connected(self) -> bool:
        """``True`` when simulating, else whether the serial device node exists.

        Checks ``os.path.exists(port)`` — cheap and, crucially, non-disruptive:
        it does *not* open the port (opening an Arduino Uno toggles DTR and resets
        the board), so status can be polled freely.
        """
        if self.use_sim:
            return True
        return os.path.exists(self.port)

    def has_model(self) -> bool:
        """Whether a trained model exists at :attr:`model_path`."""
        return Path(self.model_path).exists()

    # ------------------------------------------------------------ calibration
    def rs_of(self, raw) -> np.ndarray:
        """Live per-frame sensor resistance for one raw counts frame."""
        cfg = self.config
        return calibrate.counts_to_rs(
            raw, cfg.rl_array(), cfg.vcc, cfg.vref, cfg.bits
        )

    # ----------------------------------------------------------------- reader
    def _reader(self, *, odor=None, seed=None):
        """Build a reader: a finite simulated session, or a live serial reader."""
        if self.use_sim:
            sim_seed = self.seed if seed is None else seed
            sim = Simulator(self.config, seed=sim_seed)
            frames = sim.sniff_frames(odor or "coffee")
            return SimulatedReader(frames)
        # reconnect=False: a bounded capture must NOT loop forever on a present-but-
        # silent device (readline timing out to b"") — it ends and we report no data.
        return SerialReader(
            self.port, n_channels=self.config.n_channels, reconnect=False
        )

    # ---------------------------------------------------------------- record
    def record_one(
        self, label, *, on_phase=None, on_frame=None, seed=None
    ) -> Path:
        """Capture one bounded session for ``label`` and persist it to disk."""
        reader = self._reader(odor=label, seed=seed)
        frames = capture_session(
            reader, self.config, on_phase=on_phase, on_frame=on_frame
        )
        if not frames:
            raise RuntimeError(
                f"no data from {self.port} — is the firmware flashed and streaming? "
                "(or press s for the simulator)"
            )
        return SniffRecorder(self.config, self.out_dir).record(frames, label)

    def record_many(
        self,
        label,
        reps,
        *,
        on_phase=None,
        on_frame=None,
        on_saved=None,
        seed=None,
    ) -> list[Path]:
        """Record ``reps`` sniffs for ``label``, offsetting the seed per rep.

        The per-rep seed offset makes each simulated sniff differ (otherwise every
        rep would be byte-identical). ``on_saved(path, i)`` fires after each save.
        """
        base = self.seed if seed is None else seed
        paths: list[Path] = []
        for i in range(reps):
            path = self.record_one(
                label, on_phase=on_phase, on_frame=on_frame, seed=base + i
            )
            if on_saved is not None:
                on_saved(path, i)
            paths.append(path)
        return paths

    # ---------------------------------------------------------------- stream
    def stream(self, on_frame, should_stop, *, odor="coffee") -> None:
        """Stream frames to ``on_frame(k, frame)`` until ``should_stop()`` is true."""
        reader = self._reader(odor=odor)
        k = 0
        try:
            for frame in reader.frames():
                on_frame(k, frame)
                k += 1
                if should_stop():
                    break
        finally:
            reader.close()

    # --------------------------------------------------------------- dataset
    def dataset_counts(self) -> dict[str, int]:
        """Per-label recording counts under :attr:`out_dir` (``{}`` if empty)."""
        try:
            ds = load_dataset(self.out_dir)
        except Exception:
            return {}
        counts: dict[str, int] = {}
        for label in ds.y.tolist():
            counts[label] = counts.get(label, 0) + 1
        return counts

    # ------------------------------------------------------------------- fit
    def fit(self, classifier="knn") -> tuple[float, float]:
        """Train + persist the smell model; return cross-validated (mean, std)."""
        ds = load_dataset(self.out_dir)
        if len(ds.classes) < 2:
            raise ValueError(
                "need at least 2 labeled classes to fit a model; "
                f"found {ds.classes}"
            )
        model = SmellModel(classifier=classifier).fit(ds.X, ds.y)
        model.feature_names_ = list(ds.feature_names)
        model.save(self.model_path)
        return cross_val_accuracy(
            ds.X, ds.y, classifier=classifier, groups=ds.ids
        )

    # -------------------------------------------------------------- identify
    def identify(self, *, on_phase=None, on_frame=None, seed=None) -> dict:
        """Capture one sniff, classify it, and return a JSON-friendly verdict."""
        model = SmellModel.load(self.model_path)
        reader = self._reader(odor="coffee", seed=seed)
        frames = capture_session(
            reader, self.config, on_phase=on_phase, on_frame=on_frame
        )
        if not frames:
            raise RuntimeError(
                f"no data from {self.port} — is the firmware flashed and streaming?"
            )
        feats = SniffRecorder(self.config, self.out_dir).process(frames, "?").features
        row = feats.reshape(1, -1)

        classes, proba = model.predict_proba(row)
        proba_row = np.asarray(proba)[0]
        pick = int(np.argmax(proba_row))
        return {
            "label": str(classes[pick]),
            "proba": float(proba_row[pick]),
            "novelty": float(model.novelty(row)[0]),
            "is_novel": bool(model.is_novel(row)[0]),
            "threshold": float(model.novelty_threshold_),
            "geometry": serialize_geometry(model, new_sample=feats),
        }
