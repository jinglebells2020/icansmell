"""Command-line entry points for the sniffsniff e-nose (plain-stdout, no TUI).

Subcommands:

* ``stream``   — print a few fractional frames from the live/sim source.
* ``record``   — capture one sniff and persist it (``--label`` required).
* ``simulate`` — print a per-odor plateau fractional summary from the simulator.
* ``fit``      — build a labeled dataset (disk or ``--sim``) and fit a SmellModel.
* ``map``      — render the 2-D PCA smell map (PNG) for a fitted model.
* ``identify`` — capture/simulate one sniff and print its predicted odor + novelty.

``--sim`` selects the :mod:`sniffsniff.simulator` path; otherwise a real
:class:`sniffsniff.serialio.SerialReader` is used. The pipeline is
channel-count-agnostic: everything derives ``N`` from the loaded config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import calibrate
from .config import Config, default_config, load_config
from .record import SniffRecorder, phase_slices
from .simulator import ODOR_PROFILES, SimulatedReader, Simulator


def _load(config_path: str | None) -> Config:
    if config_path:
        return load_config(config_path)
    return default_config()


# Operator cues printed at each phase transition during a guided capture.
_PHASE_CUES = {
    "baseline": "🫧  BASELINE — hold CLEAN AIR (measuring R0)",
    "exposure": "👃  PRESENT THE SAMPLE now",
    "purge": "🌬   REMOVE the sample — purging back to clean air",
}


def _phase_cue(phase: str, k: int, n: int) -> None:
    """Print the big operator cue when a capture crosses into a new phase."""
    print(f"\n  {_PHASE_CUES.get(phase, phase)}")


def _make_ticker(cfg: Config):
    """A per-frame callback that prints a once-per-second elapsed/total line."""
    hz = max(1, cfg.scan_hz)

    def tick(k: int, n: int, phase: str, frame=None) -> None:
        if k % hz == 0:
            print(f"\r    [{k // hz:3d}s / {n // hz:3d}s]  {phase:8s}", end="", flush=True)

    return tick


def _recover_countdown(seconds: int) -> None:
    """Between reps: a clean-air recovery countdown (real capture only)."""
    import time

    for s in range(seconds, 0, -1):
        print(f"\r  recover — hold clean air, next sniff in {s:2d}s ", end="", flush=True)
        time.sleep(1)
    print()


def _cmd_simulate(args) -> int:
    """Print, for each odor (or one ``--odor``), the plateau-mean fractional vector."""
    cfg = _load(args.config)
    sim = Simulator(cfg, seed=args.seed)
    names = cfg.sensor_names()

    if args.odor:
        odors = [args.odor]
    else:
        odors = list(ODOR_PROFILES.keys())

    header = "odor".ljust(14) + "  " + "  ".join(f"{n:>8}" for n in names)
    print(header)
    print("-" * len(header))
    for odor in odors:
        frames = sim.sniff_frames(odor)
        raw = np.array([r for _, r in frames], dtype=np.int64)
        rs = calibrate.counts_to_rs(raw, cfg.rl_array(), cfg.vcc, cfg.vref, cfg.bits)
        sl = phase_slices(raw.shape[0], cfg)
        b_lo, b_hi = sl["baseline"]
        r0 = rs[b_lo:b_hi].mean(axis=0)
        fractional = rs / r0 - 1.0
        p_lo, p_hi = sl["plateau"]
        plateau = fractional[p_lo:p_hi].mean(axis=0)
        cells = "  ".join(f"{v:8.3f}" for v in plateau)
        print(f"{odor.ljust(14)}  {cells}")
    return 0


def _make_reader(args, cfg: Config, *, seed: int | None = None):
    """Build the frame source: simulator when ``--sim``, else a real serial port.

    ``seed`` overrides ``args.seed`` for the simulator path (used to vary reps).
    """
    if args.sim:
        s = args.seed if seed is None else seed
        odor = getattr(args, "odor", None) or "coffee"
        frames = Simulator(cfg, seed=s).sniff_frames(odor)
        return SimulatedReader(frames)
    from .serialio import SerialReader

    # reconnect=False so a bounded capture ends on a silent/absent device instead
    # of looping forever; startup_delay lets the Uno boot after the open-reset.
    return SerialReader(
        args.port, n_channels=cfg.n_channels, reconnect=False, startup_delay_s=2.5
    )


def _cmd_stream(args) -> int:
    """Print the first few fractional frames from the source (sim or serial)."""
    cfg = _load(args.config)
    reader = _make_reader(args, cfg)
    names = cfg.sensor_names()

    limit = args.limit
    rl = cfg.rl_array()
    printed = 0
    print("t_ms  " + "  ".join(f"{n:>8}" for n in names))
    try:
        for t_ms, raw in reader.frames():
            rs = calibrate.counts_to_rs(raw, rl, cfg.vcc, cfg.vref, cfg.bits)
            # A smoke stream has no established R0 baseline, so fractional response
            # isn't well-defined here — print the calibrated Rs (ohms) directly.
            cells = "  ".join(f"{v:8.1f}" for v in rs)
            print(f"{t_ms:6d}  {cells}")
            printed += 1
            if printed >= limit:
                break
    finally:
        reader.close()
    return 0


def _cmd_record(args) -> int:
    """Capture ``--reps`` guided sniffs from the source and persist them.

    Each sniff is a bounded capture of exactly one baseline→exposure→purge session
    (so a live serial stream is captured then stopped, never drained forever). The
    operator is cued at each phase transition; between reps on real hardware a
    recovery countdown lets the sensors return to baseline.
    """
    from .capture import capture_session

    cfg = _load(args.config)
    rec = SniffRecorder(cfg, args.out)
    reps = max(1, args.reps)
    ticker = None if args.sim else _make_ticker(cfg)

    saved = []
    for i in range(reps):
        if reps > 1:
            print(f"\n=== sniff {i + 1}/{reps}  ·  label '{args.label}' ===")
        reader = _make_reader(args, cfg, seed=args.seed + i)
        frames = capture_session(reader, cfg, on_phase=_phase_cue, on_frame=ticker)
        if ticker is not None:
            print()  # end the \r status line
        if not frames:
            print("no frames captured; nothing recorded")
            return 1

        result = rec.process(frames, args.label)  # compute once...
        path = rec.save(result)                    # ...then persist
        saved.append(path)
        print(
            f"  saved {Path(path).name}  "
            f"(samples={result.raw.shape[0]}, features={result.features.shape[0]})"
        )

        if not args.sim and i < reps - 1:
            _recover_countdown(args.gap)

    if reps > 1:
        print(f"\nrecorded {len(saved)} × '{args.label}' → {args.out}")
    else:
        print(f"recorded {args.label}: {saved[0]}")
    return 0


# The five non-clean-air odors the simulator knows — the default sim classes for
# fit/map. ``clean_air`` is the baseline, not an odor to identify against.
_DEFAULT_SIM_ODORS = ["coffee", "vinegar", "alcohol", "fresh_milk", "spoiled_milk"]


def _build_dataset(args, cfg: Config):
    """Build a :class:`~sniffsniff.dataset.Dataset` from ``--data`` or ``--sim``.

    ``--sim`` synthesises one via the simulator (``--odors``/``--reps``/``--seed``);
    otherwise ``--data DIR`` loads recorded ``.npz`` sniffs off disk. Returns
    ``(dataset, source)`` where ``source`` is ``"sim"`` or ``"real"`` (used to
    honestly label the reported accuracy).
    """
    from . import dataset as dataset_mod

    if getattr(args, "sim", False):
        if args.odors:
            odors = [o.strip() for o in args.odors.split(",") if o.strip()]
        else:
            odors = list(_DEFAULT_SIM_ODORS)
        ds = dataset_mod.simulate_dataset(
            cfg, odors, args.reps, seed=args.seed
        )
        return ds, "sim"

    if not args.data:
        raise SystemExit("one of --data DIR or --sim is required")
    ds = dataset_mod.load_dataset(args.data)
    return ds, "real"


def _cmd_fit(args) -> int:
    """Fit a SmellModel from a simulated or recorded dataset and save it."""
    from .model import SmellModel, cross_val_accuracy

    cfg = _load(args.config)
    ds, source = _build_dataset(args, cfg)
    if ds.X.shape[0] == 0:
        print("no samples in dataset; nothing to fit")
        return 1

    model = SmellModel(classifier=args.classifier)
    model.fit(ds.X, ds.y)
    # Carry the column names so geometry/identify can label features by name.
    model.feature_names_ = list(ds.feature_names)

    mean, std = cross_val_accuracy(
        ds.X, ds.y, classifier=args.classifier, groups=ds.ids
    )
    print(
        f"cross-validated accuracy ({source}): "
        f"{mean:.3f} ± {std:.3f} "
        f"({len(ds.classes)} classes, n={ds.X.shape[0]})"
    )

    model.save(args.out)
    print(f"saved model: {args.out}")
    return 0


def _cmd_map(args) -> int:
    """Render the 2-D PCA smell map to a PNG for a fitted model."""
    from .model import SmellModel
    from .smellmap import render_map

    cfg = _load(args.config)
    model = SmellModel.load(args.model)
    ds, _ = _build_dataset(args, cfg)
    saved = render_map(model, ds, path=args.out)
    print(f"saved map: {saved}")
    return 0


def _cmd_identify(args) -> int:
    """Capture/simulate one sniff, print predicted odor + probability + novelty."""
    import json as _json

    from .model import SmellModel
    from .geometry import serialize_geometry

    from .capture import capture_session

    cfg = _load(args.config)
    model = SmellModel.load(args.model)

    reader = _make_reader(args, cfg)
    ticker = None if args.sim else _make_ticker(cfg)
    frames = capture_session(reader, cfg, on_phase=_phase_cue, on_frame=ticker)
    if ticker is not None:
        print()
    if not frames:
        print("no frames captured; nothing to identify")
        return 1

    # Label is only bookkeeping for the recorder; identity comes from the model.
    label = getattr(args, "odor", None) or "unknown"
    result = SniffRecorder(cfg, ".").process(frames, label)
    feats = result.features

    feats_2d = feats.reshape(1, -1)
    classes, proba = model.predict_proba(feats_2d)
    proba_row = np.asarray(proba)[0]
    pred_idx = int(np.argmax(proba_row))
    pred_label = str(classes[pred_idx])
    pred_proba = float(proba_row[pred_idx])
    novelty = float(model.novelty(feats_2d)[0])
    is_novel = bool(model.is_novel(feats_2d)[0])

    print(f"predicted: {pred_label}  (proba={pred_proba:.3f})")
    print(
        f"novelty: {novelty:.3f}  "
        f"(threshold={model.novelty_threshold_:.3f}, is_novel={is_novel})"
    )
    if args.json:
        print(_json.dumps(serialize_geometry(model, new_sample=feats)))
    return 0


def _cmd_tui(args) -> int:
    """Launch the interactive Textual TUI over a SniffController."""
    from .tui.controller import SniffController

    cfg = _load(args.config)
    controller = SniffController(
        cfg,
        out_dir=args.out,
        use_sim=args.sim,
        port=args.port,
        seed=args.seed,
        model_path=args.model,
    )

    try:
        from .tui.app import run_tui
    except ImportError:
        print("the TUI needs textual: pip install sniffsniff[tui]")
        return 1

    run_tui(controller, reps=args.reps, label=args.label)
    return 0


def _add_dataset_source_args(parser) -> None:
    """Shared ``--data``/``--sim`` dataset-source flags for fit/map."""
    parser.add_argument("--data", default=None, help="dataset dir of recorded sniffs")
    parser.add_argument("--sim", action="store_true", help="synthesise via simulator")
    parser.add_argument(
        "--odors", default=None, help="comma-separated odors for --sim"
    )
    parser.add_argument("--reps", type=int, default=8, help="sniffs per odor for --sim")
    parser.add_argument("--config", default=None, help="path to a config TOML")
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sniffsniff", description="6-sensor MQ e-nose")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stream = sub.add_parser("stream", help="print a few frames from the source")
    p_stream.add_argument("--sim", action="store_true", help="use the simulator")
    p_stream.add_argument("--config", default=None, help="path to a config TOML")
    p_stream.add_argument("--port", default="/dev/ttyUSB0", help="serial port")
    p_stream.add_argument("--odor", default=None, help="odor for --sim")
    p_stream.add_argument("--seed", type=int, default=0)
    p_stream.add_argument("--limit", type=int, default=5, help="frames to print")
    p_stream.set_defaults(func=_cmd_stream)

    p_record = sub.add_parser("record", help="capture and save guided sniff(s)")
    p_record.add_argument("--label", required=True, help="odor label for these sniffs")
    p_record.add_argument("--sim", action="store_true", help="use the simulator")
    p_record.add_argument("--config", default=None, help="path to a config TOML")
    p_record.add_argument("--out", default="data", help="output directory")
    p_record.add_argument("--port", default="/dev/ttyUSB0", help="serial port")
    p_record.add_argument("--seed", type=int, default=0)
    p_record.add_argument("--reps", type=int, default=1, help="number of sniffs to capture")
    p_record.add_argument(
        "--gap", type=int, default=20, help="recovery seconds between reps (real capture)"
    )
    p_record.set_defaults(func=_cmd_record, odor=None)

    p_sim = sub.add_parser("simulate", help="print per-odor plateau fractional summary")
    p_sim.add_argument("--odor", default=None, help="a single odor (default: all)")
    p_sim.add_argument("--config", default=None, help="path to a config TOML")
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.set_defaults(func=_cmd_simulate)

    p_fit = sub.add_parser("fit", help="fit a SmellModel from a dataset")
    _add_dataset_source_args(p_fit)
    p_fit.add_argument("--classifier", default="knn", help="knn|svm|rf|lda")
    p_fit.add_argument("--out", default="model.joblib", help="model output path")
    p_fit.set_defaults(func=_cmd_fit)

    p_map = sub.add_parser("map", help="render the PCA smell map to a PNG")
    p_map.add_argument("--model", required=True, help="fitted model.joblib")
    _add_dataset_source_args(p_map)
    p_map.add_argument("--out", default=None, help="PNG output path")
    p_map.set_defaults(func=_cmd_map)

    p_id = sub.add_parser("identify", help="identify one sniff against a model")
    p_id.add_argument("--model", required=True, help="fitted model.joblib")
    p_id.add_argument("--sim", action="store_true", help="use the simulator")
    p_id.add_argument("--odor", default=None, help="odor to simulate for --sim")
    p_id.add_argument("--port", default="/dev/ttyUSB0", help="serial port")
    p_id.add_argument("--config", default=None, help="path to a config TOML")
    p_id.add_argument("--seed", type=int, default=0)
    p_id.add_argument("--json", action="store_true", help="also emit geometry JSON")
    p_id.set_defaults(func=_cmd_identify)

    p_tui = sub.add_parser("tui", help="launch the interactive Textual console")
    p_tui.add_argument("--sim", action="store_true", help="use the simulator")
    p_tui.add_argument(
        "--port", default="/dev/cu.usbmodem101", help="serial port"
    )
    p_tui.add_argument("--config", default=None, help="path to a config TOML")
    p_tui.add_argument("--out", default="data", help="output directory")
    p_tui.add_argument("--model", default="model.joblib", help="model path")
    p_tui.add_argument("--reps", type=int, default=1, help="sniffs per record press")
    p_tui.add_argument("--label", default="coffee", help="default record label")
    p_tui.add_argument("--seed", type=int, default=0)
    p_tui.set_defaults(func=_cmd_tui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # For record, --sim uses the label as the odor to simulate.
    if getattr(args, "command", None) == "record" and args.sim and not args.odor:
        args.odor = args.label
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
