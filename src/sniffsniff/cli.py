"""Command-line entry points for the sniffsniff e-nose (plain-stdout, no TUI).

Subcommands:

* ``stream``   — print a few fractional frames from the live/sim source.
* ``record``   — capture one sniff and persist it (``--label`` required).
* ``simulate`` — print a per-odor plateau fractional summary from the simulator.

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


def _collect_frames(reader) -> list[tuple[int, np.ndarray]]:
    """Drain a reader's ``frames()`` iterator into a list, closing it after."""
    frames = []
    try:
        for frame in reader.frames():
            frames.append(frame)
    finally:
        reader.close()
    return frames


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


def _make_reader(args, cfg: Config):
    """Build the frame source: simulator when ``--sim``, else a real serial port."""
    if args.sim:
        odor = getattr(args, "odor", None) or "coffee"
        frames = Simulator(cfg, seed=args.seed).sniff_frames(odor)
        return SimulatedReader(frames)
    from .serialio import SerialReader

    return SerialReader(args.port, n_channels=cfg.n_channels)


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
    """Capture one sniff from the source and persist it under ``--out``."""
    cfg = _load(args.config)
    reader = _make_reader(args, cfg)
    frames = _collect_frames(reader)
    if not frames:
        print("no frames captured; nothing recorded")
        return 1

    rec = SniffRecorder(cfg, args.out)
    result = rec.process(frames, args.label)  # compute once...
    path = rec.save(result)                    # ...then persist
    print(f"recorded {args.label}: {path}")
    print(f"  samples={result.raw.shape[0]} features={result.features.shape[0]}")
    return 0


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

    p_record = sub.add_parser("record", help="capture and save one sniff")
    p_record.add_argument("--label", required=True, help="odor label for this sniff")
    p_record.add_argument("--sim", action="store_true", help="use the simulator")
    p_record.add_argument("--config", default=None, help="path to a config TOML")
    p_record.add_argument("--out", default="data", help="output directory")
    p_record.add_argument("--port", default="/dev/ttyUSB0", help="serial port")
    p_record.add_argument("--seed", type=int, default=0)
    p_record.set_defaults(func=_cmd_record, odor=None)

    p_sim = sub.add_parser("simulate", help="print per-odor plateau fractional summary")
    p_sim.add_argument("--odor", default=None, help="a single odor (default: all)")
    p_sim.add_argument("--config", default=None, help="path to a config TOML")
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.set_defaults(func=_cmd_simulate)

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
