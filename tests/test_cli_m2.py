"""CLI smoke tests for the Milestone 2 subcommands: fit / map / identify.

These exercise the new subcommands end-to-end through the simulator path (no
hardware, no disk recordings needed): ``fit --sim`` trains and saves a model,
``identify --sim ... --json`` prints a valid geometry blob, and ``map --sim``
renders a PNG. The existing M1 subcommands must keep working (covered here too).
"""
from __future__ import annotations

import json

import pytest

from sniffsniff import cli
from sniffsniff.model import SmellModel


def test_cli_fit_sim_creates_model(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    rc = cli.main(["fit", "--sim", "--out", str(model_path)])
    assert rc == 0
    assert model_path.exists()

    # The saved file loads back into a fitted SmellModel.
    model = SmellModel.load(str(model_path))
    assert hasattr(model, "classes_")
    # The 5 non-clean odors are the default sim classes.
    assert set(model.classes_) == {
        "coffee",
        "vinegar",
        "alcohol",
        "fresh_milk",
        "spoiled_milk",
    }

    out = capsys.readouterr().out
    # Cross-validated accuracy is printed and labeled as simulated.
    assert "sim" in out.lower()


def test_cli_identify_sim_json(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    assert cli.main(["fit", "--sim", "--out", str(model_path)]) == 0
    capsys.readouterr()  # drop fit output

    rc = cli.main(
        [
            "identify",
            "--model",
            str(model_path),
            "--sim",
            "--odor",
            "coffee",
            "--json",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    # Find the line that parses as JSON and check it carries the geometry schema.
    blob = None
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            blob = candidate
            break
    assert blob is not None, out
    assert "novelty" in blob
    assert "pca" in blob
    assert "new_sample" in blob


def test_cli_identify_sim_no_json(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    assert cli.main(["fit", "--sim", "--out", str(model_path)]) == 0
    capsys.readouterr()

    rc = cli.main(
        [
            "identify",
            "--model",
            str(model_path),
            "--sim",
            "--odor",
            "coffee",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # A human-readable prediction line is printed even without --json.
    assert "coffee" in out.lower() or "predicted" in out.lower()
    assert "novel" in out.lower()


def test_cli_map_sim_creates_png(tmp_path):
    model_path = tmp_path / "m.joblib"
    assert cli.main(["fit", "--sim", "--out", str(model_path)]) == 0

    png_path = tmp_path / "m.png"
    rc = cli.main(
        [
            "map",
            "--model",
            str(model_path),
            "--sim",
            "--out",
            str(png_path),
        ]
    )
    assert rc == 0
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_cli_fit_custom_odors_and_reps(tmp_path):
    model_path = tmp_path / "m.joblib"
    rc = cli.main(
        [
            "fit",
            "--sim",
            "--odors",
            "coffee,vinegar,alcohol",
            "--reps",
            "5",
            "--out",
            str(model_path),
        ]
    )
    assert rc == 0
    model = SmellModel.load(str(model_path))
    assert set(model.classes_) == {"coffee", "vinegar", "alcohol"}
    assert sum(model.counts_.values()) == 15


# --- guided / bounded / multi-rep record (M2.5) ------------------------------

def test_cli_record_sim_single(tmp_path, capsys):
    rc = cli.main(["record", "--sim", "--label", "coffee", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "coffee" / "coffee_0000.npz").exists()
    assert "recorded coffee" in capsys.readouterr().out


def test_cli_record_sim_multi_rep(tmp_path, capsys):
    rc = cli.main(
        ["record", "--sim", "--label", "vinegar", "--reps", "3", "--out", str(tmp_path)]
    )
    assert rc == 0
    files = sorted((tmp_path / "vinegar").glob("*.npz"))
    assert [f.name for f in files] == [
        "vinegar_0000.npz",
        "vinegar_0001.npz",
        "vinegar_0002.npz",
    ]
    # manifest has one row per rep (plus header)
    rows = (tmp_path / "manifest.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 3
    out = capsys.readouterr().out
    assert "sniff 1/3" in out and "sniff 3/3" in out


def test_cli_record_multi_rep_seeds_vary(tmp_path):
    # Different reps must not be byte-identical sniffs (per-rep seed offset).
    import numpy as np

    cli.main(
        ["record", "--sim", "--label", "coffee", "--reps", "2", "--out", str(tmp_path)]
    )
    a = np.load(tmp_path / "coffee" / "coffee_0000.npz")["features"]
    b = np.load(tmp_path / "coffee" / "coffee_0001.npz")["features"]
    assert not np.array_equal(a, b)


def test_cli_record_bounded_on_infinite_reader(tmp_path, monkeypatch):
    # Simulate a LIVE (never-ending) serial reader and prove `record` (real path)
    # captures one session and stops instead of hanging.
    import numpy as np
    from sniffsniff import cli as climod

    class _Infinite:
        def frames(self):
            k = 0
            while True:
                yield (k * 50, np.full(6, 300, dtype=np.int64))  # realistic non-zero counts
                k += 1

        def close(self):
            pass

    monkeypatch.setattr(climod, "_make_reader", lambda args, cfg, seed=None: _Infinite())
    # short-timing config so the bounded session is small
    cfg_path = tmp_path / "fast.toml"
    cfg_path.write_text(
        """
[board]
bits = 10
vref = 5.0
[array]
vcc = 5.0
channels = [
  {ch=0,sensor="MQ3",rl=1000},{ch=1,sensor="MQ135",rl=1000},{ch=2,sensor="MQ2",rl=1000},
  {ch=3,sensor="MQ4",rl=1000},{ch=4,sensor="MQ8",rl=1000},{ch=5,sensor="MQ7",rl=1000},
]
[timing]
scan_hz=20
baseline_s=1
exposure_s=2
purge_s=1
plateau_s=0.5
[features]
ema_alphas=[0.1,0.01,0.001]
[baseline]
max_cv=0.05
recover_tol=0.02
"""
    )
    rc = cli.main(
        ["record", "--label", "coffee", "--config", str(cfg_path), "--out", str(tmp_path)]
    )
    assert rc == 0
    saved = np.load(tmp_path / "coffee" / "coffee_0000.npz")
    assert saved["raw"].shape == (80, 6)  # exactly one 80-frame session, not infinite
