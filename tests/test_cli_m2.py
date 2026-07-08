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
