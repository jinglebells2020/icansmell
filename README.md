# sniffsniff 👃

An LLM-reasoning **electronic nose**: a 6-sensor MQ gas-sensor array on an Arduino
Uno that streams to a laptop, where Python turns the raw stream into calibrated,
drift-suppressed, labeled **48-dimensional smell vectors**.

This repo is the **Foundation milestone** — the signal chain that everything else
builds on. The ML "smell map" (PCA/SOM/classifier/novelty) and the LLM reasoning
layer are later milestones (see [Roadmap](#roadmap)).

> **Design doc:** [`docs/superpowers/specs/2026-07-05-sniffsniff-foundation-design.md`](docs/superpowers/specs/2026-07-05-sniffsniff-foundation-design.md)

## The array

| Ch | Pin | Sensor | Axis it adds |
|----|-----|--------|--------------|
| C0 | A0  | MQ-3   | alcohol / ethanol — the drinks workhorse |
| C1 | A1  | MQ-135 | VOCs + ammonia — the spoiled-milk sensor |
| C2 | A2  | MQ-2   | broad smoke / VOC — general responder, baseline dimension |
| C3 | A3  | MQ-4   | methane — dairy/fermentation notes |
| C4 | A4  | MQ-8   | hydrogen — the chemical odd-one-out, spreads clusters |
| C5 | A5  | MQ-7   | carbon monoxide — different response curve |

The Python pipeline is **channel-count-agnostic** (`N` comes from config), so
resizing the array is a config edit + the firmware `NCH` constant — not a rewrite.

## Architecture

```
 [Uno firmware]  ── CSV "millis,c0,…,c5" @ ~20 Hz ──►  USB serial
        │                                                 │
        │          (SimulatedReader swaps in here)        │
        ▼                                                 ▼
   serialio  ──►  frames (t_ms, raw[6])
        ▼
   calibrate:  counts → V_RL → Rs = RL·(VCC−V_RL)/V_RL → r = Rs/R0 → y = r−1
        ▼
   record:  three-phase sniff (baseline → exposure → purge), re-baseline R0 per sniff
        ▼
   features:  sniff window → 8 feats/sensor (peak, plateau, 3+3 EMA) → 48-D vector
        ▼
   labeled dataset on disk  (.npz + manifest.csv)  ── feeds the ML milestone
```

Thin firmware, fat Python: each MQ module wires directly to an Uno analog pin
(A0–A5) and the Uno only reads A0–A5 and prints CSV; all calibration and feature
math lives in Python so you can retune `R0`/`RL` and re-extract features without
reflashing. With 6 sensors and 6 analog inputs, no multiplexer is needed.

## Quickstart (no hardware needed)

The simulator is a drop-in for the serial port, so the whole pipeline runs today.

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .          # or: pip install numpy pyserial pytest

# run the test suite (113 tests)
./.venv/bin/pytest -q

# see the simulated per-odor smell signatures (plateau fractional response)
./.venv/bin/python -m sniffsniff.cli simulate      # or: sniffsniff simulate (after pip install -e .)

# record one simulated sniff to a dataset
./.venv/bin/python -m sniffsniff.cli record --sim --label coffee --out data
```

`simulate` prints, per odor, the steady-state fractional response of each sensor —
e.g. alcohol peaks on MQ-3, vinegar and spoiled-milk on MQ-135.

## CLI

| Command | What it does |
|---------|--------------|
| `sniffsniff tui [--sim] [--port P] [--label L] [--reps N]` | interactive dashboard: live sensors + guided record→fit→identify→think |
| `sniffsniff simulate [--odor O]` | print per-odor plateau fractional summary |
| `sniffsniff record --label L [--sim] [--out DIR] [--port P]` | capture one sniff → `.npz` + manifest row |
| `sniffsniff stream [--sim] [--port P]` | print a few live calibrated frames |

`--sim` uses the simulator; without it, a real `SerialReader` opens `--port`.

### Interactive console (TUI)

`sniffsniff tui --sim` opens a live dashboard (needs the `tui` extra —
`pip install sniffsniff[tui]`). It runs **one** continuous monitor loop: the sensor
panel streams the whole time, and a record/identify is just a *window* over that
same stream — no Uno reset between sniffs.

- **SENSORS** — per-channel sparkline (recent history) + magnitude gauge + a
  rising/flat/falling trend arrow, with a `⚠` on channels whose clean-air baseline
  is too noisy.
- **CAPTURE** — a phase stepper (settle → baseline → exposure → purge) with a live
  progress bar and the response magnitude as it develops, then a return-to-baseline
  recovery readout.
- **COACH** / **LABELS** — the next suggested step, and a dot-meter of how many
  sniffs each label has toward the training target.

Keys: `r` record · `n`/`p` label · `a` add label · `+`/`-` reps · `c` classifier ·
`x` undo last · `X` clear · `f` fit · `i` identify · `t` think (LLM) · `m` map ·
`s` sim/real · `q` quit.

## Dataset format

Each sniff is one `data/<label>/<label>_NNNN.npz` holding `raw` `[T,6]`, `t_ms`
`[T]`, `rs` `[T,6]`, `r0` `[6]`, `fractional` `[T,6]`, `features` `[48]`, and a JSON
`meta` (label, phase-slice indices, feature column names, config snapshot). A
`data/manifest.csv` indexes every sniff. IDs are deterministic (no wall-clock).

## Firmware

`firmware/sniffsniff_uno/sniffsniff_uno.ino` — reads C0–C5 directly on analog pins
`A0`–`A5` (in wiring order), dummy-read + 16× averaging, prints `millis(),c0,…,c5`
at ~20 Hz. Compiles for `arduino:avr:uno` (2.4 KB flash / 7%, 184 B SRAM / 8%).

```bash
arduino-cli compile --fqbn arduino:avr:uno firmware/sniffsniff_uno
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/ttyACM0 firmware/sniffsniff_uno
```

### Wiring / hardware musts

- **Sensors → analog pins (direct, no multiplexer):** each MQ module's AO output goes
  straight to one Uno analog pin. With 6 sensors and 6 analog inputs, no mux is needed.

  | Pin | Sensor | Ch |
  |-----|--------|----|
  | A0  | MQ-3   | C0 |
  | A1  | MQ-135 | C1 |
  | A2  | MQ-2   | C2 |
  | A3  | MQ-4   | C3 |
  | A4  | MQ-8   | C4 |
  | A5  | MQ-7   | C5 |

- **Power:** heaters draw ~1 A total (6 × ~150–180 mA) — use an **external 5 V ≥ 3 A**
  supply for the sensors, common ground with the Uno. The Uno's regulator cannot feed them.
- **Burn-in:** 24–48 h continuous power on first use; 3–5 min warm-up each session.
- **Load resistor `RL`:** cheap MQ modules often ship 1 kΩ. It cancels in the Rs/R0
  ratio, but measure AO→GND (unpowered) and set it per channel in `sniffsniff.toml`.

> ⚠️ Heads-up surfaced by the build: high-resistance sensors (MQ-135, MQ-8) on a
> 1 kΩ `RL` sit at very low ADC counts and are noise-prone — the recorder emits a
> "baseline too noisy" warning when a channel's clean-air CV exceeds `max_cv`
> (default 5%). If you see it constantly, raise `RL` on those modules.

## Config

Edit [`sniffsniff.toml`](sniffsniff.toml): board ADC params, per-channel sensor +
`RL` map, `vcc`, sniff timing (baseline/exposure/purge/plateau seconds), EMA alphas,
and baseline thresholds.

## Roadmap

- **Milestone 1 — Foundation (this repo):** firmware + calibration + 48-D features + recorder. ✅
- **Milestone 2 — Smell map:** PCA/SOM, clustering, kNN/SVM/RandomForest identity, Mahalanobis novelty.
- **Milestone 3 — LLM reasoner:** serialize the learned geometry to JSON, let Claude interpret position,
  decide what to sniff next, predict-then-check, and ground clusters to words.

## Development

Built test-first — 113 pytest tests (calibration closed-form, hand-computed EMA
references, simulator determinism, dataset round-trip, end-to-end separability),
all deterministic. Run `./.venv/bin/pytest -q`.
