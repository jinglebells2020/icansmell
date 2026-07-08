# sniffsniff 👃

An LLM-reasoning **electronic nose**: a 9-sensor MQ gas-sensor array across **two
Arduino Unos** that stream to a laptop, where Python merges them and turns the raw
stream into calibrated, drift-suppressed, labeled **72-dimensional smell vectors**.

This repo is the **Foundation milestone** — the signal chain that everything else
builds on. The ML "smell map" (PCA/SOM/classifier/novelty) and the LLM reasoning
layer are later milestones (see [Roadmap](#roadmap)).

> **Design doc:** [`docs/superpowers/specs/2026-07-05-sniffsniff-foundation-design.md`](docs/superpowers/specs/2026-07-05-sniffsniff-foundation-design.md)

## The array — 9 sensors across two Arduino Unos

The array is split over **two Unos**, both on USB. The Python host reads both and
merges them (equal peers, host-clocked) into one 9-channel frame in board order.

**Uno 1** (6 sensors + the airflow servo):

| Ch | Pin | Sensor | Axis it adds |
|----|-----|--------|--------------|
| C0 | A0  | MQ-5   | LPG / natural gas — methane/fermentation axis |
| C1 | A1  | MQ-3   | alcohol / ethanol — the drinks workhorse |
| C2 | A2  | MQ-135 | VOCs + ammonia — the spoiled-milk sensor |
| C3 | A3  | MQ-7   | carbon monoxide — different response curve |
| C4 | A4  | MQ-9   | CO + combustible gas — pairs with MQ-7 |
| C5 | A5  | MQ-8   | hydrogen — the chemical odd-one-out |

**Uno 2** (3 sensors, no servo):

| Ch | Pin | Sensor | Axis it adds |
|----|-----|--------|--------------|
| C6 | A0  | MQ-2   | broad smoke / VOC — general responder |
| C7 | A1  | MQ-4   | methane — dairy/fermentation notes |
| C8 | A2  | MQ-6   | LPG / propane / butane |

The Python pipeline is **channel-count-agnostic** (`N` comes from config), so
resizing the array — or splitting it across boards — is a config edit + the
firmware `NCH` constant, not a rewrite. Feature vectors scale with `N` (9 → 72-D).
A single-board rig still works via a flat `[array].channels` table (see git history).

## Architecture

```
 [Uno 1 fw] ── "millis,c0,…,c5" ─►┐
 [Uno 2 fw] ── "millis,c0,c1,c2" ─►┤  USB serial  (SimulatedReader swaps in here)
                                    ▼
   serialio.MergedReader  ──►  merged frames (t_ms, raw[9])   ← equal peers, host-clocked
        ▼
   calibrate:  counts → V_RL → Rs = RL·(VCC−V_RL)/V_RL → r = Rs/R0 → y = r−1
        ▼
   record:  three-phase sniff (baseline → exposure → purge), re-baseline R0 per sniff
        ▼
   features:  sniff window → 8 feats/sensor (peak, plateau, 3+3 EMA) → 72-D vector
        ▼
   labeled dataset on disk  (.npz + manifest.csv)  ── feeds the ML milestone
```

Thin firmware, fat Python: each MQ module wires directly to an Uno analog pin and each
Uno only reads its pins and prints CSV; the host merges the two streams and does all
calibration + feature math, so you can retune `R0`/`RL` and re-extract without reflashing.
Two boards give 9 analog inputs — no multiplexer needed.

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

## Firmware — two sketches (one per Uno)

Both boards run the same thin firmware (dummy-read + 16× averaging, CSV at ~20 Hz,
`arduino:avr:uno`); they differ only in channel count and the servo:

- `firmware/sniffsniff_uno/` — **Uno 1**: 6 channels (`A0`–`A5`) **+ airflow servo** on D12.
- `firmware/sniffsniff_uno_b/` — **Uno 2**: 3 channels (`A0`–`A2`), no servo.

```bash
# flash Uno 1 (find its port with: arduino-cli board list)
arduino-cli compile --fqbn arduino:avr:uno firmware/sniffsniff_uno
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/cu.usbmodem101 firmware/sniffsniff_uno
# flash Uno 2
arduino-cli compile --fqbn arduino:avr:uno firmware/sniffsniff_uno_b
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/cu.usbmodem102 firmware/sniffsniff_uno_b
```

Set each board's `port` in `sniffsniff.toml` to match. The host reads both and merges
them into one 9-channel frame.

### Wiring / hardware musts

- **Sensors → analog pins (direct, no multiplexer):** each MQ module's AO output goes
  straight to one Uno analog pin — 6 on Uno 1, 3 on Uno 2 (see the array tables above).
- **Common ground across both Unos:** the external 5 V supply ground must tie to *both*
  Uno grounds, or the readings drift.
- **Power:** heaters draw ~1.5 A total across 9 sensors (~150–180 mA each) — use an
  **external 5 V ≥ 3 A** supply, common ground with the Unos. The Unos' regulators can't feed them.
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
