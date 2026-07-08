# sniffsniff — Foundation Milestone Design

**Date:** 2026-07-05
**Status:** Approved (design), pending implementation plan
**Milestone:** 1 of N — "Foundation" (firmware + serial reader + Rs→Rs/R0→fractional feature chain)

## Purpose

Build the make-or-break bedrock of an LLM-reasoning electronic nose: an Arduino Uno
that streams a clean N-channel MQ-sensor vector over USB, and a Python package that
turns that stream into calibrated, drift-suppressed, labeled feature vectors.
Everything downstream (PCA/SOM map, classifier, novelty detection, LLM reasoner)
consumes these vectors — but none of it is built in this milestone.

The hardware exists but is **not yet wired**. Therefore a **sensor simulator** is a
first-class deliverable: it lets the entire Python pipeline be developed and tested
today, with the real serial port swapping in later behind an identical interface.

## Sensor selection — 6 MQ sensors

The array is **6 sensors**, chosen for maximally *orthogonal* chemistry so clusters
spread out instead of piling redundant signal on one axis:

| Ch | Sensor | Target | Why it's in |
|----|--------|--------|-------------|
| C0 | MQ-2   | broad smoke / VOC        | General-purpose responder; reacts to almost everything — a good baseline dimension. |
| C1 | MQ-3   | alcohol / ethanol        | The drinks workhorse (coffee, vinegar, anything fermented). Non-negotiable. |
| C2 | MQ-4   | methane                  | Distinct axis; picks up dairy/fermentation notes differently than the rest. |
| C3 | MQ-7   | carbon monoxide          | Different response curve again; cheap extra dimension. |
| C4 | MQ-8   | hydrogen                 | The chemical odd-one-out — responds unlike the others, so it spreads clusters apart. |
| C5 | MQ-135 | VOCs + ammonia           | The spoiled-milk punchline sensor. Non-negotiable. |

Feature vector = **6 sensors × 8 features = 48-D per sniff**.

**Channel-count-agnostic pipeline.** The Python package derives the channel count `N`
from config (`Config.n_channels`), so resizing the array later is a config edit plus one
firmware constant (`NCH`) and the CSV field count — not a code rewrite. `N = 6` is the
concrete instance here; nothing in `calibrate`, `features`, `serialio`, or `record`
hard-codes 6.

## Scope

### In scope (this milestone)
- Arduino Uno firmware: scan 6 MQ sensors through a CD74HC4067 mux on one ADC pin,
  dummy-read + averaging, stream CSV over USB serial at ~20 Hz.
- Python `serialio`: parse the CSV stream into frames; robust to garbled lines and
  disconnects. A `SimulatedReader` with the identical interface.
- Python `simulator`: synthetic 6-MQ array with per-odor response kinetics, seeded
  for reproducibility, emitting frames in the same shape as the real device.
- Python `calibrate`: pure functions counts → V_RL → Rs → Rs/R0 → fractional.
- Python `features`: a sniff window → 8 features/sensor (2 steady-state + 6 EMA
  transient) → `N*8`-D vector.
- Python `record`: three-phase sniff protocol (baseline → exposure → purge) with
  per-sniff R0 re-baselining; writes labeled sniff records to disk.
- Config (`config` + `sniffsniff.toml`): board ADC params, per-channel sensor map +
  RL, VCC, timing, feature params.
- `cli`: `stream`, `record --label <name>`, `simulate`.
- Tests (pytest): calibration, features, simulator, and an end-to-end sanity check.

### Out of scope (explicitly deferred — YAGNI)
- PCA / SOM / clustering / classifier / Mahalanobis novelty (Milestone 2).
- LLM reasoning layer, geometry serialization, active-sniff loop (Milestone 3 — Claude API).
- Servo/fan automation, physical chamber (hardware, user's domain).
- Temperature/humidity compensation (no T/RH sensor in the kit; documented mitigation only).
- Bidirectional host→device command channel (future "Approach C" upgrade).

## Architecture

Approach **A — thin firmware, fat Python**. The Uno does only mux-scan + ADC + CSV
out (raw counts + timestamp). All calibration and feature math lives in Python, so
calibration constants (R0, RL) can be tuned and features re-extracted without
reflashing. The Uno's ~2 KB SRAM cannot hold feature buffers or a model regardless.

```
 [Uno firmware]  ── CSV "millis,c0,…,c5" @ ~20 Hz ──►  USB serial
        │                                                 │
        │          (SimulatedReader swaps in here)        │
        ▼                                                 ▼
   serialio.SerialReader / SimulatedReader  ──►  frames: (t_ms, raw[N])
        ▼
   calibrate:  counts → V_RL → Rs = RL·(VCC−V_RL)/V_RL → r = Rs/R0 → y = r−1
        ▼
   record:  three-phase sniff (baseline → exposure → purge), re-baseline R0 per sniff
        ▼
   features:  sniff window → 8 feats/sensor (2 steady + 6 EMA transient) → N*8-D vector
        ▼
   labeled dataset on disk  ── handoff to the ML milestone
```

## Data contract — the CSV wire format

One line per full-array scan, newline-terminated ASCII:

```
<millis>,<c0>,<c1>,<c2>,<c3>,<c4>,<c5>\n
```

- `millis` — unsigned `millis()` timestamp from the Uno (wraps ~every 49 days; the
  host treats it as monotonic within a session and detects wrap).
- `c0..c5` — integer raw ADC counts, 0–1023 (Uno 10-bit), one per mux channel C0–C5,
  each already the average of N samples on-device.
- ~20 Hz (one line per ~50 ms). Field count is always `NCH + 1` (7); the host skips any
  line that doesn't parse to exactly that many integers.

Raw counts (not millivolts) travel on the wire so all conversion constants live in
Python config, not firmware.

## Calibration math (`calibrate.py`, pure/stateless)

```
V_RL = counts * VREF / (2^BITS - 1)        # Uno: VREF=5.0, BITS=10  → /1023
Rs   = RL * (VCC - V_RL) / V_RL            # guard V_RL > EPS, else Rs = +inf (sensor open)
r    = Rs / R0                             # R0 = clean-air Rs per channel
y    = r - 1                               # fractional response (ΔR/R0), dimensionless
```

- `RL` and `VCC` are per-channel config. `RL` cancels in the ratio `r`, but is kept so
  `Rs` is a real resistance for inspection/logging.
- Div-by-zero guard: if `V_RL <= EPS`, treat `Rs` as `+inf` (channel reads open/rail);
  the feature/record layer flags such a channel rather than emitting NaNs downstream.

Hand-checkable case (used in tests): `VCC=5, RL=10 kΩ, V_RL=1.0 V → Rs = 10k·(5−1)/1 = 40 kΩ`.
If `R0 = 40 kΩ` then `r = 1.0`, `y = 0.0` (clean air ⇒ zero fractional response).

## R0 baseline (`record.py`)

- `R0[ch]` = mean `Rs[ch]` over the **baseline window** in clean air, re-measured
  immediately before **every** sniff (the single most important repeatability trick).
- The baseline is rejected as too noisy if any channel's coefficient of variation over
  the window exceeds a configurable threshold (default 5%); the operator is warned and
  can re-baseline.
- "Recovered" (safe to start the next sniff): every channel's `r = Rs/R0` within ±2% of
  1.0 **and** `|dy/dt| ≈ 0` (below threshold) for ≥5 consecutive seconds.

## Feature extraction (`features.py`, pure)

Per sensor, over one sniff's fractional curve `y(t)`, matching the UCI 8-feature design
(2 steady-state + 6 transient EMA):

**Steady-state (2):**
1. `peak` — max `|y|` during the exposure phase.
2. `plateau_mean` — mean `y` over the final `plateau_s` seconds of exposure.

**Transient (6):** from the difference sequence `Δy[k] = y[k] − y[k−1]`, the exponential
moving average `ema_α[k] = (1−α)·ema_α[k−1] + α·Δy[k]` for `α ∈ {0.1, 0.01, 0.001}`:
3–5. `ema_rise_α` — max `ema_α` over the **rising** (exposure) phase, one per α.
6–8. `ema_decay_α` — min `ema_α` over the **decaying** (purge/recovery) phase, one per α.

6 sensors × 8 features = **48-D vector** per sniff. Feature order is fixed
(sensor-major) and recorded in the dataset so downstream code has a stable column layout.

## Sensor simulator (`simulator.py`)

Emits frames in the exact `(t_ms, raw[N])` shape, so it is a drop-in for the serial
reader. Model per sensor:

- Clean-air baseline resistance `R_base[ch]` (deterministic per channel).
- Per-odor multiplicative response gain `g[odor][ch]` (reducing gases drop Rs): during
  exposure, `Rs` relaxes toward `R_base/(1+g)` with rise time constant `τ_rise`; during
  purge, relaxes back toward `R_base` with `τ_decay` (`τ_decay > τ_rise`, slow desorption).
- Additive Gaussian noise from a **seeded** `numpy.random.Generator` → byte-reproducible.
- Frames are produced by inverting the calibration (`Rs → V_RL → counts`, quantized to
  10-bit), so the simulator exercises the real calibration path rather than shortcutting it.

Odor profiles (initial set, tunable): `clean_air`, `coffee`, `vinegar`, `alcohol`,
`fresh_milk`, `spoiled_milk`, with gains chosen so the fractional signatures are
separable given the 6-sensor chemistry (alcohol strong on MQ-3; vinegar & spoiled-milk
on MQ-135; methane/H₂/CO axes on MQ-4/MQ-8/MQ-7 pull the dairy classes apart).

## Configuration (`sniffsniff.toml`)

```toml
[board]
bits = 10          # Uno ADC resolution
vref = 5.0         # Uno ADC reference volts

[array]
vcc = 5.0          # sensor supply volts
# one entry per mux channel C0..C5 — CONFIRM RL against your actual modules
channels = [
  { ch = 0, sensor = "MQ2",   rl = 1000 },
  { ch = 1, sensor = "MQ3",   rl = 1000 },
  { ch = 2, sensor = "MQ4",   rl = 1000 },
  { ch = 3, sensor = "MQ7",   rl = 1000 },
  { ch = 4, sensor = "MQ8",   rl = 1000 },
  { ch = 5, sensor = "MQ135", rl = 1000 },
]

[timing]
scan_hz     = 20
baseline_s  = 15
exposure_s  = 45
purge_s     = 90
plateau_s   = 10   # trailing exposure window for plateau_mean

[features]
ema_alphas = [0.1, 0.01, 0.001]

[baseline]
max_cv = 0.05      # reject baseline if any channel CV exceeds this
recover_tol = 0.02 # ±2% of R0 to count as recovered
```

`rl = 1000` reflects that cheap MQ modules commonly ship a 1 kΩ load resistor; the
operator confirms/measures per module. The number of channel entries defines `N`.

## On-disk dataset format

Per sniff, one compressed `data/<label>/<id>.npz` containing:
- `raw` — int array `[T, N]` of ADC counts.
- `t_ms` — int array `[T]` of device timestamps.
- `fractional` — float array `[T, N]` of `y(t)`.
- `r0` — float array `[N]` of the R0 used.
- `features` — float array `[N*8]`.
- `meta` — JSON blob: label, session id, sniff id, ISO timestamp, phase boundaries
  (baseline/exposure/purge sample indices), feature column names, config snapshot.

Plus a `data/manifest.csv` appended one row per sniff: `id,label,path,n_samples`.

`data/` is git-ignored.

## Module layout

```
sniffsniff/
  sniffsniff.toml
  pyproject.toml
  firmware/sniffsniff_uno/sniffsniff_uno.ino
  src/sniffsniff/
    __init__.py
    config.py        # load/validate sniffsniff.toml → typed Config object
    serialio.py      # SerialReader + SimulatedReader (shared frame interface)
    simulator.py     # synthetic N-MQ array
    calibrate.py     # pure counts→V→Rs→ratio→fractional
    features.py      # pure sniff-window → N*8-D vector
    record.py        # three-phase protocol + R0 + dataset writer
    cli.py           # stream / record / simulate entry points
  tests/
    test_config.py
    test_calibrate.py
    test_features.py
    test_serialio.py
    test_simulator.py
    test_record.py
    test_end_to_end.py
  data/              # git-ignored recorded sniffs
  docs/superpowers/specs/2026-07-05-sniffsniff-foundation-design.md
```

## Firmware (`firmware/sniffsniff_uno/sniffsniff_uno.ino`)

- `NCH = 6` (channel count constant — the one firmware value tied to array size).
- Pins: `S0..S3` on `D4,D5,D6,D7`; mux `SIG` on `A0`; mux `EN` tied to GND (active-LOW,
  always enabled).
- `selectCh(c)`: write the 4 address bits (LSB = S0).
- Per channel: set address → `delayMicroseconds(100)` settle → one `analogRead(A0)`
  discarded → average `N = 16` reads.
- Print `millis(),c0,…,c5` then `delay(50)` (~20 Hz full scan).
- No floating-point on-device; raw 10-bit counts only.
- Header comments document hardware musts: external **5 V ≥3 A** supply for the heaters
  (6 × ~150–180 mA ≈ 0.9–1.1 A; a 2 A+ supply is fine, 3 A leaves headroom), common
  ground with the Uno, unused mux channels C6–C15 tied to GND, 100 nF VCC→GND on the mux,
  24–48 h first-power burn-in and 3–5 min warm-up each session.

## Error handling

- **Serial:** malformed / short lines are skipped with a counter; on disconnect the
  reader retries with backoff; wrong field count never crashes the pipeline.
- **Calibration:** `V_RL ≤ EPS` ⇒ `Rs = +inf` and the channel is flagged, not NaN-poisoned.
- **Baseline:** refuses to emit ratios without a valid R0; warns and re-baselines if the
  clean-air window is too noisy.
- **Source-agnostic:** `SerialReader` and `SimulatedReader` share one interface, so a bad
  port or missing device degrades gracefully (and the sim path always works).

## Testing strategy (TDD — tests before implementation for the pure math)

- **`test_config.py`** — `default_config` has 6 channels in order; `load_config` round-trips
  `sniffsniff.toml`; rejects a channel table of the wrong size / duplicate `ch`.
- **`test_calibrate.py`** — closed-form cases (the 40 kΩ example above; ratio; fractional;
  the div-by-zero guard); array inputs vectorize.
- **`test_features.py`** — analytic step/exponential curves with hand-derived peak,
  plateau, and EMA values (the exact 8-feature set; no area term in this milestone);
  `feature_names` length `N*8` with correct sensor-major ordering.
- **`test_serialio.py`** — `parse_line` valid/malformed/wrong-field-count; `SerialReader`
  via an injected fake serial (no hardware).
- **`test_simulator.py`** — same seed ⇒ identical frames; distinct odor profiles produce
  separable fractional signatures; `SimulatedReader` replays exactly.
- **`test_record.py`** — three-phase segmentation, R0 computation, deterministic ids, and
  `.npz`/manifest round-trip on simulated input.
- **`test_end_to_end.py`** — simulate ≥3 classes → record → features; assert within-class
  vectors are closer (Euclidean) than across-class (sanity, not a trained model).

## Dependencies

Python 3.11+ (stdlib `tomllib` for config), `numpy`, `pyserial`, `pytest`. CLI kept to
plain stdout — no heavy TUI dependency in this milestone.

## Assumptions to confirm before wiring (not blockers for software)

1. Actual RL value on each module (measure AO→GND unpowered; many are 1 kΩ).
2. Digital pin assignments `D4–D7` for `S0–S3` (any 4 free digital pins work).
3. Sensor→channel order in the table is the assumed wiring; match your physical build to it
   (or edit the config — it only affects feature-column *labels*, since RL cancels in ratios).

## Handoff to next milestone

This milestone's output — a growing folder of labeled 48-D vectors plus the tooling to
record more — is exactly the input the ML milestone (PCA/SOM/classifier/novelty) needs.
The `features` column layout and dataset schema are the stable contract between them.
