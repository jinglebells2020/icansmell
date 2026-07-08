# sniffsniff — Technical Documentation

An **LLM-reasoning electronic nose**: a 6-sensor metal-oxide (MQ) gas-sensor array on
an Arduino Uno that streams a calibrated chemical "fingerprint" of whatever it smells,
learns a low-dimensional *smell map* of known odors, flags unknown ones as novel, and
finally hands the learned **geometry** (not the raw signals) to a language model to
produce a human-readable verdict.

This document covers the physics, the signal chain, the machine-learning pipeline, the
adaptive capture engine, the LLM layer, and the research those choices are grounded in.
It is written to be read top-to-bottom by an engineer new to the project, but every
section stands alone. Line references point into the source under
[`src/sniffsniff/`](../src/sniffsniff).

---

## 1. System at a glance

```
  PHYSICAL WORLD                      ARDUINO UNO (firmware)             PYTHON HOST
 ┌───────────────┐   headspace air   ┌────────────────────┐   USB CSV  ┌──────────────────────────┐
 │ substance jar │──►┌───────────┐──►│ 6× MQ AO → A0..A5  │──20 Hz──►  │ serialio → calibrate     │
 │  + fresh air  │   │  airflow  │   │ 16-sample average  │  115200    │   counts→V_RL→Rs→Rs/R0→y │
 └───────────────┘   │  servo D12│◄──│ S<angle> command   │◄─────────  │ record  (3-phase sniff)  │
                     └───────────┘   └────────────────────┘            │ features (48-D vector)   │
                                                                       │ model  PCA+kNN+novelty   │
                                                                       │ geometry → JSON          │
                                                                       │ reason → LLM verdict     │
                                                                       └──────────────────────────┘
```

The pipeline is **channel-count-agnostic**: the number of sensors `N` flows from the
config's channel table ([`config.py`](../src/sniffsniff/config.py)); nothing in the
calibration, feature, ML, or IO code hard-codes 6. The concrete build is `N = 6`.

The project was built in milestones, each with a design spec under
[`docs/superpowers/specs/`](superpowers/specs):

| Milestone | Scope | Key modules |
|---|---|---|
| **M1 Foundation** | firmware, serial ingest, calibration, 3-phase protocol, 48-D features, simulator | `serialio`, `calibrate`, `record`, `features`, `simulator` |
| **M2 Smell Map** | StandardScaler→PCA→classifier, Mahalanobis novelty, 2-D map, geometry JSON | `model`, `dataset`, `smellmap`, `geometry` |
| **M3 LLM Reasoner** | OpenRouter client + prompt that reasons over the geometry | `llm`, `reason` |
| **TUI / Engine** | single-connection monitor engine, adaptive capture, guided-training TUI, airflow servo | `monitor`, `recovery`, `servo`, `tui/` |

---

## 2. The physics: why a resistance tells you what you're smelling

### 2.1 Chemiresistive sensing (SnO₂)

Each MQ sensor is a **heated tin-dioxide (SnO₂) semiconductor**. In clean air, oxygen
chemisorbs on the SnO₂ grain surface and traps conduction electrons (O₂ + e⁻ → O⁻
species), forming an electron-depletion layer at every grain boundary. Those boundaries
act as potential barriers, so the sensor sits at a **high baseline resistance R₀**.

When a **reducing / combustible gas** (ethanol, methane, CO, H₂, assorted VOCs) reaches
the hot surface, it reacts with the adsorbed oxygen ions, *releases the trapped
electrons back into the conduction band*, collapses the depletion layer, and the sensor
resistance **Rs falls**. The magnitude of the fall grows (roughly log-linearly) with gas
concentration. This is the mechanism the whole instrument rests on.

> In practice a real headspace is not a single clean reducing gas — humidity and
> oxidizing components can push a channel the *other* way. The pipeline never assumes a
> sign: features use the response **magnitude** and the ML stage learns the *pattern*
> across the array. (On the real rig, fresh-milk headspace actually raised Rs on some
> channels — see §11.)

### 2.2 Baseline R₀ and the fractional response — the repeatability trick

An absolute Rs is nearly useless: it drifts with temperature, humidity, aging, and
varies chip-to-chip by large factors. The fix, standard in metal-oxide olfaction, is to
**normalize every sniff against its own freshly-measured clean-air baseline R₀**:

```
ratio       r = Rs / R0
fractional  y = Rs/R0 − 1        (≈ 0 in clean air; departs on exposure)
```

Dividing by R₀ cancels the slow common-mode drift and most inter-chip variation, leaving
a dimensionless response `y` that is comparable across sensors, chips, and days. The load
resistor `RL` (see §4) also cancels in this ratio. sniffsniff measures **R₀ anew for
every single sniff** ([`record.compute_r0`](../src/sniffsniff/record.py)), which is why
it can tolerate drift between sessions without any global recalibration.

### 2.3 Why an *array* — cross-sensitivity is a feature, not a bug

A single SnO₂ sensor is intrinsically **non-selective**: every reducing gas exploits the
same oxygen-depletion mechanism, so one sensor cannot tell ethanol from methane. The
classical e-nose answer (Persaud & Dodd, 1982) is a **distributed array of differently
tuned sensors** plus pattern recognition: each sensor has a different (overlapping)
sensitivity profile, so an odor becomes a *point in N-dimensional response space*, and
distinct odors land in distinct regions. sniffsniff's six sensors were chosen for
maximally **orthogonal chemistry** so clusters spread out rather than piling onto one
axis:

| Ch | Pin | Sensor | Nominal target | Role in the array |
|----|-----|--------|----------------|-------------------|
| C0 | A0 | **MQ-3** | alcohol / ethanol | the drinks workhorse (fermented anything) |
| C1 | A1 | **MQ-135** | VOCs + ammonia | the spoiled/dairy "off-note" axis |
| C2 | A2 | **MQ-2** | broad smoke / LPG / VOC | general responder — a good common axis |
| C3 | A3 | **MQ-4** | methane | dairy/fermentation notes, distinct axis |
| C4 | A4 | **MQ-8** | hydrogen | the chemical odd-one-out; spreads clusters |
| C5 | A5 | **MQ-7** | carbon monoxide | different response curve again |

### 2.4 Kinetics — why sniffs take tens of seconds

MOS response is **not instantaneous**. Adsorption/desorption and heater-surface
equilibration give first-order-ish dynamics with two very different time constants:

- **Rise** (τ_rise ≈ 5–60 s): on exposure, Rs relaxes toward its gas-loaded value.
- **Decay** (τ_decay ≈ 60–300 s): on return to clean air, Rs relaxes back — **much
  slower** than the rise, because desorbing gas and re-adsorbing oxygen is sluggish.

Two engineering consequences drive the whole capture design (§7–8):

1. The **exposure** window must be long enough to develop a usable response, but a weak
   odor may still be rising after a minute — so a fixed window either wastes time or
   truncates. This motivates the **growth-gated dynamic exposure** (§8.3).
2. Because decay ≫ rise, the sensors must be given time (and clean air) to **recover to
   baseline** before the next sniff, or the new R₀ is contaminated. This motivates the
   explicit **recovery** and **settle** gates (§8.2).

---

## 3. Hardware and firmware

**Wiring** ([`firmware/sniffsniff_uno/sniffsniff_uno.ino`](../firmware/sniffsniff_uno/sniffsniff_uno.ino)):
each MQ module's analog-out (AO) goes directly to an Uno analog pin, A0→MQ3 … A5→MQ7
(no multiplexer). An airflow **servo** on D12 selects which straw is open. Because the MQ
heaters draw significant current, they (and the servo) run from an **external 5 V supply**
with a common ground to the Uno — the Uno's own 5 V pin cannot source it.

**Firmware role** — deliberately thin (all physics/ML live on the host):

```
loop() every ~50 ms (~20 Hz):
  handleCommands()                     # non-blocking: parse "S<angle>\n", clamp 0..180, servo.write
  t = millis()
  for each channel c in 0..5:
    analogRead(Ac)                     # one throwaway read: let the ADC sample/hold settle
    sum += analogRead(Ac) × 16         # average 16 reads (integer)
    counts[c] = sum / 16
  Serial.println("t,c0,c1,c2,c3,c4,c5")   # CSV, 115200 8N1
```

- All on-device arithmetic is **integer**; the host does every physical conversion.
- The throwaway read matters: MQ dividers are high-impedance, so the ADC sample/hold
  needs a settle read after switching channels.
- The host tolerates `millis()` wrap and any garbled line (see §4).

> **Measured caveat:** on the real clone-Uno the full 6-channel scan + serial actually
> streams at **~16 Hz**, not the nominal 20 Hz the config assumes. All second-based
> windows therefore run ~25 % longer in wall-clock than their labels — harmless and
> conservative here, but worth knowing when reading timing (see §11).

---

## 4. Signal chain and calibration

All calibration is pure, stateless, vectorized numpy in
[`calibrate.py`](../src/sniffsniff/calibrate.py). Per channel, per frame:

```
1. counts → load-resistor volts   V_RL = counts · Vref / (2^bits − 1)
2. volts  → sensor resistance      Rs   = RL · (Vcc − V_RL) / V_RL
3. Rs     → clean-air ratio         r    = Rs / R0
4. ratio  → fractional response     y    = r − 1
```

with board defaults `bits = 10` (counts 0..1023), `Vref = Vcc = 5.0 V`, and per-channel
`RL = 1000 Ω`. Step 2 is the voltage-divider inversion: the MQ module puts Rs and RL in
series across Vcc and taps V_RL across RL.

**Open / rail guard.** If `V_RL ≤ 1e-9` (a disconnected or railed channel), step 2 would
divide by zero; instead the code returns `Rs = +inf` (never `NaN`). `+inf` propagates
cleanly through `r` and `y`, and downstream stages *detect and exclude* non-finite
channels rather than having a `NaN` silently poison a feature vector or a model fit.

---

## 5. The three-phase sniff protocol

A **sniff** is one `baseline → exposure → purge` session
([`record.py`](../src/sniffsniff/record.py), [`capture.py`](../src/sniffsniff/capture.py)):

| Phase | Airflow | Purpose |
|---|---|---|
| **baseline** | fresh air | measure the clean-air R₀ for *this* sniff |
| **exposure** | sample straw open | present the odor; capture the rising response |
| **purge** | fresh air | let the sensors recover toward baseline |

**Phase slices.** For a session of `n_frames` frames at `scan_hz`, `phase_slices()`
computes half-open `(start, end)` index ranges, all clamped to `[0, n_frames]` so a
short/truncated session never indexes out of range:

```
n_base = round(baseline_s · scan_hz);  n_exp = round(exposure_s · scan_hz)
baseline = [0, n_base)
exposure = [n_base, n_base + n_exp)
purge    = [n_base + n_exp, n_frames)
plateau  = [max(exp_start, exp_end − round(plateau_s·scan_hz)), exp_end)   # trailing slice of exposure
```

**Per-sniff R₀** = mean Rs over the baseline slice (per channel). **Baseline-quality
gate:** `baseline_cv()` computes the coefficient of variation (std/mean) per channel over
the baseline window; if any *finite* channel exceeds `max_cv = 0.05`, the sniff is
**flagged with a warning** (not rejected) so the operator can re-baseline. Non-finite CVs
(open/rail channels) are a different fault and are excluded from that check.

**Persistence.** `SniffRecorder.save()` writes `data/<label>/<label>_<NNNN>.npz`
(compressed: `raw`, `t_ms`, `rs`, `r0`, `fractional`, `features`, plus a JSON `meta`
blob with the phase slices, feature names, and a full config snapshot) and appends a row
to `data/manifest.csv` (`id,label,path,n_samples`). IDs are a deterministic per-label
sequence, so repeated records increment `label_0000`, `label_0001`, …

---

## 6. Feature engineering — the 48-D vector

Each sniff becomes a fixed **N × 8 = 48-D** feature vector
([`features.py`](../src/sniffsniff/features.py)), sensor-major (all 8 features for sensor
0, then sensor 1, …). The 8 per-sensor features follow the UCI Gas-Sensor-Array
feature design — **2 steady-state + 6 transient** — computed from the fractional curve
`y`:

**Steady-state (2):**
- `peak` = `max |y|` over the **exposure** window — the response amplitude.
- `plateau_mean` = `mean y` over the **plateau** window (trailing `plateau_s` of exposure)
  — the settled response level.

**Transient (6):** exponential moving averages of the *frame-to-frame difference*
`dy[k] = y[k] − y[k−1]` (i.e. the response *rate*), at **three time constants**
`α ∈ {0.1, 0.01, 0.001}`:

```
ema[0] = 0;   ema[k] = (1−α)·ema[k−1] + α·dy[k]

ema_rise_aα  = max( ema(dy, α) over exposure )     # fastest rise rate at scale α
ema_decay_aα = min( ema(dy, α) over purge )        # steepest decay rate at scale α
```

Different odors have different rise/decay *kinetics*, and sampling the derivative at
three widely-spaced smoothing scales captures fast transients (α=0.1) through slow trends
(α=0.001). This is what lets the classifier separate odors that reach a similar *peak*
but get there differently. Feature names are `"<sensor>__<base>"`, e.g.
`MQ135__ema_rise_a1`.

---

## 7. The adaptive, single-connection capture engine

The naïve approach — open a fresh serial connection per sniff — fails on real hardware:
**opening the port toggles DTR, which resets the Uno**, so every sniff would pay a ~2 s
boot gap and the live view would die between captures. The
[`MonitorEngine`](../src/sniffsniff/monitor.py) instead runs **one continuous frame
loop** over a single connection: every frame updates the live view, a capture is just a
*window* over that same stream, and after a sniff the same stream feeds the recovery
teller.

`MonitorEngine` is a **pure frame-by-frame state machine** — feed it one `(t_ms, raw)`
frame via `step()` and it returns an event dict (`rs`, `phase`, `phase_changed`,
`capture`, `saved`, `recovery`, `settle`, `plateau`). It owns no threads and no UI, so it
is fully deterministic and unit-testable; the TUI worker just pulls frames from a source,
calls `step()`, and renders the events.

Per-sniff phase progression:

| Phase | Length | Ends when | Airflow |
|---|---|---|---|
| **settle** | variable | the array is *at rest* (`StabilityMonitor`), or a timeout | fresh |
| **baseline** | fixed `baseline_s` | frame count reached → measure R₀ | fresh |
| **exposure** | **dynamic** | the response *stops growing* (`ResponsePlateauMonitor`), capped by `exposure_s` | **sample** |
| **purge** | fixed `purge_s` | frame count reached → arm `RecoveryMonitor` | fresh |

**Engine-owned airflow.** The servo position is intrinsic to the protocol, so the engine
owns it: `set_airflow(fn)` wires an `S<angle>` sink and the engine commands the **sample**
angle on entering exposure and the **fresh** angle otherwise. The same mechanism drives
the real servo *and* the simulator (§10), so sim and hardware run the identical path.

### 8. The three detectors ([`recovery.py`](../src/sniffsniff/recovery.py))

#### 8.1 `StabilityMonitor` — the settle gate (self-referential flatness)

Before measuring R₀ you want the array genuinely *at rest*, but there is no prior
baseline to compare against (this must work for the very first sniff). So it asks *"is Rs
flat right now?"*: over a rolling `hold_s` window it checks every channel stays within
`±recover_tol` of the window mean; when the whole window is flat it reports `settled`. A
`max_wait_s` cap makes it fall through (flagged `timed_out`) rather than wait forever on a
sensor that never fully settles. Optional EMA smoothing (`smooth_alpha`) keeps per-frame
noise from resetting the window.

#### 8.2 `RecoveryMonitor` — return-to-baseline after a sniff

After purge, the sensors are still elevated and drift back slowly (τ_decay ≫ τ_rise).
`RecoveryMonitor` watches the live `Rs/R0` against **that sniff's R₀** and reports
`recovered` once every channel has held within `±recover_tol` for a sustained `hold_s`.
Dead/rail channels are ignored so one open channel can't block or fake recovery. This is
what gates the **recovery-gated auto-reps**: the next rep won't start until the array has
returned to rest.

#### 8.3 `ResponsePlateauMonitor` — growth-gated dynamic exposure

This is the crux of the adaptive capture, and it exists because of a real bug found on
hardware (§11). The problem: **a coarse flatness test cannot tell a slow rise from a
plateau.** Fresh milk rises at only ~0.2 %/s — a rate *buried in sensor noise* — so a
"flat within ±2 % for 3 s" test reads it as already-plateaued and ends exposure after a
few seconds, capturing a fraction of the response.

The fix gates on **growth of the aggregate response magnitude**, not flatness:

```
m = mean over channels of |Rs/R0 − 1|          # one robust scalar; per-channel noise averages out
track running peak of (EMA-smoothed) m
  · if m exceeds peak + eps        → new high  → reset the "held" timer
  · else                           → held += 1 frame
plateaued  ⟺  elapsed ≥ min_exposure_s   AND   held ≥ plateau_hold_s
```

Intuitively: **keep exposing while the response is still setting new highs; stop once it
has stopped growing for `plateau_hold_s`** — never before the `min_exposure_s` floor, and
always capped by `exposure_s`. Defaults (tuned on the real rig): `min_exposure_s = 15`,
`plateau_hold_s = 8`, `plateau_eps = 0.005` (0.5 pp), which sits ~20× above the measured
magnitude noise floor. The effective slope threshold is `eps / plateau_hold_s`. A
still-rising weak odor therefore rides to the cap (captures everything); a strong odor
that genuinely flattens stops early. The floor is clamped to the cap so a config with
`exposure_s < min_exposure_s` can't silently disable the gate.

---

## 9. The machine-learning pipeline

[`model.py`](../src/sniffsniff/model.py) — `SmellModel` is `StandardScaler → PCA →
classifier`, with a **decoupled** novelty detector living in the same PCA space.

```
48-D features ──StandardScaler──► z-scored ──PCA(k=5)──► working space ℝ^k
                                                          ├─► classifier  (kNN | SVM | RF | LDA)
                                                          ├─► per-class Mahalanobis novelty
                                                          └─► first 2 PCs → 2-D "smell map" (visual only)
```

**Standardize → PCA.** Each of the 48 features is z-scored, then PCA reduces to
`k = n_components` (default **5**), clamped to `min(k, n_features, n_samples − 1)` so it
is always valid on tiny datasets. Crucially, the **classifier and novelty work in the
full 5-D space**, while only the **first 2 PCs** are used for the human-facing map —
early experiments showed 2 PCs captured too little variance to classify reliably, so the
map (2-D) and the working space (5-D) were deliberately decoupled.

**Classifiers** (`--classifier`): `knn` (k = min(3, n), default), `svm` (RBF,
`probability=True`), `rf` (200 trees, seeded), `lda`.

**Mahalanobis novelty — "is this an odor I've never seen?"** For each known class `c`,
in PCA space, the model stores a centroid `μ_c` and an inverse covariance `Σ_c⁻¹` (via
Moore–Penrose `pinv`); a class with too few members (`n_c < k+1`) falls back to the
**pooled** covariance so it is never singular. The novelty score of a sample `x` is the
**minimum** Mahalanobis distance to any class:

```
D_c(x) = sqrt( (x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c) )        novelty(x) = min_c D_c(x)
is_novel  ⟺  novelty(x) > τ,   τ = sqrt( χ²_ppf(α, df=k) ),  α = 0.975
```

So a sample that is far (in standardized, correlation-aware distance) from *every* known
cluster is reported **novel** instead of being force-fit to the nearest label — the
"unknown smell" case.

**Honest accuracy on tiny data.** `cross_val_accuracy()` runs the *whole* pipeline inside
each CV fold (no leakage): **GroupKFold** keyed on sniff `ids` when groups are available
(so repeated sniffs of one sample can't leak across folds), `StratifiedKFold` otherwise,
falling back to `LeaveOneOut` when a class has < 3 samples. PCA components are re-clamped
to the smallest fold's training size. On the real 3-class set (alcohol / vinegar /
fresh_milk) this reports **~0.93 ± 0.13** cross-validated accuracy with kNN (§11).

---

## 10. Smell map and geometry serialization (the bridge to the LLM)

[`geometry.py`](../src/sniffsniff/geometry.py) turns the *learned model* — not the raw
signals — into a compact JSON blob that a language model can reason over
(`serialize_geometry(model, new_sample=x)`):

- `pca`: map vs working component counts, explained-variance ratios.
- `axis_interpretation`: for each map axis, the top-3 features by absolute PCA loading,
  signed — e.g. `"PC1": "MQ3__peak(+), MQ2__peak(+), MQ135__peak(−)"` — so the LLM can
  say *"this sample is pushed along the ethanol axis."*
- `known_clusters`: each class's 2-D centroid, radius, count, and (for a new sample) its
  2-D Euclidean distance.
- `new_sample`: 2-D coordinates, the classifier's predicted label + probability, and the
  top-3 most extreme **z-scored** features.
- `novelty`: the per-class Mahalanobis in the **full k-D** space, the nearest class, the
  threshold, and `is_novel`.

[`smellmap.py`](../src/sniffsniff/smellmap.py) renders the same 2-D projection as a PNG
(class scatter + centroid "X" markers + a gold "★" for a new sample) for the human.

---

## 11. The LLM reasoning layer (M3)

[`llm.py`](../src/sniffsniff/llm.py) + [`reason.py`](../src/sniffsniff/reason.py) — the
part with no precedent in the e-nose literature.

- **Client:** a thin, **stdlib-only** OpenRouter client (`urllib`, no `requests`/SDK).
  Default model `moonshotai/kimi-k2-thinking`; response parsing falls back to the
  message's `reasoning` field when `content` is empty (accommodating thinking models).
  The `transport` callable is injectable, so tests never touch the network.
- **What the LLM sees:** *only the geometry JSON from §10* — never raw signals. The
  system prompt tells it to (1) name the most likely odor, (2) **ground its confidence in
  the numbers** — centroid distance vs. radius, novelty vs. threshold, classifier
  probability — (3) interpret the sample's position via the axis interpretations, (4)
  judge novelty honestly (report "unfamiliar" rather than forcing a label when the
  Mahalanobis score exceeds threshold), and (5) recommend a next action (commit /
  re-sniff / adjust). "Be concise and concrete. Cite the numbers you reason from."
- **Security:** the API key is read **only** from `OPENROUTER_API_KEY` (or an explicit
  param), placed solely in the `Authorization` header, and **never logged, returned, or
  embedded in exception text**; error messages are sanitized.

---

## 12. The simulator — hardware-free development and CI

The simulator ([`simulator.py`](../src/sniffsniff/simulator.py)) is a first-class
deliverable: the entire Python pipeline runs, is tested, and demos with **no hardware**,
behind the identical `frames()` / `write_command()` interface as the real port.

- **`ODOR_PROFILES`**: per-odor multiplicative gains keyed by sensor *name*
  (e.g. `coffee = {MQ2:0.9, MQ3:0.6, MQ135:0.7, …}`, plus `vinegar`, `alcohol`,
  `fresh_milk`, `spoiled_milk`, `clean_air`). Keying by name (not channel index) makes
  the simulator independent of wiring order.
- **Physical model**: a first-order relaxation. On exposure the target is
  `r_base / (1 + gain)` (reducing gas lowers Rs) approached with `τ_rise = 5 s`; on purge
  it relaxes back to `r_base` with `τ_decay = 20 s`. It then **inverts the real
  calibration chain** (Rs → V_RL → counts, + seeded Gaussian noise) so the genuine
  `calibrate` math is exercised.
- **`ContinuousSim`** (in [`monitor.py`](../src/sniffsniff/monitor.py)): a *servo-driven*
  variant that keeps continuous state and reacts to the engine's `S<angle>` commands
  exactly like the rig — presenting the odor while the servo is at the sample angle. This
  is what lets the adaptive dynamic-exposure logic behave identically in sim and on
  hardware.

---

## 13. Interfaces: CLI and guided-training TUI

- **CLI** ([`cli.py`](../src/sniffsniff/cli.py)): `stream`, `record --label <name>`,
  `simulate`, `fit`, `identify`, `servo`, plus the `tui` launcher.
- **TUI** ([`tui/`](../src/sniffsniff/tui)): a Textual app with an animated ASCII nose,
  a **continuous** live sensor-bar view, and a guided workflow — record (with
  **recovery-gated auto-reps**), fit, identify, "think" (LLM narrative), plus delete/clear
  and a smell-map view. The `SniffController` holds the non-UI logic; the app drives the
  `MonitorEngine` on a background thread and renders its events (settle progress, live
  exposure response %, recovery teller). It is headless-testable via Textual's `Pilot`.

---

## 14. Research foundations

sniffsniff is a faithful implementation of four decades of electronic-nose methodology,
with one novel layer on top:

- **Sensor-array + pattern-recognition olfaction** — Persaud & Dodd (1982) established
  that selectivity comes from an *array of broadly-tuned sensors* plus pattern
  recognition, not from one specific sensor; Gardner & Bartlett formalized the "electronic
  nose." §2.3 is a direct application.
- **Transient + steady-state feature engineering** — the 2 steady-state + 6 EMA-transient
  per-sensor features (§6) follow the design used with the UCI **Gas Sensor Array Drift**
  dataset (Vergara et al., 2012), which showed that transient dynamics across multiple
  time scales carry much of the discriminative signal.
- **PCA + simple classifiers** — dimensionality reduction followed by kNN/LDA/SVM is the
  standard e-nose discrimination recipe (§9).
- **Mahalanobis novelty detection** — per-class Mahalanobis distance with a χ²-based
  threshold (Mahalanobis, 1936) is the classical statistical test for "does this belong
  to any known distribution?" (§9), giving principled unknown-odor rejection.
- **Drift mitigation** — per-sniff R₀ re-baselining (§2.2) is the standard first-line
  defense against MOS baseline drift.
- **Honest small-data evaluation** — grouped / stratified / leave-one-out cross-validation
  (§9) follows standard CV practice (Kohavi, 1995) for the tiny datasets a hobby rig
  produces.
- **The LLM reasoning layer (novel)** — reasoning over the *learned geometry as JSON* to
  produce a grounded natural-language verdict has no precedent in the e-nose literature;
  it is the project's original contribution (§11).

*(References are the standard, well-established works in the field; they situate the
design rather than being cited for specific numeric claims.)*

---

## 15. Empirical notes from the real rig

- **Milk kinetics.** Fresh-milk headspace produces a **slow, weak, still-rising** response
  over 45–60 s (it does not cleanly plateau); on clean sensors the aggregate response
  climbed from ~1 % to ~22 % over a full 45 s window (MQ7 alone reached ~60–98 %). This is
  exactly the regime that broke the old fixed/flatness exposure and drove the growth-gated
  redesign (§8.3). Note some channels' Rs *rose* under milk (humidity/oxidizing effects),
  which is why features are magnitude-based.
- **Recovery is slow.** After a strong milk exposure the array can take a minute or more
  to return to baseline — the recovery/settle gates (§8.1–8.2) exist precisely to prevent
  a contaminated R₀ on the next sniff.
- **Effective sample rate ≈ 16 Hz** on the clone-Uno (vs. the config's nominal 20 Hz), so
  time-based windows run ~25 % longer in wall-clock than their labels. Conservative, but
  factor it in when interpreting durations.
- **A real 3-class model** (alcohol / vinegar / fresh_milk) trained end-to-end on the rig
  reports **~0.93 ± 0.13** cross-validated accuracy (kNN, grouped CV). A live identify
  sniff after retraining predicted `fresh_milk` at **p = 1.00** with novelty 0.30 (well
  below threshold — not flagged novel), confirming the full record → fit → identify loop.
- **Two confusion classes, opposite fixes (6-class rig data — verified by testing each pair
  in full 48-D vs PCA-5D):** *peppermint ↔ cilantro* is a **representation** artifact — the
  MQ3 (alcohol) axis separates them perfectly (peppermint's menthol is a terpene alcohol),
  but the unsupervised PCA-5D diluted it. Grafting the raw MQ3 features onto the classifier
  space (`SmellModel.augment_features`, novelty/map stay pure-PCA) lifts grouped-CV **0.80 →
  0.90** at zero cost to other pairs. *fresh_milk ↔ cinnamon* is **structural** — no software
  feature (raw, ratio, or kinetic) separates them in honest CV (both near-identical on this
  array); breaking it needs a new sensing dimension (an aldehyde-selective sensor such as
  **MQ-9** — cinnamaldehyde drives it, milk does not). Chemistry-proposed ratio features were
  tested and **refuted** (2-class small-sample artifacts that hurt honest 6-class CV).

---

## 16. Testing and reproducibility

- **~316 tests** (pytest) cover calibration, features, the 3-phase protocol, the ML
  pipeline and novelty, the monitor engine and all three detectors, the servo-driven
  simulator, the LLM client (with an injected fake transport), and the TUI (headless via
  `Pilot`).
- The **simulator runs the whole pipeline in CI** with no hardware, behind the same
  interface as the real port, so every stage is exercised deterministically (seeded RNG).
- A **real-milk regression fixture** ([`tests/data/milk_exposure.npz`](../tests/data/milk_exposure.npz))
  replays the actual captured milk curve to guarantee the dynamic-exposure gate never
  truncates a slow rise again.

---

## 17. Module map

| Module | Responsibility |
|---|---|
| [`config.py`](../src/sniffsniff/config.py) | immutable `Config` from `sniffsniff.toml`; `N` flows from the channel table |
| [`serialio.py`](../src/sniffsniff/serialio.py) | robust CSV serial reader (lazy open, reconnect/backoff, `write_command`) |
| [`servo.py`](../src/sniffsniff/servo.py) | `ServoLink` — low-level `S<angle>` airflow control |
| [`calibrate.py`](../src/sniffsniff/calibrate.py) | pure counts → V_RL → Rs → ratio → fractional |
| [`capture.py`](../src/sniffsniff/capture.py) | bounded single-session capture + phase cues (legacy fixed path) |
| [`record.py`](../src/sniffsniff/record.py) | 3-phase protocol, per-sniff R₀, CV gate, `.npz`/manifest persistence |
| [`features.py`](../src/sniffsniff/features.py) | the 48-D (N×8) feature vector |
| [`dataset.py`](../src/sniffsniff/dataset.py) | load/simulate labeled datasets for training |
| [`model.py`](../src/sniffsniff/model.py) | `SmellModel` — scaler→PCA→classifier + Mahalanobis novelty + CV |
| [`geometry.py`](../src/sniffsniff/geometry.py) | serialize the learned geometry to JSON for the LLM |
| [`smellmap.py`](../src/sniffsniff/smellmap.py) | render the 2-D smell map PNG |
| [`monitor.py`](../src/sniffsniff/monitor.py) | `MonitorEngine` state machine + servo-driven `ContinuousSim` |
| [`recovery.py`](../src/sniffsniff/recovery.py) | `StabilityMonitor`, `ResponsePlateauMonitor`, `RecoveryMonitor` |
| [`simulator.py`](../src/sniffsniff/simulator.py) | odor profiles + physical MQ-array simulator |
| [`llm.py`](../src/sniffsniff/llm.py) | stdlib OpenRouter client |
| [`reason.py`](../src/sniffsniff/reason.py) | geometry → prompt → narrative verdict |
| [`tui/`](../src/sniffsniff/tui) | Textual guided-training UI + `SniffController` |
