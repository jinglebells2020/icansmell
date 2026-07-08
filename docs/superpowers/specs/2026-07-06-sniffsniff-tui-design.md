# sniffsniff — TUI Control Center — Design

**Date:** 2026-07-06
**Status:** Approved (design), building
**Component:** `sniffsniff tui` — a full-screen terminal app to run the whole workflow

## Purpose

A single terminal cockpit that walks you through the e-nose workflow correctly:
**connect → watch live sensors → record labeled sniffs (guided) → train → identify**,
with a live view of the serial input, a workflow checklist so you don't skip steps,
and an animated nose that breathes when idle and sniffs during exposure.

Built on **[Textual](https://textual.textualize.io/)** (async widgets, styling,
animation, headless-testable via `Pilot`). Reuses the entire existing pipeline —
the TUI is an orchestration layer, not new signal logic.

## Layout

```
┌ sniffsniff 👃  ── SIM ▸ or /dev/cu.usbmodem101 ─────────────────────────────┐
│ ┌───────────────┐  ┌───────────────────────────────────────────────────┐   │
│ │   ANIMATED    │  │ LIVE SENSORS            phase: BASELINE  12.3s      │   │
│ │     NOSE      │  │ MQ3   ███████░░░░░░  41.2 kΩ                        │   │
│ │  (breathe /   │  │ MQ135 ██████████░░  58.9 kΩ   ⚠ noisy              │   │
│ │   sniff)      │  │ MQ2   █████░░░░░░░  22.1 kΩ                        │   │
│ │               │  │ MQ4 … MQ8 … MQ7 …                                  │   │
│ ├───────────────┤  └───────────────────────────────────────────────────┘   │
│ │ WORKFLOW      │  ┌ LOG ──────────────────────────────────────────────┐   │
│ │ ✓ connected   │  │ recorded coffee_0007  (48-D)                       │   │
│ │ coffee    8   │  │ fit: cross-val 0.94 ± 0.05 (5 classes, n=40)       │   │
│ │ vinegar   8   │  │ identify: coffee (0.92)   novelty 1.8  not novel   │   │
│ │ alcohol   6   │  │ …                                                  │   │
│ │ model: ✗      │  └───────────────────────────────────────────────────┘   │
│ └───────────────┘                                                           │
└ (c)onnect (s)im  (l)abel  (r)ecord  (f)it  (m)ap  (i)dentify  (q)uit ────────┘
```

## Modules

- **`tui/controller.py` — `SniffController`** (no Textual imports; fully unit-testable).
  Wraps the pipeline: build a reader (sim or real), `stream`, `record_one` /
  `record_many`, `fit`, `identify`, `dataset_counts`. Callbacks (`on_phase`,
  `on_frame`, `on_saved`) let the UI react; a `should_stop` predicate bounds live
  streaming. All heavy work is plain Python so tests never touch the UI.
- **`tui/nose.py` — `NoseWidget`.** An ASCII/Unicode nose with two states: `idle`
  (slow breathe) and `sniffing` (fast, with air-intake marks). Cycles frames on an
  interval timer; `set_state("sniffing"|"idle")` switches. Pure frame list → testable.
- **`tui/widgets.py`.** `SensorBars` (one live bar per sensor, value + noisy flag),
  `WorkflowPanel` (connection ✓, per-class sniff counts, model fitted ✗/✓),
  `LogPanel` (scrolling messages/results).
- **`tui/app.py` — `SniffApp`.** Composes the layout, key bindings, and runs
  controller actions in **thread workers** (`@work(thread=True)`), marshaling frame
  updates to the widgets via `app.call_from_thread`. The nose switches to `sniffing`
  on the exposure phase.
- **`cli.py`:** a `tui` subcommand (`sniffsniff tui [--sim] [--port P] [--config P]
  [--out DIR] [--model M]`) that launches `SniffApp`.

## Key bindings

`c` connect · `s` toggle sim/real · `l` cycle/enter label · `r` record (uses `--reps`
via an input) · `f` fit (train on the collected dataset) · `m` render map PNG · `i`
identify one sniff · `q` quit. All actions are non-blocking (worker threads); the UI
and nose keep animating during a capture.

## Concurrency

One reader is active at a time. Idle+connected → a `stream` worker feeds `SensorBars`
(nose idle). `record`/`identify` → a capture worker runs `capture.capture_session`
(bars update via `on_frame`, nose `sniffing` during exposure), saves, then idle stream
resumes. Frame → widget updates always go through `call_from_thread` (thread-safe).

## Testing

- **`test_tui_controller.py`** (the important one): with `--sim`, `record_one` writes
  an `.npz`; `record_many(reps=3)` writes 3 with incrementing ids; `dataset_counts`
  reflects them; `fit` returns a model + cross-val accuracy and saves it; `identify`
  returns a prediction+novelty dict; `on_phase` fires baseline→exposure→purge.
- **`test_tui_nose.py`**: `NoseWidget` frame list non-empty; `set_state` switches
  state; advancing the animation cycles frames deterministically.
- **`test_tui_app.py`** (headless `Pilot`): app mounts with the nose + sensor bars +
  workflow + log present; pressing `r` in sim mode records a sniff (dataset count
  goes up / log line appears); `q` exits. Uses `async with app.run_test()`.

## Dependencies

Add **`textual`** as a `[tui]` optional extra (pulls `rich`). Everything else reused.
The core library and other CLI commands keep working without `textual` installed
(the `tui` subcommand imports it lazily and prints an install hint if missing).

## Out of scope

Live embedded PCA scatter inside the TUI (the `map` command already renders a PNG);
mouse-drag plots; remote/streaming dashboards. Milestone 3 (LLM reasoner) is separate.
