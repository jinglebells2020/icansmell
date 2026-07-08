# sniffsniff — TUI v2: Guided Training Workflow — Design

**Date:** 2026-07-06
**Status:** Approved (design), building
**Component:** `sniffsniff tui` — make it *lead* the operator through training an ML model.

## Purpose

The v1 TUI exposed raw actions. v2 turns it into a **guided training console**: every
function needed to build a working smell model, plus a coach that always tells you the
next step, live per-label progress, and the ability to fix mistakes (delete a bad sniff,
clear and restart). Someone who's never used it should reach a trained model by just
following the on-screen guidance.

## The flow it guides

```
connect ──► pick a label ──► record N sniffs ──► switch label, repeat (≥2 labels)
   ──► [coach: "enough data — press f"] ──► fit ──► identify / map
```

A **coach line** is always visible and reflects state:
- not connected → "Connect a device, or press s for the simulator."
- <2 labels with data → "Record sniffs for another label. Pick a label (n/p), press r."
- some labels below target → "Collect more: coffee (2/3), vinegar (1/3). Press r."
- enough → "Enough data — press f to train."
- model trained → "Trained ✓ — press i to identify, m for the smell map."

## New/changed functions (every step covered)

| Key | Function |
|-----|----------|
| `r` | record the current label (`reps` sniffs), guided baseline→exposure→purge |
| `n` / `p` | select next / previous label |
| `a` | add a custom label (inline input) |
| `+` / `-` | increase / decrease `reps` per record |
| `x` | delete the **last** sniff of the current label (undo a bad capture) |
| `X` | clear the whole dataset (requires a confirm press) |
| `c` | cycle classifier: knn → svm → rf → lda |
| `f` | fit/train on the collected data (shows cross-val accuracy) |
| `i` | identify one sniff (needs a model) |
| `m` | render the 2-D smell-map PNG |
| `s` | toggle sim / real |
| `q` | quit |

## Layout

```
┌ sniffsniff 👃  SIM ▸ | label: coffee | reps: 3 | clf: knn ───────────────────┐
│  ┌─ nose ─┐  ┌─ LABELS ──────────┐  ┌─ SENSORS ───────────────────────────┐ │
│  │  (o)   │  │ ▸ coffee     3    │  │ phase: baseline  6.2s               │ │
│  │  breathe│ │   vinegar    2    │  │ MQ3   ██████░░░  5.2k               │ │
│  └────────┘  │   alcohol    0    │  │ …                                   │ │
│  ┌─ COACH ─┐  │   fresh_milk 0    │  └─────────────────────────────────────┘ │
│  │NEXT:    │  │   spoiled…   0    │  ┌─ LOG ───────────────────────────────┐ │
│  │collect  │  └───────────────────┘  │ saved coffee_0002                   │ │
│  │alcohol… │  model: ✗                │ cross-val 0.91 ± 0.06               │ │
│  └─────────┘                          └─────────────────────────────────────┘ │
└ r rec  n/p label  a add  +/- reps  c clf  x del  X clear  f fit  i id  m map ┘
```

## Module changes

### `record.py` (additive; dataset management)
```python
def delete_last_sniff(out_dir, label) -> Path | None
    # remove the most-recent sniff for `label`: delete its .npz and drop its manifest row.
    # most-recent = highest sequence id for that label. Returns the deleted path or None.
def clear_dataset(out_dir) -> int
    # delete every <label>/*.npz and manifest.csv under out_dir; return #sniffs removed.
```

### `tui/controller.py`
```python
DEFAULT_LABELS = ["coffee","vinegar","alcohol","fresh_milk","spoiled_milk"]
CLASSIFIERS = ["knn","svm","rf","lda"]
GOOD_REPS = 3           # per-label target the coach nudges toward
MIN_CLASSES = 2
SniffController:
    self.classifier: str = "knn"     # settable; fit() uses it
    def known_labels(self) -> list[str]        # DEFAULT_LABELS ∪ recorded, stable order
    def delete_last(self, label) -> Path | None
    def clear(self) -> int
    def cycle_classifier(self) -> str          # advance + return new
    def ready_to_fit(self) -> bool             # ≥ MIN_CLASSES labels with ≥1 sniff
    def next_step(self) -> str                 # the coach guidance string
    # fit() now defaults to self.classifier
```

### `tui/widgets.py`
- `LabelList` (Static): `update_labels(labels, counts, current)` → one row per label
  `"▸ coffee     3"` / `"  vinegar    0"`, current marked, count shown; pure `render_label_list()` helper for tests.
- `CoachPanel` (Static): `update_coach(next_step, connected, label, reps, classifier, has_model)`
  → a header status line + a wrapped `NEXT: …` line; pure `render_coach()` helper.

### `tui/app.py`
- New bindings (table above); `self.reps`, `self.controller.classifier`, `self.label` are
  live and shown in the header/coach. Every action refreshes the coach + label list.
- `a` (add label) uses a tiny `ModalScreen` with an `Input`; on submit the label becomes
  current (and joins the list). `X` needs a second `X` within the same focus to confirm
  (a `_clear_armed` flag), logging "press X again to confirm".
- All long actions keep the existing `_busy` guard and thread workers.

## Testing

- `test_record_dataset_ops.py`: `delete_last_sniff` removes the right npz + manifest row
  (and returns None on an empty/unknown label); `clear_dataset` empties the dir and returns
  the count; both round-trip against `SniffRecorder`-written data.
- `test_tui_controller.py` (extend): `delete_last`/`clear`/`cycle_classifier`/`ready_to_fit`/
  `next_step` transitions across the states above; `known_labels` includes defaults + recorded;
  `fit` uses `self.classifier`.
- `test_tui_widgets.py` (extend): `render_label_list` marks current + shows counts;
  `render_coach` contains the next-step text.
- `test_tui_app.py` (extend, headless Pilot): pressing `n`/`p` changes the current label;
  `+`/`-` changes reps; `c` cycles classifier; `x` after a sim record removes it (count → 0);
  the coach text updates after a record. Keep it deterministic (drive actions, await workers).

## Out of scope

Editing an existing sniff, multi-select label ops, in-TUI PCA scatter (the `m` PNG stays).
Milestone 3 (LLM reasoner) remains separate.
