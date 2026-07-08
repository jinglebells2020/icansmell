# sniffsniff TUI — modern-dashboard redesign

**Date:** 2026-07-08
**Goal:** Polish-led UI/UX overhaul of the Textual e-nose console, for demos/showcase.
**Aesthetic:** Modern dashboard — titled bordered cards, muted palette + one accent,
status pills, per-sensor sparkline + bar + trend. The animated ASCII nose is removed.

## Decisions (from brainstorming)

- **Direction:** Modern dashboard, visual-polish-led, audience = demo/showcase.
- **Sensors:** sparkline (rolling history) + magnitude bar + rising/flat/falling trend arrow.
- **Nose mascot:** dropped entirely (module, widget, tests).
- **Architecture preserved:** every panel stays a *pure render helper* (unit-testable,
  no running App) behind a thin `Static`/`RichLog` widget. This is the repo's core
  pattern and the redesign keeps it — only the render contracts and layout change.

## Layout

```
┌ HeaderBar ─ 👃 sniffsniff · e-nose        ● sim   model ✓   ◉ recording ┐
├───────────────────────────────┬────────────────────────────────────────┤
│ SENSORS (3fr)                 │ COACH (2fr)                             │
│  MQ3  ▁▂▄▆█▇▅ ██████░░ 41.2k ▲│  press f to train …                     │
│  …                            │  sim · label coffee · reps 1 · clf knn  │
│                               ├─────────────────────────────────────────┤
│ CAPTURE                       │ LABELS                                  │
│  ✓settle → ✓baseline →        │  ▸ coffee   ●●● 3 ✓                      │
│    ▐exposure▌ → ○purge        │    vinegar  ●●○ 2                        │
│  ████████████░░░ 78%          │    alcohol  ○○○ 0                        │
│  ⏺ 'coffee' exposure 4s …     │  …                                      │
├───────────────────────────────┴────────────────────────────────────────┤
│ LOG                                                                      │
│  ready — label 'coffee', sim …                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ r Rec  n/p Label  a Add  f Fit  i Id  t Think  m Map  s Sim  q Quit  (Footer)│
└──────────────────────────────────────────────────────────────────────────┘
```

Left column (3fr) = live signal: **SENSORS** + **CAPTURE** progress.
Right column (2fr) = guidance: **COACH** (next step + state) + **LABELS** (dot-meter counts).
**LOG** spans full width below; Textual **Footer** shows the keymap. A slim **HeaderBar**
tops it with the wordmark and status pills.

## Widgets & pure render helpers (`tui/widgets.py`)

| Widget | Helper | Renders |
|--------|--------|---------|
| `HeaderBar` | `render_header(source, connected, has_model, phase_label)` | wordmark + source/model/phase pills |
| `SensorBars` | `render_sensors(names, values, histories, vmax, noisy)` | one row/sensor: name · sparkline · bar · value · trend · ⚠ |
| `CapturePanel` (`id="status"`) | `render_capture(phase, frac, detail)` | 4-step stepper (settle→baseline→exposure→purge) + progress bar + detail line |
| `LabelList` | `render_label_list(labels, counts, current, good_reps)` | dot-meter (●/○) progress toward `good_reps` + count, current marked ▸ |
| `CoachPanel` | `render_coach(...)` *(signature unchanged)* | NEXT guidance + dim state line |
| `LogPanel` | — | scrolling `RichLog` (unchanged) |

New primitives: `sparkline(values, width)` (8-level ▁–█, normalized over the window),
`trend(values, deadband)` → `"rising"|"flat"|"falling"`, `_bar(frac, width)` (shared fill).
`bar_row(...)` keeps its exact contract (still a tested primitive). Rows carry Textual
markup (`[green]…[/]`) for inline color; helpers stay pure — tests assert on visible
substrings, which survive markup.

**Removed:** `render_sensor_bars`, `render_workflow`, `WorkflowPanel` (its info — connected,
model, counts — now lives in the HeaderBar pills and the LABELS dot-meters), and the whole
`tui/nose.py`.

## App wiring (`tui/app.py`)

- New `compose()` layout above; `#status` id retained on `CapturePanel` (tests query it).
- `on_mount`: set `theme = "nord"`, set each card's `border_title`, seed zeroed sensors,
  init header, start monitor. No nose init.
- `_on_engine_event`: update sensors (keeps rolling history), then fold the settle /
  capture / exposure / recovery branches into a single `(step_phase, frac, detail)` and push
  it to the `CapturePanel`; also update the HeaderBar phase pill. `_status`/`_set_nose`
  removed. Recovery detail keeps the word "recover" (regression-tested).
- `_refresh_all` = labels + coach + header (workflow refresh removed).

## Styling

Textual `nord` theme; cards use `border: round` with `border-title-align: left` and design
tokens (`$primary`/`$accent`/`$success`/`$panel`). Trend/pill/stepper colors via inline
markup so partial-line coloring works (CSS can't color a substring).

## Tests

- Delete `tests/test_tui_nose.py`; drop `NoseWidget`/`WorkflowPanel` from
  `test_tui_app.py` and `test_tui_widgets.py`.
- Rewrite `test_tui_widgets.py`: keep `bar_row` tests as-is; add tests for `sparkline`,
  `trend`, `render_sensors`, `render_capture`, `render_header`, updated `render_label_list`,
  `render_coach`, and `test_widgets_construct` over the new widget set.
- All existing app/controller/fixes tests must stay green (behavior unchanged; only the
  view layer changed).

## Verification

`pytest -q` green, then a headless screenshot: mount the app under `run_test()`, feed frames
to populate sparklines and drive a capture into exposure, `save_screenshot(...)`, render the
SVG to PNG and eyeball the result — iterate on spacing/color until it reads as a polished
dashboard.
