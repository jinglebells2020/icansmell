# Dual-Uno 9-sensor support — design

**Date:** 2026-07-08
**Goal:** Add a second Arduino Uno so the array grows from 6 to **9 sensors across two
boards**, merged equally into one 9-channel frame that flows through the existing
calibrate → features → recorder → engine pipeline unchanged.

## Hardware

| Uno 1 (6 ch, has the airflow servo) | Uno 2 (3 ch) |
|---|---|
| A0 MQ-5 · A1 MQ-3 · A2 MQ-135 · A3 MQ-7 · A4 MQ-9 · A5 MQ-8 | A0 MQ-2 · A1 MQ-4 · A2 MQ-6 |

Combined flat channel order (board order): `MQ5 MQ3 MQ135 MQ7 MQ9 MQ8 MQ2 MQ4 MQ6` → N=9.

## Key insight

Everything downstream of ingest already derives `N` from config (`calibrate`,
`features`, `record`, `simulator`, `monitor`), so a second board only touches **config**
(describe two boards → one flat 9-ch array) and **ingest** (merge two serial streams).
Feature vectors auto-grow 48 → 72. No changes to the calibration/feature math.

## 1. Config — dual-board (`config.py`, `sniffsniff.toml`)

New `Board` dataclass: `port: str | None`, `n_channels: int`, `servo: bool`, `start: int`
(offset of this board's first channel in the flat vector). `Config` gains
`boards: tuple[Board, ...]`; `Channel` gains `board: int`. `n_channels`, `sensor_names()`,
`rl_array()` stay flat and unchanged.

TOML shape (array-of-tables), `ch` is the local A-pin per board:
```toml
[[array.board]]                      # Uno 1
port = "/dev/cu.usbmodemXXXX"
servo = true
channels = [ {ch=0,sensor="MQ5",rl=1000}, ... 6 rows ... ]

[[array.board]]                      # Uno 2
port = "/dev/cu.usbmodemYYYY"
channels = [ {ch=0,sensor="MQ2",rl=1000}, {ch=1,sensor="MQ4",rl=1000}, {ch=2,sensor="MQ6",rl=1000} ]
```
The loader flattens boards in order, assigning global `ch` 0..8 and per-board `start`.
Validation: each board's local `ch` set must be `{0..nb-1}`.

**Backward compatibility (critical):** a legacy flat `[array].channels` table (today's
file, no `[[array.board]]`) still parses as a single board (`port=None`, filled from the
CLI `--port`; `servo` from legacy `[servo].enabled`). `default_config()` stays the 6-ch
single board, so **every existing test is untouched**. The shipped `sniffsniff.toml` is
updated to the 9-ch dual-board rig.

## 2. Ingest — `MergedReader` (new, `serialio.py`)

Symmetric merge, both Unos peers:
- One `SerialReader` per board (reuses parse/reconnect/`opener` seam), each pumped by its
  own daemon thread that keeps only its *latest* `(t_ms, raw)` under a lock.
- A host-clocked emit loop yields the combined frame at `config.scan_hz`:
  `frame k → (k·step_ms, concat(latest board0 counts, latest board1 counts))`, width 9.
  Host clock because the two Unos have unrelated `millis()` epochs.
- **Startup:** first emit waits until *every* board has produced ≥1 frame (bounded by a
  timeout; on timeout it raises so the caller reports "no data").
- **Hold-last:** a board that stalls keeps its last values (≤50 ms staleness, negligible
  vs. seconds-slow gas dynamics).
- `write_command()` routes to the board whose `servo=true` (Uno 1); `close()` stops the
  threads and closes both readers.
- Presents the same `frames()`/`close()`/`write_command()` interface as `SerialReader`, so
  the `MonitorEngine` and controller are unchanged.

A pure `merge_counts(latest, widths)` helper (concatenate boards, zero-fill a not-yet-seen
board) is unit-tested directly; the threaded reader is integration-tested with
constant-value fake readers so timing/interleaving doesn't make tests flaky.

## 3. Reader construction — `build_reader(config, ...)`

A small factory (in `serialio.py`) returns a plain `SerialReader` for a single-board
config or a `MergedReader` for multi-board. The controller's real-capture path and the TUI
`_build_source` both call it. **Sim mode is unaffected** — `Simulator`/`ContinuousSim`
already emit `N=9` channels directly from config, so no merge is needed there.

## 4. Firmware — two sketches

- `firmware/sniffsniff_uno/` — 6 ch + servo (existing sketch; wiring header updated to the
  Uno-1 order MQ5/3/135/7/9/8).
- `firmware/sniffsniff_uno_b/` — new: 3 ch, **no servo** (`NCH=3`, `PINS={A0,A1,A2}`, servo
  code removed; header maps A0→MQ2, A1→MQ4, A2→MQ6).

## 5. Simulator — new-sensor odor gains (`simulator.py`)

`ODOR_PROFILES` gets chemistry-plausible gains for the three new sensors so they aren't
dead in sim: **MQ-5 / MQ-6** (LPG/methane — track the dairy/fermentation axis near MQ-4),
**MQ-9** (CO + combustibles — near MQ-7/MQ-2). Existing sensors' gains are unchanged, so
the current 6-ch sim behaviour and its tests are preserved.

## 6. Servo / airflow

One physical airflow path. `Config` exposes `servo_enabled = any(board.servo)`; the engine
still calls `source.write_command`, which `MergedReader` routes to the servo board. Legacy
single-board behaviour is identical.

## 7. Testing

- Config: dual-board TOML → 9 flat channels in the right order; local-`ch` validation;
  legacy flat config still parses; `default_config()` still 6 ch.
- `merge_counts` pure helper: order, widths, zero-fill.
- `MergedReader` with fake dual streams: 9-wide output, board slices correct, hold-last
  when one stalls, startup raises when a board never streams.
- Simulator: 9-ch separability incl. the new sensors; existing 6-ch determinism intact.
- End-to-end: a 9-ch sim record produces a 72-D feature vector and round-trips.
- Full existing suite (328) stays green.

## Non-goals / caveats

- A `model.joblib` trained on 6-ch (48-D) data won't load against 9-ch (72-D) vectors —
  re-record + re-fit on the new rig (expected).
- No tight cross-board time sync (hold-last is sufficient for this signal).
