"""The Textual app that wires the sniffsniff pipeline into an interactive TUI.

:class:`SniffApp` runs ONE persistent frame loop (a :class:`~sniffsniff.monitor.
MonitorEngine` in a worker thread): every frame updates the live sensor bars, a
record/identify is just a *window* over that same stream (no reopen, no Uno
reset), and after each sniff the bars keep flowing while a status line reports
return-to-baseline. Guided-training v2 (label list, coach, undo/clear) and the M3
`think` reasoner sit on top. All slow work runs in ``@work(thread=True)`` workers;
widgets are touched only via :meth:`textual.app.App.call_from_thread`.

``textual`` is an optional extra, imported lazily from the CLI.
"""
from __future__ import annotations

import time

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, Static

from .controller import SniffController
from .nose import NoseWidget
from .widgets import CoachPanel, LabelList, LogPanel, SensorBars, WorkflowPanel

__all__ = ["SniffApp", "run_tui"]


class _AddLabelScreen(ModalScreen[str]):
    """A tiny modal with an ``Input`` for adding a custom label."""

    CSS = """
    _AddLabelScreen { align: center middle; }
    #box { width: 50; height: auto; border: round $accent; padding: 1; background: $surface; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label("New label (Enter to add, Esc to cancel):")
            yield Input(placeholder="e.g. garlic", id="label_input")

    def on_mount(self) -> None:
        self.query_one("#label_input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def key_escape(self) -> None:
        self.dismiss("")


class SniffApp(App):
    """Interactive e-nose console over a :class:`SniffController`."""

    CSS = """
    #body { height: 1fr; }
    #left { width: 1fr; }
    #right { width: 2fr; }
    NoseWidget { height: auto; border: round $accent; padding: 1; }
    LabelList { height: 1fr; border: round $primary; padding: 1; }
    CoachPanel { height: auto; border: round $warning; padding: 1; }
    WorkflowPanel { height: auto; border: round $primary; padding: 1; }
    SensorBars { height: auto; border: round $secondary; padding: 1; }
    #status { height: auto; border: round $success; padding: 0 1; }
    LogPanel { height: 1fr; border: round $panel; }
    """

    BINDINGS = [
        ("r", "record", "Rec"),
        ("n", "next_label", "Next label"),
        ("p", "prev_label", "Prev label"),
        ("a", "add_label", "Add label"),
        ("plus", "more_reps", "+reps"),
        ("minus", "fewer_reps", "-reps"),
        ("c", "cycle_classifier", "Clf"),
        ("x", "delete_last", "Del"),
        ("X", "clear", "Clear"),
        ("f", "fit", "Fit"),
        ("i", "identify", "Identify"),
        ("t", "think", "Think"),
        ("m", "map", "Map"),
        ("s", "toggle_sim", "Sim/Real"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        controller: SniffController,
        *,
        reps: int = 1,
        label: str | None = None,
        source_factory=None,
        paced: bool | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.reps = reps
        self.label = label or controller.known_labels()[0]
        self._clear_armed = False       # first X arms, second X confirms
        self._busy = False              # a fit/map/delete/clear worker is running
        self._last_geometry = None      # geometry of the last identified sniff (for `t`)
        self._last_verdict = None
        # monitor-engine state
        self._engine = None
        self._source = None
        self._monitor_stop = False
        self._identify_pending = False
        self._active_label = None
        self._model = None              # cached loaded SmellModel (invalidated on fit)
        # recovery-gated auto-reps: `r` records `reps` sniffs, each after recovery
        self._rep_label = None
        self._reps_remaining = 0
        self._rep_total = 0
        # test hooks: inject a bounded frame source + disable real-time pacing
        self._source_factory = source_factory
        self._paced = paced

    # ------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield NoseWidget(id="nose")
                yield LabelList(id="labels")
                yield CoachPanel(id="coach")
                yield WorkflowPanel(id="workflow")
            with Vertical(id="right"):
                yield SensorBars(id="sensors")
                yield Static("", id="status")
                yield LogPanel(id="log")
        yield Footer()

    def on_mount(self) -> None:
        import numpy as np

        names = self.controller.config.sensor_names()
        self.query_one("#sensors", SensorBars).update_values(
            names, np.zeros(len(names)), phase="idle"
        )
        self._refresh_all()
        mode = "sim" if self.controller.use_sim else "real"
        self._log(f"ready — label '{self.label}', {mode}")
        if not self.controller.use_sim and not self.controller.connected:
            self._log(
                f"⚠ no device at {self.controller.port} — reconnect it, or press s for the simulator"
            )
        self._start_monitor()

    def on_unmount(self) -> None:
        self._monitor_stop = True

    # -------------------------------------------------------- ui helpers
    def _log(self, msg: str) -> None:
        self.query_one("#log", LogPanel).write_line(msg)

    def _status(self, msg: str) -> None:
        self.query_one("#status", Static).update(msg)

    def _refresh_workflow(self) -> None:
        ctrl = self.controller
        self.query_one("#workflow", WorkflowPanel).update_state(
            ctrl.connected, ctrl.dataset_counts(), ctrl.has_model()
        )

    def _refresh_labels(self) -> None:
        ctrl = self.controller
        self.query_one("#labels", LabelList).update_labels(
            ctrl.known_labels(), ctrl.dataset_counts(), self.label
        )

    def _refresh_coach(self) -> None:
        ctrl = self.controller
        self.query_one("#coach", CoachPanel).update_coach(
            ctrl.next_step(), ctrl.connected, self.label, self.reps,
            ctrl.classifier, ctrl.has_model(),
        )

    def _refresh_all(self) -> None:
        self._refresh_labels()
        self._refresh_coach()
        self._refresh_workflow()

    def _set_nose(self, state: str) -> None:
        self.query_one("#nose", NoseWidget).set_state(state)

    def _clear_busy(self) -> None:
        self._busy = False

    def _reject_if_busy(self) -> bool:
        if self._busy:
            self._log("busy — a fit/map is running; please wait")
            return True
        return False

    def _disarm_clear(self) -> None:
        self._clear_armed = False

    # -------------------------------------------------- monitor engine
    def _build_source(self):
        """Build the persistent frame source for the current mode (or a test hook)."""
        if self._source_factory is not None:
            return self._source_factory(self.controller)
        ctrl = self.controller
        cfg = ctrl.config
        if ctrl.use_sim:
            from ..monitor import ContinuousSim

            # noise-free so the recovery teller (±2%) settles cleanly for the demo;
            # values still move during a sniff. Real hardware has its own noise.
            return ContinuousSim(cfg, seed=ctrl.seed, noise_counts=0.0)
        from ..serialio import SerialReader

        return SerialReader(
            ctrl.port, n_channels=cfg.n_channels, reconnect=False, startup_delay_s=2.5
        )

    def _start_monitor(self) -> None:
        from ..monitor import MonitorEngine
        from ..record import SniffRecorder

        self._monitor_stop = False
        self._engine = MonitorEngine(
            self.controller.config, SniffRecorder(self.controller.config, self.controller.out_dir)
        )
        self._source = self._build_source()
        # Let the engine drive the airflow straw per phase (fresh vs sample). In sim
        # this is how the odor is presented; on hardware it needs a write-capable
        # source and servo_enabled. Without it, the operator switches straws by hand.
        cfg = self.controller.config
        if hasattr(self._source, "write_command") and (
            self.controller.use_sim or cfg.servo_enabled
        ):
            self._engine.set_airflow(self._source.write_command)
        self._monitor_worker()

    @work(thread=True, exclusive=True, group="monitor")
    def _monitor_worker(self) -> None:
        engine = self._engine
        source = self._source
        names = self.controller.config.sensor_names()
        hz = max(1, self.controller.config.scan_hz)
        paced = self.controller.use_sim if self._paced is None else self._paced
        try:
            for frame in source.frames():
                if self._monitor_stop:
                    break
                ev = engine.step(frame)
                self.call_from_thread(self._on_engine_event, ev, names)
                if paced:
                    time.sleep(1.0 / hz)
        except Exception as exc:  # pragma: no cover - device/stream error
            self.call_from_thread(self._log, f"monitor stopped: {exc}")
        finally:
            try:
                source.close()
            except Exception:  # pragma: no cover
                pass

    def _on_engine_event(self, ev: dict, names) -> None:
        """Render one engine event (runs on the UI thread via call_from_thread)."""
        phase = ev["phase"]
        self.query_one("#sensors", SensorBars).update_values(names, ev["rs"], phase)
        self._set_nose("sniffing" if phase == "exposure" else "idle")

        if ev["settle"] is not None:
            st = ev["settle"]
            if st["timed_out"]:
                self._status(f"⏳ settle timed out ({st['waited_s']:.0f}s) — proceeding")
            else:
                dev = f"{st['max_dev'] * 100:.1f}%" if st["max_dev"] is not None else "…"
                self._status(
                    f"⏳ settling — waiting for a stable baseline ({dev}, {st['waited_s']:.1f}s)"
                )

        if ev["capture"] is not None:
            k, n = ev["capture"]
            pct = k * 100 // n
            self._status(f"⏺ capturing '{self._active_label}' — {phase} {k}/{n} ({pct}%)")

        if ev["saved"] is not None:
            self._on_capture_complete(ev["saved"])

        rec = ev["recovery"]
        if rec is not None:
            if rec["recovered"]:
                if rec["just_recovered"]:
                    self._status("✓ sensors recovered — ready for the next sniff")
            else:
                wc = names[rec["worst_channel"]] if rec["worst_channel"] >= 0 else "?"
                self._status(
                    f"… recovering — {rec['max_dev'] * 100:.1f}% off ({wc}), "
                    f"held {rec['held_s']:.1f}/{rec['target_s']:.0f}s"
                )


    def _on_capture_complete(self, saved) -> None:
        result, path = saved
        if self._identify_pending:
            self._identify_pending = False
            try:
                verdict = self._classify(result.features)
            except Exception as exc:
                self._log(f"identify failed: {exc}")
            else:
                self._last_geometry = verdict["geometry"]
                self._last_verdict = verdict
                self._log(
                    f"predicted {verdict['label']} (p={verdict['proba']:.3f}) "
                    f"novelty {verdict['novelty']:.3f} is_novel {verdict['is_novel']}"
                )
        elif path is not None:
            self._reps_remaining -= 1
            done = self._rep_total - self._reps_remaining
            if self._reps_remaining > 0:
                self._log(f"saved {path.name} ({done}/{self._rep_total}) — settling for next…")
                self._arm_next_rep()  # its SETTLE waits for the sensors to return to rest
            else:
                self._log(f"saved {path.name} ({done}/{self._rep_total}) — done")
        self._active_label = None
        self._refresh_all()

    def _classify(self, features) -> dict:
        import numpy as np

        from ..geometry import serialize_geometry
        from ..model import SmellModel

        if self._model is None:
            self._model = SmellModel.load(self.controller.model_path)
        m = self._model
        row = np.asarray(features).reshape(1, -1)
        classes, proba = m.predict_proba(row)
        pr = np.asarray(proba)[0]
        k = int(pr.argmax())
        return {
            "label": str(classes[k]),
            "proba": float(pr[k]),
            "novelty": float(m.novelty(row)[0]),
            "is_novel": bool(m.is_novel(row)[0]),
            "geometry": serialize_geometry(m, new_sample=np.asarray(features)),
        }

    # ------------------------------------------------------- label / reps
    def action_next_label(self) -> None:
        self._disarm_clear()
        labels = self.controller.known_labels()
        idx = labels.index(self.label) if self.label in labels else -1
        self.label = labels[(idx + 1) % len(labels)]
        self._log(f"label → {self.label}")
        self._refresh_labels()
        self._refresh_coach()

    def action_prev_label(self) -> None:
        self._disarm_clear()
        labels = self.controller.known_labels()
        idx = labels.index(self.label) if self.label in labels else 0
        self.label = labels[(idx - 1) % len(labels)]
        self._log(f"label → {self.label}")
        self._refresh_labels()
        self._refresh_coach()

    def action_add_label(self) -> None:
        self._disarm_clear()

        def _done(value: str | None) -> None:
            if value:
                self.label = value
                self._log(f"added label → {self.label}")
                self._refresh_labels()
                self._refresh_coach()

        self.push_screen(_AddLabelScreen(), _done)

    def action_more_reps(self) -> None:
        self._disarm_clear()
        self.reps += 1
        self._log(f"reps → {self.reps}")
        self._refresh_coach()

    def action_fewer_reps(self) -> None:
        self._disarm_clear()
        if self.reps > 1:
            self.reps -= 1
        self._log(f"reps → {self.reps}")
        self._refresh_coach()

    def action_cycle_classifier(self) -> None:
        self._disarm_clear()
        new = self.controller.cycle_classifier()
        self._log(f"classifier → {new}")
        self._refresh_coach()

    def action_toggle_sim(self) -> None:
        """Flip sim/real: rebuild the controller and restart the monitor loop."""
        self._disarm_clear()
        self._monitor_stop = True  # stop the current monitor before switching ports
        ctrl = self.controller
        new = SniffController(
            ctrl.config, out_dir=ctrl.out_dir, use_sim=not ctrl.use_sim,
            port=ctrl.port, seed=ctrl.seed, model_path=ctrl.model_path,
        )
        new.classifier = ctrl.classifier
        self.controller = new
        self._model = None
        self._log(f"source → {'sim' if new.use_sim else 'real'}")
        self._refresh_all()
        self._start_monitor()

    # ---------------------------------------------------------- record
    def action_record(self) -> None:
        self._disarm_clear()
        if self._engine is None or self._engine.busy:
            self._log("busy — a capture sequence is already running")
            return
        self._rep_label = self.label
        self._rep_total = max(1, self.reps)
        self._reps_remaining = self._rep_total
        if self._rep_total > 1:
            self._log(
                f"recording {self._rep_total}× '{self.label}' — auto, recovery-gated"
            )
        self._arm_next_rep()

    def _arm_next_rep(self) -> None:
        """Arm the next sniff in the current record sequence (UI thread)."""
        label = self._rep_label
        if self._engine.arm_capture(label, save=True):
            self._active_label = label
            idx = self._rep_total - self._reps_remaining + 1
            self._log(f"recording '{label}' {idx}/{self._rep_total} — follow the cues")
            # sim: tell the source which odor its sample straw presents (the engine's
            # airflow reveals it during exposure). Real hardware ignores this.
            if self.controller.use_sim and hasattr(self._source, "set_odor"):
                self._source.set_odor(label)

    # -------------------------------------------------------- identify
    def action_identify(self) -> None:
        self._disarm_clear()
        if self._engine is None or self._engine.busy:
            return
        if not self.controller.has_model():
            self._log("identify: no model yet — press f to fit first")
            return
        if self._engine.arm_capture("?", save=False):
            self._identify_pending = True
            self._active_label = "?"
            self._log("identifying one sniff …")
            if self.controller.use_sim and hasattr(self._source, "set_odor"):
                self._source.set_odor(self.label)

    # ----------------------------------------------------- delete / clear
    def action_delete_last(self) -> None:
        self._disarm_clear()
        if self._reject_if_busy():
            return
        self._busy = True
        self._delete_worker(self.label)

    @work(thread=True)
    def _delete_worker(self, label: str) -> None:
        try:
            deleted = self.controller.delete_last(label)
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.call_from_thread(self._log, f"delete failed: {exc}")
        else:
            if deleted is None:
                self.call_from_thread(self._log, f"nothing to delete for '{label}'")
            else:
                self.call_from_thread(self._log, f"deleted {deleted.name} (undo capture)")
        finally:
            self.call_from_thread(self._refresh_all)
            self.call_from_thread(self._clear_busy)

    def action_clear(self) -> None:
        if self._reject_if_busy():
            return
        if not self._clear_armed:
            self._clear_armed = True
            self._log("press X again to confirm clearing the whole dataset")
            return
        self._clear_armed = False
        self._busy = True
        self._log("clearing dataset …")
        self._clear_worker()

    @work(thread=True)
    def _clear_worker(self) -> None:
        try:
            removed = self.controller.clear()
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.call_from_thread(self._log, f"clear failed: {exc}")
        else:
            self.call_from_thread(self._log, f"cleared {removed} sniff(s)")
        finally:
            self.call_from_thread(self._refresh_all)
            self.call_from_thread(self._clear_busy)

    # ------------------------------------------------------------- fit
    def action_fit(self) -> None:
        self._disarm_clear()
        if self._reject_if_busy():
            return
        if not self.controller.ready_to_fit():
            self._log("fit: need ≥2 labels with data — record more first")
            return
        self._busy = True
        self._log(f"fitting model ({self.controller.classifier}) …")
        self._fit_worker()

    @work(thread=True)
    def _fit_worker(self) -> None:
        try:
            mean, std = self.controller.fit()
        except Exception as exc:
            self.call_from_thread(self._log, f"fit failed: {exc}")
        else:
            self._model = None  # invalidate the cached model
            self.call_from_thread(
                self._log, f"trained ✓ cross-val accuracy {mean:.3f} ± {std:.3f}"
            )
        finally:
            self.call_from_thread(self._refresh_all)
            self.call_from_thread(self._clear_busy)

    # ----------------------------------------------------------- think
    def action_think(self) -> None:
        self._disarm_clear()
        if self._reject_if_busy():
            return
        if self._last_geometry is None:
            self._log("think: identify a sniff first (press i)")
            return
        self._busy = True
        self._log("thinking about the last sniff …")
        self._think_worker()

    @work(thread=True)
    def _think_worker(self) -> None:
        try:
            from .. import llm
            from ..reason import reason

            client = llm.OpenRouterClient()
            narrative = reason(self._last_geometry, client)
        except Exception as exc:
            self.call_from_thread(self._log, str(exc))
        else:
            for line in str(narrative).splitlines() or [str(narrative)]:
                self.call_from_thread(self._log, line)
        finally:
            self.call_from_thread(self._clear_busy)

    # ------------------------------------------------------------- map
    def action_map(self) -> None:
        self._disarm_clear()
        if self._reject_if_busy():
            return
        if not self.controller.has_model():
            self._log("map: no model yet — press f to fit first")
            return
        if not self.controller.dataset_counts():
            self._log("map: no recorded data — press r to record first")
            return
        self._busy = True
        self._log("rendering smell map …")
        self._map_worker()

    @work(thread=True)
    def _map_worker(self) -> None:
        ctrl = self.controller
        try:
            from ..dataset import load_dataset
            from ..model import SmellModel
            from ..smellmap import render_map

            model = SmellModel.load(ctrl.model_path)
            ds = load_dataset(ctrl.out_dir)
            saved = render_map(model, ds)
        except Exception as exc:
            self.call_from_thread(self._log, f"map failed: {exc}")
        else:
            self.call_from_thread(self._log, f"saved map: {saved}")
        finally:
            self.call_from_thread(self._clear_busy)


def run_tui(
    controller: SniffController, *, reps: int = 1, label: str | None = None
) -> None:
    """Launch the Textual app over ``controller`` (blocks until the user quits)."""
    SniffApp(controller, reps=reps, label=label).run()
