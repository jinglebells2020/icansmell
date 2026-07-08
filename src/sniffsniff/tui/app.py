"""The Textual app that wires the sniffsniff pipeline into an interactive TUI.

:class:`SniffApp` owns *no* pipeline logic — it delegates everything to a
:class:`~sniffsniff.tui.controller.SniffController` and merely reflects progress
in widgets. All the slow work (capturing a bounded session, fitting, identifying,
rendering the map) runs in ``@work(thread=True)`` worker threads so the event
loop stays responsive; those workers touch widgets *only* through
:meth:`textual.app.App.call_from_thread`.

v2 turns it into a *guided training console*: a live per-label list, a coach line
that always says the next step, and the ability to fix mistakes (delete the last
sniff, clear + restart).

``textual`` is an optional extra, so this module is imported lazily from the CLI
(``_cmd_tui``). Importing it without ``textual`` installed raises ``ImportError``,
which the CLI turns into a friendly install hint.
"""
from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label

from .controller import CLASSIFIERS, SniffController
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
        self, controller: SniffController, *, reps: int = 1, label: str | None = None
    ) -> None:
        super().__init__()
        self.controller = controller
        self.reps = reps
        self.label = label or controller.known_labels()[0]
        self._busy = False  # a capture/fit/map is running — refuse a second
        self._clear_armed = False  # first X arms, second X within focus confirms
        self._last_geometry = None  # geometry of the last identified sniff (for `t`)
        self._last_verdict = None  # the last identify verdict dict (for `t`)

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
                yield LogPanel(id="log")
        yield Footer()

    def on_mount(self) -> None:
        import numpy as np

        # Populate the sensor panel immediately so it isn't an empty, collapsed box
        # before the first capture (a zeroed bar per configured sensor).
        names = self.controller.config.sensor_names()
        self.query_one("#sensors", SensorBars).update_values(
            names, np.zeros(len(names)), phase="idle"
        )
        self._refresh_all()
        mode = "sim" if self.controller.use_sim else "real"
        self._log(f"ready — label '{self.label}', {mode}")
        # In real mode with no device present, say so loudly instead of failing later.
        if not self.controller.use_sim and not self.controller.connected:
            self._log(
                f"⚠ no device at {self.controller.port} — reconnect it, or press s for the simulator"
            )

    # -------------------------------------------------------- ui helpers
    def _log(self, msg: str) -> None:
        self.query_one("#log", LogPanel).write_line(msg)

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
            ctrl.next_step(),
            ctrl.connected,
            self.label,
            self.reps,
            ctrl.classifier,
            ctrl.has_model(),
        )

    def _refresh_all(self) -> None:
        """Refresh label list + coach + workflow (the three state panels)."""
        self._refresh_labels()
        self._refresh_coach()
        self._refresh_workflow()

    def _set_nose(self, state: str) -> None:
        self.query_one("#nose", NoseWidget).set_state(state)

    def _clear_busy(self) -> None:
        self._busy = False

    def _reject_if_busy(self) -> bool:
        """Log + return True when a long action is already running (guard)."""
        if self._busy:
            self._log("busy — a capture/fit is already running; please wait")
            return True
        return False

    def _disarm_clear(self) -> None:
        """Any action other than a second X cancels a pending clear confirmation."""
        self._clear_armed = False

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
        """Flip the sim/real flag by rebuilding the controller in place."""
        self._disarm_clear()
        ctrl = self.controller
        new = SniffController(
            ctrl.config,
            out_dir=ctrl.out_dir,
            use_sim=not ctrl.use_sim,
            port=ctrl.port,
            seed=ctrl.seed,
            model_path=ctrl.model_path,
        )
        new.classifier = ctrl.classifier  # preserve the chosen classifier
        self.controller = new
        self._log(f"source → {'sim' if self.controller.use_sim else 'real'}")
        self._refresh_all()

    # ---------------------------------------------------------- record
    def action_record(self) -> None:
        self._disarm_clear()
        if self._reject_if_busy():
            return
        self._busy = True
        self._log(f"recording {self.reps} × '{self.label}' …")
        self._record_worker(self.label, self.reps)

    @work(thread=True)
    def _record_worker(self, label: str, reps: int) -> None:
        ctrl = self.controller
        names = ctrl.config.sensor_names()

        import numpy as np

        def on_phase(phase: str, k: int, n: int) -> None:
            self.call_from_thread(
                self._set_nose, "sniffing" if phase == "exposure" else "idle"
            )
            self.call_from_thread(
                self.query_one("#sensors", SensorBars).update_values,
                names,
                np.zeros(ctrl.config.n_channels),
                phase,
            )

        def on_frame(k: int, n: int, phase: str, frame) -> None:
            self.call_from_thread(
                self.query_one("#sensors", SensorBars).update_values,
                names,
                ctrl.rs_of(frame[1]),
                phase,
            )

        def on_saved(path, i: int) -> None:
            self.call_from_thread(self._log, f"saved {path.name} ({i + 1}/{reps})")
            self.call_from_thread(self._refresh_all)

        try:
            ctrl.record_many(
                label,
                reps,
                on_phase=on_phase,
                on_frame=on_frame,
                on_saved=on_saved,
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.call_from_thread(self._log, f"record failed: {exc}")
        finally:
            self.call_from_thread(self._set_nose, "idle")
            self.call_from_thread(self._refresh_all)
            self.call_from_thread(self._clear_busy)

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
                self.call_from_thread(
                    self._log, f"nothing to delete for '{label}'"
                )
            else:
                self.call_from_thread(
                    self._log, f"deleted {deleted.name} (undo capture)"
                )
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
            self.call_from_thread(
                self._log, f"trained ✓ cross-val accuracy {mean:.3f} ± {std:.3f}"
            )
        finally:
            self.call_from_thread(self._refresh_all)
            self.call_from_thread(self._clear_busy)

    # -------------------------------------------------------- identify
    def action_identify(self) -> None:
        self._disarm_clear()
        if self._reject_if_busy():
            return
        if not self.controller.has_model():
            self._log("identify: no model yet — press f to fit first")
            return
        self._busy = True
        self._log("identifying one sniff …")
        self._identify_worker()

    @work(thread=True)
    def _identify_worker(self) -> None:
        ctrl = self.controller
        names = ctrl.config.sensor_names()

        def on_phase(phase: str, k: int, n: int) -> None:
            self.call_from_thread(
                self._set_nose, "sniffing" if phase == "exposure" else "idle"
            )

        def on_frame(k: int, n: int, phase: str, frame) -> None:
            self.call_from_thread(
                self.query_one("#sensors", SensorBars).update_values,
                names,
                ctrl.rs_of(frame[1]),
                phase,
            )

        try:
            result = ctrl.identify(on_phase=on_phase, on_frame=on_frame)
        except Exception as exc:
            self.call_from_thread(self._log, f"identify failed: {exc}")
        else:
            # Stash for the `t` (think) action, which reasons over this geometry.
            self._last_geometry = result["geometry"]
            self._last_verdict = result
            self.call_from_thread(
                self._log,
                f"predicted {result['label']} "
                f"(p={result['proba']:.3f}) "
                f"novelty {result['novelty']:.3f} is_novel {result['is_novel']}",
            )
        finally:
            self.call_from_thread(self._set_nose, "idle")
            self.call_from_thread(self._refresh_coach)
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
            # Includes LLMError (missing key hint) — never a traceback or a key.
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
