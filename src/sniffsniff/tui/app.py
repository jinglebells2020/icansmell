"""The Textual app that wires the sniffsniff pipeline into an interactive TUI.

:class:`SniffApp` owns *no* pipeline logic — it delegates everything to a
:class:`~sniffsniff.tui.controller.SniffController` and merely reflects progress
in widgets. All the slow work (capturing a bounded session, fitting, identifying,
rendering the map) runs in ``@work(thread=True)`` worker threads so the event
loop stays responsive; those workers touch widgets *only* through
:meth:`textual.app.App.call_from_thread`.

``textual`` is an optional extra, so this module is imported lazily from the CLI
(``_cmd_tui``). Importing it without ``textual`` installed raises ``ImportError``,
which the CLI turns into a friendly install hint.
"""
from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from .controller import SniffController
from .nose import NoseWidget
from .widgets import LogPanel, SensorBars, WorkflowPanel

__all__ = ["SniffApp", "run_tui"]

# The five demo odors; the first is the default record label. Pressing ``l``
# cycles through them.
_ODORS = ["coffee", "vinegar", "alcohol", "fresh_milk", "spoiled_milk"]


class SniffApp(App):
    """Interactive e-nose console over a :class:`SniffController`."""

    CSS = """
    #body { height: 1fr; }
    #left { width: 1fr; }
    #right { width: 1fr; }
    NoseWidget { height: auto; border: round $accent; padding: 1; }
    WorkflowPanel { height: 1fr; border: round $primary; padding: 1; }
    SensorBars { height: auto; border: round $secondary; padding: 1; }
    LogPanel { height: 1fr; border: round $panel; }
    """

    BINDINGS = [
        ("r", "record", "Record"),
        ("f", "fit", "Fit"),
        ("i", "identify", "Identify"),
        ("m", "map", "Map"),
        ("s", "toggle_sim", "Sim/Real"),
        ("l", "cycle_label", "Label"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self, controller: SniffController, *, reps: int = 8, label: str | None = None
    ) -> None:
        super().__init__()
        self.controller = controller
        self.reps = reps
        self.label = label or _ODORS[0]

    # ------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield NoseWidget(id="nose")
                yield WorkflowPanel(id="workflow")
            with Vertical(id="right"):
                yield SensorBars(id="sensors")
                yield LogPanel(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_workflow()
        self._log(f"ready — label '{self.label}', {'sim' if self.controller.use_sim else 'real'}")

    # -------------------------------------------------------- ui helpers
    def _log(self, msg: str) -> None:
        self.query_one("#log", LogPanel).write_line(msg)

    def _refresh_workflow(self) -> None:
        ctrl = self.controller
        self.query_one("#workflow", WorkflowPanel).update_state(
            ctrl.connected, ctrl.dataset_counts(), ctrl.has_model()
        )

    def _set_nose(self, state: str) -> None:
        self.query_one("#nose", NoseWidget).set_state(state)

    # ------------------------------------------------------- no-op-ish
    def action_cycle_label(self) -> None:
        idx = (_ODORS.index(self.label) + 1) % len(_ODORS) if self.label in _ODORS else 0
        self.label = _ODORS[idx]
        self._log(f"label → {self.label}")

    def action_toggle_sim(self) -> None:
        """Flip the sim/real flag by rebuilding the controller in place."""
        ctrl = self.controller
        self.controller = SniffController(
            ctrl.config,
            out_dir=ctrl.out_dir,
            use_sim=not ctrl.use_sim,
            port=ctrl.port,
            seed=ctrl.seed,
            model_path=ctrl.model_path,
        )
        self._log(f"source → {'sim' if self.controller.use_sim else 'real'}")
        self._refresh_workflow()

    # ---------------------------------------------------------- record
    def action_record(self) -> None:
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
            self.call_from_thread(self._refresh_workflow)

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
            self.call_from_thread(self._refresh_workflow)

    # ------------------------------------------------------------- fit
    def action_fit(self) -> None:
        self._log("fitting model …")
        self._fit_worker()

    @work(thread=True)
    def _fit_worker(self) -> None:
        try:
            mean, std = self.controller.fit()
        except Exception as exc:
            self.call_from_thread(self._log, f"fit failed: {exc}")
        else:
            self.call_from_thread(
                self._log, f"cross-val accuracy {mean:.3f} ± {std:.3f}"
            )
        finally:
            self.call_from_thread(self._refresh_workflow)

    # -------------------------------------------------------- identify
    def action_identify(self) -> None:
        if not self.controller.has_model():
            self._log("identify: no model yet — press f to fit first")
            return
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
            self.call_from_thread(
                self._log,
                f"predicted {result['label']} "
                f"(p={result['proba']:.3f}) "
                f"novelty {result['novelty']:.3f} is_novel {result['is_novel']}",
            )
        finally:
            self.call_from_thread(self._set_nose, "idle")

    # ------------------------------------------------------------- map
    def action_map(self) -> None:
        if not self.controller.has_model():
            self._log("map: no model yet — press f to fit first")
            return
        if not self.controller.dataset_counts():
            self._log("map: no recorded data — press r to record first")
            return
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


def run_tui(
    controller: SniffController, *, reps: int = 8, label: str | None = None
) -> None:
    """Launch the Textual app over ``controller`` (blocks until the user quits)."""
    SniffApp(controller, reps=reps, label=label).run()
