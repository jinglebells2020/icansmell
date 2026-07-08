# sniffsniff — Milestone 3: LLM Reasoner — Design

**Date:** 2026-07-06
**Status:** Approved (design), building
**Milestone:** 3 of 3 — the LLM reasons over the learned smell-space geometry.

## Purpose

The payoff layer: an LLM reads the **geometry JSON** that Milestone 2 already
produces (`geometry.serialize_geometry(model, new_sample=feats)`) and turns it into a
human narrative — *"most likely coffee (nearest centroid, distance 0.6, within its
radius), but nudged toward the MQ-135 acid axis suggesting mild acidity; not novel;
confident"* — plus a suggested next action (commit / re-sniff / raise temperature).

Architectural principle (from the research doc): the LLM reasons over the **learned
geometry as structured text, never over raw volts**. We already serialize exactly that.

**Backend:** OpenRouter (OpenAI-compatible), model **`moonshotai/kimi-k2-thinking`**.
The API key comes **only** from the `OPENROUTER_API_KEY` environment variable — never
hardcoded, printed, logged, or committed.

## Scope

### In scope
- `llm.py`: `OpenRouterClient` — a thin, dependency-free (stdlib `urllib`) chat client;
  key from env; injectable transport for tests; clear `LLMError` on missing key / HTTP
  failure; extracts `choices[0].message.content` (falls back to `.reasoning` for the
  thinking model).
- `reason.py`: the olfactory-reasoning system prompt, `build_messages(geometry, question)`,
  and `reason(geometry, client, question=None) -> str`.
- CLI `reason` subcommand: capture/sim one sniff → geometry → LLM → print the narrative.
- TUI: a `t` (think) action that reasons about the most-recently identified sniff and
  logs the narrative (runs in a worker; guarded on a model + key being present).

### Out of scope (a possible v2)
- Active-sensing tool loop ("what to sniff next" driving the servo/fan).
- Predict-then-check (LLM predicts coordinates before sensing).
- Grounding unlabeled clusters to words (unsupervised cluster naming).

## Module interfaces

### `llm.py`
```python
DEFAULT_MODEL = "moonshotai/kimi-k2-thinking"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

class LLMError(Exception): ...

class OpenRouterClient:
    def __init__(self, model=DEFAULT_MODEL, api_key=None, *, base_url=BASE_URL,
                 timeout=120.0, referer="https://github.com/sniffsniff",
                 title="sniffsniff", transport=None): ...
        # api_key defaults to os.environ.get("OPENROUTER_API_KEY").
        # transport: optional callable(url, headers:dict, payload:dict) -> dict (parsed JSON),
        #   so tests inject a fake and NO network is touched. Default transport uses urllib.
    def complete(self, messages: list[dict], *, temperature=0.3, max_tokens=None) -> str
        # raise LLMError("set OPENROUTER_API_KEY ...") if no key.
        # POST {model, messages, temperature[, max_tokens]}; Authorization: Bearer <key>.
        # return message.content, or message.reasoning if content is empty.
        # raise LLMError on transport/HTTP/parse failure (never include the key in the message).
```
The key is only ever placed in the `Authorization` header; it is never returned, logged,
or embedded in any exception text.

### `reason.py`
```python
SYSTEM_PROMPT: str   # "You are an olfactory reasoning agent for an electronic nose. You
                     #  never smell directly — you reason over the smell-space geometry
                     #  given as JSON: 2-D map axes interpreted from PCA loadings, known
                     #  odor clusters (centroid/radius/distance), a new sample's coords +
                     #  classifier prediction, and a Mahalanobis novelty score+threshold.
                     #  Identify the sample, state confidence grounded in the distances/
                     #  novelty (not just the label), describe its position relative to
                     #  clusters using the axis interpretations, judge novelty honestly,
                     #  and suggest the next action (commit / re-sniff / raise temperature)."
def build_messages(geometry: dict, question: str | None = None) -> list[dict]
    # [{"role":"system", ...}, {"role":"user", ...}] where the user message embeds the
    # geometry as pretty JSON plus the question (default: "Interpret this new sniff.").
def reason(geometry: dict, client, *, question=None) -> str
    # client.complete(build_messages(geometry, question))  — client is any object with
    # .complete(messages)->str, so a fake drives tests.
```

### `cli.py` — `reason` subcommand
`sniffsniff reason --model model.joblib (--sim --odor X | --port P) [--llm-model M] [--config P] [--seed S]`
- capture/sim one sniff → `SmellRecorder.process(...).features` → `serialize_geometry` →
  `reason(geometry, OpenRouterClient(llm_model))` → print the narrative.
- If `OPENROUTER_API_KEY` is unset: print
  `"the reasoner needs an OpenRouter key — export OPENROUTER_API_KEY=... (get one at openrouter.ai)"`
  and return 1 (no traceback).

### `tui/app.py` — `t` action
- `identify` stashes the geometry of the last sniff on the app.
- `t` runs `reason(...)` in a `@work(thread=True)` worker and logs the narrative via
  `call_from_thread`; if no model or no key, logs a friendly hint instead.

## Security

- The key is read **only** from `OPENROUTER_API_KEY`; if absent, every entry point degrades
  gracefully with an instruction to set it.
- The key is placed **only** in the `Authorization` header — never in logs, exceptions,
  `repr`, the geometry JSON, or committed files. Tests assert the key never appears in an
  `LLMError` message.

## Testing (no network)

- `test_llm.py`: `OpenRouterClient` with an injected `transport` returns the content; falls
  back to `reasoning` when content is empty; raises `LLMError` (mentioning the env var, NOT
  the key) when no key; sends `Authorization: Bearer <key>` and the right model/messages to
  the transport; a transport that raises surfaces as `LLMError` without leaking the key.
- `test_reason.py`: `build_messages` has a system + user message, embeds the geometry
  (axis_interpretation, known_clusters, novelty) as JSON, and the question; `reason` passes
  those messages to a fake client and returns its text.
- `test_cli_reason.py`: with a fake/monkeypatched client, `reason --sim` prints the narrative
  and returns 0; with the key unset it prints the hint and returns 1 (no traceback).

## Live verification (manual, after the build)

With `export OPENROUTER_API_KEY=…` set by the operator:
`sniffsniff reason --model model.joblib --sim --odor coffee` → a real kimi-k2-thinking
narrative. (Cannot be unit-tested — it needs the key + network.)
