"""Milestone 3 — the olfactory-reasoning prompt and message assembly.

The LLM reasons over the **learned smell-space geometry as structured text**,
never over raw volts. :func:`build_messages` embeds the geometry JSON (from
:func:`sniffsniff.geometry.serialize_geometry`) into a user turn; :func:`reason`
drives any client exposing ``.complete(messages) -> str``.
"""
from __future__ import annotations

import json

__all__ = ["SYSTEM_PROMPT", "build_messages", "reason"]

SYSTEM_PROMPT = (
    "You are an olfactory reasoning agent for an electronic nose. You never "
    "smell directly — you reason over the smell-space geometry given to you as "
    "JSON: a 2-D map whose axes are interpreted from PCA loadings, the known "
    "odor clusters (each with a centroid, radius, and count, plus the new "
    "sample's Euclidean distance to it), the new sample's map coordinates, the "
    "classifier's predicted label and probability, its most extreme z-scored "
    "features, and a Mahalanobis novelty score with its threshold.\n\n"
    "Your job:\n"
    "1. Identify the sample — name the most likely odor.\n"
    "2. State confidence grounded in the geometry (nearest centroid distance vs. "
    "its radius, the novelty score vs. threshold, the classifier probability) — "
    "not just the bare label.\n"
    "3. Describe where the sample sits relative to the clusters using the axis "
    "interpretations (e.g. nudged toward the MQ-135 acid axis suggests acidity).\n"
    "4. Judge novelty honestly: if the Mahalanobis score exceeds the threshold, "
    "say the smell looks unfamiliar rather than forcing a known label.\n"
    "5. Suggest the next action: commit (confident, within a cluster), re-sniff "
    "(borderline/ambiguous), or raise the temperature / adjust sensing "
    "(novel or off-map).\n\n"
    "Be concise and concrete. Cite the numbers you reason from."
)

DEFAULT_QUESTION = "Interpret this new sniff."


def build_messages(geometry: dict, question: str | None = None) -> list[dict]:
    """Build the ``[system, user]`` chat turns for a geometry blob.

    The user turn embeds ``geometry`` as pretty-printed JSON followed by the
    ``question`` (defaulting to :data:`DEFAULT_QUESTION`).
    """
    if question is None:
        question = DEFAULT_QUESTION
    geometry_json = json.dumps(geometry, indent=2)
    user = (
        "Here is the current smell-space geometry as JSON:\n\n"
        f"{geometry_json}\n\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def reason(geometry: dict, client, *, question: str | None = None) -> str:
    """Reason over ``geometry`` via ``client.complete`` and return the narrative.

    ``client`` is any object with ``.complete(messages) -> str`` (the real
    :class:`~sniffsniff.llm.OpenRouterClient` or a fake in tests).
    """
    return client.complete(build_messages(geometry, question))
