"""Tests for reason.py (Milestone 3) — message building + reason(). NO network."""
from __future__ import annotations

from sniffsniff.reason import SYSTEM_PROMPT, build_messages, reason


GEOMETRY = {
    "pca": {"map_components": 2},
    "axis_interpretation": {"PC1": "MQ3__peak(+)", "PC2": "MQ135__peak(-)"},
    "known_clusters": {"coffee": {"centroid": [0.1, 0.2], "radius": 0.5, "n": 8}},
    "new_sample": {"predicted": {"label": "coffee", "proba": 0.9}},
    "novelty": {"min_mahalanobis": 1.2, "threshold": 3.0, "is_novel": False},
}


def test_build_messages_shape():
    msgs = build_messages(GEOMETRY)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_user_message_embeds_geometry_and_default_question():
    msgs = build_messages(GEOMETRY)
    user = msgs[1]["content"]
    # Known geometry keys appear in the pretty-printed JSON.
    assert "axis_interpretation" in user
    assert "known_clusters" in user
    assert "novelty" in user
    # Default question.
    assert "Interpret this new sniff." in user


def test_user_message_embeds_custom_question():
    msgs = build_messages(GEOMETRY, question="Is this coffee?")
    assert "Is this coffee?" in msgs[1]["content"]


def test_reason_passes_messages_to_client_and_returns_text():
    class FakeClient:
        def __init__(self):
            self.seen = None

        def complete(self, messages):
            self.seen = messages
            return "canned narrative"

    client = FakeClient()
    out = reason(GEOMETRY, client)
    assert out == "canned narrative"
    assert client.seen == build_messages(GEOMETRY)
