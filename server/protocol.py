"""Standardized recording protocol (feature F1): one fixed reading passage, versioned, never changed automatically.

Recordings are only compared when their prompt and processing protocol match the baseline's. Check-ins saved before
this protocol existed (no stored prompt) count as the legacy free-speech task ("default").
"""
from __future__ import annotations

STANDARD_ID = "standard_reading_v1"
LEGACY_ID = "default"

# Original passage written for this project. About 120 words: roughly 40-50 s read aloud at a normal pace, which gives
# the pipeline several 8 s windows and comfortably more than the 15 s of speech the quality gate requires.
PASSAGE = (
    "Every morning the bakery on the corner opens its doors a little after six. The first customers are usually "
    "nurses finishing a night shift and builders starting an early one. They buy warm bread, a coffee, and sometimes "
    "a slice of apple cake. The owner knows most of them by name. On rainy days the windows fog up and the queue "
    "spills out under the green awning, but nobody seems to mind waiting. By nine o'clock the shelves are half empty, "
    "the radio is playing quietly, and the smell of cinnamon has reached the bus stop across the road. Later, when "
    "the shop is calm again, the owner sits by the window, writes tomorrow's list, and plans what to bake next."
)

PROMPTS = {
    STANDARD_ID: {"prompt_id": STANDARD_ID, "prompt_version": 1, "protocol_version": "1.0", "task_type": "read_speech",
                  "text": PASSAGE,
                  "instructions": ["Use your normal speaking voice.",
                                   "Keep the microphone at about the same distance as last time.",
                                   "Read the passage shown on screen, from the beginning.",
                                   "Don't deliberately change how you speak unless you are running a controlled experiment."]},
    LEGACY_ID: {"prompt_id": LEGACY_ID, "prompt_version": 0, "protocol_version": "0", "task_type": "free_speech",
                "text": "Tell me about your day so far - how you slept, how your energy has been, and what you have "
                        "been looking forward to.",
                "instructions": ["Speak naturally about your day."]},
}


def get(prompt_id: str | None) -> dict:
    return PROMPTS.get(prompt_id or LEGACY_ID, PROMPTS[LEGACY_ID])


def meta(prompt_id: str | None) -> dict:
    """What every check-in stores (no passage text)."""
    p = get(prompt_id)
    return {k: p[k] for k in ("prompt_id", "prompt_version", "protocol_version", "task_type")}


def of_checkin(ck) -> dict:
    """Protocol of a stored check-in; anything recorded before protocols existed is the legacy free-speech task."""
    return (ck.extra or {}).get("protocol") or meta(LEGACY_ID)


def key(m: dict) -> tuple:
    return (m.get("prompt_id"), m.get("prompt_version"), m.get("protocol_version"))


def compatible(a: dict, b: dict) -> bool:
    return key(a) == key(b)
