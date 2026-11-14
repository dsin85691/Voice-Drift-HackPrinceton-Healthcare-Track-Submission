"""Language branch: one LLM call -> PHQ-9 evidence + safety_flag. Quotes are verified verbatim."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..config import get_config

log = logging.getLogger("vd.language")

PHQ9_LABELS = {1: "anhedonia", 2: "low mood", 3: "sleep", 4: "energy", 5: "appetite",
               6: "self-worth", 7: "concentration", 8: "psychomotor", 9: "self-harm"}


class LLMItem(BaseModel):
    phq9_item: int = Field(ge=1, le=9)
    label: str = ""
    direction: Literal["worse", "better", "neutral"] = "neutral"
    quotes: list[str] = []
    confidence: float = Field(default=0.0, ge=0, le=1)


class LLMOut(BaseModel):
    safety_flag: bool = False
    items: list[LLMItem] = []
    summary_for_clinician: str = ""


SYSTEM = f"""You help a clinician review a patient's spoken check-in transcript.
Return ONLY a JSON object: {{"safety_flag": bool, "items": [{{"phq9_item": 1-9, "label": str,
"direction": "worse"|"better"|"neutral", "quotes": [str], "confidence": 0-1}}],
"summary_for_clinician": str}}.
Rules:
- Quote VERBATIM only: every quote must be an exact substring of the transcript. Never paraphrase.
- Only output items that have evidence in the transcript. Output nothing for other items.
- PHQ-9 items: {json.dumps(PHQ9_LABELS)}. Item 9 (self-harm) must NEVER appear in items; if the
  transcript contains any self-harm or suicidal content, set safety_flag to true instead.
- direction is relative to the person's wellbeing as expressed in the quote.
- summary_for_clinician: at most 2 sentences, descriptive, never use diagnosis words
  (depressed, depression, disorder, suicidal)."""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def verify_quote(quote: str, transcript: str) -> bool:
    """Exact substring after normalizing whitespace and case."""
    q = norm(quote)
    return bool(q) and q in norm(transcript)


def verify_items(out: LLMOut, transcript: str) -> tuple[list[dict], int]:
    """Drops quotes that are not verbatim; returns (evidence rows, dropped_count). Item 9 never shown."""
    evidence, dropped = [], 0
    for it in out.items:
        if it.phq9_item == 9:
            continue
        for q in it.quotes:
            if verify_quote(q, transcript):
                evidence.append({"item": it.label or PHQ9_LABELS[it.phq9_item], "phq9": it.phq9_item,
                                 "direction": it.direction, "quote": q.strip(), "verified": True,
                                 "confidence": it.confidence})
            else:
                dropped += 1
    return evidence, dropped


class LanguageBranch:
    def __init__(self):
        self.cfg = get_config().llm

    @property
    def reachable(self) -> bool:
        return bool(self.cfg["enabled"] and os.environ.get("ANTHROPIC_API_KEY"))

    def analyze(self, transcript: str) -> dict:
        """Returns {available, safety_flag, evidence, evidence_dropped, summary}. Never raises."""
        empty = {"available": False, "safety_flag": False, "evidence": [], "evidence_dropped": 0, "summary": ""}
        if not transcript.strip() or not self.reachable:
            return empty
        try:
            r = httpx.post(
                self.cfg["api_url"],
                headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                json={"model": self.cfg["model"], "max_tokens": 1024, "temperature": 0, "system": SYSTEM,
                      "messages": [{"role": "user", "content": f"Transcript:\n{transcript}"}]},
                timeout=float(self.cfg["timeout_s"]),
            )
            r.raise_for_status()
            text = "".join(b.get("text", "") for b in r.json()["content"])
            return self.parse(text, transcript)
        except (httpx.HTTPError, KeyError, ValueError, ValidationError) as e:
            log.warning("LLM branch failed: %s", e)
            return empty

    @staticmethod
    def parse(text: str, transcript: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        out = LLMOut.model_validate_json(m.group(0) if m else text)
        evidence, dropped = verify_items(out, transcript)
        return {"available": True, "safety_flag": out.safety_flag, "evidence": evidence,
                "evidence_dropped": dropped, "summary": out.summary_for_clinician}
