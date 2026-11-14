"""Safety screen: lexicon regex on the transcript. The LLM safety_flag is OR-ed in by the pipeline."""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from ..config import get_config, resolve


@lru_cache(maxsize=1)
def _patterns() -> list[re.Pattern]:
    path = resolve(get_config().safety["lexicon"])
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [re.compile(p, re.I) for p in data["patterns"]]


def lexicon_patterns() -> list[str]:
    return [p.pattern for p in _patterns()]


def lexicon_match(transcript: str) -> str | None:
    """Returns the matched phrase, or None."""
    t = re.sub(r"\s+", " ", transcript or "")
    for p in _patterns():
        m = p.search(t)
        if m:
            return m.group(0)
    return None


def screen(transcript: str, llm_flag: bool = False) -> dict:
    """{triggered, source}. Either source is enough."""
    hit = lexicon_match(transcript)
    if hit:
        return {"triggered": True, "source": "lexicon"}
    if llm_flag:
        return {"triggered": True, "source": "llm"}
    return {"triggered": False, "source": None}
