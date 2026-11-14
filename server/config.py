"""Config loading. config.yaml at repo root; VD_CONFIG overrides the path."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Cfg(dict):
    """dict with attribute access; nested dicts are wrapped lazily."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


@lru_cache(maxsize=1)
def get_config() -> Cfg:
    _load_env()
    path = Path(os.environ.get("VD_CONFIG", ROOT / "config.yaml"))
    return Cfg(yaml.safe_load(path.read_text(encoding="utf-8")))


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q
