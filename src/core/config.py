from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return _ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
