"""Configuration loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration and resolve relative paths against the project root."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section, keys in {"data": ("raw_dir", "processed_dir"), "reports": ("dir", "figures_dir")}.items():
        for k in keys:
            cfg[section][k] = str(PROJECT_ROOT / cfg[section][k])
    validate(cfg)
    return cfg


IMPLEMENTED = {("target",): "Global_active_power", ("forecast", "horizon"): 24, ("forecast", "issue_hour"): 0}


def validate(cfg: dict[str, Any]) -> None:
    """Refuse values that the code does not implement, instead of silently ignoring them."""
    for keys, expected in IMPLEMENTED.items():
        v = cfg
        for k in keys:
            v = v[k]
        if v != expected:
            raise NotImplementedError(f"config {'.'.join(keys)} = {v!r}: only {expected!r} is implemented")
