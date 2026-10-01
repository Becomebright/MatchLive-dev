"""Load a YAML config and resolve its paths against the repo root.

The config stays a plain nested dict; only the `paths` section is rewritten into
absolute `Path`s so no module ever depends on the current working directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

# matchlive/common/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml", overrides: Iterable[str] = ()) -> dict[str, Any]:
    """`overrides` are "section.key=value" settings applied on top of the file (value parsed as
    YAML, so "loop.arbiter=false" sets a bool), e.g. one ablation's switches."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        config = yaml.safe_load(f)
    for item in overrides:
        dotted, value = item.split("=", 1)
        *parents, key = dotted.split(".")
        node = config
        for name in parents:
            node = node[name]
        if key not in node:
            raise KeyError(f"override {item!r}: no setting {dotted!r} in {path.name}")
        node[key] = yaml.safe_load(value)
    config["paths"] = {name: REPO_ROOT / value for name, value in config["paths"].items()}
    return config


def run_overrides(run_dir: Path) -> list[str]:
    """The overrides a live run (scripts/run_live.py) was made with, from its run.json, so the
    scripts reading the run (QA, evaluation) use the same settings, e.g. loop.cascade."""
    path = Path(run_dir) / "run.json"
    return json.loads(path.read_text()).get("set", []) if path.exists() else []
