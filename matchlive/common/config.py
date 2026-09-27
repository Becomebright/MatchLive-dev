"""Load a YAML config and resolve its paths against the repo root.

The config stays a plain nested dict; only the `paths` section is rewritten into
absolute `Path`s so no module ever depends on the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# matchlive/common/config.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        config = yaml.safe_load(f)
    config["paths"] = {name: REPO_ROOT / value for name, value in config["paths"].items()}
    return config
