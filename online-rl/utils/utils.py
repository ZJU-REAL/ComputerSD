"""Stateless helper functions shared across the GUI rollout modules.

Only pure, dependency-free utilities live here (no module-level logger/state
coupling), so they can be imported from both the rollout entrypoint and the
trajectory/episode layer without circular imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_image(image_bytes: bytes | bytearray | None, out_path: Path) -> None:
    """Write raw image bytes to ``out_path`` (creating parent dirs)."""
    if not image_bytes:
        raise ValueError(f"Cannot save empty screenshot to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(image_bytes)


def load_task_config(base_dir: str, domain: str, example_id: str) -> dict[str, Any]:
    """Load one task config JSON from ``<base_dir>/examples/<domain>/<id>.json``."""
    cfg_path = Path(base_dir) / "examples" / domain / f"{example_id}.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)
