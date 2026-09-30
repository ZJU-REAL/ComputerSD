"""Offline analyzer guidance, keyed by task ID and zero-based policy step.

The eval coordinator is the only writer; workers receive only their task's
guidance in Sample.metadata. Empty/failed steps retain their original indices.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from reward.prm_utils import append_privileged_guidance


def load_guidance(path: str) -> dict[str, dict[int, str]]:
    """Accept {task_id: {steps: [...]}} or [{task_id: ..., steps: ...}].

    A task value can also be a list of guidance strings, where list position is
    the step index. Explicit step records use ``step_index`` and ``guidance``.
    """
    with open(path, encoding="utf-8") as stream:
        document = json.load(stream)
    if isinstance(document, list):
        entries = []
        for record in document:
            if not isinstance(record, dict) or not record.get("task_id"):
                raise ValueError("Guidance list entries require task_id and steps")
            entries.append((record["task_id"], record))
    elif isinstance(document, dict):
        entries = document.items()
    else:
        raise ValueError("Guidance file must contain a task map or list")
    result: dict[str, dict[int, str]] = {}
    for task_id, record in entries:
        task_id = str(task_id)
        if task_id in result:
            raise ValueError(f"Duplicate guidance task: {task_id}")
        steps = record.get("steps") if isinstance(record, dict) else record
        if not isinstance(steps, list):
            raise ValueError(f"Guidance task {task_id}: steps must be a list")
        parsed: dict[int, str] = {}
        for position, step in enumerate(steps):
            index = step.get("step_index", position) if isinstance(step, dict) else position
            guidance = step.get("guidance", "") if isinstance(step, dict) else step
            if type(index) is not int or index < 0 or index in parsed:
                raise ValueError(f"Guidance task {task_id}: invalid/duplicate step_index {index!r}")
            if not isinstance(guidance, str):
                raise ValueError(f"Guidance task {task_id}, step {index}: guidance must be a string")
            parsed[index] = guidance.strip()
        result[task_id] = parsed
    return result


def trajectory_guidance(path: str, task_id: str) -> str:
    """Return the fixed summary guidance for one task, if present.

    Trajectory-level guidance files contain one record at step zero.  Keeping
    this lookup here gives training workers the same parsing semantics as eval
    replay without exposing the full guidance map to every trajectory object.
    """
    records = load_guidance(path)
    steps = records.get(str(task_id), {})
    if len(steps) == 1 and 0 in steps:
        return steps[0]
    return ""


def inject_step_guidance(
    messages: list[dict[str, Any]], steps: dict[int, str], step_index: int,
) -> list[dict[str, Any]]:
    """Match OPD: g_t goes into the last user turn before generating action t.

    Guidance is not accumulated or shifted to fill holes. Missing tasks, empty
    steps, and steps beyond the recorded trajectory use the ordinary context.
    """
    # A trajectory-level guidance file stores exactly one item at step 0. Keep
    # the older multi-step format unchanged, while replaying the single summary
    # at every turn so later actions receive the same global lesson.
    guidance = steps.get(step_index, "")
    if not guidance and len(steps) == 1 and 0 in steps:
        guidance = steps[0]
    return append_privileged_guidance(messages, guidance) if guidance else messages


def guidance_record(sample: Any) -> dict[str, Any]:
    metadata = sample.metadata or {}
    return {
        "task_id": str(metadata["example_id"]),
        "domain": str(metadata.get("domain", "")),
        "status": getattr(sample.status, "value", str(sample.status)),
        "error": metadata.get("gui_error_message"),
        "steps": metadata.get("analyzer_guidance_steps", []),
    }


def write_guidance(path: str, records: dict[str, Any]) -> None:
    """Atomically checkpoint completed tasks without concurrent worker writes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
