"""Shared helpers for PRM (process reward model) reward agents.

Both the external-API reward agent and the local-sglang reward agent parse the
same ``\\boxed{...}`` verdict format and read trajectory images off disk. These
helpers were duplicated verbatim across the two agents; they live here now.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

# A PRM judges each step and emits a verdict as ``\boxed{1}`` (good) or
# ``\boxed{-1}`` (bad). These patterns extract and validate that scalar.
_PRM_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_PRM_STRICT_NUMBER_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")
_GUIDANCE_PATTERN = re.compile(
    r"\[GUIDANCE_START\](.*?)\[GUIDANCE_END\]",
    re.DOTALL | re.IGNORECASE,
)
_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _json_objects(text: str):
    """Yield JSON objects embedded in an otherwise free-form PRM response."""
    decoder = json.JSONDecoder()
    seen: set[int] = set()
    candidates = [match.group(1) for match in _FENCED_JSON_PATTERN.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        for pos, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[pos:])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(obj, dict) and id(obj) not in seen:
                seen.add(id(obj))
                yield obj


def _sign(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if abs(number - 1.0) < 1e-9:
        return 1
    if abs(number + 1.0) < 1e-9:
        return -1
    return 0


def extract_prm_guidance_from_text(text: str) -> str:
    """Extract action guidance from JSON or ``[GUIDANCE_*]`` markers."""
    if not text:
        return ""
    for obj in _json_objects(text):
        for key in ("guidance", "action_guidance", "advice", "hint"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    match = _GUIDANCE_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def extract_prm_sign_from_text(text: str) -> int:
    """Parse a PRM verdict into ``+1`` / ``-1`` / ``0`` (unparseable/neutral)."""
    if not text:
        return 0
    for obj in _json_objects(text):
        for key in ("score", "step_score", "reward", "verdict"):
            score = _sign(obj.get(key))
            if score:
                return score
    match = _PRM_BOXED_PATTERN.search(text)
    if not match:
        return 0
    boxed_content = match.group(1).strip()
    strict_number_match = _PRM_STRICT_NUMBER_PATTERN.fullmatch(boxed_content)
    if not strict_number_match:
        return 0
    return _sign(strict_number_match.group(1))


def validate_opd_weight_versions(
    original_version: Any,
    privileged_version: Any,
    rollout_version: Any,
) -> tuple[bool, int | None]:
    """Describe rollout/teacher staleness without rejecting q0/q+ drift.

    q0 and q+ are prefills issued during rollout-target construction and may
    observe different actor snapshots in a fully asynchronous run. Their
    versions are recorded for monitoring, but a mismatch (or a missing version
    from an older SGLang endpoint) does not mask the target.

    Returns ``(rollout_mismatch, teacher_minus_rollout_version)``. The numeric
    gap is ``None`` when a version is absent or not integer-like.
    """

    rollout_mismatch = (
        rollout_version is not None
        and privileged_version is not None
        and str(rollout_version) != str(privileged_version)
    )
    version_gap = None
    if rollout_version is not None and privileged_version is not None:
        try:
            version_gap = int(privileged_version) - int(rollout_version)
        except (TypeError, ValueError):
            pass
    return rollout_mismatch, version_gap


def select_prm_guidance(votes: list[dict[str, Any]]) -> str:
    """Select non-empty guidance from valid votes aligned with the majority."""
    scored = [
        v
        for v in votes
        if v.get("ok", True)
        and not v.get("format_error", False)
        and int(v.get("score", 0)) in (-1, 1)
    ]
    if not scored:
        return ""
    score_sum = sum(int(v["score"]) for v in scored)
    majority = 1 if score_sum > 0 else -1 if score_sum < 0 else int(scored[0]["score"])
    aligned = [
        v
        for v in scored
        if int(v["score"]) == majority and str(v.get("guidance", "") or "").strip()
    ]
    if not aligned:
        return ""
    # Prefer the most informative vote while keeping selection reproducible.
    return max(aligned, key=lambda v: len(str(v["guidance"]))).get("guidance", "").strip()


def append_privileged_guidance(messages: list[dict[str, Any]], guidance: str) -> list[dict[str, Any]]:
    """Append teacher-only guidance to the last user turn without mutating input."""
    guidance = str(guidance or "").strip()
    if not guidance:
        raise ValueError("OPD privileged guidance must not be empty")
    enhanced = copy.deepcopy(messages)
    user_index = next((i for i in range(len(enhanced) - 1, -1, -1) if enhanced[i].get("role") == "user"), None)
    if user_index is None:
        raise ValueError("Cannot attach OPD guidance: no user message in trajectory snapshot")
    block = (
        "\n\n[PRIVILEGED_ACTION_GUIDANCE]\n"
        f"{guidance}\n"
        "[/PRIVILEGED_ACTION_GUIDANCE]"
    )
    content = enhanced[user_index].get("content", "")
    if isinstance(content, list):
        content.append({"type": "text", "text": block})
    elif isinstance(content, str):
        enhanced[user_index]["content"] = content + block
    else:
        raise TypeError(f"Unsupported user message content: {type(content).__name__}")
    return enhanced


def build_sglang_prefill_payload(
    *,
    prompt_text: str,
    input_ids: list[int],
    image_data: list[str],
    response_length: int,
    top_logprobs_num: int | None,
) -> dict[str, Any]:
    """Build a teacher-forcing request without expanding vision tokens twice."""
    payload: dict[str, Any] = {
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Request one preceding token so SGLang's first undefined/BOS row is
        # outside the response span. The parser then drops that row and keeps
        # one complete distribution for every response token.
        "logprob_start_len": max(0, len(input_ids) - response_length - 1),
    }
    if top_logprobs_num is not None:
        payload["top_logprobs_num"] = top_logprobs_num
    if image_data:
        # SGLang must perform the multimodal expansion exactly once. Sending
        # processor-expanded input_ids together with image_data makes Qwen3-VL
        # interpret repeated vision tokens as additional image placeholders.
        payload["text"] = prompt_text
        payload["image_data"] = image_data
    else:
        payload["input_ids"] = input_ids
    return payload


def parse_sglang_topk_response(
    output: Any,
    *,
    response_length: int,
    topk: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Strictly decode SGLang teacher-forcing top-k rows for response tokens."""
    if response_length <= 0 or topk <= 0:
        raise ValueError(f"Invalid OPD dimensions: response_length={response_length}, topk={topk}")
    meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
    rows = meta.get("input_top_logprobs") if isinstance(meta, dict) else None
    if not isinstance(rows, list):
        raise ValueError("SGLang response has no meta_info.input_top_logprobs list")

    # SGLang includes one undefined row for the first token in the scored span.
    if len(rows) > response_length:
        rows = rows[1:]
    if len(rows) < response_length:
        raise ValueError(f"SGLang returned {len(rows)} top-k rows for {response_length} response tokens")
    rows = rows[-response_length:]

    log_probs: list[list[float]] = []
    indices: list[list[int]] = []
    for row_idx, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < topk:
            raise ValueError(f"Invalid top-k row {row_idx}: expected at least {topk} entries")
        row_lp: list[float] = []
        row_ids: list[int] = []
        for entry in row[:topk]:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                lp, token_id = entry[0], entry[1]
            elif isinstance(entry, dict):
                lp, token_id = entry.get("logprob"), entry.get("token_id")
            else:
                raise ValueError(f"Invalid top-k entry in row {row_idx}: {entry!r}")
            lp = float(lp)
            token_id = int(token_id)
            if not math.isfinite(lp) or token_id < 0:
                raise ValueError(f"Invalid top-k value in row {row_idx}: logprob={lp}, token_id={token_id}")
            row_lp.append(lp)
            row_ids.append(token_id)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"Duplicate teacher token ids in top-k row {row_idx}")
        # The top-k mass must leave a non-negative tail bucket (allow tiny FP error).
        if sum(math.exp(lp) for lp in row_lp) > 1.0001:
            raise ValueError(f"Teacher top-k probability mass exceeds one in row {row_idx}")
        log_probs.append(row_lp)
        indices.append(row_ids)
    return log_probs, indices


def parse_sglang_token_log_probs(
    output: Any,
    *,
    response_length: int,
    expected_token_ids: list[int] | None = None,
) -> list[float]:
    """Extract teacher-forced log-probs for the sampled response tokens.

    SGLang's ``input_token_logprobs`` normally contains an undefined first
    position for the first token in the requested span.  Taking the final
    ``response_length`` rows handles both that format and versions which omit
    the undefined row.  Token ids are checked when supplied so a stale or
    differently-tokenized target cannot silently produce a confidence gate.
    """
    if response_length <= 0:
        raise ValueError(f"Invalid response length: {response_length}")
    meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
    rows = meta.get("input_token_logprobs") if isinstance(meta, dict) else None
    if not isinstance(rows, list):
        raise ValueError("SGLang response has no meta_info.input_token_logprobs list")
    if len(rows) < response_length:
        raise ValueError(
            f"SGLang returned {len(rows)} input logprob rows for {response_length} response tokens"
        )
    rows = rows[-response_length:]
    if expected_token_ids is not None and len(expected_token_ids) != response_length:
        raise ValueError(
            f"Expected token id count {len(expected_token_ids)} != response length {response_length}"
        )

    values: list[float] = []
    for row_idx, row in enumerate(rows):
        if isinstance(row, dict):
            logprob = row.get("logprob")
            token_id = row.get("token_id")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            logprob, token_id = row[0], row[1]
        else:
            raise ValueError(f"Invalid input logprob row {row_idx}: {row!r}")
        try:
            logprob = float(logprob)
            token_id = int(token_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid input logprob row {row_idx}: {row!r}") from exc
        if not math.isfinite(logprob) or token_id < 0:
            raise ValueError(
                f"Invalid input logprob row {row_idx}: logprob={logprob}, token_id={token_id}"
            )
        if expected_token_ids is not None and token_id != int(expected_token_ids[row_idx]):
            raise ValueError(
                "SGLang teacher-forced token ids do not match the sampled response "
                f"at offset {row_idx}: got {token_id}, expected {expected_token_ids[row_idx]}"
            )
        values.append(logprob)
    return values


def opd_confidence_weights(
    privileged_log_probs: list[float],
    original_log_probs: list[float],
    *,
    beta: float = 5.0,
) -> list[float]:
    """Compute the frozen-snapshot OPD confidence gate ``sigmoid(beta * Δ)``."""
    if len(privileged_log_probs) != len(original_log_probs):
        raise ValueError(
            "Privileged/original logprob lengths differ: "
            f"{len(privileged_log_probs)} != {len(original_log_probs)}"
        )
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"OPD gate beta must be finite and non-negative, got {beta}")

    weights: list[float] = []
    for privileged, original in zip(privileged_log_probs, original_log_probs):
        delta = beta * (float(privileged) - float(original))
        # Stable sigmoid for large positive/negative shifts.
        if delta >= 0:
            exp_neg = math.exp(-delta)
            weights.append(1.0 / (1.0 + exp_neg))
        else:
            exp_pos = math.exp(delta)
            weights.append(exp_pos / (1.0 + exp_pos))
    return weights


def resolve_opd_step_polarity(detail: dict[str, Any]) -> int:
    """Map reliable majority labels to positive/corrective OPD direction.

    ``1`` confirms a fully consistent and effective step, ``-1`` marks a step
    requiring correction, and ``0`` means either majority is missing/tied so
    the OPD target must be masked.
    """
    consistency = detail.get("consistency_majority")
    effectiveness = detail.get("effectiveness_majority")
    if consistency not in (0, 1) or effectiveness not in (0, 1):
        return 0
    return 1 if consistency == 1 and effectiveness == 1 else -1


def orient_opd_confidence_weights(weights: list[float], polarity: int) -> list[float]:
    """Orient sampled-token confidence toward confirmation or correction."""
    if polarity not in (-1, 0, 1):
        raise ValueError(f"OPD polarity must be -1, 0, or 1, got {polarity}")
    oriented: list[float] = []
    for weight in weights:
        weight = float(weight)
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"OPD confidence weight must be in [0, 1], got {weight}")
        if polarity > 0:
            oriented.append(weight)
        elif polarity < 0:
            oriented.append(1.0 - weight)
        else:
            oriented.append(0.0)
    return oriented


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
