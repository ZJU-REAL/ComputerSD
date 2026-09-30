"""Build step-level GUI privileged-context OPD targets.

Each dynamic-history sample represents one action response.  The frozen actor
snapshot is teacher-forced twice: once with the original context (q0) and once
with the analyzer's GUIDE/AVOID context (q+).  The q+ sampled-token log-probs
provide the sampled-token OPD target; q+ top-k distributions provide the
optional projected reverse-KL target and its KL gate. The q+ - q0 shift is
retained only for rollout diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any

from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.types import Sample

from reward.prm_utils import (
    append_privileged_guidance,
    build_sglang_prefill_payload,
    opd_confidence_weights,
    orient_opd_confidence_weights,
    parse_sglang_token_log_probs,
    parse_sglang_topk_response,
    resolve_opd_step_polarity,
    validate_opd_weight_versions,
)

logger = logging.getLogger(__name__)

_TEACHER_SEMAPHORES: dict[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = {}


def _analyzer_privileged_info(detail: dict[str, Any]) -> str:
    """Return the analyzer's canonical non-empty OPD guidance.

    ``guidance`` is the representative vote selected by the analyzer.  Do not
    reconstruct it from arbitrary ``guide``/``avoid`` fields here: that could
    turn a result explicitly marked ``missing_guidance`` into a teacher target.
    """
    return str(detail.get("guidance", "") or "").strip()


def _teacher_semaphore(args: Any) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = max(1, int(getattr(args, "gui_opd_teacher_max_concurrency", 1)))
    current = _TEACHER_SEMAPHORES.get(loop)
    if current is None:
        current = (limit, asyncio.Semaphore(limit))
        _TEACHER_SEMAPHORES[loop] = current
    elif current[0] != limit:
        logger.warning(
            "GUI OPD teacher concurrency already fixed at %d; ignoring %d",
            current[0],
            limit,
        )
    return current[1]


def _teacher_input(
    state: Any,
    agent: Any,
    messages: list[dict[str, Any]],
) -> tuple[str, list[int], list[str]]:
    """Tokenize one multimodal conversation for SGLang prefill."""
    tokenizer = state.tokenizer
    processor = state.processor
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image_data: list[str] = []
    if processor:
        multimodal_inputs = agent._extract_multimodal(messages, processor) or {}
        images = multimodal_inputs.get("images") or []
        proc_out = processor(text=[prompt_text], return_tensors="pt", **multimodal_inputs)
        input_ids = proc_out["input_ids"][0].tolist()
        image_data = [encode_image_for_rollout_engine(image) for image in images]
    else:
        input_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return prompt_text, input_ids, image_data


async def _attach_one_target(
    args: Any,
    state: Any,
    agent: Any,
    sample: Sample,
    snapshot: dict[str, Any],
    guidance: str,
    polarity: int,
    consistency_majority: int,
    effectiveness_majority: int,
) -> None:
    response_length = int(sample.response_length)
    loss_mode = str(getattr(args, "gui_opd_loss_mode", "sampled_token"))
    topk = int(getattr(args, "gui_opd_topk", 50))
    if response_length <= 0 or (loss_mode == "topk" and topk <= 0):
        raise ValueError(
            f"invalid OPD dimensions: response_length={response_length}, topk={topk}, mode={loss_mode}"
        )

    original_messages = snapshot["train_messages"]
    privileged_messages = append_privileged_guidance(original_messages, guidance)
    original_prompt_text, original_input_ids, original_image_data = await asyncio.to_thread(
        _teacher_input,
        state,
        agent,
        original_messages,
    )
    privileged_prompt_text, privileged_input_ids, privileged_image_data = await asyncio.to_thread(
        _teacher_input,
        state,
        agent,
        privileged_messages,
    )

    sampled_suffix = list(sample.tokens[-response_length:])
    for name, input_ids in (
        ("original", original_input_ids),
        ("privileged", privileged_input_ids),
    ):
        if len(input_ids) < response_length:
            raise ValueError(
                f"invalid {name} response span: total={len(input_ids)}, "
                f"response={response_length}"
            )
        if list(input_ids[-response_length:]) != sampled_suffix:
            raise ValueError(
                f"{name} teacher-forced context does not match the sampled response tokens"
            )

    privileged_payload = build_sglang_prefill_payload(
        prompt_text=privileged_prompt_text,
        input_ids=privileged_input_ids,
        image_data=privileged_image_data,
        response_length=response_length,
        top_logprobs_num=topk if loss_mode == "topk" else None,
    )
    original_payload = build_sglang_prefill_payload(
        prompt_text=original_prompt_text,
        input_ids=original_input_ids,
        image_data=original_image_data,
        response_length=response_length,
        top_logprobs_num=None,
    )
    # Explicitly select the updatable actor route when multiple named SGLang
    # models (actor + analyzer) are deployed.
    url = get_model_url(args, "actor")

    # The two requests share one concurrency slot. Weight synchronization can
    # still happen between HTTP requests in fully asynchronous training, so the
    # returned versions are recorded below for asynchronous staleness monitoring.
    configured_teacher_retries = getattr(args, "gui_opd_teacher_http_max_retries", None)
    if configured_teacher_retries is None:
        configured_teacher_retries = os.environ.get("GUI_OPD_TEACHER_HTTP_MAX_RETRIES", "30")
    teacher_http_max_retries = max(
        1,
        int(configured_teacher_retries or 30),
    )
    original_output = await post(
        url,
        original_payload,
        max_retries=teacher_http_max_retries,
    )
    privileged_output = await post(
        url,
        privileged_payload,
        max_retries=teacher_http_max_retries,
    )

    if loss_mode == "topk":
        topk_log_probs, topk_indices = parse_sglang_topk_response(
            privileged_output,
            response_length=response_length,
            topk=topk,
        )
    else:
        topk_log_probs, topk_indices = None, None
    original_token_log_probs = parse_sglang_token_log_probs(
        original_output,
        response_length=response_length,
        expected_token_ids=sampled_suffix,
    )
    privileged_token_log_probs = parse_sglang_token_log_probs(
        privileged_output,
        response_length=response_length,
        expected_token_ids=sampled_suffix,
    )
    raw_gate = opd_confidence_weights(
        privileged_token_log_probs,
        original_token_log_probs,
        beta=float(getattr(args, "gui_opd_gate_beta", 5.0)),
    )
    gate = orient_opd_confidence_weights(raw_gate, polarity)

    original_meta = original_output.get("meta_info", {}) if isinstance(original_output, dict) else {}
    privileged_meta = privileged_output.get("meta_info", {}) if isinstance(privileged_output, dict) else {}
    original_version = original_meta.get("weight_version") if isinstance(original_meta, dict) else None
    privileged_version = privileged_meta.get("weight_version") if isinstance(privileged_meta, dict) else None
    rollout_version = snapshot.get("rollout_weight_version")
    q0_qplus_version_mismatch = (
        original_version is not None
        and privileged_version is not None
        and str(original_version) != str(privileged_version)
    )
    if q0_qplus_version_mismatch:
        logger.warning(
            "GUI OPD q0/q+ teacher versions differ; keeping OPD target: "
            "sample=%s step=%s q0=%s q+=%s",
            sample.index,
            snapshot.get("step_idx"),
            original_version,
            privileged_version,
        )
    rollout_version_mismatch, teacher_rollout_version_gap = validate_opd_weight_versions(
        original_version,
        privileged_version,
        rollout_version,
    )
    if rollout_version_mismatch:
        logger.warning(
            "GUI OPD teacher version differs from rollout snapshot; keeping OPD: "
            "sample=%s step=%s rollout=%s q0=%s q+=%s gap=%s",
            sample.index,
            snapshot.get("step_idx"),
            rollout_version,
            original_version,
            privileged_version,
            teacher_rollout_version_gap,
        )

    # The sampled-token target is already returned by the same q+ request and
    # is the default SEED-compatible OPD target.
    sample.teacher_log_probs = privileged_token_log_probs
    sample.teacher_topk_log_probs = topk_log_probs
    sample.teacher_topk_indices = topk_indices
    # Both loss modes recompute their mode-specific gate at training time. This
    # field is only a binary target-validity mask.
    sample.opd_token_weights = [1.0] * response_length
    sample.opd_step_polarity = polarity
    # q0 is retained only for asynchronous diagnostics. It is never used as
    # the OPD teacher target or as a loss input when versions differ.
    sample.opd_q0_log_probs = original_token_log_probs
    sample.opd_q0_valid = bool(
        len(original_token_log_probs) == response_length
        and original_version is not None
        and privileged_version is not None
        and str(original_version) == str(privileged_version)
    )
    sample.metadata = sample.metadata or {}
    shifts = [p - o for p, o in zip(privileged_token_log_probs, original_token_log_probs)]
    sample.metadata["opd"] = {
        "status": "ok",
        "step_index": int(snapshot["step_idx"]),
        "topk": topk,
        "loss_mode": loss_mode,
        "guidance": guidance,
        "teacher_prompt_length": len(privileged_input_ids) - response_length,
        "response_length": response_length,
        "gate_beta": float(getattr(args, "gui_opd_gate_beta", 5.0)),
        "teacher_http_max_retries": teacher_http_max_retries,
        "step_polarity": "positive" if polarity > 0 else "corrective",
        "consistency_majority": int(consistency_majority),
        "effectiveness_majority": int(effectiveness_majority),
        "raw_gate_mean": sum(raw_gate) / len(raw_gate) if raw_gate else 0.0,
        "raw_gate_active_ratio": (
            sum(weight > 0.5 for weight in raw_gate) / len(raw_gate) if raw_gate else 0.0
        ),
        "effective_gate_mean": sum(gate) / len(gate) if gate else 0.0,
        "effective_gate_active_ratio": (
            sum(weight > 0.5 for weight in gate) / len(gate) if gate else 0.0
        ),
        # Compatibility alias for the rollout-snapshot q0/q+ diagnostic. The
        # training loss recomputes its effective gate against the current actor.
        "gate_mean": sum(gate) / len(gate) if gate else 0.0,
        "logprob_shift_mean": sum(shifts) / len(shifts) if shifts else 0.0,
        "rollout_weight_version": rollout_version,
        "teacher_weight_version": privileged_version,
        "original_weight_version": original_version,
        "q0_qplus_version_mismatch": q0_qplus_version_mismatch,
        "teacher_rollout_version_mismatch": rollout_version_mismatch,
        "teacher_rollout_version_gap": teacher_rollout_version_gap,
    }
    sample.metadata["opd"]["judge_polarity"] = polarity


def _mask_one_target(sample: Sample, *, topk: int, step_idx: int, error: str) -> None:
    """Keep a failed step in GRPO while making its OPD contribution zero."""
    response_length = max(0, int(sample.response_length))
    safe_logprob = -math.log(topk + 1)
    sample.teacher_log_probs = [0.0] * response_length
    sample.teacher_topk_log_probs = [[safe_logprob] * topk for _ in range(response_length)]
    sample.teacher_topk_indices = [list(range(topk)) for _ in range(response_length)]
    sample.opd_token_weights = [0.0] * response_length
    sample.opd_step_polarity = 0
    sample.opd_q0_log_probs = None
    sample.opd_q0_valid = False
    sample.metadata = sample.metadata or {}
    sample.metadata["opd"] = {
        "status": "masked",
        "step_index": int(step_idx),
        "topk": topk,
        "loss_mode": "masked",
        "response_length": response_length,
        "step_polarity": "masked",
        "raw_gate_mean": 0.0,
        "raw_gate_active_ratio": 0.0,
        "effective_gate_mean": 0.0,
        "effective_gate_active_ratio": 0.0,
        "gate_mean": 0.0,
        "error": str(error),
    }


async def attach_gui_opd_target_for_step(
    args: Any,
    state: Any,
    agent: Any,
    sample: Sample,
    snapshot: dict[str, Any],
    detail: dict[str, Any],
) -> bool:
    """Attach one OPD target, masking only this step when it is unusable.

    This is deliberately usable as soon as an individual PRM task completes.
    q0/q+ remain serial within the teacher semaphore. Since updates can occur
    between the two HTTP requests, their versions are recorded for monitoring
    but are not required to match.
    """
    topk = int(getattr(args, "gui_opd_topk", 50))
    step_idx = int(snapshot.get("step_idx", (sample.metadata or {}).get("dynamic_step_index", -1)))
    guidance = _analyzer_privileged_info(detail)
    try:
        if detail.get("status") != "ok":
            raise ValueError(f"PRM step status is {detail.get('status', 'missing')}")
        polarity = resolve_opd_step_polarity(detail)
        if polarity == 0:
            raise ValueError(
                "PRM majority direction is unavailable or tied: "
                f"consistency={detail.get('consistency_majority')}, "
                f"effectiveness={detail.get('effectiveness_majority')}"
            )
        if not guidance:
            raise ValueError("PRM returned no canonical action guidance")
        # Bound multimodal preprocessing and both serial teacher prefills.
        async with _teacher_semaphore(args):
            await _attach_one_target(
                args,
                state,
                agent,
                sample,
                snapshot,
                guidance,
                polarity,
                int(detail["consistency_majority"]),
                int(detail["effectiveness_majority"]),
            )
        return True
    except Exception as exc:
        _mask_one_target(sample, topk=topk, step_idx=step_idx, error=str(exc))
        logger.warning(
            "Masking GUI OPD step sample=%s step=%d: %s",
            sample.index,
            step_idx,
            exc,
        )
        return False


def attach_precomputed_gui_opd_targets(
    args: Any,
    samples: list[Sample],
    targets_by_step: dict[int, Sample],
    errors_by_step: dict[int, str] | None = None,
) -> dict[str, int]:
    """Transfer pipelined per-step targets onto final dynamic-history samples.

    The target was built from the same immutable trajectory snapshot while the
    rollout continued.  Recheck the response suffix before copying so a future
    tokenizer or mask change cannot silently attach a target to different
    sampled tokens.
    """
    loss_mode = str(getattr(args, "gui_opd_loss_mode", "sampled_token"))
    topk = int(getattr(args, "gui_opd_topk", 50))
    errors_by_step = errors_by_step or {}
    attached = 0
    masked = 0

    for sample in samples:
        step_idx = int((sample.metadata or {}).get("dynamic_step_index", -1))
        target = targets_by_step.get(step_idx)
        error = errors_by_step.get(step_idx)
        if target is None:
            _mask_one_target(
                sample,
                topk=topk,
                step_idx=step_idx,
                error=error or "OPD target was unavailable after PRM completion",
            )
            masked += 1
            continue

        target_opd = (target.metadata or {}).get("opd", {})
        target_status = target_opd.get("status") if isinstance(target_opd, dict) else None
        target_polarity = getattr(target, "opd_step_polarity", None)
        valid_shape = (
            target.response_length == sample.response_length
            and len(target.tokens) >= target.response_length
            and len(sample.tokens) >= sample.response_length
            and target.tokens[-target.response_length :] == sample.tokens[-sample.response_length :]
            and target.teacher_log_probs is not None
            and len(target.teacher_log_probs) == sample.response_length
            and target.opd_token_weights is not None
            and len(target.opd_token_weights) == sample.response_length
        )
        if loss_mode == "topk":
            valid_shape = valid_shape and (
                target.teacher_topk_log_probs is not None
                and target.teacher_topk_indices is not None
                and len(target.teacher_topk_log_probs) == sample.response_length
                and len(target.teacher_topk_indices) == sample.response_length
            )
        else:
            valid_shape = valid_shape and target_polarity in (-1, 0, 1)
            if target_status == "ok":
                valid_shape = valid_shape and target_polarity in (-1, 1)
            elif target_status == "masked":
                valid_shape = valid_shape and target_polarity == 0
        if target_status not in {"ok", "masked"} or not valid_shape:
            _mask_one_target(
                sample,
                topk=topk,
                step_idx=step_idx,
                error="Pipelined OPD target does not match final dynamic sample",
            )
            masked += 1
            continue

        # One trajectory snapshot maps to exactly one dynamic child. Transfer
        # these large [T, K] rows by reference; the caller releases the temporary
        # target container immediately after fan-out.
        sample.teacher_log_probs = target.teacher_log_probs
        sample.teacher_topk_log_probs = target.teacher_topk_log_probs if loss_mode == "topk" else None
        sample.teacher_topk_indices = target.teacher_topk_indices if loss_mode == "topk" else None
        sample.opd_token_weights = target.opd_token_weights
        sample.opd_step_polarity = target_polarity
        sample.opd_q0_log_probs = target.opd_q0_log_probs
        sample.opd_q0_valid = bool(target.opd_q0_valid)
        sample.metadata = sample.metadata or {}
        sample.metadata["opd"] = target_opd
        if target_status == "ok":
            attached += 1
        else:
            masked += 1

    return {
        "requested": len(samples),
        "attached": attached,
        "masked": masked,
        # Compatibility name for existing dashboards. This never means the
        # Sample was removed from the GRPO batch.
        "dropped": masked,
    }


async def attach_gui_opd_targets(
    args: Any,
    state: Any,
    agent: Any,
    samples: list[Sample],
    step_snapshots: list[dict[str, Any]],
    prm_step_details: list[dict[str, Any]],
) -> dict[str, int]:
    """Attach per-step teacher targets; failures mask OPD only."""
    if not getattr(args, "gui_opd_enable", False):
        return {"requested": 0, "attached": 0, "masked": 0, "dropped": 0}
    if not getattr(args, "dynamic_history", False):
        raise RuntimeError("GUI OPD requires dynamic_history=true (one training sample per GUI step)")
    if not getattr(args, "prm_enable", False):
        raise RuntimeError("GUI OPD requires prm_enable=true so every step has privileged guidance")

    snapshots = {int(item["step_idx"]): item for item in step_snapshots}
    details = {
        int(item.get("step_index", i)): item
        for i, item in enumerate(prm_step_details)
        if isinstance(item, dict)
    }

    async def _run(sample: Sample) -> bool:
        step_idx = int((sample.metadata or {}).get("dynamic_step_index", -1))
        snapshot = snapshots.get(step_idx)
        detail = details.get(step_idx, {})
        if snapshot is None:
            _mask_one_target(
                sample,
                topk=int(getattr(args, "gui_opd_topk", 50)),
                step_idx=step_idx,
                error=f"missing trajectory snapshot for step {step_idx}",
            )
            logger.warning("Masking GUI OPD step sample=%s step=%d: missing trajectory snapshot", sample.index, step_idx)
            return False
        return await attach_gui_opd_target_for_step(
            args,
            state,
            agent,
            sample,
            snapshot,
            detail,
        )

    outcomes = await asyncio.gather(*[_run(sample) for sample in samples])
    attached = sum(bool(item) for item in outcomes)
    masked = len(samples) - attached
    return {
        "requested": len(samples),
        "attached": attached,
        "masked": masked,
        # Compatibility name for existing dashboards. This never means the
        # Sample was removed from the GRPO batch.
        "dropped": masked,
    }
