"""Joint policy + step-level GUI privileged-context OPD losses.

Each loss mode computes its detached confidence gate from the same discrepancy
that it optimizes. ``sampled_token`` uses the signed sampled-token gap, while
``topk`` uses projected top-k-plus-tail reverse KL.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.nn.functional as F
from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_responses, policy_loss_function
from slime.utils.ppo_utils import compute_log_probs


def _masked_sample_mean(
    rows: list[torch.Tensor],
    valid_rows: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Sum per-sample means over tokens valid for the selected OPD target."""
    total = rows[0].new_zeros(()) if rows else torch.tensor(0.0)
    for row, valid, loss_mask in zip(rows, valid_rows, loss_masks, strict=True):
        mask = valid.to(device=row.device, dtype=row.dtype) * loss_mask.to(
            device=row.device, dtype=row.dtype
        )
        denominator = mask.sum().clamp_min(1.0)
        total = total + (row * mask).sum() / denominator
    return total


def _validity_and_direction(
    valid: torch.Tensor,
    expected: torch.Tensor,
    polarity_value: object,
    *,
    sample_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the shared target mask and judge direction."""
    if valid.ndim != 1 or valid.shape != expected.shape:
        raise ValueError(
            f"Invalid GUI OPD mask at sample {sample_index}: {tuple(valid.shape)}; "
            f"expected {tuple(expected.shape)}"
        )
    if isinstance(polarity_value, torch.Tensor):
        if polarity_value.numel() != 1:
            raise ValueError(
                f"Invalid GUI OPD polarity shape at sample {sample_index}: "
                f"{tuple(polarity_value.shape)}"
            )
        polarity = int(polarity_value.item())
    else:
        polarity = int(polarity_value)
    if polarity not in (-1, 0, 1):
        raise ValueError(
            f"Invalid GUI OPD polarity at sample {sample_index}: {polarity}; "
            "expected -1, 0, or 1"
        )
    valid = (valid > 0).to(dtype=torch.float32) * float(polarity != 0)
    return valid, torch.full_like(valid, float(polarity))


def _sampled_token_gate_parts(
    args: Namespace,
    teacher: torch.Tensor,
    student: torch.Tensor,
    valid: torch.Tensor,
    polarity_value: object,
    *,
    sample_index: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the detached sampled-token gate against the current student."""
    if teacher.ndim != 1 or teacher.shape != student.shape:
        raise ValueError(
            f"Sampled-token teacher length mismatch at sample {sample_index}: "
            f"teacher={tuple(teacher.shape)}, student={tuple(student.shape)}"
        )
    valid, judge_direction = _validity_and_direction(
        valid, student, polarity_value, sample_index=sample_index
    )
    if bool(getattr(args, "gui_opd_judge_gate", True)):
        loss_direction = judge_direction
    else:
        loss_direction = torch.ones_like(judge_direction)
    gap = (teacher - student.detach()).detach()
    directed_gap = (loss_direction * gap).detach()
    gate_input = float(getattr(args, "gui_opd_gate_beta", 5.0)) * directed_gap
    if bool(getattr(args, "gui_opd_gate_reverse", False)):
        gate_input = -gate_input
    gate = ((gate_input > 0).to(dtype=directed_gap.dtype)
            if bool(getattr(args, "gui_opd_hard_gate", False))
            else torch.sigmoid(gate_input)).detach()
    if not bool(getattr(args, "gui_opd_gate", True)):
        gate = torch.ones_like(gate)
    return valid, judge_direction, loss_direction, gap, directed_gap, gate


def _sampled_token_opd_parts(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute judge-directed, SEED-style sampled-token OPD.

    ``teacher_log_probs`` are q+ log-probs for the sampled response tokens.
    The current actor log-probs are recomputed from the q0 logits.  The
    precomputed token weights are used only as a validity mask (zero means a
    failed/masked teacher target); their magnitude is not applied a second
    time. With the judge gate enabled, for judge direction ``d`` in
    ``{+1, -1}``, the token loss is::

        gate = sigmoid(beta * d * (teacher - student.detach()))
        loss = d * gate * (teacher - student)

    Thus confirmed steps increase the sampled token's probability, while
    corrective steps decrease it. The directional gap also makes updates
    strongest when q+ agrees with the judge. With the judge gate disabled,
    ``d=+1`` for every valid target, exactly recovering SEED's sampled-token
    gate and loss; the original polarity remains available for diagnostics.
    """
    teacher_logps = batch.get("teacher_log_probs")
    if not teacher_logps:
        raise ValueError("Sampled-token GUI OPD requires teacher_log_probs")
    weight_rows = batch.get("opd_token_weights")
    if weight_rows is None:
        weight_rows = [
            [1.0] * int(response_length)
            for response_length in batch["response_lengths"]
        ]
    if len(weight_rows) != len(teacher_logps):
        raise ValueError("GUI OPD token-weight and sampled-token target counts differ")
    polarity_values = batch.get("opd_step_polarity")
    if polarity_values is None or len(polarity_values) != len(teacher_logps):
        raise ValueError(
            "Sampled-token GUI OPD requires one opd_step_polarity per sample"
        )

    tp_group = mpu.get_tensor_model_parallel_group()
    student_rows: list[torch.Tensor] = []
    teacher_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    gap_rows: list[torch.Tensor] = []
    directed_gap_rows: list[torch.Tensor] = []
    gate_rows: list[torch.Tensor] = []
    judge_direction_rows: list[torch.Tensor] = []
    loss_direction_rows: list[torch.Tensor] = []

    for i, (logits_chunk, tokens_chunk) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )
    ):
        teacher = torch.as_tensor(
            teacher_logps[i], device=logits_chunk.device, dtype=torch.float32
        )
        student = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group).squeeze(-1)
        valid = torch.as_tensor(
            weight_rows[i], device=logits_chunk.device, dtype=torch.float32
        )
        (
            valid,
            judge_direction,
            loss_direction,
            gap,
            directed_gap,
            gate,
        ) = _sampled_token_gate_parts(
            args,
            teacher,
            student,
            valid,
            polarity_values[i],
            sample_index=i,
        )

        student_rows.append(student)
        teacher_rows.append(teacher)
        valid_rows.append(valid)
        gap_rows.append(gap)
        directed_gap_rows.append(directed_gap)
        gate_rows.append(gate)
        judge_direction_rows.append(judge_direction)
        loss_direction_rows.append(loss_direction)

    loss_rows = [
        direction * gate * (teacher - student)
        for direction, gate, teacher, student in zip(
            loss_direction_rows, gate_rows, teacher_rows, student_rows, strict=True
        )
    ]
    raw_rows = [
        direction * (teacher - student)
        for direction, teacher, student in zip(
            loss_direction_rows, teacher_rows, student_rows, strict=True
        )
    ]
    active_rows = [
        ((gate > 0.5).to(dtype=gate.dtype) * valid)
        for gate, valid in zip(gate_rows, valid_rows, strict=True)
    ]

    opd_loss = _masked_sample_mean(loss_rows, valid_rows, batch["loss_masks"])
    raw_loss = _masked_sample_mean(raw_rows, valid_rows, batch["loss_masks"])
    # ``_masked_sample_mean`` applies the validity mask itself.  Do not include
    # ``valid`` in the row a second time, otherwise this metric would report
    # mean(gate²) instead of the actual mean gate.
    gate_mean = _masked_sample_mean(gate_rows, valid_rows, batch["loss_masks"])
    gate_active_ratio = _masked_sample_mean(active_rows, valid_rows, batch["loss_masks"])
    teacher_gap_mean = _masked_sample_mean(gap_rows, valid_rows, batch["loss_masks"])
    directed_gap_mean = _masked_sample_mean(
        directed_gap_rows, valid_rows, batch["loss_masks"]
    )
    positive_rows = [
        (direction > 0).to(direction.dtype) for direction in judge_direction_rows
    ]
    corrective_rows = [
        (direction < 0).to(direction.dtype) for direction in judge_direction_rows
    ]
    positive_ratio = _masked_sample_mean(positive_rows, valid_rows, batch["loss_masks"])
    corrective_ratio = _masked_sample_mean(
        corrective_rows, valid_rows, batch["loss_masks"]
    )
    return (
        opd_loss,
        raw_loss,
        gate_mean,
        gate_active_ratio,
        teacher_gap_mean,
        directed_gap_mean,
        positive_ratio,
        corrective_ratio,
    )


def _async_residual_parts(args: Namespace, batch: dict, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token-mean |q0-rollout - current-student| diagnostics."""
    q0_rows = batch.get("opd_q0_log_probs")
    valid_rows = batch.get("opd_q0_valid")
    if not q0_rows or valid_rows is None:
        zero = logits.sum() * 0.0
        return zero, zero
    tp_group = mpu.get_tensor_model_parallel_group()
    values: list[torch.Tensor] = []
    directed: list[torch.Tensor] = []
    for i, (logits_chunk, tokens_chunk) in enumerate(get_responses(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )):
        is_valid = bool(valid_rows[i].item() if isinstance(valid_rows[i], torch.Tensor) else valid_rows[i])
        if not is_valid:
            continue
        q0 = torch.as_tensor(q0_rows[i], device=logits_chunk.device, dtype=torch.float32)
        student = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group).squeeze(-1).detach()
        if q0.numel() != student.numel():
            continue
        mask = batch["loss_masks"][i].to(device=student.device, dtype=student.dtype)
        denom = mask.sum().clamp_min(1.0)
        values.append(((q0 - student) * mask).sum() / denom)
        direction = float(batch.get("opd_step_polarity", [0])[i])
        directed.append((direction * (q0 - student) * mask).sum() / denom)
    if not values:
        zero = logits.sum() * 0.0
        return zero, zero
    return torch.stack(values).mean(), torch.stack(directed).mean()


def _topk_reverse_kl_parts(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return current-student-gated KL and its diagnostics."""
    teacher_logps = batch.get("teacher_topk_log_probs")
    teacher_indices = batch.get("teacher_topk_indices")
    if not teacher_logps or not teacher_indices:
        raise ValueError("GUI OPD requires teacher_topk_log_probs and teacher_topk_indices")
    if len(teacher_logps) != len(teacher_indices):
        raise ValueError("GUI OPD teacher top-k field counts differ")

    polarity_values = batch.get("opd_step_polarity")
    if polarity_values is None or len(polarity_values) != len(teacher_logps):
        raise ValueError("Top-k GUI OPD requires one opd_step_polarity per sample")

    weight_rows = batch.get("opd_token_weights")
    if weight_rows is None:
        # The weights are a validity mask; the gate itself is recomputed below.
        weight_rows = [
            [1.0] * int(response_length)
            for response_length in batch["response_lengths"]
        ]
    if len(weight_rows) != len(teacher_logps):
        raise ValueError("GUI OPD token-weight and target counts differ")

    tp_group = mpu.get_tensor_model_parallel_group()
    student_rows: list[torch.Tensor] = []
    teacher_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    direction_rows: list[torch.Tensor] = []

    for i, (logits_chunk, _tokens_chunk) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens"),
        )
    ):
        t_logps = torch.as_tensor(
            teacher_logps[i], device=logits_chunk.device, dtype=torch.float32
        )
        t_indices = torch.as_tensor(
            teacher_indices[i], device=logits_chunk.device, dtype=torch.long
        )
        if t_logps.ndim != 2 or t_indices.shape != t_logps.shape:
            raise ValueError(
                f"Invalid GUI OPD target shapes at sample {i}: {t_logps.shape}, {t_indices.shape}"
            )
        if t_logps.size(0) != logits_chunk.size(0):
            raise ValueError(
                f"GUI OPD response length mismatch at sample {i}: "
                f"teacher={t_logps.size(0)}, student={logits_chunk.size(0)}"
            )

        gathered = []
        for k in range(t_indices.size(1)):
            # Fused vocab-parallel CE may mutate its logits argument.
            gathered.append(
                compute_log_probs(
                    logits_chunk.clone(), t_indices[:, k], tp_group
                ).squeeze(-1)
            )
        student_topk = torch.stack(gathered, dim=-1)

        # Aggregate every token outside teacher top-k into one tail bucket.
        student_log_mass = torch.logsumexp(student_topk, dim=-1, keepdim=True).clamp(max=-1e-7)
        teacher_log_mass = torch.logsumexp(t_logps, dim=-1, keepdim=True).clamp(max=-1e-7)
        student_tail = torch.log(-torch.expm1(student_log_mass))
        teacher_tail = torch.log(-torch.expm1(teacher_log_mass))
        student_rows.append(torch.cat([student_topk, student_tail], dim=-1))
        teacher_rows.append(torch.cat([t_logps, teacher_tail], dim=-1))

        valid = torch.as_tensor(
            weight_rows[i], device=logits_chunk.device, dtype=torch.float32
        )
        valid, direction = _validity_and_direction(
            valid,
            t_logps[:, 0],
            polarity_values[i],
            sample_index=i,
        )
        valid_rows.append(valid)
        direction_rows.append(direction)

    per_token_rows = [
        F.kl_div(
            teacher,
            student,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        for student, teacher in zip(student_rows, teacher_rows, strict=True)
    ]
    kl_gap_rows = [row.detach() for row in per_token_rows]
    gate_rows = []
    for gap in kl_gap_rows:
        gate_input = float(getattr(args, "gui_opd_gate_beta", 5.0)) * gap
        if bool(getattr(args, "gui_opd_gate_reverse", False)):
            gate_input = -gate_input
        gate = ((gate_input > 0).to(dtype=gap.dtype)
                if bool(getattr(args, "gui_opd_hard_gate", False))
                else torch.sigmoid(gate_input)).detach()
        if not bool(getattr(args, "gui_opd_gate", True)):
            gate = torch.ones_like(gate)
        gate_rows.append(gate)
    raw_kl = _masked_sample_mean(per_token_rows, valid_rows, batch["loss_masks"])

    # Match SEED's masked-token normalization: the absolute gate magnitude
    # controls OPD strength instead of being cancelled by a gated-mass divisor.
    weighted_rows = [
        row * gate for row, gate in zip(per_token_rows, gate_rows, strict=True)
    ]
    gated_kl = _masked_sample_mean(weighted_rows, valid_rows, batch["loss_masks"])

    gate_mean = _masked_sample_mean(gate_rows, valid_rows, batch["loss_masks"])
    active_rows = [(gate > 0.5).to(dtype=gate.dtype) for gate in gate_rows]
    gate_active_ratio = _masked_sample_mean(
        active_rows, valid_rows, batch["loss_masks"]
    )
    teacher_gap_mean = _masked_sample_mean(
        kl_gap_rows, valid_rows, batch["loss_masks"]
    )
    directed_gap_rows = [
        direction * gap
        for direction, gap in zip(direction_rows, kl_gap_rows, strict=True)
    ]
    directed_gap_mean = _masked_sample_mean(
        directed_gap_rows, valid_rows, batch["loss_masks"]
    )
    positive_rows = [
        (direction > 0).to(direction.dtype) for direction in direction_rows
    ]
    corrective_rows = [
        (direction < 0).to(direction.dtype) for direction in direction_rows
    ]
    positive_ratio = _masked_sample_mean(positive_rows, valid_rows, batch["loss_masks"])
    corrective_ratio = _masked_sample_mean(
        corrective_rows, valid_rows, batch["loss_masks"]
    )
    return (
        gated_kl,
        raw_kl,
        gate_mean,
        gate_active_ratio,
        teacher_gap_mean,
        directed_gap_mean,
        positive_ratio,
        corrective_ratio,
    )


def _topk_reverse_kl(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute the confidence-gated top-k-plus-tail reverse KL."""
    return _topk_reverse_kl_parts(args, batch, logits, sum_of_sample_mean)[0]


def gui_opd_topk_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine trajectory GRPO/PPO policy loss with step-level GUI OPD."""
    policy_coef = float(getattr(args, "gui_opd_policy_loss_coef", 1.0))
    kl_coef = float(getattr(args, "gui_opd_kl_coef", 1.0))
    if policy_coef < 0 or kl_coef < 0 or (policy_coef == 0 and kl_coef == 0):
        raise ValueError(
            f"Invalid GUI OPD coefficients: policy={policy_coef}, kl={kl_coef}"
        )

    if policy_coef:
        policy_loss, metrics = policy_loss_function(args, batch, logits, sum_of_sample_mean)
    else:
        policy_loss = logits.sum() * 0.0
        metrics = {}
    loss_mode = str(getattr(args, "gui_opd_loss_mode", "sampled_token"))
    if loss_mode == "sampled_token":
        (
            opd_loss,
            raw_opd_loss,
            gate_mean,
            gate_active_ratio,
            teacher_gap_mean,
            directed_gap_mean,
            positive_ratio,
            corrective_ratio,
        ) = _sampled_token_opd_parts(args, batch, logits, sum_of_sample_mean)
        loss = policy_coef * policy_loss + kl_coef * opd_loss
    elif loss_mode == "topk":
        (
            opd_loss,
            raw_opd_loss,
            gate_mean,
            gate_active_ratio,
            teacher_gap_mean,
            directed_gap_mean,
            positive_ratio,
            corrective_ratio,
        ) = _topk_reverse_kl_parts(args, batch, logits, sum_of_sample_mean)
        loss = policy_coef * policy_loss + kl_coef * opd_loss
    else:
        raise ValueError(
            f"Unknown GUI OPD loss mode {loss_mode!r}; expected 'sampled_token' or 'topk'"
        )

    async_mean, directed_async_mean = _async_residual_parts(args, batch, logits)

    metrics = dict(metrics)
    metrics.update(
        {
            "loss": loss.detach().clone(),
            "opd_loss_mode_sampled_token": torch.tensor(
                float(loss_mode == "sampled_token"), device=loss.device
            ),
            "opd_loss_mode_topk": torch.tensor(float(loss_mode == "topk"), device=loss.device),
            "opd_judge_gate_enabled": torch.tensor(
                float(bool(getattr(args, "gui_opd_judge_gate", True))),
                device=loss.device,
            ),
            "opd_gate_mean": gate_mean.detach().clone(),
            "opd_gate_active_ratio": gate_active_ratio.detach().clone(),
            "opd_policy_component": (policy_coef * policy_loss).detach().clone(),
            "opd_component": (kl_coef * opd_loss).detach().clone(),
            # Compatibility name: this is the OPD component regardless of the
            # selected sampled-token or top-k implementation.
            "opd_kl_component": (kl_coef * opd_loss).detach().clone(),
            "opd_async_residual_mean": async_mean.detach().clone(),
            "opd_directed_async_residual_mean": directed_async_mean.detach().clone(),
            "opd_teacher_gap_mean": teacher_gap_mean.detach().clone(),
            "opd_directed_gap_mean": directed_gap_mean.detach().clone(),
            "opd_positive_token_ratio": positive_ratio.detach().clone(),
            "opd_corrective_token_ratio": corrective_ratio.detach().clone(),
        }
    )
    if loss_mode == "topk":
        metrics.update(
            {
                "opd_topk_reverse_kl": opd_loss.detach().clone(),
                "opd_raw_reverse_kl": raw_opd_loss.detach().clone(),
                "opd_topk_gate_kl_mean": teacher_gap_mean.detach().clone(),
            }
        )
    else:
        metrics.update(
            {
                "opd_sampled_token_loss": opd_loss.detach().clone(),
                "opd_sampled_token_raw_loss": raw_opd_loss.detach().clone(),
            }
        )
    return loss, metrics
