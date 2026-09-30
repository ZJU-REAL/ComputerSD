"""Majority-vote rewards for optional analyzer/PRM GRPO training.

The rollout analyzer already produces one record per vote.  This module keeps
the PRM reward contract independent from the GUI actor reward:

* ``consistency`` and ``effectiveness`` are majority-voted independently;
* a valid vote gets one point for matching each corresponding majority label;
* a malformed JSON vote gets ``0`` for both fields;
* an even-vote tie has no majority and therefore gives zero for that field.

The regular GUI GRPO+OPD actor path does not call ``prm_reward_func``.  The
optional dual-model PRM trainer uses the same reward contract while keeping
PRM samples and actor samples in separate train batches.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _majority_label(votes: Iterable[dict[str, Any]], field: str) -> int | None:
    labels = [
        int(vote[field])
        for vote in votes
        if not vote.get("format_error")
        and vote.get("ok")
        and isinstance(vote.get(field), int)
        and vote[field] in (0, 1)
    ]
    if not labels:
        return None
    ones = sum(labels)
    zeros = len(labels) - ones
    if ones == zeros:
        return None
    return 1 if ones > zeros else 0


def assign_majority_vote_rewards(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return independent majority labels and per-vote PRM rewards."""
    consistency_majority = _majority_label(votes, "consistency")
    effectiveness_majority = _majority_label(votes, "effectiveness")
    consistency_rewards: list[int] = []
    effectiveness_rewards: list[int] = []
    for vote in votes:
        valid = bool(vote.get("ok")) and not bool(vote.get("format_error"))
        consistency_rewards.append(
            int(valid and consistency_majority is not None and vote.get("consistency") == consistency_majority)
        )
        effectiveness_rewards.append(
            int(valid and effectiveness_majority is not None and vote.get("effectiveness") == effectiveness_majority)
        )

    for vote, c_reward, e_reward in zip(
        votes,
        consistency_rewards,
        effectiveness_rewards,
    ):
        vote["consistency_reward"] = c_reward
        vote["effectiveness_reward"] = e_reward
        vote["majority_reward"] = c_reward + e_reward

    return {
        "consistency_majority": consistency_majority,
        "effectiveness_majority": effectiveness_majority,
        "consistency_rewards": consistency_rewards,
        "effectiveness_rewards": effectiveness_rewards,
        "majority_rewards": [c + e for c, e in zip(consistency_rewards, effectiveness_rewards)],
    }


def select_representative_vote(
    votes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the longest non-empty guidance among valid best-reward votes."""
    valid = [
        vote
        for vote in votes
        if vote.get("ok") and not vote.get("format_error")
        and str(vote.get("guidance", "") or "").strip()
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda vote: (
            int(vote.get("majority_reward", 0) or 0),
            len(str(vote.get("guidance", "") or "")),
        ),
    )


def _sample_prm_reward(sample: Any) -> dict[str, float]:
    metadata = getattr(sample, "metadata", None)
    record = metadata.get("prm_vote") if isinstance(metadata, dict) else None
    if not isinstance(record, dict):
        return {"score": 0.0, "consistency": 0.0, "effectiveness": 0.0}
    return {
        "score": float(record.get("majority_reward", 0.0) or 0.0),
        "consistency": float(record.get("consistency_reward", 0.0) or 0.0),
        "effectiveness": float(record.get("effectiveness_reward", 0.0) or 0.0),
    }


async def prm_reward_func(args: Any, sample: Any, **_kwargs: Any) -> dict[str, float]:
    """Slime custom reward interface for a PRM vote training sample."""
    del args
    return _sample_prm_reward(sample)


def prm_post_process_rewards(
    args: Any,
    samples: list[Any],
    **_kwargs: Any,
) -> tuple[list[Any], list[Any]]:
    """Return raw and normalized PRM rewards without changing their values."""
    del args
    rewards = [_sample_prm_reward(sample) for sample in samples]
    return rewards, rewards


def collect_prm_vote_samples(actor_samples: list[Any]) -> list[Any]:
    """Materialize one trainable Sample for each analyzer response vote."""
    from slime.utils.types import Sample

    step_records: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    seen_steps: set[tuple[int, int, int]] = set()
    for position, actor_sample in enumerate(actor_samples):
        metadata = getattr(actor_sample, "metadata", None)
        prm = metadata.get("prm") if isinstance(metadata, dict) else None
        details = prm.get("step_details") if isinstance(prm, dict) else None
        if not isinstance(details, list):
            continue
        actor_group = int(actor_sample.group_index) if actor_sample.group_index is not None else -1
        actor_index = int(actor_sample.index) if actor_sample.index is not None else position
        for detail in details:
            if not isinstance(detail, dict):
                continue
            step_index = int(detail.get("step_index", -1))
            step_key = (actor_group, actor_index, step_index)
            if step_key in seen_steps:
                continue
            seen_steps.add(step_key)
            step_records.append((step_key, detail))

    prm_samples: list[Sample] = []
    for prm_group_index, (step_key, detail) in enumerate(step_records):
        prompt = detail.get("prm_train_prompt")
        votes = detail.get("votes")
        if not isinstance(prompt, dict) or not isinstance(votes, list):
            continue
        prompt_tokens = prompt.get("tokens")
        if not isinstance(prompt_tokens, list) or not prompt_tokens:
            continue
        multimodal_train_inputs = prompt.get("multimodal_train_inputs")
        for vote_id, vote in enumerate(votes):
            if not isinstance(vote, dict):
                continue
            output_token_ids = vote.get("output_token_ids")
            if not isinstance(output_token_ids, list) or not output_token_ids:
                # HTTP/request failures have no generated response and are not
                # policy samples. Malformed JSON responses do have tokens and
                # remain in the group with reward zero.
                continue
            response_length = len(output_token_ids)
            vote_metadata = {
                "actor_group_index": step_key[0],
                "actor_sample_index": step_key[1],
                "step_index": step_key[2],
                "vote_id": vote_id,
                "ok": bool(vote.get("ok")),
                "format_error": bool(vote.get("format_error")),
                "consistency": vote.get("consistency"),
                "effectiveness": vote.get("effectiveness"),
                "consistency_reward": int(vote.get("consistency_reward", 0) or 0),
                "effectiveness_reward": int(vote.get("effectiveness_reward", 0) or 0),
                "majority_reward": int(vote.get("majority_reward", 0) or 0),
            }
            prm_samples.append(
                Sample(
                    group_index=prm_group_index,
                    index=len(prm_samples),
                    tokens=[int(token) for token in prompt_tokens + output_token_ids],
                    response=str(vote.get("raw_text", "") or ""),
                    response_length=response_length,
                    reward={
                        "score": float(vote_metadata["majority_reward"]),
                        "consistency": float(vote_metadata["consistency_reward"]),
                        "effectiveness": float(vote_metadata["effectiveness_reward"]),
                    },
                    loss_mask=[1] * response_length,
                    multimodal_train_inputs=multimodal_train_inputs,
                    status=Sample.Status.COMPLETED,
                    metadata={"prm_vote": vote_metadata},
                )
            )
    return prm_samples


def build_prm_train_data(
    args: Any,
    prm_samples: list[Any],
    *,
    min_samples: int = 1,
) -> dict[str, Any]:
    """Convert PRM vote samples into one pre-normalized GRPO train batch."""
    from slime.rollout.grpo_utils import normalize_non_dynamic_rewards

    raw_rewards = [float(sample.reward["score"]) for sample in prm_samples]
    rewards = (
        normalize_non_dynamic_rewards(
            prm_samples,
            raw_rewards,
            n_samples_per_prompt=max(2, int(getattr(args, "prm_m", 8) or 8)),
            std_normalization=bool(getattr(args, "grpo_std_normalization", True)),
            singleton_coef=0.0,
        )
        if prm_samples
        else []
    )
    rollout_ids = list(range(len(prm_samples)))
    loss_masks = [sample.loss_mask for sample in prm_samples]
    data: dict[str, Any] = {
        "tokens": [sample.tokens for sample in prm_samples],
        "response_lengths": [sample.response_length for sample in prm_samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [0] * len(prm_samples),
        "sample_indices": [sample.index for sample in prm_samples],
        "rollout_ids": rollout_ids,
        "loss_masks": loss_masks,
        "rollout_mask_sums": [sum(mask) for mask in loss_masks],
    }
    if any(sample.multimodal_train_inputs is not None for sample in prm_samples):
        data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in prm_samples]
    # Keep a PRM optimizer step alive even when every analyzer request failed
    # before producing output tokens. These samples are deliberately masked
    # out, so they exercise the same scheduling/update path without changing
    # the model.
    dummy_count = max(0, int(min_samples) - len(prm_samples))
    for dummy_id in range(dummy_count):
        data["tokens"].append([0, 0])
        data["response_lengths"].append(1)
        data["rewards"].append(0.0)
        data["raw_reward"].append(0.0)
        data["truncated"].append(0)
        data["sample_indices"].append(-(dummy_id + 1))
        data["rollout_ids"].append(len(data["rollout_ids"]))
        data["loss_masks"].append([0])
        data["rollout_mask_sums"].append(0)
        if "multimodal_train_inputs" in data:
            data["multimodal_train_inputs"].append(None)
    return data


__all__ = [
    "assign_majority_vote_rewards",
    "select_representative_vote",
    "prm_reward_func",
    "prm_post_process_rewards",
    "collect_prm_vote_samples",
    "build_prm_train_data",
]
