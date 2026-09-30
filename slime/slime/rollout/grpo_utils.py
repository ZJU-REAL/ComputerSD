import math
from collections import Counter, defaultdict
from typing import Any, Hashable, Mapping, Sequence


def prepare_removed_samples(samples: Sequence[Any]) -> int:
    """Turn removed trajectories into model-safe, zero-gradient placeholders.

    The samples stay in their original positions so rollout batching and prompt
    grouping remain stable. Sequence-aligned fields that are already present
    are shortened to the placeholder response length as well.
    """
    has_rollout_log_probs = any(
        not sample.remove_sample and sample.rollout_log_probs is not None for sample in samples
    )
    has_teacher_log_probs = any(
        not sample.remove_sample and sample.teacher_log_probs is not None for sample in samples
    )

    prepared = 0
    for sample in samples:
        if not sample.remove_sample:
            continue

        sample.tokens = [0, 0]
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.multimodal_train_inputs = None
        sample.rollout_log_probs = [0.0] if has_rollout_log_probs else None
        sample.teacher_log_probs = [0.0] if has_teacher_log_probs else None
        sample.teacher_topk_log_probs = None
        sample.teacher_topk_indices = None
        if hasattr(sample, "opd_token_weights"):
            sample.opd_token_weights = None
        if hasattr(sample, "opd_step_polarity"):
            sample.opd_step_polarity = 0
        prepared += 1
    return prepared


def normalize_non_dynamic_rewards(
    samples: Sequence[Any],
    raw_rewards: Sequence[float],
    *,
    n_samples_per_prompt: int,
    std_normalization: bool,
    singleton_coef: float = 1.0,
) -> list[float]:
    """Normalize trajectory rewards per prompt while ignoring removed samples.

    ``group_index`` is the authoritative prompt identity. Older/custom rollout
    paths that do not set it fall back to positional groups of
    ``n_samples_per_prompt`` instead of collapsing the whole batch into one
    reward group.
    """
    if len(samples) != len(raw_rewards):
        raise ValueError(f"samples/rewards length mismatch: {len(samples)} != {len(raw_rewards)}")
    if n_samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples_per_prompt}")
    if singleton_coef < 0:
        raise ValueError(f"singleton_coef must be non-negative, got {singleton_coef}")

    groups: dict[Hashable, list[tuple[int, float]]] = defaultdict(list)
    for position, (sample, reward) in enumerate(zip(samples, raw_rewards, strict=True)):
        if sample.remove_sample:
            continue
        if sample.group_index is not None:
            group_key: Hashable = ("group_index", int(sample.group_index))
        else:
            group_key = ("position", position // n_samples_per_prompt)
        groups[group_key].append((position, float(reward)))

    normalized = [0.0] * len(samples)
    for entries in groups.values():
        if len(entries) == 1:
            position, reward = entries[0]
            normalized[position] = singleton_coef * reward
            continue

        mean = sum(reward for _, reward in entries) / len(entries)
        centered = [(position, reward - mean) for position, reward in entries]

        if std_normalization:
            variance = sum(value * value for _, value in centered) / (len(centered) - 1)
            denominator = math.sqrt(variance) + 1e-6
            centered = [(position, value / denominator) for position, value in centered]

        for position, reward in centered:
            normalized[position] = reward

    return normalized


def broadcast_dynamic_trajectory_advantages(
    sample_keys: Sequence[Hashable],
    normalized_by_key: Mapping[Hashable, float],
    *,
    scaling: str,
) -> list[float]:
    """Broadcast trajectory advantages to dynamic-history step samples.

    ``dynamic_history`` turns one trajectory into ``T_i`` independently reduced
    step samples. Let ``mean_T`` be the mean retained step count across distinct
    trajectories in the current rollout batch. ``linear`` uses
    ``A_traj * mean_T / T_i`` and ``sqrt`` uses the softer
    ``A_traj * sqrt(mean_T / T_i)``. The batch mean factor keeps the average
    advantage scale unchanged while removing (or reducing) relative long-
    trajectory overweighting. Inputs must already be trajectory-normalized, so
    scaling never changes the prompt-level GRPO baseline or standard deviation.
    """
    if scaling not in {"none", "sqrt", "linear"}:
        raise ValueError(
            "dynamic trajectory advantage scaling must be one of none, sqrt, linear; "
            f"got {scaling!r}"
        )

    counts = Counter(sample_keys)
    if not counts:
        return []
    mean_step_count = len(sample_keys) / len(counts)
    values: list[float] = []
    for key in sample_keys:
        advantage = float(normalized_by_key[key])
        if scaling == "sqrt":
            advantage *= math.sqrt(mean_step_count / counts[key])
        elif scaling == "linear":
            advantage *= mean_step_count / counts[key]
        values.append(advantage)
    return values


def combine_dynamic_gigpo_advantages(
    sample_keys: Sequence[Hashable],
    trajectory_advantage_by_key: Mapping[Hashable, float],
    weighted_step_advantages: Sequence[float],
    *,
    scaling: str,
) -> tuple[list[float], list[float]]:
    """Scale only GiGPO's trajectory branch, then add its step branch."""
    if len(sample_keys) != len(weighted_step_advantages):
        raise ValueError(
            "sample keys and weighted step advantages must have equal length: "
            f"{len(sample_keys)} != {len(weighted_step_advantages)}"
        )
    scaled_trajectory_advantages = broadcast_dynamic_trajectory_advantages(
        sample_keys, trajectory_advantage_by_key, scaling=scaling
    )
    advantages = [
        trajectory_advantage + float(step_advantage)
        for trajectory_advantage, step_advantage in zip(
            scaled_trajectory_advantages, weighted_step_advantages, strict=True
        )
    ]
    return scaled_trajectory_advantages, advantages
