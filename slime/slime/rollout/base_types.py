from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample


@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


def flatten_rollout_samples(node: Any) -> list[Sample]:
    """Flatten possibly mixed rollout fan-out output into ``list[Sample]``.

    A dynamic-history group can contain both ``list[Sample]`` from completed
    trajectories and a bare ``Sample`` from an early-abort path. Flattening one
    rectangular nesting level at a time cannot handle that mixed shape.
    """
    flattened: list[Sample] = []

    def _visit(item: Any, path: str) -> None:
        if isinstance(item, Sample):
            flattened.append(item)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                _visit(child, f"{path}[{index}]")
            return
        raise TypeError(
            f"Unexpected rollout output at {path}: expected Sample/list/tuple, "
            f"got {type(item).__name__}"
        )

    _visit(node, "samples")
    return flattened


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
