"""Serial GRPO+OPD trainer with optional analyzer self-training."""

from slime.utils.arguments import parse_args
from train_fully_async import train


if __name__ == "__main__":
    args = parse_args()
    if args.rollout_function_path != "rollout_fast.phased_rollout.generate_rollout_phased":
        raise ValueError("The serial OPD trainer requires the batch-phased rollout entry")
    train(args, fully_async=False)
