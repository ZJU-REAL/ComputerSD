import logging

import ray

from slime.ray.placement_group import (
    create_placement_groups,
    create_prm_training_model,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action


logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args, *, fully_async=True):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor, optional critic, and optional independent PRM trainer
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    prm_model = create_prm_training_model(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()
    if prm_model is not None:
        prm_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare", model_name="actor"))
        if prm_model is not None:
            ray.get(
                rollout_manager.check_weights.remote(
                    action="compare",
                    model_name=str(getattr(args, "prm_model_name", "analyzer")),
                )
            )

    # The serial OPD baseline reuses model setup/update logic without prefetch.
    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

    def role_data(data_bundle, role):
        if isinstance(data_bundle, dict) and "actor" in data_bundle:
            return data_bundle.get(role)
        if role == "actor":
            return data_bundle
        return None

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if not fully_async and rollout_data_next_future is None:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id)
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if fully_async and rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        actor_data_ref = role_data(rollout_data_curr_ref, "actor")
        prm_data_ref = role_data(rollout_data_curr_ref, "prm")
        skip_actor_train = (
            isinstance(rollout_data_curr_ref, dict)
            and bool(rollout_data_curr_ref.get("skip_actor_train", False))
        )

        actor_trains_this_step = True
        if skip_actor_train:
            # The rollout contained only constant-reward trajectories whose
            # OPD gates were all zero. Advancing the optimizer or scheduler
            # here would be a pure no-op with substantial compute cost.
            actor_trains_this_step = False
            logger.warning(
                "Skipping actor optimizer update for rollout %s: no trainable GRPO+OPD samples.",
                rollout_id,
            )
        elif args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, actor_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, actor_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, actor_data_ref))

        if prm_model is not None:
            # PRM training data is independent of whether this rollout happened
            # to contain an actor GRPO/OPD gradient.
            if prm_data_ref is None:
                raise RuntimeError(
                    f"TRAIN_PRM rollout {rollout_id} did not return PRM train data"
                )
            ray.get(prm_model.async_train(rollout_id, prm_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if prm_model is not None:
                prm_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if not fully_async or (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            if fully_async:
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()
            if prm_model is not None:
                prm_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
