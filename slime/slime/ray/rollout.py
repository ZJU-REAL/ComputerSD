import dataclasses
import logging
import math
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.external import start_external_rollout_servers
from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn, flatten_rollout_samples
from slime.rollout.grpo_utils import (
    broadcast_dynamic_trajectory_advantages,
    combine_dynamic_gigpo_advantages,
    normalize_non_dynamic_rewards,
    prepare_removed_samples,
)
from slime.utils import logging_utils
from slime.utils.dp_schedule import build_dp_schedule, dynamic_alignment_padding_rollout_ids
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .rollout_validation import validate_server_group_gpu_indices
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", "decode", or "placeholder"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
        validate_server_group_gpu_indices(
            worker_type=self.worker_type,
            gpu_offset=self.gpu_offset,
            num_gpus_per_engine=self.num_gpus_per_engine,
            num_gpu_per_engine=num_gpu_per_engine,
            num_engines=len(self.all_engines),
            num_available_gpus=len(reordered_gpu_ids),
            rollout_num_gpus=self.args.rollout_num_gpus,
            rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
        )

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
                    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    "SLIME_ENABLE_PROFILING": "true",
                }.items()
            }
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        # Compute base_port from the maximum cursor across all nodes that
        # this group's engines may land on (conservative: just use global max).
        base_port = max(port_cursors.values()) if port_cursors else 15000
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            args=self.args,
            rollout_engines=rollout_engines,
            worker_type=self.worker_type,
            num_gpus_per_engine=self.num_gpus_per_engine,
            rank_offset=self.rank_offset,
            base_port=base_port,
        )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups.

        Used instead of ``resume_memory_occupation(tags=[WEIGHTS])`` so that
        CPU memory is not consumed by offloaded weight copies.
        """
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, Any] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _get_metrics_router_addr(self) -> str | None:
        """Return the router address for scraping SGLang engine metrics.

        The sglang_router gateway exposes ``/engine_metrics`` on its main port,
        which aggregates Prometheus metrics from all backend sglang servers.
        Returns ``http://{ip}:{port}`` for the first server, or ``None`` when
        metrics are disabled or no servers are running.
        """
        srv = self.server
        if srv is None or srv.router_ip is None:
            return None
        return f"http://{srv.router_ip}:{srv.router_port}"

    def get_metrics_router_addr(self) -> str | None:
        """Public wrapper for remote calls from the driver process."""
        return self._get_metrics_router_addr()

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if (
            self.server
            and self.server.server_groups
            and self.server.server_groups[0].all_engines
            and self.server.server_groups[0].all_engines[0]
        ):
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()
        logging_utils.finish_tracking(self.args)

    @property
    def server(self) -> Any | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self, model_name: str | None = None) -> Any | None:
        """Return the server with ``update_weights=True``.

        ``model_name`` makes dual actor/PRM training explicit; callers that do
        not provide one retain the legacy first-updatable-model behavior.
        """
        if model_name is not None:
            srv = self.servers.get(model_name)
            if srv is None:
                raise KeyError(
                    f"unknown rollout model {model_name!r}; available={list(self.servers)}"
                )
            if not srv.update_weights:
                raise ValueError(f"rollout model {model_name!r} is not configured for weight updates")
            return srv
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self, model_name: str | None = None):
        """Return engines eligible for weight updates.

        With ``model_name``, returns that model's engines after validating it is
        updatable. Without a name, retains the legacy first-updatable behavior.
        """
        srv = self._get_updatable_server(model_name)
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        prm_samples = None
        if getattr(self.args, "train_prm", False):
            from reward.prm_grpo import collect_prm_vote_samples

            prm_samples = collect_prm_vote_samples(data)

        actor_data = self._convert_samples_to_train_data(data)
        if actor_data is None:
            logger.warning(
                "Skipping actor update for rollout %s: no GRPO or active GUI OPD signal remained.",
                rollout_id,
            )
            actor_refs = None
        else:
            actor_refs = self._split_train_data_by_dp(actor_data)
        if not getattr(self.args, "train_prm", False):
            if actor_refs is None:
                # Keep the return shape explicit so train drivers can skip this
                # optimizer update without treating it as a Ray failure.
                return {"actor": None, "skip_actor_train": True}
            return actor_refs

        from reward.prm_grpo import build_prm_train_data

        dp_size = int(self.train_parallel_config["dp_size"])
        prm_data = build_prm_train_data(
            self.args,
            prm_samples or [],
            min_samples=dp_size,
        )
        prm_refs = self._split_train_data_by_dp(
            prm_data,
            global_batch_size_override=len(prm_data["tokens"]),
        )
        return {
            "actor": actor_refs,
            "prm": prm_refs,
            "prm_sample_count": len(prm_samples or []),
            "skip_actor_train": actor_refs is None,
        }

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self, model_name: str | None = None):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server(model_name)
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self, model_name: str | None = None):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server(model_name)
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str, model_name: str | None = None):
        engines = (
            self.servers[model_name].engines
            if model_name is not None
            else self.rollout_engines
        )
        return ray.get([engine.check_weights.remote(action=action) for engine in engines])

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            # Enforce the rollout_id contract before flattening: any list[Sample]
            # encountered in the nested output must have rollout_id set on every
            # element. Default rollouts inherit it from the data source; compact /
            # subagent paths that split one rollout into N training samples must
            # set the same rollout_id on every sibling so the loss reducer counts
            # the rollout once instead of N times.
            #
            # dynamic_history is exempt: it intentionally expands one trajectory
            # into many independent step samples (cua-style per-step loss) and
            # leaves rollout_id unset, so each step falls back to a unique id
            # (rollout_id == range(len)) and is trained as its own rollout. The
            # contract above would otherwise reject that expanded shape.
            if not getattr(self.args, "dynamic_history", False):
                _validate_rollout_id_annotated(data)
            # Dynamic-history fan-out may be ragged: completed trajectories
            # return list[Sample], while an early-abort path can return a bare
            # Sample in the same group. Flatten by leaf type instead of assuming
            # every node at a nesting level has the same shape.
            data = flatten_rollout_samples(data)

        return data, metrics

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]

        # GiGPO computes trajectory and state-relative step advantages inside
        # the prompt-group reward function, but trajectory-length scaling needs
        # the complete cleaned rollout batch to obtain one batch-wide mean(T).
        # Recombine the two recorded components here after scaling only A_traj.
        if getattr(self.args, "dynamic_history", False) and self.args.advantage_estimator == "gigpo":
            key_by_sample: list[tuple[int, int]] = []
            trajectory_advantage_by_key: dict[tuple[int, int], float] = {}
            weighted_step_advantages: list[float] = []
            diagnostics: list[dict] = []
            for i, sample in enumerate(samples):
                group_idx = int(sample.group_index) if sample.group_index is not None else -1
                traj_idx = int(sample.index) if sample.index is not None else i
                key = (group_idx, traj_idx)
                key_by_sample.append(key)

                meta = sample.metadata if isinstance(sample.metadata, dict) else {}
                gigpo_meta = meta.get("gigpo") if isinstance(meta.get("gigpo"), dict) else {}
                diag = gigpo_meta.get("gigpo_diag") if isinstance(gigpo_meta.get("gigpo_diag"), dict) else {}
                if "trajectory_advantage" not in diag or "weighted_step_advantage" not in diag:
                    raise RuntimeError(
                        "Dynamic GiGPO requires trajectory_advantage and weighted_step_advantage diagnostics"
                    )
                diagnostics.append(diag)
                trajectory_advantage_by_key.setdefault(key, float(diag["trajectory_advantage"]))
                weighted_step_advantages.append(float(diag["weighted_step_advantage"]))

            scaled_trajectory_advantages, rewards = combine_dynamic_gigpo_advantages(
                key_by_sample,
                trajectory_advantage_by_key,
                weighted_step_advantages,
                scaling=getattr(self.args, "dynamic_trajectory_advantage_scaling", "none"),
            )
            for diag, trajectory_advantage, advantage in zip(
                diagnostics, scaled_trajectory_advantages, rewards, strict=True
            ):
                diag["trajectory_advantage_scaled"] = trajectory_advantage
                diag["advantage"] = advantage
            return raw_rewards, rewards

        if not (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            return raw_rewards, raw_rewards

        std_norm = (self.args.advantage_estimator in ["grpo", "gspo"]) and self.args.grpo_std_normalization

        if getattr(self.args, "dynamic_history", False):
            # dynamic_history + GRPO/GSPO:
            # One environment trajectory is expanded into many step-level
            # training samples that all share the same ``(group_index, index)``.
            # Reward/advantage normalization stays at the TRAJECTORY level:
            #   1. de-duplicate by (group_index, index) so each distinct
            #      trajectory contributes exactly one outcome reward;
            #   2. run GRPO normalization (subtract group mean, optionally
            #      divide by std) over those distinct-trajectory outcomes;
            #   3. optionally scale it by mean(T)/T or sqrt(mean(T)/T), where T
            #      is the retained step count and mean(T) is batch-wide;
            #   4. broadcast the result to every step sample of the trajectory.
            # This keeps trajectory-level RL semantics: a long trajectory that
            # produces more step samples is NOT counted multiple times in the
            # group mean/std.
            def _normalize_vals(vals: torch.Tensor) -> torch.Tensor:
                vals = vals - vals.mean()
                if std_norm:
                    if len(vals) > 1:
                        vals = vals / (vals.std() + 1e-6)
                    else:
                        vals = torch.zeros_like(vals)
                return vals

            traj_reward_by_key: dict[tuple[int, int], float] = {}
            group_to_keys: dict[int, list[tuple[int, int]]] = {}
            key_by_sample: list[tuple[int, int]] = []
            for i, sample in enumerate(samples):
                group_idx = int(sample.group_index) if sample.group_index is not None else -1
                traj_idx = int(sample.index) if sample.index is not None else i
                key = (group_idx, traj_idx)
                key_by_sample.append(key)
                if key not in traj_reward_by_key:
                    traj_reward_by_key[key] = float(raw_rewards[i])
                    group_to_keys.setdefault(group_idx, []).append(key)

            normalized_by_key: dict[tuple[int, int], float] = {}
            for _, keys in group_to_keys.items():
                vals = torch.tensor([traj_reward_by_key[k] for k in keys], dtype=torch.float32)
                vals = _normalize_vals(vals)
                for j, key in enumerate(keys):
                    normalized_by_key[key] = float(vals[j].item())

            rewards = broadcast_dynamic_trajectory_advantages(
                key_by_sample,
                normalized_by_key,
                scaling=getattr(self.args, "dynamic_trajectory_advantage_scaling", "none"),
            )
            return raw_rewards, rewards

        rewards = normalize_non_dynamic_rewards(
            samples,
            raw_rewards,
            n_samples_per_prompt=self.args.n_samples_per_prompt,
            std_normalization=std_norm,
            singleton_coef=float(getattr(self.args, "grpo_singleton_reinforce_coef", 1.0)),
        )
        return raw_rewards, rewards

    def _clean_dynamic_history_samples(self, samples: list[Sample]) -> list[Sample]:
        """Three-step cleaning before reward normalization (dynamic_history).

        1. Mark step samples missing ``multimodal_train_inputs`` as removable
           (only when the batch otherwise carries multimodal inputs); this may
           drop a subset of one trajectory's steps.
        2. Drop marked samples. For GUI OPD, retain a zero-std reward group
           only where an unmasked OPD token can still supply a gradient.
        3. If fewer than dp_size samples remain, pad with dummy samples whose
           loss_mask is all zeros so they contribute no gradient.
        """

        def _mark_missing_multimodal(samples: list[Sample]) -> None:
            if any(sample.multimodal_train_inputs is not None for sample in samples):
                missing = []
                for sample in samples:
                    if sample.multimodal_train_inputs is None and not sample.remove_sample:
                        sample.remove_sample = True
                        missing.append(sample.index)
                if missing:
                    logger.warning(
                        "Marked %d samples non-trainable due to missing multimodal_train_inputs: indices=%s",
                        len(missing),
                        missing[:20],
                    )

        def _make_dummy_samples(count: int) -> list[Sample]:
            reward = {self.args.reward_key or "score": 0.0}
            return [
                Sample(
                    group_index=-(i + 1),
                    index=-(i + 1),
                    tokens=[0, 0],
                    response_length=1,
                    loss_mask=[0],
                    rollout_log_probs=[0.0],
                    reward=reward,
                    remove_sample=True,
                    status=Sample.Status.FAILED,
                    metadata={"dummy_removed_sample": True},
                )
                for i in range(count)
            ]

        _mark_missing_multimodal(samples)
        samples = [sample for sample in samples if not sample.remove_sample]
        if getattr(self.args, "gui_opd_enable", False):
            samples = self._drop_zero_gradient_gui_opd_samples(samples)
        else:
            samples = self._drop_constant_reward_groups(samples)

        # There is no GRPO or OPD gradient left in this rollout. Let the
        # async driver skip the optimizer step rather than scheduling a
        # model-wide forward/backward over only zero-loss placeholders.
        if not samples:
            logger.warning("No trainable dynamic-history samples remain after GRPO+OPD cleaning.")
            return []

        dp_size = self.train_parallel_config["dp_size"]
        if len(samples) < dp_size:
            logger.warning("Injecting %d dummy samples.", dp_size - len(samples))
            samples.extend(_make_dummy_samples(dp_size - len(samples)))
        return samples

    def _drop_zero_gradient_gui_opd_samples(self, samples: list[Sample]) -> list[Sample]:
        """Remove only steps that have neither GRPO nor GUI-OPD gradient.

        Dynamic-history GRPO normalizes once per trajectory outcome and then
        broadcasts the result to every retained step.  Therefore individual
        steps from a non-constant prompt group must remain intact.  A constant
        group has a zero GRPO advantage for every trajectory, however; within
        those groups we keep exactly the steps with a valid OPD target on at
        least one loss-contributing token.
        """
        if not samples:
            return samples
        if self.args.advantage_estimator not in ["grpo", "gspo"] or not self.args.rewards_normalization:
            return samples

        def _has_active_opd_token(sample: Sample) -> bool:
            weights = sample.opd_token_weights
            if weights is None:
                return False
            if int(getattr(sample, "opd_step_polarity", 0) or 0) == 0:
                return False
            mask = sample.loss_mask
            if mask is None:
                mask = [1] * int(sample.response_length)
            if len(weights) != int(sample.response_length) or len(mask) != int(sample.response_length):
                # Malformed targets are rejected later as well. They must not
                # keep an otherwise all-zero reward group alive here.
                return False
            return any(
                int(mask_value) != 0
                and math.isfinite(float(weight))
                and float(weight) > 0.0
                for mask_value, weight in zip(mask, weights, strict=True)
            )

        raw_rewards = [float(sample.get_reward_value(self.args)) for sample in samples]
        group_to_indices: dict[int, list[int]] = {}
        for i, sample in enumerate(samples):
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            group_to_indices.setdefault(group_idx, []).append(i)

        keep_positions: set[int] = set()
        constant_groups: list[int] = []
        dropped_steps = 0
        retained_opd_steps = 0
        for group_idx, idxs in group_to_indices.items():
            values = [raw_rewards[i] for i in idxs]
            is_constant = bool(values) and max(values) - min(values) <= 1e-12
            if not is_constant:
                keep_positions.update(idxs)
                continue

            constant_groups.append(group_idx)
            for i in idxs:
                if _has_active_opd_token(samples[i]):
                    keep_positions.add(i)
                    retained_opd_steps += 1
                else:
                    dropped_steps += 1

        filtered = [sample for i, sample in enumerate(samples) if i in keep_positions]
        if dropped_steps:
            logger.warning(
                "Dropped zero-gradient GUI OPD steps from constant-reward groups: "
                "groups=%s retained_opd_steps=%d dropped_steps=%d samples %d -> %d",
                constant_groups,
                retained_opd_steps,
                dropped_steps,
                len(samples),
                len(filtered),
            )
        return filtered

    def _drop_constant_reward_groups(self, samples: list[Sample]) -> list[Sample]:
        """Drop GRPO/GSPO groups whose (trajectory) rewards are all identical.

        A constant-reward group has zero std, so GRPO advantages are all 0 and
        the group contributes no gradient. Because every step sample of one
        trajectory shares the same outcome reward, deduplicating is unnecessary
        here: judging "all equal" over the expanded step samples is equivalent
        to judging it over distinct trajectories. Keep at least one group so the
        training batch is never empty.
        """
        if not samples:
            return samples
        if self.args.advantage_estimator not in ["grpo", "gspo"] or not self.args.rewards_normalization:
            return samples

        raw_rewards = [float(sample.get_reward_value(self.args)) for sample in samples]
        group_to_indices: dict[int, list[int]] = {}
        for i, sample in enumerate(samples):
            group_idx = int(sample.group_index) if sample.group_index is not None else -1
            group_to_indices.setdefault(group_idx, []).append(i)

        constant_groups: list[int] = []
        for group_idx, idxs in group_to_indices.items():
            vals = [raw_rewards[i] for i in idxs]
            if len(vals) == 0:
                continue
            if max(vals) - min(vals) <= 1e-12:
                constant_groups.append(group_idx)

        if not constant_groups:
            return samples

        keep_groups = [g for g in group_to_indices.keys() if g not in set(constant_groups)]
        dropped_groups = list(constant_groups)
        if not keep_groups:
            # Keep one full group so the batch is never empty.
            keep_group = next(iter(group_to_indices.keys()))
            keep_groups = [keep_group]
            dropped_groups = [g for g in constant_groups if g != keep_group]

        keep_set = set(keep_groups)
        filtered_samples = [
            sample
            for sample in samples
            if (int(sample.group_index) if sample.group_index is not None else -1) in keep_set
        ]

        if len(filtered_samples) != len(samples):
            logger.warning(
                "Dropped constant-reward groups for %s: dropped=%s kept=%s samples %d -> %d",
                self.args.advantage_estimator,
                dropped_groups,
                keep_groups,
                len(samples),
                len(filtered_samples),
            )
        return filtered_samples

    def _compute_dynamic_global_batch_size(self, num_samples: int, target_steps: int | None = None) -> int:
        """Derive a per-rollout global_batch_size from the actual collected
        step-sample count when dynamic_history expands one trajectory into a
        variable number of step samples.

        ``build_dp_schedule`` treats ``global_batch_size`` as "rollouts per
        step", and under dynamic_history each step sample carries a unique
        ``rollout_id`` (no sharing), so a rollout == one step sample here.
        We choose gbs so the realized number of training steps stays close to
        ``target_steps`` (``num_steps_per_rollout``). For the GUI default of one
        optimizer step, the realized global batch is exactly ``num_samples``;
        uneven DP partitioning is supported by ``build_dp_schedule`` and no
        trailing GUI step is discarded. Falls back to one step if target_steps
        is unset.
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        desired_steps = int(target_steps) if target_steps is not None and target_steps > 0 else 1
        per_step_target = max(1, num_samples // desired_steps)
        dynamic_gbs = per_step_target

        if dynamic_gbs < dp_size:
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        realized_steps = max(1, num_samples // dynamic_gbs)
        wasted = num_samples % dynamic_gbs
        if dynamic_gbs != original_gbs or wasted > 0 or realized_steps != desired_steps:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, "
                f"target_steps={desired_steps}, realized_steps={realized_steps}, wasted={wasted})"
            )
        return dynamic_gbs

    def _append_dynamic_alignment_dummies(self, data: dict, *, global_batch_size: int) -> None:
        """Pad dynamic DP steps with zero-loss samples after reward processing.

        Each dummy shares an existing rollout id in the step it pads. This
        keeps the number of distinct rollouts, loss denominator, and LR
        scheduler increment based on real rollouts only.
        """
        sample_count = len(data["tokens"])
        if sample_count == 0:
            return
        if "alignment_dummy_mask" in data:
            raise RuntimeError("dynamic alignment dummies were added more than once")

        dp_size = int(self.train_parallel_config["dp_size"])
        vpp_size = int(self.train_parallel_config["vpp_size"])
        mb_group = int(self.train_parallel_config["microbatch_group_size_per_vp_stage"])
        align_to = dp_size * (mb_group if vpp_size > 1 else 1)
        padding_rollout_ids = dynamic_alignment_padding_rollout_ids(
            data["rollout_ids"],
            total_lengths=[len(tokens) for tokens in data["tokens"]],
            args=self.args,
            train_parallel_config=self.train_parallel_config,
            global_batch_size=global_batch_size,
        )
        data["alignment_dummy_mask"] = [False] * sample_count
        if not padding_rollout_ids:
            return

        anchor_position_by_rollout_id: dict[int, int] = {}
        for position, rollout_id in enumerate(data["rollout_ids"]):
            anchor_position_by_rollout_id.setdefault(rollout_id, position)

        topk = None
        if "teacher_topk_log_probs" in data:
            first_rows = data["teacher_topk_log_probs"][0]
            if not first_rows or not first_rows[0]:
                raise ValueError("cannot build an OPD alignment dummy without top-k targets")
            topk = len(first_rows[0])
            if "teacher_topk_indices" not in data or "opd_token_weights" not in data:
                raise ValueError("incomplete GUI OPD targets while building alignment dummies")

        existing_sample_indices = set(data["sample_indices"])
        next_sample_index = min([0, *existing_sample_indices]) - 1
        for rollout_id in padding_rollout_ids:
            anchor_position = anchor_position_by_rollout_id[rollout_id]
            while next_sample_index in existing_sample_indices:
                next_sample_index -= 1
            existing_sample_indices.add(next_sample_index)

            data["tokens"].append([0, 0])
            data["response_lengths"].append(1)
            data["rewards"].append(0.0)
            data["truncated"].append(0)
            data["loss_masks"].append([0])
            data["sample_indices"].append(next_sample_index)
            data["rollout_ids"].append(rollout_id)
            data["rollout_mask_sums"].append(data["rollout_mask_sums"][anchor_position])
            data["alignment_dummy_mask"].append(True)
            next_sample_index -= 1

            if "round_number" in data:
                data["round_number"].append(data["round_number"][anchor_position])
            if "rollout_log_probs" in data:
                data["rollout_log_probs"].append([0.0])
            if "teacher_log_probs" in data:
                data["teacher_log_probs"].append([0.0])
            if "opd_q0_log_probs" in data:
                data["opd_q0_log_probs"].append([0.0])
            if "opd_q0_valid" in data:
                data["opd_q0_valid"].append(False)
            if "opd_token_weights" in data:
                data["opd_token_weights"].append([0.0])
            if "opd_step_polarity" in data:
                data["opd_step_polarity"].append(0)
            if "multimodal_train_inputs" in data:
                data["multimodal_train_inputs"].append(None)
            if "prompt" in data:
                data["prompt"].append("")
            if "rollout_routed_experts" in data:
                prototype = data["rollout_routed_experts"][anchor_position]
                data["rollout_routed_experts"].append(
                    np.zeros((2, *prototype.shape[1:]), dtype=prototype.dtype)
                )
            if topk is not None:
                safe_logprob = -float(np.log(topk + 1))
                data["teacher_topk_log_probs"].append([[safe_logprob] * topk])
                data["teacher_topk_indices"].append([list(range(topk))])
                if "opd_token_weights" not in data:
                    data["opd_token_weights"] = [[0.0] for _ in range(len(data["tokens"]))]

        logger.info(
            "Injected %d zero-loss dynamic alignment dummies: real_samples=%d, "
            "align_to=%d, global_batch_size=%d",
            len(padding_rollout_ids),
            sample_count,
            align_to,
            global_batch_size,
        )

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        # dynamic_history expands a trajectory into step samples and can safely
        # drop entries before deriving its dynamic batch size. The default path
        # keeps fixed rollout slots, replacing removed trajectories with tiny
        # zero-mask placeholders so DP scheduling and prompt grouping stay
        # stable while model forward remains well-formed.
        if getattr(self.args, "dynamic_history", False):
            samples = self._clean_dynamic_history_samples(samples)
            if not samples:
                # A fully constant GRPO batch with no active OPD tokens has no
                # mathematical update. The caller returns a skip marker instead
                # of constructing zero-mask placeholders for a useless update.
                return None
        else:
            prepared = prepare_removed_samples(samples)
            if prepared:
                logger.warning(
                    "Prepared %d removed trajectories as zero-mask placeholders before reward normalization.",
                    prepared,
                )

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        # Rollout id (one per rollout execution). Default rollouts emit one
        # sample per rollout, so we fall back to ``sample.index`` (unique).
        # Compact / subagent paths that emit multiple training samples per
        # rollout set ``rollout_id`` explicitly so all siblings share a
        # value; the loss reducer then aggregates them as one rollout.
        if samples[0].rollout_id is None:
            rollout_ids = list(range(len(samples)))
        else:
            rollout_ids = [sample.rollout_id for sample in samples]

        # dynamic_history: each step sample is its own rollout (unique
        # rollout_id), so the number of distinct rollouts equals the cleaned
        # step-sample count. Derive the per-rollout global_batch_size now so
        # build_dp_schedule keeps the realized training-step count near
        # num_steps_per_rollout regardless of how many steps each trajectory
        # produced. Stored on self for _split_train_data_by_dp and the trainer.
        if getattr(self.args, "dynamic_history", False):
            num_distinct_rollouts = len(set(rollout_ids))
            self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(
                num_distinct_rollouts,
                target_steps=getattr(self.args, "num_steps_per_rollout", None),
            )

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "rollout_ids": rollout_ids,
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # Per-rollout aggregate, precomputed at the step level (where we can
        # see every sample of every rollout) and broadcast per-sample so the
        # per-mb loss reducer uses the correct whole-rollout denominator even
        # when a rollout's samples land in different micro-batches (first-fit
        # packing can split a rollout across mbs):
        #
        #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
        #   sample in sample i's rollout. Used as the reducer's denominator
        #   so summing partial contributions across mbs yields one
        #   token-weighted mean per rollout.
        rollout_id_list = train_data["rollout_ids"]
        mask_sums_per_sample = [sum(m) for m in loss_masks]
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]

        # Overwrite raw_reward when available. Mixed-source batches may only
        # populate this field for a subset of samples (e.g. SWE but not code).
        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        gui_opd_loss_mode = str(getattr(self.args, "gui_opd_loss_mode", "sampled_token"))
        gui_opd_enabled = bool(getattr(self.args, "gui_opd_enable", False))
        has_teacher_log_probs = any(sample.teacher_log_probs is not None for sample in samples)
        needs_sampled_teacher = (
            has_teacher_log_probs and not gui_opd_enabled
        ) or (
            gui_opd_enabled and gui_opd_loss_mode == "sampled_token"
        )
        if needs_sampled_teacher:
            teacher_log_probs = []
            sampled_opd_weights = []
            for sample in samples:
                log_probs = sample.teacher_log_probs
                weights = getattr(sample, "opd_token_weights", None)
                if log_probs is None:
                    logger.warning(
                        "Sample %s is missing GUI OPD sampled-token teacher targets; masking OPD",
                        sample.index,
                    )
                    log_probs = [0.0] * sample.response_length
                    weights = [0.0] * sample.response_length
                if len(log_probs) != sample.response_length:
                    raise ValueError(
                        f"GUI OPD sampled-token target length mismatch for sample {sample.index}: "
                        f"logps={len(log_probs)}, response={sample.response_length}"
                    )
                if gui_opd_enabled and gui_opd_loss_mode == "sampled_token":
                    if weights is None:
                        weights = [1.0] * sample.response_length
                    if len(weights) != sample.response_length:
                        raise ValueError(
                            f"GUI OPD token-weight length mismatch for sample {sample.index}: "
                            f"weights={len(weights)}, response={sample.response_length}"
                        )
                    sampled_opd_weights.append([float(weight) for weight in weights])
                teacher_log_probs.append([float(log_prob) for log_prob in log_probs])
            train_data["teacher_log_probs"] = teacher_log_probs
            if sampled_opd_weights:
                train_data["opd_token_weights"] = sampled_opd_weights

        if gui_opd_enabled:
            train_data["opd_step_polarity"] = [
                int(getattr(sample, "opd_step_polarity", 0) or 0)
                for sample in samples
            ]
            train_data["opd_q0_log_probs"] = [
                sample.opd_q0_log_probs
                if sample.opd_q0_log_probs is not None
                else [0.0] * sample.response_length
                for sample in samples
            ]
            train_data["opd_q0_valid"] = [
                bool(getattr(sample, "opd_q0_valid", False))
                for sample in samples
            ]

        topk_samples = [sample for sample in samples if sample.teacher_topk_log_probs is not None]
        if gui_opd_loss_mode == "topk" and (topk_samples or gui_opd_enabled):
            if topk_samples:
                first_rows = topk_samples[0].teacher_topk_log_probs
                if not first_rows or not first_rows[0]:
                    raise ValueError("GUI OPD received an empty teacher top-k target")
                topk = len(first_rows[0])
            else:
                topk = int(getattr(self.args, "gui_opd_topk", 50))
            safe_logprob = -float(np.log(topk + 1))
            teacher_topk_log_probs = []
            teacher_topk_indices = []
            opd_token_weights = []
            for sample in samples:
                logps = sample.teacher_topk_log_probs
                indices = sample.teacher_topk_indices
                if logps is None or indices is None:
                    if not sample.remove_sample:
                        logger.warning(
                            "Sample %s is missing GUI OPD teacher targets; masking OPD while "
                            "preserving its GRPO step",
                            sample.index,
                        )
                    logps = [[safe_logprob] * topk for _ in range(sample.response_length)]
                    indices = [list(range(topk)) for _ in range(sample.response_length)]
                if len(logps) != sample.response_length or len(indices) != sample.response_length:
                    raise ValueError(
                        f"GUI OPD target length mismatch for sample {sample.index}: "
                        f"logps={len(logps)}, indices={len(indices)}, response={sample.response_length}"
                    )
                teacher_topk_log_probs.append(logps)
                teacher_topk_indices.append(indices)
                weights = getattr(sample, "opd_token_weights", None)
                if weights is None:
                    # A missing validity mask defaults to valid only when the
                    # top-k target exists; the training-time gate is recomputed.
                    weights = (
                        [1.0] * sample.response_length
                        if sample.teacher_topk_log_probs is not None and not sample.remove_sample
                        else [0.0] * sample.response_length
                    )
                if len(weights) != sample.response_length:
                    raise ValueError(
                        f"GUI OPD token-weight length mismatch for sample {sample.index}: "
                        f"weights={len(weights)}, response={sample.response_length}"
                    )
                opd_token_weights.append([float(weight) for weight in weights])
            train_data["teacher_topk_log_probs"] = teacher_topk_log_probs
            train_data["teacher_topk_indices"] = teacher_topk_indices
            train_data["opd_token_weights"] = opd_token_weights

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(
        self,
        data,
        *,
        global_batch_size_override: int | None = None,
    ):
        """Compute the DP/mbs schedule and package each rank's rollout_data
        into a Ray Box. The schedule itself is computed by
        :func:`build_dp_schedule` so it stays unit-testable without Ray/sglang.

        Step split is by rollout id (``samples[i].rollout_id``, falling back
        to ``samples[i].index``); each step holds exactly
        ``args.global_batch_size`` rollouts so the training-step count per
        rollout is fixed at ``rollout_batch_size * n_samples_per_prompt //
        global_batch_size`` regardless of how many training samples each
        rollout produced.
        """
        # Under dynamic_history the per-rollout step count is variable, so use
        # the dynamically-derived global_batch_size computed in
        # _convert_samples_to_train_data; otherwise keep the static config value.
        global_batch_size = self.args.global_batch_size
        if global_batch_size_override is not None:
            global_batch_size = int(global_batch_size_override)
        elif getattr(self.args, "dynamic_history", False) and hasattr(self, "_dynamic_global_batch_size"):
            global_batch_size = self._dynamic_global_batch_size

        # Dynamic-history fan-out yields a variable number of samples after
        # invalid/constant-reward groups are removed.  Both packed and static
        # schedules must be aligned to DP (and VPP microbatch groups).  This is
        # especially important for Qwen3.5: GDN requires static bshd batches,
        # so an odd sample count cannot be repaired by splitting packed bins.
        # _append_dynamic_alignment_dummies adds only zero-loss samples and
        # preserves the number of real rollout ids / optimizer-step semantics.
        if getattr(self.args, "dynamic_history", False):
            self._append_dynamic_alignment_dummies(
                data,
                global_batch_size=global_batch_size,
            )

        dp_size = self.train_parallel_config["dp_size"]
        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
            self.args,
            self.train_parallel_config,
            total_lengths,
            global_batch_size=global_batch_size,
            rollout_indices=data["rollout_ids"],
        )

        # Debug: split layout logging is ON by default ("1"). Override with
        # SLIME_DEBUG_SPLIT=/path/dump.pt to also dump the full per-rank data,
        # or SLIME_DEBUG_SPLIT=0 (or "") to disable.
        debug_split = os.environ.get("SLIME_DEBUG_SPLIT", "1")
        if debug_split and debug_split != "0":
            n = len(total_lengths)
            loss_masks_dbg = data.get("loss_masks")
            mask_sums = [int(sum(m)) for m in loss_masks_dbg] if loss_masks_dbg is not None else None
            alignment_dummy_count = sum(data.get("alignment_dummy_mask", []))
            logger.info(
                "[split] num_samples=%d gbs=%d num_steps=%d dp_size=%d num_microbatches=%s "
                "global_batch_sizes=%s alignment_dummies=%d total_len[min/mean/max]=%s/%s/%s",
                n,
                global_batch_size,
                len(num_microbatches),
                dp_size,
                num_microbatches,
                global_batch_sizes,
                alignment_dummy_count,
                min(total_lengths) if n else 0,
                (sum(total_lengths) // n) if n else 0,
                max(total_lengths) if n else 0,
            )
            if mask_sums is not None:
                resp = data.get("response_lengths") or [0] * n
                logger.info(
                    "[split] mask_sum[min/mean/max]=%s/%s/%s  resp_len[min/mean/max]=%s/%s/%s  "
                    "zero_mask_samples=%d  rewards[:8]=%s",
                    min(mask_sums),
                    sum(mask_sums) // max(n, 1),
                    max(mask_sums),
                    min(resp),
                    sum(resp) // max(n, 1),
                    max(resp),
                    sum(1 for s in mask_sums if s == 0),
                    [round(float(x), 3) for x in (data.get("rewards") or [])[:8]],
                )
            for r in range(dp_size):
                part = partitions[r]
                logger.info(
                    "[split] rank=%d num_samples=%d num_mbs=%d sample_token_sum=%d mbs_sizes=%s",
                    r,
                    len(part),
                    len(micro_batch_indices[r]),
                    sum(total_lengths[j] for j in part),
                    [len(mb) for mb in micro_batch_indices[r]],
                )
            if debug_split != "1":
                dump = {
                    "global_batch_size": global_batch_size,
                    "num_microbatches": num_microbatches,
                    "global_batch_sizes": global_batch_sizes,
                    "partitions": [list(p) for p in partitions],
                    "micro_batch_indices": micro_batch_indices,
                    "total_lengths": total_lengths,
                    "rollout_ids": data.get("rollout_ids"),
                    "rewards": data.get("rewards"),
                    "response_lengths": data.get("response_lengths"),
                    "loss_masks": data.get("loss_masks"),
                    "sample_indices": data.get("sample_indices"),
                    "alignment_dummy_mask": data.get("alignment_dummy_mask"),
                }
                Path(debug_split).parent.mkdir(parents=True, exist_ok=True)
                torch.save(dump, debug_split)
                logger.info("[split] dumped full split data to %s", debug_split)

        # Package per-rank rollout_data
        rollout_data_refs = []
        for r in range(dp_size):
            partition = partitions[r]
            rollout_data = {"partition": partition}
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
                "teacher_topk_log_probs",
                "teacher_topk_indices",
                "opd_token_weights",
                "opd_step_polarity",
                "alignment_dummy_mask",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = [data[key][j] for j in partition]
            # keys that need to be splited at train side
            for key in ["raw_reward", "total_lengths"]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            rollout_data["global_batch_sizes"] = global_batch_sizes
            rollout_data["num_microbatches"] = num_microbatches
            rollout_data["micro_batch_indices"] = micro_batch_indices[r]
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _validate_rollout_id_annotated(node, depth=0):
    """Walk the rollout function's nested output and validate ``rollout_id`` only
    when a compact / subagent pattern is detected.

    "Compact" = the rollout function wraps multiple training samples from one
    rollout execution into a ``list[Sample]``. In slime's convention the
    default rollout shape is ``list[list[Sample]]`` (depth-2: prompt × rollout)
    so its leaf ``list[Sample]`` lands at depth 1 and we skip validation,
    preserving backward compatibility. A compact rollout adds a third level:
    ``list[list[list[Sample]]]`` (prompt × rollout × samples-from-one-rollout),
    so the leaf ``list[Sample]`` lands at depth ≥ 2. At that point we require
    every sibling to carry a non-None ``rollout_id`` and to share the same
    value, so the loss reducer counts the rollout once instead of N times.
    """
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        _validate_rollout_id_annotated(item, depth + 1)


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sglang_router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values.
    When ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        # Disable circuit breaker to prevent RDMA transfer timeouts from
        # marking decode workers as dead. Timeouts are transient (PCIe
        # contention under high load) and do not indicate a dead server.
        router_args.disable_circuit_breaker = True

    # We will not use the health check from router.
    router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, Any]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        port_cursors: dict[int, int] = {}

        has_epd = model_cfg.has_encoder_disaggregation

        def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
            nonlocal engine_offset, gpu_offset
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if overrides_extra:
                for k, v in overrides_extra.items():
                    overrides.setdefault(k, v)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus
            return group

        if has_epd:
            # --- Phase 1: start encoder groups, wait, collect URLs ---
            encoder_urls: list[str] = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type != "encoder":
                    continue
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                if handles:
                    ray.get(handles)
                urls = ray.get([e.get_url.remote() for e in group.engines])
                encoder_urls.extend(u for u in urls if u is not None)
                server_groups.append(group)

            logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

            # --- Phase 2: start non-encoder groups, injecting encoder URLs into
            # language-only LLM workers. Prefill groups use this for full EPD,
            # while regular groups allow encoder/LLM split without PD.
            non_encoder_handles: list = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type == "encoder":
                    continue
                overrides_extra = {}
                if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                    overrides_extra["language_only"] = True
                    overrides_extra["encoder_urls"] = encoder_urls
                group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                handles, port_cursors = group.start_engines(port_cursors)
                non_encoder_handles.extend(handles)
                server_groups.append(group)

            if non_encoder_handles:
                ray.get(non_encoder_handles)
        else:
            # No EPD — start all groups in one pass (original path).
            all_init_handles: list = []
            for group_cfg in model_cfg.server_groups:
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                all_init_handles.extend(handles)
                server_groups.append(group)

            if all_init_handles:
                ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    # Expose per-model router info for custom rollout functions.
    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
