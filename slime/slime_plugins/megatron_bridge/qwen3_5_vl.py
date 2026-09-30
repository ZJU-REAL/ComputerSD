"""Register the official Megatron-Bridge implementations for Qwen3.5-VL.

The RL launcher uses ``megatron.bridge.AutoBridge`` (``--megatron-to-hf-mode
bridge``), not the legacy ``mbridge`` adapter.  Importing this module triggers
the official Qwen3.5 dense/MoE bridge decorators and makes the architectures
discoverable by ``AutoBridge.from_hf_pretrained``.
"""

from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)


def _add_legacy_mtp_aliases(registry):
    """Add aliases for older Megatron MTP parameter names."""
    if registry is None:
        return registry
    original = list(registry.mappings)
    extra = []
    for mapping in original:
        name = getattr(mapping, "megatron_param", None)
        if isinstance(name, str) and ".mtp_model_layer." in name:
            alias = copy.copy(mapping)
            alias.megatron_param = name.replace(".mtp_model_layer.", ".transformer_layer.")
            extra.append(alias)
    if not extra:
        return registry
    return registry.__class__(*original, *extra)


def _patch_bridge_mapping_registry(bridge_cls):
    """Patch MTP mapping aliases once, while preserving the original method."""
    original = bridge_cls.mapping_registry
    if getattr(original, "_slime_mtp_alias_patched", False):
        return

    def patched(self, *args, **kwargs):
        return _add_legacy_mtp_aliases(original(self, *args, **kwargs))

    patched._slime_mtp_alias_patched = True  # type: ignore[attr-defined]
    bridge_cls.mapping_registry = patched


try:
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (  # noqa: F401
        Qwen35VLMoEBridge,
        Qwen35VLBridge,
    )

    _patch_bridge_mapping_registry(Qwen35VLBridge)
    _patch_bridge_mapping_registry(Qwen35VLMoEBridge)
except (ImportError, AttributeError) as exc:  # pragma: no cover - env dependent
    logger.warning(
        "Qwen3.5-VL Megatron bridges are unavailable. Install a Megatron Bridge "
        "version containing qwen35_vl_bridge and a Transformers version exposing "
        "Qwen3_5ForConditionalGeneration: %s",
        exc,
    )
