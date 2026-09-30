from megatron.bridge.models.conversion.param_mapping import AutoMapping

AutoMapping.register_module_type("LinearCrossEntropyModule", "column")