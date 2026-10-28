# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from typing import Any

from megatron.bridge.models.gpt_provider import GPTModelProvider, transformer_engine_layer_spec
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.spec_utils import ModuleSpec

from .attention import KeyeCoreAttention, KeyeSelfAttention


def keye_layer_spec(config: GPTModelProvider) -> ModuleSpec:
    spec = transformer_engine_layer_spec(config)
    spec.submodules.input_layernorm = TENorm
    spec.submodules.self_attention.module = KeyeSelfAttention
    spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
    spec.submodules.self_attention.submodules.core_attention = KeyeCoreAttention
    return spec


@dataclass
class KeyeModelProvider(GPTModelProvider):
    transformer_layer_spec = staticmethod(keye_layer_spec)
    keye_indexer_loss_coeff: float = 0.01
    keye_attention_chunk_size: int = 1024
    hf_model_path: str | None = None
    hf_config: Any = None
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def configure_runtime(self, args: Any) -> None:
        if getattr(args, "dynamic_context_parallel", False):
            raise ValueError("Keye currently supports static context parallelism only")
        if self.context_parallel_size > 1 and args.cp_partition_mode != "zigzag":
            raise ValueError("Keye context parallelism requires zigzag partitioning")
        if args.multimodal_keys:
            self.position_embedding_type = "mrope"
            self.apply_rope_fusion = False
            args.multimodal_input_key_map = {"image_grid_thw": "grid_thws"}
            args.uses_unsplit_forward = True

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> Any:
        language = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if self.position_embedding_type != "mrope":
            return language
        from .model import KeyeVLModel

        model = KeyeVLModel(self, language)
        model.freeze(self.freeze_language_model, self.freeze_vision_model, self.freeze_vision_projection)
        return model

    def finalize(self) -> None:
        self.transformer_layer_spec = keye_layer_spec
        super().finalize()
        self.experimental_attention_variant_loss_scale_func = DSAIndexerLossAutoScaler.set_loss_scale
