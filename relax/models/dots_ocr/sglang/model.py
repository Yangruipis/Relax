# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang external model for dotsocr2."""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens, general_mm_embed_routine
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.utils import add_prefix, logging

from relax.utils.logging_utils import get_logger


_dbg_logger = get_logger(__name__)
_dbg_logger.debug(f"[dbg] relax.models.dots_ocr.sglang.model TOP-OF-MODULE pid={__import__('os').getpid()}")

_dbg_logger.debug("[dbg] importing relax.models.dots_ocr.configuration.DotsVisionConfig")
from relax.models.dots_ocr.configuration import DotsVisionConfig  # noqa: E402

_dbg_logger.debug("[dbg] importing relax.models.dots_ocr.vision.DotsVisionTransformer")
from relax.models.dots_ocr.vision import DotsVisionTransformer  # noqa: E402

_dbg_logger.debug("[dbg] relax.models.dots_ocr.sglang.model module IMPORT COMPLETE")
# Log only first N forward/get_image_feature calls to avoid spamming decode loop.
_DBG_MAX_LOG_CALLS = 5
_forward_call_count = 0
_get_image_feature_call_count = 0
_pad_input_ids_call_count = 0


class DotsOCRForCausalLM(nn.Module):
    default_bitsandbytes_target_modules = [
        ".fc2.",
        ".fc1.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        _dbg_logger.debug(
            f"[dbg] DotsOCRForCausalLM.__init__ START prefix={prefix!r} "
            f"quant={type(quant_config).__name__ if quant_config else None}"
        )
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        if isinstance(vision_config, dict):
            vision_config = DotsVisionConfig(**vision_config)
        _dbg_logger.debug(f"[dbg] DotsOCRForCausalLM.__init__ building vision_tower (cfg={type(vision_config).__name__})")
        self.vision_tower = DotsVisionTransformer(vision_config)
        _dbg_logger.debug("[dbg] DotsOCRForCausalLM.__init__ building Qwen2Model")
        self.model = Qwen2Model(config, quant_config, prefix=add_prefix("model", prefix))
        if config.tie_word_embeddings:
            logging.warning("tied word embeddings are not supported in SGLang DotsOCRForCausalLM.")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        _dbg_logger.debug("[dbg] DotsOCRForCausalLM.__init__ DONE")

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        global _pad_input_ids_call_count
        if _pad_input_ids_call_count < _DBG_MAX_LOG_CALLS:
            _dbg_logger.debug(
                f"[dbg] pad_input_ids call#{_pad_input_ids_call_count} "
                f"len(input_ids)={len(input_ids)} "
                f"mm_inputs.image_offsets={getattr(mm_inputs, 'image_offsets', None)}"
            )
        _pad_input_ids_call_count += 1
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        out = pattern.pad_input_tokens(input_ids, mm_inputs)
        if _pad_input_ids_call_count <= _DBG_MAX_LOG_CALLS:
            _dbg_logger.debug(f"[dbg] pad_input_ids call#{_pad_input_ids_call_count - 1} DONE len(out)={len(out)}")
        return out

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        global _get_image_feature_call_count
        verbose = _get_image_feature_call_count < _DBG_MAX_LOG_CALLS
        if verbose:
            _dbg_logger.debug(
                f"[dbg] get_image_feature call#{_get_image_feature_call_count} n_items={len(items)} "
                f"feature_shapes={[tuple(it.feature.shape) for it in items[:3]]}"
            )
        _get_image_feature_call_count += 1
        target_device = self.vision_tower.device
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=target_device, dtype=self.vision_tower.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0).to(target_device)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        if verbose:
            _dbg_logger.debug(
                f"[dbg] get_image_feature call#{_get_image_feature_call_count - 1} "
                f"pixel_values={tuple(pixel_values.shape)} grid_thw={tuple(image_grid_thw.shape)} "
                f"-> calling vision_tower"
            )
        out = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        if verbose:
            _dbg_logger.debug(
                f"[dbg] get_image_feature call#{_get_image_feature_call_count - 1} "
                f"DONE vision_embeds={tuple(out.shape)} dtype={out.dtype}"
            )
        return out

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        target_device = self.vision_tower.device
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=target_device, dtype=self.vision_tower.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0).to(target_device)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        return self.vision_tower(pixel_values, grid_thw=video_grid_thw)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        global _forward_call_count
        verbose = _forward_call_count < _DBG_MAX_LOG_CALLS
        if verbose:
            _dbg_logger.debug(
                f"[dbg] forward call#{_forward_call_count} "
                f"input_ids={tuple(input_ids.shape)} positions={tuple(positions.shape) if positions is not None else None} "
                f"get_embedding={get_embedding} "
                f"forward_mode={getattr(forward_batch, 'forward_mode', None)} "
                f"mm_inputs={'yes' if getattr(forward_batch, 'mm_inputs', None) else 'no'}"
            )
        _forward_call_count += 1
        if verbose:
            _dbg_logger.debug(f"[dbg] forward call#{_forward_call_count - 1} -> general_mm_embed_routine")
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if verbose:
            _dbg_logger.debug(
                f"[dbg] forward call#{_forward_call_count - 1} "
                f"general_mm_embed_routine DONE hidden_states={tuple(hidden_states.shape) if hidden_states is not None else None}"
            )
        if not get_embedding:
            out = self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        else:
            out = self.pooler(hidden_states, forward_batch)
        if verbose:
            _dbg_logger.debug(f"[dbg] forward call#{_forward_call_count - 1} DONE")
        return out

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        _dbg_logger.debug("[dbg] load_weights START")
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        _dbg_logger.debug(f"[dbg] load_weights total_params={len(params_dict)}")
        n = 0
        for name, loaded_weight in weights:
            n += 1
            if n % 100 == 0:
                _dbg_logger.debug(f"[dbg] load_weights processed {n} weights, latest={name}")
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "vision_tower" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        _dbg_logger.debug(f"[dbg] load_weights DONE total_weights_seen={n}")


EntryClass = [DotsOCRForCausalLM]
