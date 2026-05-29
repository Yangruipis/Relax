# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass
from typing import Optional

import torch
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from relax.models.dots_ocr.configuration import DotsVisionConfig
from relax.models.dots_ocr.vision import DotsVisionTransformer


@dataclass
class DotsOCRTransformerConfig(TransformerConfig):
    vocab_size: int = 151936
    language_max_sequence_length: int = 131072
    image_token_id: int = 151665
    video_token_id: int = 151656
    vision_config: Optional[DotsVisionConfig] = None
    fp16_lm_cross_entropy: bool = False
    rotary_percent: float = 1.0
    scatter_embedding_sequence_parallel: bool = False


class DotsOCRGPTModel(GPTModel):
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params=None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
        )


class DotsOCRModel(MegatronModule):
    def __init__(
        self,
        language_transformer_config: DotsOCRTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config: DotsVisionConfig,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.encoder_hidden_state = None
        self.vision_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.share_embeddings_and_output_weights = language_transformer_config.share_embeddings_and_output_weights

        if self.pre_process:
            self.vision_model = DotsVisionTransformer(vision_transformer_config)
            self.vision_model.gradient_checkpointing = True
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to(device="cuda", dtype=torch.bfloat16)

        self.language_model = DotsOCRGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="rope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for DotsOCRModel"
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.extend([self.vision_model.patch_embed, self.vision_model.blocks, self.vision_model.rotary_pos_emb])
            if hasattr(self.vision_model, "post_trunk_norm"):
                modules.append(self.vision_model.post_trunk_norm)
        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.merger)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _position_ids(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            return position_ids
        return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert inference_params is None, "Inference is not supported in Megatron DotsOCRModel"

        if self.pre_process:
            combined_embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None).clone()
            if pixel_values is not None and image_grid_thw is not None and image_grid_thw.shape[0] > 0:
                image_mask = image_input_mask
                if image_mask is None:
                    image_mask = (input_ids == self.image_token_id).contiguous()
                vision_embeds = self.vision_model(
                    hidden_states=pixel_values,
                    grid_thw=image_grid_thw,
                )
                emb_bsh = combined_embeddings.transpose(0, 1).contiguous()
                emb_bsh = emb_bsh.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(emb_bsh),
                    vision_embeds.to(emb_bsh.device).type(emb_bsh.dtype),
                )
                combined_embeddings = emb_bsh.transpose(0, 1).contiguous()
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        if position_ids is None:
            position_ids = self._position_ids(input_ids, attention_mask)

        return self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
