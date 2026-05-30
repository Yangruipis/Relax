# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Image processor for the Relax external dots.mocr SGLang model.

Mirrors sglang.srt.multimodal.processors.dots_vlm.DotsVLMImageProcessor — uses
the dots-native ``<|img|><|imgpad|><|endofimg|>`` token triple, NOT Qwen-VL's
``<|vision_start|><|image_pad|><|vision_end|>``. Inheriting from
``QwenVLImageProcessor`` would inject the wrong tokens and access fields
(vision_start_token_id, …) that don't exist on dots.mocr's hf_config.
"""

import os
import re
from typing import Dict, List, Union

from relax.utils.logging_utils import get_logger

_p_logger = get_logger(__name__)
_p_logger.info(f"[dbg] relax.models.dots_ocr.sglang.processor TOP-OF-MODULE pid={os.getpid()}")

_p_logger.info("[dbg] importing sglang BaseMultimodalProcessor + MultimodalSpecialTokens")
from sglang.srt.multimodal.processors.base_processor import (  # noqa: E402
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

_p_logger.info("[dbg] importing relax.models.dots_ocr.sglang.model.DotsOCRForCausalLM")
from relax.models.dots_ocr.sglang.model import DotsOCRForCausalLM  # noqa: E402


class DotsOCRImageProcessor(BaseMultimodalProcessor):
    models = [DotsOCRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        _p_logger.info(f"[dbg] DotsOCRImageProcessor.__init__ START pid={os.getpid()}")
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IMAGE_TOKEN = "<|img|><|imgpad|><|endofimg|>"
        self.IMAGE_TOKEN_REGEX = re.compile(r"<\|img\|>(?:<\|imgpad\|>)+<\|endofimg\|>")

        assert len(_processor.tokenizer.encode("<|img|>")) == 1
        self.im_start_id = _processor.tokenizer.encode("<|img|>")[0]
        self.im_end_id = _processor.tokenizer.encode("<|endofimg|>")[0]
        self.image_token_id = _processor.tokenizer.encode("<|imgpad|>")[0]
        self.IM_TOKEN_ID = self.image_token_id
        self.IM_START_TOKEN_ID = self.im_start_id
        self.IM_END_TOKEN_ID = self.im_end_id

        vision_config = hf_config.vision_config
        self.IMAGE_FACTOR = vision_config.patch_size * vision_config.spatial_merge_size
        self.MIN_PIXELS = _processor.image_processor.min_pixels
        self.MAX_PIXELS = _processor.image_processor.max_pixels
        self.MAX_RATIO = 200

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.image_token_id,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
        ).build(_processor)
        _p_logger.info(f"[dbg] DotsOCRImageProcessor.__init__ DONE pid={os.getpid()}")

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if isinstance(image_data, str):
            image_data = [image_data]

        if isinstance(image_data, list) and image_data and isinstance(image_data[0], list):
            image_data = sum(image_data, [])

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        combined_mm_item, input_ids, _ = self.process_and_combine_mm_data(base_output, self.mm_tokens)
        if combined_mm_item is None:
            return None

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": combined_mm_item,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "im_token_id": self.image_token_id,
        }


_p_logger.info("[dbg] relax.models.dots_ocr.sglang.processor module IMPORT COMPLETE")
