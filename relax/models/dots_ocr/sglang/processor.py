# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from relax.models.dots_ocr.sglang.model import DotsOCRForCausalLM


class DotsOCRImageProcessor(QwenVLImageProcessor):
    models = [DotsOCRForCausalLM]
