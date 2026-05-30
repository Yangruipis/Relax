# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare per-input-token log-probabilities between SGLang and Megatron for
DotsOCR2 on the same multimodal input (text + image).

Single-GPU debugging utility (tp=pp=cp=ep=1). The goal is to localize the
sglang ↔ megatron mismatch users hit during RL training by replaying the
*same* loading paths that production uses:

  * Megatron side: ``AutoBridge.from_hf_pretrained`` + ``to_megatron_provider``
    + ``load_hf_weights`` — identical to ``relax.backends.megatron.model_provider``
    when ``--megatron-to-hf-mode bridge`` is used.
  * SGLang side: offline ``sglang.Engine`` started with the
    ``SGLANG_EXTERNAL_MODEL_PACKAGE=relax.models.dots_ocr.sglang`` env var,
    so ``DotsOCRForCausalLM`` from this repo is the actual implementation
    loaded — identical to what the rollout engine instantiates at runtime.

Run modes (``--side``):
  * ``megatron``: build the megatron model, forward, dump logprobs to disk.
  * ``sglang``:   launch the sglang engine, generate with ``return_logprob``,
                  dump logprobs to disk.
  * ``compare``:  load both dumps and print summary stats + worst positions.
  * ``all`` (default): run all three sequentially in one process.

The two sides do NOT share GPU memory simultaneously in ``all`` mode — we
free the megatron model before launching SGLang's worker subprocess.

Example::

    python scripts/debug/compare_sglang_megatron_dotsocr.py \\
        --hf-checkpoint /data/rednote-hilab/dots.mocr/ \\
        --image /data/sample.png \\
        --prompt "Describe this image."

Use ``--side megatron`` / ``--side sglang`` for incremental debugging when
one side fails and you don't want to re-run the other.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class DumpRecord:
    """Per-token logprob dump persisted to disk for cross-side comparison."""

    side: str  # "megatron" | "sglang"
    input_ids: list[int]  # post-image-expansion token sequence (length T)
    # logprobs[i] = log P(input_ids[i] | input_ids[:i]), length T;
    # entry 0 is None (no conditioning available).
    logprobs: list[Optional[float]]
    meta: dict


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process "
    "here </think><answer> answer here </answer>"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-checkpoint", required=True, help="Path to dots.mocr HF checkpoint dir.")
    p.add_argument(
        "--image",
        default=None,
        help=(
            "Image file path or http(s) URL (jpg/png/webp). "
            "Omit for text-only mode (skips vision tower on all sides; useful "
            "to isolate whether a mismatch is multimodal-specific or hits the "
            "language backbone too)."
        ),
    )
    p.add_argument("--prompt", default="Describe this image.", help="User text prompt.")
    p.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (defaults to the GRPO system prompt used in run-dotsocr2-8xgpu.sh).",
    )
    p.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Skip system prompt entirely (debug only).",
    )
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP), help="Model dtype.")
    p.add_argument(
        "--side",
        default="all",
        choices=["megatron", "sglang", "hf", "compare", "all"],
        help=(
            "Which side to run; 'all' does megatron→hf→sglang→compare in one shot. "
            "'hf' uses transformers.AutoModelForCausalLM as a third ground-truth "
            "reference so you can tell which engine is the broken one."
        ),
    )
    p.add_argument(
        "--dump-dir",
        default="/tmp/relax_dotsocr_debug",
        help="Directory for per-side logprob dumps and comparison report.",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.65,
        help="SGLang static memory fraction (lower this when sharing GPU with megatron).",
    )
    p.add_argument(
        "--top-n-worst",
        type=int,
        default=20,
        help="Report the N positions with the largest |Δ logprob|.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for ``model_parallel_cuda_manual_seed`` (matches relax default).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared input prep — both sides must agree on the (text, image, input_ids)
# tuple, otherwise the comparison is meaningless.
# ---------------------------------------------------------------------------


def _build_messages(args: argparse.Namespace) -> list[dict]:
    """Mirror the rollout-side chat template (apply-chat-template + system
    prompt + multimodal user turn) so we exercise the same prefill SGLang
    would see during RL training. When ``--image`` is omitted, the user
    turn is a plain text string and no vision tokens are inserted."""
    messages = []
    if not args.no_system_prompt and args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    if args.image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": args.image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": args.prompt})
    return messages


def _load_image_any(src: str):
    """Open a PIL image from a local path or an http(s) URL."""
    from io import BytesIO

    from PIL import Image

    if src.startswith(("http://", "https://")):
        import requests

        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(src).convert("RGB")


def _build_processor_inputs(args: argparse.Namespace):
    """Run the HF processor *once* to get the canonical post-expansion
    ``input_ids`` (+ ``pixel_values`` / ``image_grid_thw`` when an image is
    provided).

    Both sides will use these exact ``input_ids``; the megatron forward
    consumes ``pixel_values`` + ``image_grid_thw`` directly, while sglang
    re-tokenizes from text + image_data (and we assert its expanded ids
    match). When ``--image`` is omitted, this falls back to the raw
    tokenizer path so we exercise the language backbone only."""
    from transformers import AutoProcessor, AutoTokenizer

    messages = _build_messages(args)
    if args.image:
        processor = AutoProcessor.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        image = _load_image_any(args.image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_out = processor(text=[text], images=[image], padding=False, return_tensors="pt")
        return processor, text, image, proc_out

    # Text-only path: the multimodal AutoProcessor for dots.mocr requires an
    # image; fall back to the tokenizer + a dict that mimics processor output.
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    proc_out = {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask")}
    return tokenizer, text, None, proc_out


# ---------------------------------------------------------------------------
# Megatron side
# ---------------------------------------------------------------------------


def _init_single_gpu_distributed(seed: int) -> None:
    """Bring up tp=pp=cp=ep=1 mpu state. Crucially seeds the model-parallel
    RNG via ``tensor_parallel.model_parallel_cuda_manual_seed`` — skipping
    this triggers ``cuda rng state model-parallel-rng is not added`` the
    first time a TP layer forwards (mirrors ``relax/backends/megatron/
    initialize.py:32``)."""
    import torch.distributed as dist
    from megatron.core import mpu, tensor_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=1, rank=0)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def _apply_single_gpu_provider_overrides(provider, dtype: torch.dtype) -> None:
    """Force the bridge provider into a 1-GPU configuration that mirrors
    relax's CLI overrides in ``relax.backends.megatron.model_provider``."""
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = False
    provider.variable_seq_lengths = False
    provider.apply_rope_fusion = False
    provider.attention_softmax_in_fp32 = True
    if hasattr(provider, "attention_backend"):
        provider.attention_backend = "flash"
    provider.fp16 = dtype == torch.float16
    provider.bf16 = dtype == torch.bfloat16
    provider.params_dtype = dtype


# ---------------------------------------------------------------------------
# Layer-by-layer intermediate capture. We register forward hooks on the
# embedding, every transformer layer, the final norm, and the lm_head so we
# can localize the first point of divergence between HF and Megatron. All
# captured tensors are normalized to BSH/BSV (batch-first) and saved as
# bf16 CPU tensors to keep dump size manageable.
# ---------------------------------------------------------------------------


def _to_bsh(t: torch.Tensor, kind: str) -> torch.Tensor:
    """Megatron decoder layers + final_layernorm + output_layer produce SBH/SBV;
    HF produces BSH/BSV. Normalize both to BSH/BSV using ``kind`` as the hint
    of which side this tensor came from."""
    if kind == "megatron" and t.dim() == 3:
        return t.transpose(0, 1).contiguous()
    return t


def _capture_hook(store: dict, name: str, kind: str):
    def hook(_mod, _inp, out):
        # Many megatron layers return (hidden, residual_or_None) or
        # (logits, bias). Take the first tensor of any tuple/list.
        if isinstance(out, (tuple, list)):
            out = next((o for o in out if isinstance(o, torch.Tensor)), None)
            if out is None:
                return
        if not isinstance(out, torch.Tensor):
            return
        store[name] = _to_bsh(out, kind).detach().to(torch.bfloat16).cpu()
    return hook


def _install_megatron_hooks(model) -> tuple[dict, list]:
    """Register hooks on relax DotsOCRModel → returns (captured_dict, handles)."""
    captured: dict = {}
    handles: list = []
    lm = model.language_model
    handles.append(lm.embedding.register_forward_hook(_capture_hook(captured, "embed", "megatron")))
    for i, layer in enumerate(lm.decoder.layers):
        handles.append(layer.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}", "megatron")))
    if getattr(lm.decoder, "final_layernorm", None) is not None:
        handles.append(
            lm.decoder.final_layernorm.register_forward_hook(_capture_hook(captured, "final_norm", "megatron"))
        )
    handles.append(lm.output_layer.register_forward_hook(_capture_hook(captured, "lm_head", "megatron")))
    return captured, handles


def _install_hf_hooks(model) -> tuple[dict, list]:
    """Register hooks on HF DotsOCRForCausalLM (Qwen2-like layout)."""
    captured: dict = {}
    handles: list = []
    inner = model.model  # Qwen2Model
    handles.append(inner.embed_tokens.register_forward_hook(_capture_hook(captured, "embed", "hf")))
    for i, layer in enumerate(inner.layers):
        handles.append(layer.register_forward_hook(_capture_hook(captured, f"layer.{i:02d}", "hf")))
    final_norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None)
    if final_norm is not None:
        handles.append(final_norm.register_forward_hook(_capture_hook(captured, "final_norm", "hf")))
    handles.append(model.lm_head.register_forward_hook(_capture_hook(captured, "lm_head", "hf")))
    return captured, handles


def _intermediates_path(dump_dir: Path, side: str) -> Path:
    return dump_dir / f"{side}.intermediates.pt"


@torch.no_grad()
def run_megatron(args: argparse.Namespace) -> DumpRecord:
    """Build the Megatron DotsOCR model the same way relax does for the
    actor, forward a single (text, image) sample, return per-token logprobs."""
    import torch.nn.functional as F
    from megatron.bridge import AutoBridge

    # Side-effect import: relax/models/__init__.py runs the
    # @MegatronModelBridge.register_bridge(source="DotsOCRForCausalLM", ...)
    # decorator on DotsOCRBridge. Without this AutoBridge.from_hf_pretrained
    # raises "Model architecture 'DotsOCRForCausalLM' is not yet supported".
    import relax.models  # noqa: F401

    dtype = DTYPE_MAP[args.dtype]
    _init_single_gpu_distributed(args.seed)

    processor, text, image, proc_out = _build_processor_inputs(args)
    print(f"[megatron] chat_template text (first 300 chars): {text[:300]!r}")

    print(f"[megatron] loading AutoBridge from {args.hf_checkpoint}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    _apply_single_gpu_provider_overrides(provider, dtype)
    provider.finalize()

    print("[megatron] building model via provider.provide()")
    model = provider.provide(pre_process=True, post_process=True)
    model = model.cuda().to(dtype).eval()

    print("[megatron] loading HF weights -> Megatron via bridge.load_hf_weights")
    bridge.load_hf_weights([model])

    device = torch.device("cuda")
    input_ids = proc_out["input_ids"].to(device=device, dtype=torch.long)  # [1, T]
    attention_mask = proc_out.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    pixel_values = None
    image_grid_thw = None
    if args.image:
        pixel_values = proc_out["pixel_values"].to(device=device, dtype=dtype)
        image_grid_thw = proc_out["image_grid_thw"].to(device=device, dtype=torch.long)
        # Sanity: the vision tower derives cu_seqlens from grid_thw.device, and
        # flash_attn requires cu_seqlens on CUDA. Bail loudly if anything is CPU.
        for name, t in [
            ("input_ids", input_ids),
            ("pixel_values", pixel_values),
            ("image_grid_thw", image_grid_thw),
        ]:
            assert t.is_cuda, f"{name} ended up on {t.device}; expected CUDA"
        print(
            f"[megatron] forward input_ids={tuple(input_ids.shape)}@{input_ids.dtype} "
            f"pixel_values={tuple(pixel_values.shape)}@{pixel_values.dtype} "
            f"grid_thw={image_grid_thw.tolist()}"
        )
    else:
        print(f"[megatron] text-only forward input_ids={tuple(input_ids.shape)}@{input_ids.dtype}")

    captured, handles = _install_megatron_hooks(model)
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
    finally:
        for h in handles:
            h.remove()
    # GPTModel returns either (s, b, v) or (b, s, v) depending on parallel_output;
    # with tp=1, sp=False, parallel_output=True the layout is (s, b, v).
    if logits.dim() == 3 and logits.shape[0] == input_ids.shape[1]:
        logits = logits.transpose(0, 1).contiguous()  # -> (1, T, V)
    assert logits.shape[:2] == (1, input_ids.shape[1]), (
        f"unexpected logits shape {tuple(logits.shape)} for input_ids {tuple(input_ids.shape)}"
    )

    intermediates_path = _intermediates_path(Path(args.dump_dir), "megatron")
    intermediates_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, intermediates_path)
    print(f"[megatron] dumped {len(captured)} intermediate tensors -> {intermediates_path}")

    logprobs_all = F.log_softmax(logits.float(), dim=-1)  # (1, T, V)
    ids = input_ids[0].tolist()
    per_token: list[Optional[float]] = [None]
    for t in range(1, len(ids)):
        per_token.append(float(logprobs_all[0, t - 1, ids[t]].item()))

    rec = DumpRecord(
        side="megatron",
        input_ids=ids,
        logprobs=per_token,
        meta={
            "dtype": args.dtype,
            "vocab_size": int(logits.shape[-1]),
            "prompt_text_first_300": text[:300],
            "pixel_values_shape": (list(pixel_values.shape) if pixel_values is not None else None),
            "image_grid_thw": (image_grid_thw.tolist() if image_grid_thw is not None else None),
            "image": args.image,
        },
    )

    # Free GPU memory before sglang spins up its worker subprocess.
    del logits, logprobs_all, model, bridge
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# HuggingFace reference side — independent third opinion. Loaded via
# ``AutoModelForCausalLM`` + ``trust_remote_code`` so we get whatever the
# checkpoint shipped as its canonical implementation. When megatron and
# sglang disagree, this side tells you which one matches the reference.
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_hf(args: argparse.Namespace) -> DumpRecord:
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device("cuda")

    _, text, _, proc_out = _build_processor_inputs(args)
    print(f"[hf] chat_template text (first 300 chars): {text[:300]!r}")

    print(f"[hf] loading AutoModelForCausalLM from {args.hf_checkpoint} (trust_remote_code)")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    input_ids = proc_out["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = proc_out.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    fwd_kwargs = dict(input_ids=input_ids, use_cache=False)
    if args.image:
        pixel_values = proc_out["pixel_values"].to(device=device, dtype=dtype)
        image_grid_thw = proc_out["image_grid_thw"].to(device=device, dtype=torch.long)
        fwd_kwargs["pixel_values"] = pixel_values
        fwd_kwargs["image_grid_thw"] = image_grid_thw
        print(
            f"[hf] forward input_ids={tuple(input_ids.shape)} "
            f"pixel_values={tuple(pixel_values.shape)} grid_thw={image_grid_thw.tolist()}"
        )
    else:
        pixel_values = None
        image_grid_thw = None
        print(f"[hf] text-only forward input_ids={tuple(input_ids.shape)}")
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask
    captured, handles = _install_hf_hooks(model)
    try:
        outputs = model(**fwd_kwargs)
    finally:
        for h in handles:
            h.remove()
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    assert logits.dim() == 3 and logits.shape[:2] == (1, input_ids.shape[1]), (
        f"unexpected HF logits shape {tuple(logits.shape)} for input_ids {tuple(input_ids.shape)}"
    )

    intermediates_path = _intermediates_path(Path(args.dump_dir), "hf")
    intermediates_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, intermediates_path)
    print(f"[hf] dumped {len(captured)} intermediate tensors -> {intermediates_path}")

    logprobs_all = F.log_softmax(logits.float(), dim=-1)
    ids = input_ids[0].tolist()
    per_token: list[Optional[float]] = [None]
    for t in range(1, len(ids)):
        per_token.append(float(logprobs_all[0, t - 1, ids[t]].item()))

    rec = DumpRecord(
        side="hf",
        input_ids=ids,
        logprobs=per_token,
        meta={
            "dtype": args.dtype,
            "vocab_size": int(logits.shape[-1]),
            "prompt_text_first_300": text[:300],
            "pixel_values_shape": (list(pixel_values.shape) if pixel_values is not None else None),
            "image_grid_thw": (image_grid_thw.tolist() if image_grid_thw is not None else None),
            "image": args.image,
        },
    )

    del logits, logprobs_all, model, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# SGLang side
# ---------------------------------------------------------------------------


def _resolve_external_pkg() -> str:
    """Matches the value passed via ``--sglang-external-model-package`` in
    ``run-dotsocr2-8xgpu.sh``."""
    return "relax.models.dots_ocr.sglang"


@torch.no_grad()
def run_sglang(args: argparse.Namespace) -> DumpRecord:
    """Launch the offline ``sglang.Engine`` with the relax external model
    package, fire one generate() call with ``return_logprob=True``, and
    pull per-input-token logprobs out of meta_info."""
    # MUST be set *before* importing sglang so registry picks up the external
    # model package (same as relax/backends/sglang/sglang_engine.py:_init_normal).
    external_pkg = _resolve_external_pkg()
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = external_pkg
    print(f"[sglang] SGLANG_EXTERNAL_MODEL_PACKAGE={external_pkg}")

    from sglang import Engine

    _, text, _, proc_out = _build_processor_inputs(args)
    expected_ids = proc_out["input_ids"][0].tolist()
    print(f"[sglang] chat_template text (first 300 chars): {text[:300]!r}")

    sglang_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
    engine = Engine(
        model_path=args.hf_checkpoint,
        tp_size=1,
        dtype=sglang_dtype,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        # Greedy & deterministic — sampling decisions don't affect input-token
        # logprobs but max_new_tokens=1 keeps the call cheap.
        random_seed=args.seed,
        # Disable cuda graphs for fair comparison with megatron forward path;
        # they shouldn't change numerics but rule them out.
        disable_cuda_graph=True,
    )

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    generate_kwargs = dict(
        prompt=text,
        sampling_params=sampling_params,
        return_logprob=True,
        # logprob_start_len=0 → return logprobs for *every* input token; the
        # first position is always None (no preceding context).
        logprob_start_len=0,
    )
    if args.image:
        generate_kwargs["image_data"] = [args.image]
    out = engine.generate(**generate_kwargs)
    engine.shutdown()

    # SGLang's input_token_logprobs entries are (logprob, token_id, token_text)
    # tuples; the first one is (None, first_token_id, ...).
    meta = out["meta_info"]
    raw = meta["input_token_logprobs"]
    sglang_ids = [int(item[1]) for item in raw]
    per_token: list[Optional[float]] = [
        (None if item[0] is None else float(item[0])) for item in raw
    ]

    if sglang_ids != expected_ids:
        # Not fatal — image-token expansion can differ depending on processor
        # registration. Surface it loudly so the user knows the comparison is
        # not strictly token-aligned.
        n_match = sum(1 for a, b in zip(sglang_ids, expected_ids) if a == b)
        print(
            f"[sglang] WARNING: input_ids differ from HF-processor expectation "
            f"len(sglang)={len(sglang_ids)} len(expected)={len(expected_ids)} "
            f"matching_prefix_count={n_match}",
            file=sys.stderr,
        )

    return DumpRecord(
        side="sglang",
        input_ids=sglang_ids,
        logprobs=per_token,
        meta={
            "dtype": args.dtype,
            "prompt_text_first_300": text[:300],
            "expected_input_ids_first_50": expected_ids[:50],
        },
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _dump_path(dump_dir: Path, side: str) -> Path:
    return dump_dir / f"{side}.json"


def save_record(rec: DumpRecord, dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = _dump_path(dump_dir, rec.side)
    with path.open("w") as f:
        json.dump(
            {
                "side": rec.side,
                "input_ids": rec.input_ids,
                "logprobs": rec.logprobs,
                "meta": rec.meta,
            },
            f,
        )
    print(f"[{rec.side}] dumped {len(rec.input_ids)} tokens -> {path}")


def load_record(dump_dir: Path, side: str) -> DumpRecord:
    path = _dump_path(dump_dir, side)
    with path.open("r") as f:
        data = json.load(f)
    return DumpRecord(side=data["side"], input_ids=data["input_ids"], logprobs=data["logprobs"], meta=data["meta"])


def _compare_pair(a: DumpRecord, b: DumpRecord, top_n_worst: int) -> None:
    """Print id-prefix alignment + |Δ| stats + cosine + worst positions for
    one (a, b) pair. Each pair gets its own alignment because mm-image-pad
    sentinels may differ across engines and the divergence point is pair-
    specific."""
    label = f"{a.side:>8} vs {b.side:<8}"
    print(f"\n---------- {label} ----------")
    print(f"len({a.side})={len(a.input_ids)}  len({b.side})={len(b.input_ids)}")

    n = min(len(a.input_ids), len(b.input_ids))
    aligned_n = 0
    for i in range(n):
        if a.input_ids[i] != b.input_ids[i]:
            break
        aligned_n = i + 1
    print(f"aligned id-prefix = {aligned_n} / {n}")
    if aligned_n < n:
        i = aligned_n
        print(
            f"first id mismatch at pos={i}: {a.side}={a.input_ids[i]} "
            f"{b.side}={b.input_ids[i]} (truncating)"
        )

    diffs: list[tuple[int, int, float, float, float]] = []
    for i in range(1, aligned_n):  # skip position 0 (no preceding context)
        x = a.logprobs[i]
        y = b.logprobs[i]
        if x is None or y is None:
            continue
        diffs.append((i, a.input_ids[i], x, y, x - y))

    if not diffs:
        print("no comparable positions (both sides have None or no overlap)")
        return

    abs_diffs_t = torch.tensor([abs(d[4]) for d in diffs])
    print(f"compared {len(diffs)} positions")
    print(
        f"|Δ| mean={abs_diffs_t.mean().item():.6f} "
        f"max={abs_diffs_t.max().item():.6f} "
        f"p50={abs_diffs_t.median().item():.6f} "
        f"p90={abs_diffs_t.quantile(0.9).item():.6f} "
        f"p99={abs_diffs_t.quantile(0.99).item():.6f}"
    )

    a_lp = torch.tensor([d[2] for d in diffs], dtype=torch.float64)
    b_lp = torch.tensor([d[3] for d in diffs], dtype=torch.float64)
    cos_lp = torch.nn.functional.cosine_similarity(a_lp, b_lp, dim=0).item()
    cos_p = torch.nn.functional.cosine_similarity(a_lp.exp(), b_lp.exp(), dim=0).item()
    print(f"cosine(logprob) = {cos_lp:.8f}")
    print(f"cosine(prob)    = {cos_p:.8f}")

    diffs.sort(key=lambda d: abs(d[4]), reverse=True)
    n_show = min(top_n_worst, len(diffs))
    print(f"\ntop {n_show} worst positions ({a.side} - {b.side}):")
    print(f"{'pos':>6} {'token_id':>10} {a.side:>12} {b.side:>12} {'delta':>12}")
    for pos, tok, x, y, d in diffs[:n_show]:
        print(f"{pos:>6d} {tok:>10d} {x:>12.4f} {y:>12.4f} {d:>+12.4f}")


def compare(records: dict[str, DumpRecord], top_n_worst: int) -> None:
    """Pairwise-compare every available side. With megatron+sglang+hf this
    prints 3 blocks; with only 2 sides present it prints 1 block. The
    pairwise view is what lets you triangulate which engine is the broken
    one — e.g. if megatron-vs-hf and megatron-vs-sglang are both bad but
    sglang-vs-hf is near 1.0, megatron is the culprit."""
    print("\n========== compare summary ==========")
    available = sorted(records.keys())
    print(f"sides present: {available}")
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i + 1 :]]
    if not pairs:
        print("nothing to compare — need at least 2 sides")
        return
    for a_name, b_name in pairs:
        _compare_pair(records[a_name], records[b_name], top_n_worst)


def _stats_one_tensor(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-tensor diagnostics: flatten cosine + |Δ|/relative-error stats.

    Both inputs are upcast to float32 for stable accumulation; the relative
    error denominator uses |b| with a small floor to avoid div-by-zero at
    near-zero entries (typical in attention masks and post-RMSNorm tails)."""
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    n = min(a32.numel(), b32.numel())
    a32, b32 = a32[:n], b32[:n]
    cos = torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1).item()
    diff = (a32 - b32).abs()
    return {
        "n": n,
        "shape_a": tuple(a.shape),
        "shape_b": tuple(b.shape),
        "cos": cos,
        "abs_mean": diff.mean().item(),
        "abs_max": diff.max().item(),
        "rel_mean": (diff / (b32.abs() + 1e-6)).mean().item(),
        "norm_a": a32.norm().item(),
        "norm_b": b32.norm().item(),
    }


def _trim_to_aligned_prefix(a: torch.Tensor, b: torch.Tensor, aligned_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For BSH/BSV tensors, restrict to the first ``aligned_n`` positions so
    we don't include image-pad sentinels that differ between engines."""
    if a.dim() >= 2 and b.dim() >= 2:
        n = min(a.shape[1], b.shape[1], aligned_n)
        return a[:, :n], b[:, :n]
    return a, b


def compare_intermediates(dump_dir: Path, aligned_n: int) -> None:
    """Walk per-layer dumps from megatron/{side}.intermediates.pt + hf/...
    and report cosine + |Δ| per layer so we can localize the first layer
    where the two diverge meaningfully (cos drops below ~0.99)."""
    mg_path = _intermediates_path(dump_dir, "megatron")
    hf_path = _intermediates_path(dump_dir, "hf")
    if not mg_path.exists() or not hf_path.exists():
        print(f"\n[compare-intermediates] skipped: need both {mg_path.name} and {hf_path.name}")
        return

    mg = torch.load(mg_path, map_location="cpu", weights_only=True)
    hf = torch.load(hf_path, map_location="cpu", weights_only=True)
    common = [k for k in mg.keys() if k in hf]
    print("\n========== per-layer intermediates (megatron vs hf) ==========")
    print(f"dump_dir={dump_dir}  trim_to_aligned_prefix={aligned_n}")
    print(f"{'layer':<14} {'shape_mg':<22} {'shape_hf':<22} {'cos':>10} {'|Δ|mean':>10} {'|Δ|max':>10} {'rel':>10}")

    def sort_key(name: str) -> tuple:
        # embed < layer.NN < final_norm < lm_head
        order = {"embed": (0, 0), "final_norm": (2, 0), "lm_head": (3, 0)}
        if name.startswith("layer."):
            return (1, int(name.split(".", 1)[1]))
        return order.get(name, (4, 0))

    first_bad: Optional[str] = None
    for name in sorted(common, key=sort_key):
        a, b = mg[name], hf[name]
        a, b = _trim_to_aligned_prefix(a, b, aligned_n)
        s = _stats_one_tensor(a, b)
        marker = "" if s["cos"] > 0.999 else " <- DIVERGES" if first_bad is None else ""
        if s["cos"] <= 0.999 and first_bad is None:
            first_bad = name
        print(
            f"{name:<14} {str(s['shape_a']):<22} {str(s['shape_b']):<22} "
            f"{s['cos']:>10.6f} {s['abs_mean']:>10.4f} {s['abs_max']:>10.4f} {s['rel_mean']:>10.4f}{marker}"
        )
    if first_bad:
        print(f"\nfirst layer with cos<=0.999: {first_bad}")
    else:
        print("\nall layers cos > 0.999 — divergence must be downstream of lm_head (sampling/dtype)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    dump_dir = Path(args.dump_dir)

    # Run order matters: HF + megatron load the full model into our process
    # before sglang launches its subprocess, so we go megatron→hf→sglang and
    # rely on each side's del+empty_cache to free memory before the next.
    if args.side in ("megatron", "all"):
        save_record(run_megatron(args), dump_dir)

    if args.side in ("hf", "all"):
        save_record(run_hf(args), dump_dir)

    if args.side in ("sglang", "all"):
        save_record(run_sglang(args), dump_dir)

    if args.side in ("compare", "all"):
        records: dict[str, DumpRecord] = {}
        for side in ("megatron", "hf", "sglang"):
            path = _dump_path(dump_dir, side)
            if path.exists():
                records[side] = load_record(dump_dir, side)
            else:
                print(f"[compare] skip {side}: no dump at {path}")
        compare(records, top_n_worst=args.top_n_worst)

    return 0


if __name__ == "__main__":
    sys.exit(main())
