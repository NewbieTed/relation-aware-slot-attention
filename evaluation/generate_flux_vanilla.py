"""Generate vanilla FLUX samples for T2I-CompBench-style prompt files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from training.config import parse_args_with_config
from training.flux_inference_runtime import set_pipeline_execution_device
from training.runtime import DEFAULT_FLUX_MODEL_ID, choose_weight_dtype, resolve_torch_device, set_seed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate vanilla FLUX images for a prompt file.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--flux-quantization", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _read_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _safe_prompt_for_filename(prompt: str) -> str:
    # T2I-CompBench extracts the prompt from the substring before the first
    # underscore, so keep spaces but avoid underscores and path-hostile chars.
    cleaned = re.sub(r"[_/\\:*?\"<>|]+", " ", prompt.strip())
    return re.sub(r"\s+", " ", cleaned)


def _build_flux_quantization_config(mode: str, dtype: torch.dtype) -> Any | None:
    if mode == "none":
        return None
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("Quantized FLUX generation needs diffusers BitsAndBytesConfig.") from exc
    try:
        __import__("bitsandbytes")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install bitsandbytes before using quantized FLUX.") from exc
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    compute_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def _load_pipeline(args: argparse.Namespace, device: str, dtype: torch.dtype) -> Any:
    from diffusers import FluxPipeline, FluxTransformer2DModel

    quantization_config = _build_flux_quantization_config(args.flux_quantization, dtype)
    if quantization_config is not None:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map={"": device},
        )
        pipeline = FluxPipeline.from_pretrained(
            args.model_id,
            transformer=transformer,
            torch_dtype=dtype,
        )
    else:
        pipeline = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype)

    pipeline.set_progress_bar_config(disable=True)
    if args.low_vram and device == "cuda":
        if quantization_config is not None:
            # Match the relation-aware loader: keep the large frozen text
            # encoders on CPU, while the quantized transformer and VAE stay on
            # CUDA. This avoids bitsandbytes/offload hook interactions that can
            # still OOM or move tensors to the wrong device.
            pipeline.text_encoder.to("cpu")
            pipeline.text_encoder_2.to("cpu")
            pipeline.vae.to(device=device, dtype=dtype)
            set_pipeline_execution_device(pipeline, device)
        else:
            pipeline.enable_sequential_cpu_offload()
    elif quantization_config is None:
        pipeline.to(device)
    else:
        pipeline.vae.to(device=device, dtype=dtype)
        pipeline.text_encoder.to(device)
        pipeline.text_encoder_2.to(device)
    return pipeline


def main() -> int:
    args = parse_args_with_config(make_parser(), section="generate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    prompts = _read_prompts(args.prompt_file, args.limit_prompts)
    pipeline = _load_pipeline(args, device=device, dtype=dtype)

    records: list[dict[str, object]] = []
    sample_index = 0
    execution_device = torch.device(getattr(pipeline, "_execution_device", device))
    for prompt_index, prompt in enumerate(tqdm(prompts, desc="VanillaFluxGeneration")):
        prompt_name = _safe_prompt_for_filename(prompt)
        for repeat_index in range(args.samples_per_prompt):
            seed = args.seed + prompt_index * args.samples_per_prompt + repeat_index
            generator = torch.Generator(device=execution_device).manual_seed(seed) if device != "mps" else None
            image = pipeline(
                prompt=prompt,
                height=args.image_size,
                width=args.image_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=args.max_sequence_length,
            ).images[0]
            filename = f"{prompt_name}_{sample_index:06d}.png"
            image.save(samples_dir / filename)
            records.append(
                {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "file_name": filename,
                }
            )
            sample_index += 1

    run_config = {
        "model_id": args.model_id,
        "prompt_file": str(args.prompt_file),
        "num_prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "image_size": args.image_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "mixed_precision": args.mixed_precision,
        "flux_quantization": args.flux_quantization,
        "low_vram": args.low_vram,
        "seed": args.seed,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (args.output_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    print(f"Generated {len(records)} samples into {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
