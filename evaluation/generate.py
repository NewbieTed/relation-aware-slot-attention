from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

MODEL_REGISTRY = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd21": "stabilityai/stable-diffusion-2-1",
}


def resolve_torch_device(device_preference: str = "auto") -> str:
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_preference == "cuda" and not torch.cuda.is_available():
        print("Evaluation: CUDA requested but not available, falling back to CPU.")
        return "cpu"

    if (
        device_preference == "mps"
        and not (torch.backends.mps.is_built() and torch.backends.mps.is_available())
    ):
        print("Evaluation: MPS requested but not available, falling back to CPU.")
        return "cpu"

    return device_preference


def load_prompt_lines(path: Path) -> list[str]:
    prompts = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        prompts.append(line)
    return prompts


def build_pipeline(model_name: str, device: str):
    from diffusers import StableDiffusionPipeline

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def save_run_config(
    output_dir: Path,
    *,
    model_key: str,
    model_name: str,
    prompt_file: Path,
    device: str,
    seed: int,
    num_images_per_prompt: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
    prompt_count: int,
) -> None:
    payload = {
        "model_key": model_key,
        "model_name": model_name,
        "prompt_file": str(prompt_file),
        "device": device,
        "seed": seed,
        "num_images_per_prompt": num_images_per_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "image_size": image_size,
        "prompt_count": prompt_count,
        "output_layout": "t2i-compbench-style",
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


def prompt_to_filename(prompt: str, image_index: int) -> str:
    """Match T2I-CompBench's flat sample naming convention."""
    return f"{prompt}_{image_index:06d}.png"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate images for a plain-text prompt file using a vanilla Stable Diffusion baseline."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="sd15",
        help="Vanilla Stable Diffusion model key to evaluate (default: sd15).",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="Plain-text prompt file with one prompt per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated images and run metadata will be written.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Torch device for generation (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt (default: 1).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of diffusion steps (default: 30).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square output size in pixels (default: 512).",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Optional cap on the number of prompts for a dry run.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting global sample index used in output filenames (default: 0).",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    model_name = MODEL_REGISTRY[args.model]
    device = resolve_torch_device(args.device)
    prompts = load_prompt_lines(args.prompts_file)
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        raise ValueError("No prompts found in prompt file")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(model_name, device)
    save_run_config(
        args.output_dir,
        model_key=args.model,
        model_name=model_name,
        prompt_file=args.prompts_file,
        device=device,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        prompt_count=len(prompts),
    )

    global_image_index = args.start_index
    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(args.num_images_per_prompt):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + prompt_index * args.num_images_per_prompt + image_index
            )
            result = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.image_size,
                width=args.image_size,
                generator=generator,
            )
            image: Image.Image = result.images[0]
            image.save(samples_dir / prompt_to_filename(prompt, global_image_index))
            global_image_index += 1

    print(f"Generated {len(prompts)} prompts into {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
