from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.graph_modules import GraphSlotEncoder, build_slot_conditioning
from training.relation_attention import install_relation_aware_processors
from training.scene_graph import build_batched_scene_graphs

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


def resolve_lora_weight_name(lora_path: Path) -> str | None:
    if lora_path.is_file():
        return lora_path.name

    candidate_names = (
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    for candidate in candidate_names:
        if (lora_path / candidate).exists():
            return candidate
    return None


def infer_graph_encoder_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    slot_dim = int(state_dict["node_proj.weight"].shape[0])
    layer_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    gnn_layers = max(layer_ids) + 1 if layer_ids else 0
    return slot_dim, gnn_layers


def load_graph_encoder(
    *,
    path: Path,
    text_hidden_dim: int,
    device: str,
    dtype: torch.dtype,
) -> GraphSlotEncoder:
    state_dict = torch.load(path, map_location="cpu")
    slot_dim, gnn_layers = infer_graph_encoder_config(state_dict)
    encoder = GraphSlotEncoder(
        text_hidden_dim=text_hidden_dim,
        slot_dim=slot_dim,
        num_layers=gnn_layers,
    ).to(device=device, dtype=dtype)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def resolve_relation_aware_artifacts(
    *,
    relation_aware_dir: Path | None,
    unet_path: Path | None,
    lora_path: Path | None,
    graph_encoder_path: Path | None,
    relation_attention_path: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    if relation_aware_dir is None:
        return unet_path, lora_path, graph_encoder_path, relation_attention_path

    candidate_unet = relation_aware_dir / "unet"
    resolved_unet = unet_path or (candidate_unet if candidate_unet.exists() else None)
    if lora_path is not None:
        resolved_lora = lora_path
    else:
        candidate_lora = relation_aware_dir / "lora"
        resolved_lora = candidate_lora if candidate_lora.exists() else None
    resolved_graph = graph_encoder_path or (relation_aware_dir / "graph_encoder.pt")
    resolved_attention = relation_attention_path or (relation_aware_dir / "relation_attention.pt")
    return resolved_unet, resolved_lora, resolved_graph, resolved_attention


def build_pipeline(
    model_name: str,
    device: str,
    unet_path: Path | None = None,
    lora_path: Path | None = None,
    relation_attention_path: Path | None = None,
    *,
    disable_progress_bar: bool = True,
):
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    if unet_path is not None:
        if not unet_path.exists():
            raise FileNotFoundError(f"Missing UNet checkpoint path: {unet_path}")
        pipeline.unet = UNet2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch_dtype,
        )
        print(f"Evaluation: loaded full UNet weights from {unet_path}")
    pipeline = pipeline.to(device)

    relation_attention_processors: dict[str, torch.nn.Module] | None = None
    if relation_attention_path is not None:
        relation_attention_processors = install_relation_aware_processors(pipeline.unet)
        if relation_attention_path.exists():
            processor_state = torch.load(relation_attention_path, map_location="cpu")
            for name, module in relation_attention_processors.items():
                state = processor_state.get(name)
                if state is not None:
                    module.load_state_dict(state)
            print(f"Evaluation: loaded relation-attention weights from {relation_attention_path}")
        else:
            print(
                "Evaluation warning: relation-attention weights were requested but not found at "
                f"{relation_attention_path}. Using freshly initialized relation-attention processors."
            )

    if lora_path is not None:
        if not lora_path.exists():
            raise FileNotFoundError(f"Missing LoRA adapter path: {lora_path}")
        adapter_name = "scopdepth"
        weight_name = resolve_lora_weight_name(lora_path)
        pipeline.unet.load_lora_adapter(
            str(lora_path),
            prefix=None,
            adapter_name=adapter_name,
            weight_name=weight_name,
        )
        pipeline.unet.set_adapters(adapter_name)
        active_adapters = list(pipeline.unet.active_adapters())
        if adapter_name not in active_adapters:
            raise RuntimeError(
                f"LoRA adapter was loaded from {lora_path} but is not active. "
                f"Active adapters: {active_adapters}"
            )
        print(f"Evaluation: loaded LoRA adapter from {lora_path}")
        if weight_name is not None:
            print(f"Evaluation: LoRA weight file = {weight_name}")
        print(f"Evaluation: active UNet adapters = {active_adapters}")

    pipeline.set_progress_bar_config(disable=disable_progress_bar)
    return pipeline


def save_run_config(
    output_dir: Path,
    *,
    model_key: str,
    model_name: str,
    unet_path: Path | None,
    lora_path: Path | None,
    graph_encoder_path: Path | None,
    relation_attention_path: Path | None,
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
        "unet_path": str(unet_path) if unet_path is not None else None,
        "lora_path": str(lora_path) if lora_path is not None else None,
        "graph_encoder_path": str(graph_encoder_path) if graph_encoder_path is not None else None,
        "relation_attention_path": str(relation_attention_path) if relation_attention_path is not None else None,
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
    return f"{prompt}_{image_index:06d}.png"


def build_relation_aware_conditioning(
    *,
    prompt: str,
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    scene_graph = parse_prompt_to_scene_graph(prompt)
    node_count = len(scene_graph["nodes"])
    slot_targets = torch.zeros(1, node_count, 3, device=device)
    slot_mask = torch.ones(1, node_count, dtype=torch.bool, device=device)
    scene_graph_batch = build_batched_scene_graphs(
        [scene_graph],
        slot_targets=slot_targets,
        slot_mask=slot_mask,
    )
    conditioning = build_slot_conditioning(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        scene_graph_batch=scene_graph_batch,
        graph_encoder=graph_encoder,
        device=device,
    )
    return (
        conditioning.slot_embeddings,
        conditioning.slot_positions,
        conditioning.slot_log_sigmas,
        int(node_count),
    )


def describe_prompt_parse_support(prompt: str) -> tuple[bool, str | None]:
    try:
        parse_prompt_to_scene_graph(prompt)
        return True, None
    except ValueError as exc:
        return False, str(exc)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate images for a plain-text prompt file using vanilla or relation-aware Stable Diffusion."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="sd15",
        help="Stable Diffusion model key to evaluate (default: sd15).",
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
        "--unet-path",
        type=Path,
        default=None,
        help="Optional full UNet checkpoint directory to load on top of the base model.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Optional LoRA adapter directory to load on top of the base model.",
    )
    parser.add_argument(
        "--relation-aware-dir",
        type=Path,
        default=None,
        help="Optional directory containing relation-aware artifacts: lora/, graph_encoder.pt, relation_attention.pt.",
    )
    parser.add_argument(
        "--graph-encoder-path",
        type=Path,
        default=None,
        help="Optional graph encoder checkpoint for relation-aware generation.",
    )
    parser.add_argument(
        "--relation-attention-path",
        type=Path,
        default=None,
        help="Optional relation-attention processor checkpoint for relation-aware generation.",
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
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="Show the per-image diffusion progress bar during generation.",
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

    unet_path, lora_path, graph_encoder_path, relation_attention_path = resolve_relation_aware_artifacts(
        relation_aware_dir=args.relation_aware_dir,
        unet_path=args.unet_path,
        lora_path=args.lora_path,
        graph_encoder_path=args.graph_encoder_path,
        relation_attention_path=args.relation_attention_path,
    )
    relation_aware_enabled = graph_encoder_path is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(
        model_name,
        device,
        unet_path=unet_path,
        lora_path=lora_path,
        relation_attention_path=relation_attention_path if relation_aware_enabled else None,
        disable_progress_bar=not args.show_progress_bar,
    )
    graph_encoder: GraphSlotEncoder | None = None
    if relation_aware_enabled:
        if not graph_encoder_path.exists():
            raise FileNotFoundError(f"Missing graph encoder checkpoint: {graph_encoder_path}")
        graph_encoder = load_graph_encoder(
            path=graph_encoder_path,
            text_hidden_dim=pipeline.text_encoder.config.hidden_size,
            device=device,
            dtype=pipeline.text_encoder.dtype,
        )
        print(f"Evaluation: loaded graph encoder from {graph_encoder_path}")

    save_run_config(
        args.output_dir,
        model_key=args.model,
        model_name=model_name,
        unet_path=unet_path,
        lora_path=lora_path,
        graph_encoder_path=graph_encoder_path,
        relation_attention_path=relation_attention_path if relation_aware_enabled else None,
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
    do_classifier_free_guidance = args.guidance_scale > 1.0
    fallback_count = 0
    for prompt_index, prompt in enumerate(prompts):
        cross_attention_kwargs: dict[str, Any] | None = None
        prompt_embeds = None
        negative_prompt_embeds = None
        use_relation_aware_for_prompt = relation_aware_enabled

        if relation_aware_enabled:
            supported, reason = describe_prompt_parse_support(prompt)
            if not supported:
                fallback_count += 1
                use_relation_aware_for_prompt = False
                print(
                    "Evaluation warning: falling back to vanilla prompt conditioning for "
                    f"unsupported relation prompt: {prompt!r}. Reason: {reason}"
                )
            else:
                assert graph_encoder is not None
                prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=args.num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=None,
                )
                text_token_count = int(prompt_embeds.shape[1])
                slot_embeddings, slot_positions, slot_log_sigmas, _ = build_relation_aware_conditioning(
                    prompt=prompt,
                    pipeline=pipeline,
                    graph_encoder=graph_encoder,
                    device=device,
                )
                slot_embeddings = slot_embeddings.to(prompt_embeds.dtype).repeat_interleave(
                    args.num_images_per_prompt,
                    dim=0,
                )
                prompt_embeds = torch.cat([prompt_embeds, slot_embeddings], dim=1)
                if negative_prompt_embeds is not None:
                    zero_slots = torch.zeros_like(slot_embeddings)
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, zero_slots], dim=1)
                cross_attention_kwargs = {
                    "slot_positions": slot_positions.repeat_interleave(args.num_images_per_prompt, dim=0),
                    "slot_log_sigmas": slot_log_sigmas.repeat_interleave(args.num_images_per_prompt, dim=0),
                    "slot_mask": torch.ones(
                        args.num_images_per_prompt,
                        slot_positions.shape[1],
                        dtype=torch.bool,
                        device=slot_positions.device,
                    ),
                    "text_token_count": text_token_count,
                }

        for image_index in range(args.num_images_per_prompt):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + prompt_index * args.num_images_per_prompt + image_index
            )
            if use_relation_aware_for_prompt:
                image_prompt_embeds = prompt_embeds[image_index : image_index + 1]
                image_negative_prompt_embeds = (
                    negative_prompt_embeds[image_index : image_index + 1]
                    if negative_prompt_embeds is not None
                    else None
                )
                image_cross_kwargs = {
                    key: value[image_index : image_index + 1]
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == args.num_images_per_prompt
                    else value
                    for key, value in (cross_attention_kwargs or {}).items()
                }
                result = pipeline(
                    prompt_embeds=image_prompt_embeds,
                    negative_prompt_embeds=image_negative_prompt_embeds,
                    cross_attention_kwargs=image_cross_kwargs,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.image_size,
                    width=args.image_size,
                    generator=generator,
                )
            else:
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

    if relation_aware_enabled and fallback_count > 0:
        print(
            "Evaluation summary: "
            f"{fallback_count} prompt(s) used vanilla fallback because the rule-based parser "
            "does not yet support their relation wording."
        )
    print(f"Generated {len(prompts)} prompts into {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
