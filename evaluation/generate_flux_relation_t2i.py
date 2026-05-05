"""Generate relation-aware FLUX samples for T2I-CompBench prompt files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.graph_modules import GraphSlotEncoder, build_slot_conditioning
from training.oscr_renderer import render_oscr_boxes
from training.runtime import (
    DEFAULT_FLUX_MODEL_ID,
    choose_weight_dtype,
    normalize_graph_encoder_state_dict,
    resolve_torch_device,
    set_seed,
)
from training.scene_graph import build_batched_scene_graphs
from training.seethrough_condition import (
    build_binding_prompt,
    call_ids_from_binding_prompt,
    render_blender_oscr_conditions,
    render_seethrough_oscr_and_masks,
)
from training.train_relation_flux_lora import (
    _build_flux_quantization_config,
    _import_seethrough3d_flux,
    _install_condition_lora_processors,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate relation-aware FLUX images for a T2I prompt file.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--flux-quantization", choices=("none", "8bit", "4bit"), default="4bit")
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--oscr-size", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--condition-renderer", choices=("seethrough", "legacy", "blender"), default="seethrough")
    parser.add_argument("--oscr-face-alpha", type=float, default=0.10)
    parser.add_argument("--oscr-azimuth-degrees", type=float, default=0.0)
    parser.add_argument("--blender-bin", type=str, default="blender")
    parser.add_argument("--blender-cache-dir", type=Path, default=None)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    return parser


def _read_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _safe_prompt_for_filename(prompt: str) -> str:
    cleaned = re.sub(r"[_/\\:*?\"<>|]+", " ", prompt.strip())
    return re.sub(r"\s+", " ", cleaned)


def _tensor_to_pil(image: torch.Tensor) -> Any:
    from PIL import Image

    array = image.detach().float().add(1.0).mul(127.5).clamp(0, 255)
    array = array.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array)


def _load_graph_encoder(
    *,
    path: Path,
    text_hidden_dim: int,
    slot_dim: int,
    gnn_layers: int,
    device: str,
) -> GraphSlotEncoder:
    encoder = GraphSlotEncoder(
        text_hidden_dim=text_hidden_dim,
        slot_dim=slot_dim,
        num_layers=gnn_layers,
    ).to(device)
    encoder.load_state_dict(
        normalize_graph_encoder_state_dict(torch.load(path, map_location=device)),
        strict=False,
    )
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _load_pipeline(args: argparse.Namespace, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
    (
        FluxPipeline,
        FluxTransformer2DModel,
        MultiDoubleStreamBlockLoraProcessor,
        MultiSingleStreamBlockLoraProcessor,
        FluxAttnProcessor2_0,
    ) = _import_seethrough3d_flux()

    pipeline = FluxPipeline.from_pretrained(args.model_id, transformer=None, torch_dtype=dtype)
    quantization_config = _build_flux_quantization_config(args.flux_quantization, dtype)
    transformer_kwargs: dict[str, Any] = {"subfolder": "transformer", "torch_dtype": dtype}
    if quantization_config is not None:
        transformer_kwargs["quantization_config"] = quantization_config
        transformer_kwargs["device_map"] = {"": device}
    pipeline.transformer = FluxTransformer2DModel.from_pretrained(args.model_id, **transformer_kwargs)

    if args.low_vram:
        pipeline.text_encoder.to("cpu")
        pipeline.text_encoder_2.to("cpu")
        pipeline.vae.to("cpu")
        if quantization_config is None:
            pipeline.transformer.to(device=device, dtype=dtype)
    else:
        if quantization_config is None:
            pipeline.to(device)
        else:
            pipeline.vae.to(device=device, dtype=dtype)
            pipeline.text_encoder.to(device)
            pipeline.text_encoder_2.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    _install_condition_lora_processors(
        transformer=pipeline.transformer,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        cond_size=args.oscr_size,
        device=device,
        dtype=dtype,
        double_processor_cls=MultiDoubleStreamBlockLoraProcessor,
        single_processor_cls=MultiSingleStreamBlockLoraProcessor,
        base_processor_cls=FluxAttnProcessor2_0,
    )
    lora_path = args.checkpoint_dir / "flux_lora.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing FLUX LoRA checkpoint: {lora_path}")
    pipeline.transformer.load_state_dict(torch.load(lora_path, map_location=device), strict=False)
    pipeline.transformer.eval()
    _set_pipeline_execution_device(pipeline, device)
    return pipeline, quantization_config


def _pipeline_execution_device(pipeline: Any, fallback: str) -> torch.device:
    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is None:
        return torch.device(fallback)
    return torch.device(execution_device)


def _set_pipeline_execution_device(pipeline: Any, device: str) -> None:
    """Force SeeThrough3D FLUX to prepare inference latents on the transformer device."""

    try:
        object.__setattr__(pipeline, "_execution_device", torch.device(device))
    except Exception:
        pipeline._execution_device = torch.device(device)


@torch.no_grad()
def _predict_condition(
    *,
    prompt: str,
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    device: str,
    oscr_size: int,
    max_sequence_length: int,
    condition_renderer: str,
    oscr_face_alpha: float,
    oscr_azimuth_degrees: float,
    blender_bin: str,
    blender_cache_dir: Path,
    prompt_prefix: str,
) -> tuple[Any, Any, str, list[list[torch.Tensor]], torch.Tensor, dict[str, Any]]:
    scene_graph = parse_prompt_to_scene_graph(prompt)
    node_count = len(scene_graph["nodes"])
    targets = torch.zeros(1, node_count, 3, device=device)
    slot_mask = torch.ones(1, node_count, dtype=torch.bool, device=device)
    batched_graph = build_batched_scene_graphs([scene_graph], slot_targets=targets, slot_mask=slot_mask)
    graph_device = "cpu" if next(graph_encoder.parameters()).device.type == "cpu" else device
    conditioning = build_slot_conditioning(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        scene_graph_batch=batched_graph,
        graph_encoder=graph_encoder,
        device=graph_device,
    )
    centers = conditioning.slot_positions.to(device)
    log_sizes = conditioning.slot_log_sizes_3d.to(device)
    slot_mask = conditioning.slot_mask.to(device)
    cond_grid = (oscr_size // 16, oscr_size // 16)
    if condition_renderer == "seethrough":
        oscr, cuboids_segmasks = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_grid,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
        )
        oscr_viz, _ = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_grid,
            face_alpha=max(oscr_face_alpha, 0.25),
            azimuth_degrees=oscr_azimuth_degrees,
        )
    elif condition_renderer == "blender":
        oscr = render_blender_oscr_conditions(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            scene_graphs=[scene_graph],
            prompts=[prompt],
            image_size=oscr_size,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
            blender_bin=blender_bin,
            cache_dir=blender_cache_dir,
        )
        oscr_viz = render_blender_oscr_conditions(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            scene_graphs=[scene_graph],
            prompts=[prompt],
            image_size=oscr_size,
            face_alpha=max(oscr_face_alpha, 0.25),
            azimuth_degrees=oscr_azimuth_degrees,
            blender_bin=blender_bin,
            cache_dir=blender_cache_dir,
        )
        _, cuboids_segmasks = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_grid,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
        )
    else:
        oscr = render_oscr_boxes(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
        )
        _, cuboids_segmasks = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_grid,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
        )
        oscr_viz, _ = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=oscr_size,
            mask_size=cond_grid,
            face_alpha=0.25,
            azimuth_degrees=oscr_azimuth_degrees,
        )
    binding_prompt = build_binding_prompt(
        original_prompt=prompt,
        scene_graph=scene_graph,
        prefix=prompt_prefix,
    )
    call_ids = [
        call_ids_from_binding_prompt(
            tokenizer=pipeline.tokenizer_2,
            binding_prompt=binding_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )
    ]
    cuboids_segmasks = cuboids_segmasks.to(device=device, dtype=torch.uint8)
    layout = {
        "prompt": prompt,
        "binding_prompt": binding_prompt.prompt,
        "nodes": scene_graph["nodes"],
        "edges": scene_graph["edges"],
        "predicted_centers": centers[0].detach().cpu().tolist(),
        "predicted_sizes_3d": log_sizes[0].exp().detach().cpu().tolist(),
        "binding_token_count": sum(len(ids) for sample in call_ids for ids in sample),
        "binding_mask_pct": float(cuboids_segmasks.float().mean().mul(100.0).item()),
    }
    return _tensor_to_pil(oscr[0]), _tensor_to_pil(oscr_viz[0]), binding_prompt.prompt, call_ids, cuboids_segmasks, layout


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    prompts = _read_prompts(args.prompt_file, args.limit_prompts)
    pipeline, _quantization_config = _load_pipeline(args, device=device, dtype=dtype)

    graph_device = "cpu" if args.low_vram else device
    graph_path = args.checkpoint_dir / "graph_encoder.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph encoder checkpoint: {graph_path}")
    graph_encoder = _load_graph_encoder(
        path=graph_path,
        text_hidden_dim=pipeline.text_encoder.config.hidden_size,
        slot_dim=args.slot_dim,
        gnn_layers=args.gnn_layers,
        device=graph_device,
    )

    records: list[dict[str, object]] = []
    sample_index = 0
    for prompt_index, prompt in enumerate(tqdm(prompts, desc="RelationFluxGeneration")):
        prompt_name = _safe_prompt_for_filename(prompt)
        oscr_image, oscr_viz_image, binding_prompt, call_ids, cuboids_segmasks, layout = _predict_condition(
            prompt=prompt,
            pipeline=pipeline,
            graph_encoder=graph_encoder,
            device=device,
            oscr_size=args.oscr_size,
            max_sequence_length=args.max_sequence_length,
            condition_renderer=args.condition_renderer,
            oscr_face_alpha=args.oscr_face_alpha,
            oscr_azimuth_degrees=args.oscr_azimuth_degrees,
            blender_bin=args.blender_bin,
            blender_cache_dir=args.blender_cache_dir or (args.output_dir / "blender_condition_cache"),
            prompt_prefix=args.prompt_prefix,
        )
        condition_dir = args.output_dir / "conditions"
        condition_dir.mkdir(parents=True, exist_ok=True)
        oscr_viz_name = f"{prompt_name}_oscr_viz.png"
        oscr_viz_image.save(condition_dir / oscr_viz_name)
        layout["oscr_viz_file"] = str(Path("conditions") / oscr_viz_name)
        for repeat_index in range(args.samples_per_prompt):
            seed = args.seed + prompt_index * args.samples_per_prompt + repeat_index
            generator_device = _pipeline_execution_device(pipeline, device)
            generator = (
                torch.Generator(device=generator_device).manual_seed(seed)
                if generator_device.type != "mps"
                else None
            )
            image = pipeline(
                prompt=binding_prompt,
                prompt_2=binding_prompt,
                height=args.image_size,
                width=args.image_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=args.max_sequence_length,
                spatial_images=[oscr_image],
                subject_images=[],
                cond_size=args.oscr_size,
                call_ids=call_ids,
                cuboids_segmasks=cuboids_segmasks,
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
                    "layout": layout,
                }
            )
            sample_index += 1

    run_config = {
        "model_id": args.model_id,
        "checkpoint_dir": str(args.checkpoint_dir),
        "prompt_file": str(args.prompt_file),
        "num_prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "image_size": args.image_size,
        "oscr_size": args.oscr_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "mixed_precision": args.mixed_precision,
        "flux_quantization": args.flux_quantization,
        "low_vram": args.low_vram,
        "condition_renderer": args.condition_renderer,
        "oscr_face_alpha": args.oscr_face_alpha,
        "oscr_azimuth_degrees": args.oscr_azimuth_degrees,
        "blender_bin": args.blender_bin,
        "blender_cache_dir": str(args.blender_cache_dir or (args.output_dir / "blender_condition_cache")),
        "prompt_prefix": args.prompt_prefix,
        "seed": args.seed,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (args.output_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    print(f"Generated {len(records)} relation-aware samples into {samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
