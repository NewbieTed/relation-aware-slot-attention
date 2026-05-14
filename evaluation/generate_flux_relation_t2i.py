"""Generate relation-aware FLUX samples for T2I-CompBench prompt files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.config import parse_args_with_config
from training.flux_inference_runtime import (
    build_flux_quantization_config,
    import_seethrough3d_flux,
    install_condition_lora_processors,
    pipeline_execution_device,
    set_pipeline_execution_device,
    text_encoder_device,
)
from training.graph_modules import GraphSlotEncoder, build_slot_conditioning
from training.oscr_renderer import render_oscr_boxes
from training.runtime import (
    DEFAULT_FLUX_MODEL_ID,
    choose_weight_dtype,
    infer_graph_encoder_config,
    infer_text_encoder_type,
    load_graph_encoder,
    load_graph_label_encoder,
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

OFFICIAL_SEETHROUGH3D_LORA_REPO = "va1bhavagrawa1/seethrough3d-flux.1-weights"
OFFICIAL_SEETHROUGH3D_LORA_FILENAME = "checkpoints/seethrough3d_release/lora.safetensors"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate relation-aware FLUX images for a T2I prompt file.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Relation-aware checkpoint directory. Used for flux_lora.pt and graph_encoder.pt unless overridden.",
    )
    parser.add_argument(
        "--graph-encoder-path",
        type=Path,
        default=None,
        help="Optional graph encoder checkpoint path. Useful when using an external SeeThrough3D LoRA.",
    )
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
    parser.add_argument(
        "--gnn-layout-sample-mode",
        choices=("auto", "prior_mean", "prior_sample"),
        default="auto",
        help="For CVAE graph checkpoints, use prior_mean for deterministic boxes or prior_sample for stochastic boxes.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument(
        "--external-lora-safetensors",
        type=Path,
        default=None,
        help="Optional SeeThrough3D-format LoRA .safetensors checkpoint to use instead of checkpoint-dir/flux_lora.pt.",
    )
    parser.add_argument(
        "--use-official-seethrough3d-lora",
        action="store_true",
        help="Download and use the released SeeThrough3D FLUX LoRA from Hugging Face.",
    )
    parser.add_argument("--official-seethrough3d-lora-repo", type=str, default=OFFICIAL_SEETHROUGH3D_LORA_REPO)
    parser.add_argument("--official-seethrough3d-lora-filename", type=str, default=OFFICIAL_SEETHROUGH3D_LORA_FILENAME)
    parser.add_argument(
        "--official-lora-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for the released SeeThrough3D LoRA.",
    )
    parser.add_argument("--condition-renderer", choices=("seethrough", "legacy", "blender"), default="seethrough")
    parser.add_argument("--oscr-face-alpha", type=float, default=0.10)
    parser.add_argument("--oscr-azimuth-degrees", type=float, default=0.0)
    parser.add_argument("--oscr-render-size", type=int, default=None)
    parser.add_argument("--blender-bin", type=str, default="blender")
    parser.add_argument("--blender-cache-dir", type=Path, default=None)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument(
        "--generation-prompt-suffix",
        type=str,
        default="",
        help=(
            "Optional qualitative-only text appended after the binding prompt. "
            "The GNN/OSCR layout still uses the original prompt."
        ),
    )
    parser.add_argument(
        "--generation-scene-prefix",
        type=str,
        default="",
        help=(
            "Optional qualitative-only scene text inserted after the subject anchors "
            "and before the relation prompt, while preserving token binding offsets."
        ),
    )
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
    device: str,
) -> tuple[GraphSlotEncoder, str, int]:
    state_dict = normalize_graph_encoder_state_dict(torch.load(path, map_location="cpu"))
    _slot_dim, text_hidden_dim, _gnn_layers, _layout_mode, _latent_dim = infer_graph_encoder_config(state_dict)
    text_encoder_type = infer_text_encoder_type(text_hidden_dim)
    encoder = load_graph_encoder(
        path=path,
        text_hidden_dim=text_hidden_dim,
        device=device,
        dtype=torch.float32,
    )
    encoder.requires_grad_(False)
    return encoder, text_encoder_type, text_hidden_dim


def _download_official_lora(args: argparse.Namespace) -> Path:
    """Download the released SeeThrough3D LoRA using the model-card path."""

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Downloading the official SeeThrough3D LoRA requires huggingface_hub. "
            "Install the FLUX extras with `python -m pip install -e '.[flux]'`."
        ) from exc
    path = hf_hub_download(
        repo_id=args.official_seethrough3d_lora_repo,
        filename=args.official_seethrough3d_lora_filename,
        repo_type="model",
        cache_dir=str(args.official_lora_cache_dir) if args.official_lora_cache_dir is not None else None,
    )
    return Path(path)


def _resolve_lora_path(args: argparse.Namespace) -> Path:
    if args.use_official_seethrough3d_lora:
        return _download_official_lora(args)
    if args.external_lora_safetensors is not None:
        if not args.external_lora_safetensors.exists():
            raise FileNotFoundError(f"Missing external LoRA checkpoint: {args.external_lora_safetensors}")
        return args.external_lora_safetensors
    if args.checkpoint_dir is None:
        raise ValueError(
            "Pass --checkpoint-dir for a local relation-aware checkpoint, or use "
            "--use-official-seethrough3d-lora / --external-lora-safetensors."
        )
    lora_path = args.checkpoint_dir / "flux_lora.pt"
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing FLUX LoRA checkpoint: {lora_path}")
    return lora_path


def _load_lora_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading SeeThrough3D .safetensors LoRA weights requires safetensors."
            ) from exc
        state: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
        return state
    return torch.load(path, map_location="cpu")


def _infer_lora_rank(state: dict[str, torch.Tensor]) -> int | None:
    for key, value in state.items():
        if key.endswith(".down.weight") and value.ndim >= 1:
            return int(value.shape[0])
    return None


def _load_pipeline(args: argparse.Namespace, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
    (
        FluxPipeline,
        FluxTransformer2DModel,
        MultiDoubleStreamBlockLoraProcessor,
        MultiSingleStreamBlockLoraProcessor,
        FluxAttnProcessor2_0,
    ) = import_seethrough3d_flux()

    pipeline = FluxPipeline.from_pretrained(args.model_id, transformer=None, torch_dtype=dtype)
    quantization_config = build_flux_quantization_config(args.flux_quantization, dtype)
    transformer_kwargs: dict[str, Any] = {"subfolder": "transformer", "torch_dtype": dtype}
    if quantization_config is not None:
        transformer_kwargs["quantization_config"] = quantization_config
        transformer_kwargs["device_map"] = {"": device}
    pipeline.transformer = FluxTransformer2DModel.from_pretrained(args.model_id, **transformer_kwargs)

    if args.low_vram:
        pipeline.text_encoder.to("cpu")
        pipeline.text_encoder_2.to("cpu")
        # SeeThrough3D encodes spatial_images inside pipeline.__call__ after
        # moving them to the execution device, so the VAE must stay there too.
        # Keeping only the text encoders on CPU preserves most of the low-VRAM
        # benefit without triggering CPU/CUDA conv mismatches.
        pipeline.vae.to(device=device, dtype=dtype)
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

    lora_path = _resolve_lora_path(args)
    lora_state = _load_lora_state(lora_path)
    inferred_rank = _infer_lora_rank(lora_state)
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    if inferred_rank is not None and inferred_rank != args.lora_rank:
        print(f"Using LoRA rank {inferred_rank} inferred from {lora_path} instead of --lora-rank={args.lora_rank}.")
        lora_rank = inferred_rank
    if args.use_official_seethrough3d_lora and inferred_rank is not None:
        lora_alpha = float(inferred_rank)

    install_condition_lora_processors(
        transformer=pipeline.transformer,
        rank=lora_rank,
        alpha=lora_alpha,
        cond_size=args.oscr_size,
        device=device,
        dtype=dtype,
        double_processor_cls=MultiDoubleStreamBlockLoraProcessor,
        single_processor_cls=MultiSingleStreamBlockLoraProcessor,
        base_processor_cls=FluxAttnProcessor2_0,
    )
    incompatible = pipeline.transformer.load_state_dict(lora_state, strict=False)
    print(
        f"Loaded LoRA from {lora_path} "
        f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})."
    )
    pipeline.transformer.eval()
    set_pipeline_execution_device(pipeline, device)
    return pipeline, quantization_config

@torch.no_grad()
def _predict_condition(
    *,
    prompt: str,
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    graph_tokenizer: object,
    graph_text_encoder: object,
    device: str,
    oscr_size: int,
    oscr_render_size: int | None,
    max_sequence_length: int,
    condition_renderer: str,
    oscr_face_alpha: float,
    oscr_azimuth_degrees: float,
    blender_bin: str,
    blender_cache_dir: Path,
    prompt_prefix: str,
    gnn_layout_sample_mode: str,
    generation_scene_prefix: str = "",
) -> tuple[Any, Any, str, list[list[torch.Tensor]], torch.Tensor, dict[str, Any]]:
    scene_graph = parse_prompt_to_scene_graph(prompt)
    node_count = len(scene_graph["nodes"])
    targets = torch.zeros(1, node_count, 3, device=device)
    slot_mask = torch.ones(1, node_count, dtype=torch.bool, device=device)
    batched_graph = build_batched_scene_graphs([scene_graph], slot_targets=targets, slot_mask=slot_mask)
    graph_device = "cpu" if next(graph_encoder.parameters()).device.type == "cpu" else device
    conditioning = build_slot_conditioning(
        tokenizer=graph_tokenizer,
        text_encoder=graph_text_encoder,
        scene_graph_batch=batched_graph,
        graph_encoder=graph_encoder,
        device=graph_device,
        layout_sample_mode=gnn_layout_sample_mode,
    )
    centers = conditioning.slot_positions.to(device)
    log_sizes = conditioning.slot_log_sizes_3d.to(device)
    slot_mask = conditioning.slot_mask.to(device)
    cond_grid = (oscr_size // 16, oscr_size // 16)
    render_size = oscr_render_size or oscr_size
    if condition_renderer == "seethrough":
        oscr, cuboids_segmasks = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=render_size,
            mask_size=cond_grid,
            face_alpha=oscr_face_alpha,
            azimuth_degrees=oscr_azimuth_degrees,
        )
        oscr_viz, _ = render_seethrough_oscr_and_masks(
            centers=centers,
            log_sizes=log_sizes,
            slot_mask=slot_mask,
            image_size=render_size,
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
            image_size=render_size,
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
            image_size=render_size,
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
    if generation_scene_prefix.strip():
        subject_list = binding_prompt.prompt[: binding_prompt.subject_spans[-1][1]]
        generation_prompt_for_binding = build_binding_prompt(
            original_prompt=f"{generation_scene_prefix.strip()}, {prompt}",
            scene_graph=scene_graph,
            prefix=prompt_prefix,
        )
        # Keep the same subject-anchor prefix, but let the relation phrase carry
        # the richer scene text. This preserves the object token offsets.
        if generation_prompt_for_binding.prompt.startswith(subject_list):
            binding_prompt = generation_prompt_for_binding
    call_ids = [
        call_ids_from_binding_prompt(
            tokenizer=pipeline.tokenizer_2,
            binding_prompt=binding_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )
    ]
    cuboids_segmasks = cuboids_segmasks.to(device=device, dtype=torch.uint8)
    oscr_for_model = oscr
    if oscr_for_model.shape[-1] != oscr_size or oscr_for_model.shape[-2] != oscr_size:
        oscr_for_model = F.interpolate(
            oscr_for_model,
            size=(oscr_size, oscr_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(-1.0, 1.0)
    layout = {
        "prompt": prompt,
        "binding_prompt": binding_prompt.prompt,
        "graph_layout_mode": getattr(graph_encoder, "layout_mode", "deterministic"),
        "gnn_layout_sample_mode": gnn_layout_sample_mode,
        "nodes": scene_graph["nodes"],
        "edges": scene_graph["edges"],
        "predicted_centers": centers[0].detach().cpu().tolist(),
        "predicted_sizes_3d": log_sizes[0].exp().detach().cpu().tolist(),
        "binding_token_count": sum(len(ids) for sample in call_ids for ids in sample),
        "binding_mask_pct": float(cuboids_segmasks.float().mean().mul(100.0).item()),
    }
    return _tensor_to_pil(oscr_for_model[0]), _tensor_to_pil(oscr_viz[0]), binding_prompt.prompt, call_ids, cuboids_segmasks, layout


def main() -> int:
    args = parse_args_with_config(make_parser(), section="generate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    prompts = _read_prompts(args.prompt_file, args.limit_prompts)
    pipeline, _quantization_config = _load_pipeline(args, device=device, dtype=dtype)

    graph_device = "cpu" if args.low_vram else device
    if args.graph_encoder_path is not None:
        graph_path = args.graph_encoder_path
    elif args.checkpoint_dir is not None:
        graph_path = args.checkpoint_dir / "graph_encoder.pt"
    else:
        raise ValueError("Pass --graph-encoder-path when using an external/official SeeThrough3D LoRA.")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph encoder checkpoint: {graph_path}")
    graph_encoder, graph_text_encoder_type, _graph_text_hidden_dim = _load_graph_encoder(
        path=graph_path,
        device=graph_device,
    )
    graph_tokenizer, graph_text_encoder, _encoder_hidden_dim = load_graph_label_encoder(
        model_id=args.model_id,
        text_encoder_type=graph_text_encoder_type,
        torch_dtype=dtype,
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
            graph_tokenizer=graph_tokenizer,
            graph_text_encoder=graph_text_encoder,
            device=device,
            oscr_size=args.oscr_size,
            oscr_render_size=args.oscr_render_size,
            max_sequence_length=args.max_sequence_length,
            condition_renderer=args.condition_renderer,
            oscr_face_alpha=args.oscr_face_alpha,
            oscr_azimuth_degrees=args.oscr_azimuth_degrees,
            blender_bin=args.blender_bin,
            blender_cache_dir=args.blender_cache_dir or (args.output_dir / "blender_condition_cache"),
            prompt_prefix=args.prompt_prefix,
            gnn_layout_sample_mode=args.gnn_layout_sample_mode,
            generation_scene_prefix=args.generation_scene_prefix,
        )
        condition_dir = args.output_dir / "conditions"
        condition_dir.mkdir(parents=True, exist_ok=True)
        oscr_viz_name = f"{prompt_name}_oscr_viz.png"
        oscr_viz_image.save(condition_dir / oscr_viz_name)
        layout["oscr_viz_file"] = str(Path("conditions") / oscr_viz_name)
        generation_prompt = " ".join([binding_prompt, args.generation_prompt_suffix]).strip()
        encoder_device = text_encoder_device(pipeline)
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipeline.encode_prompt(
            prompt=generation_prompt,
            prompt_2=generation_prompt,
            device=encoder_device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        for repeat_index in range(args.samples_per_prompt):
            seed = args.seed + prompt_index * args.samples_per_prompt + repeat_index
            generator_device = pipeline_execution_device(pipeline, device)
            generator = (
                torch.Generator(device=generator_device).manual_seed(seed)
                if generator_device.type != "mps"
                else None
            )
            image = pipeline(
                prompt=None,
                prompt_2=None,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
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
                    "generation_prompt": generation_prompt,
                    "layout": layout,
                }
            )
            sample_index += 1

    run_config = {
        "model_id": args.model_id,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir is not None else None,
        "graph_encoder_path": str(graph_path),
        "external_lora_safetensors": str(args.external_lora_safetensors) if args.external_lora_safetensors else None,
        "use_official_seethrough3d_lora": args.use_official_seethrough3d_lora,
        "official_seethrough3d_lora_repo": args.official_seethrough3d_lora_repo,
        "official_seethrough3d_lora_filename": args.official_seethrough3d_lora_filename,
        "prompt_file": str(args.prompt_file),
        "num_prompts": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "image_size": args.image_size,
        "oscr_size": args.oscr_size,
        "oscr_render_size": args.oscr_render_size,
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
        "generation_prompt_suffix": args.generation_prompt_suffix,
        "generation_scene_prefix": args.generation_scene_prefix,
        "gnn_layout_sample_mode": args.gnn_layout_sample_mode,
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
