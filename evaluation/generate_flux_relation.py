"""Generate FLUX.1-dev images with GNN-predicted SeeThrough3D OSCR conditions.

It parses a spatial prompt into a two-object scene graph, asks the frozen graph
encoder for 3D centers and box sizes, renders those predictions as an OSCR
condition image, and sends that condition image through the SeeThrough3D FLUX
condition stream with LoRA processors loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from training.flux_inference_runtime import import_seethrough3d_flux, install_condition_lora_processors
from training.graph_modules import GraphSlotEncoder, build_slot_conditioning
from training.oscr_renderer import render_oscr_boxes
from training.scene_graph import build_batched_scene_graphs
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

from .prompt_parser import parse_prompt_to_scene_graph


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a relation-aware FLUX image from one prompt.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--oscr-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--gnn-layout-sample-mode",
        choices=("auto", "prior_mean", "prior_sample"),
        default="auto",
        help="For CVAE graph checkpoints, use prior_mean for deterministic boxes or prior_sample for stochastic boxes.",
    )
    return parser


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().add(1.0).mul(127.5).clamp(0, 255)
    array = array.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array)


def _load_graph_encoder(
    *,
    path: Path,
    device: str,
) -> tuple[GraphSlotEncoder, str]:
    state_dict = normalize_graph_encoder_state_dict(torch.load(path, map_location="cpu"))
    _slot_dim, text_hidden_dim, _gnn_layers, _layout_mode, _latent_dim = infer_graph_encoder_config(state_dict)
    text_encoder_type = infer_text_encoder_type(text_hidden_dim)
    graph_encoder = load_graph_encoder(
        path=path,
        text_hidden_dim=text_hidden_dim,
        device=device,
        dtype=torch.float32,
    )
    graph_encoder.requires_grad_(False)
    return graph_encoder, text_encoder_type


@torch.no_grad()
def _predict_layout(
    *,
    prompt: str,
    pipeline: Any,
    graph_encoder: GraphSlotEncoder,
    graph_tokenizer: object,
    graph_text_encoder: object,
    device: str,
    oscr_size: int,
    gnn_layout_sample_mode: str,
) -> tuple[Image.Image, dict[str, Any]]:
    scene_graph = parse_prompt_to_scene_graph(prompt)
    node_count = len(scene_graph["nodes"])
    targets = torch.zeros(1, node_count, 3, device=device)
    slot_mask = torch.ones(1, node_count, dtype=torch.bool, device=device)
    scene_graph_batch = build_batched_scene_graphs([scene_graph], slot_targets=targets, slot_mask=slot_mask)
    conditioning = build_slot_conditioning(
        tokenizer=graph_tokenizer,
        text_encoder=graph_text_encoder,
        scene_graph_batch=scene_graph_batch,
        graph_encoder=graph_encoder,
        device=device,
        layout_sample_mode=gnn_layout_sample_mode,
    )
    oscr = render_oscr_boxes(
        centers=conditioning.slot_positions,
        log_sizes=conditioning.slot_log_sizes_3d,
        slot_mask=conditioning.slot_mask,
        image_size=oscr_size,
    )
    layout = {
        "prompt": prompt,
        "graph_layout_mode": getattr(graph_encoder, "layout_mode", "deterministic"),
        "gnn_layout_sample_mode": gnn_layout_sample_mode,
        "nodes": scene_graph["nodes"],
        "edges": scene_graph["edges"],
        "predicted_centers": conditioning.slot_positions[0].detach().cpu().tolist(),
        "predicted_log_sizes_3d": conditioning.slot_log_sizes_3d[0].detach().cpu().tolist(),
        "predicted_sizes_3d": conditioning.slot_log_sizes_3d[0].exp().detach().cpu().tolist(),
    }
    return _tensor_to_pil(oscr[0]), layout


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)
    dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)

    (
        FluxPipeline,
        FluxTransformer2DModel,
        MultiDoubleStreamBlockLoraProcessor,
        MultiSingleStreamBlockLoraProcessor,
        FluxAttnProcessor2_0,
    ) = import_seethrough3d_flux()

    pipeline = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipeline.transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    pipeline.to(device)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    install_condition_lora_processors(
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

    graph_path = args.checkpoint_dir / "graph_encoder.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph encoder checkpoint: {graph_path}")
    graph_encoder, graph_text_encoder_type = _load_graph_encoder(
        path=graph_path,
        device=device,
    )
    graph_tokenizer, graph_text_encoder, _encoder_hidden_dim = load_graph_label_encoder(
        model_id=args.model_id,
        text_encoder_type=graph_text_encoder_type,
        torch_dtype=dtype,
        device=device,
    )

    oscr_image, layout = _predict_layout(
        prompt=args.prompt,
        pipeline=pipeline,
        graph_encoder=graph_encoder,
        graph_tokenizer=graph_tokenizer,
        graph_text_encoder=graph_text_encoder,
        device=device,
        oscr_size=args.oscr_size,
        gnn_layout_sample_mode=args.gnn_layout_sample_mode,
    )
    oscr_image.save(args.output_dir / "oscr_condition.png")
    (args.output_dir / "predicted_layout.json").write_text(json.dumps(layout, indent=2))

    generator = torch.Generator(device=device).manual_seed(args.seed) if device != "mps" else None
    image = pipeline(
        prompt=args.prompt,
        height=args.image_size,
        width=args.image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
        spatial_images=[oscr_image],
        subject_images=[],
        cond_size=args.oscr_size,
    ).images[0]
    image.save(args.output_dir / "generated.png")
    print(f"Generated image: {args.output_dir / 'generated.png'}")
    print(f"OSCR condition: {args.output_dir / 'oscr_condition.png'}")
    print(f"Predicted layout: {args.output_dir / 'predicted_layout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
