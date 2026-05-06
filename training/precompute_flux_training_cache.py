from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import parse_args_with_config
from .dataset import build_dataset_splits, collate_training_items
from .flux_training_cache import build_expected_manifest, file_sha256, write_manifest
from .runtime import choose_weight_dtype, resolve_torch_device, set_seed
from .train_relation_flux_lora import (
    DEFAULT_FLUX_MODEL_ID,
    _build_binding_inputs,
    _build_condition_latents,
    _encode_packed_latents,
    _import_seethrough3d_flux,
    _load_graph_encoder,
    _resize_condition_ids,
    _text_encoder_device,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute frozen FLUX relation-training inputs.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--init-graph-encoder", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--cache-dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--oscr-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--condition-renderer", choices=("seethrough", "legacy", "blender"), default="seethrough")
    parser.add_argument("--oscr-face-alpha", type=float, default=0.10)
    parser.add_argument("--oscr-azimuth-degrees", type=float, default=0.0)
    parser.add_argument("--blender-bin", type=str, default="blender")
    parser.add_argument("--blender-cache-dir", type=Path, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _cache_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def _to_cache_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.detach().cpu().to(dtype=dtype).contiguous()
    return tensor.detach().cpu().contiguous()


def _prepare_cache_dir(cache_dir: Path, overwrite: bool) -> None:
    if cache_dir.exists() and any(cache_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Cache directory already exists and is not empty: {cache_dir}. "
                "Pass --overwrite or choose a new --cache-dir."
            )
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval", "test"):
        (cache_dir / split).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def _precompute_batch(
    *,
    batch: dict[str, Any],
    pipeline: Any,
    graph_encoder: Any,
    device: str,
    model_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pixel_values = batch["pixel_values"].to(device=device, dtype=model_dtype)
    clean_latents, image_ids, image_grid = _encode_packed_latents(
        pipeline=pipeline,
        images=pixel_values,
        image_size=args.image_size,
        device=device,
        dtype=model_dtype,
    )
    cond_latents, _cond_ids, cond_grid_tensor, centers, log_sizes, slot_mask, cuboid_masks = _build_condition_latents(
        batch=batch,
        pipeline=pipeline,
        graph_encoder=graph_encoder,
        device=device,
        dtype=model_dtype,
        oscr_size=args.oscr_size,
        condition_renderer=args.condition_renderer,
        oscr_face_alpha=args.oscr_face_alpha,
        oscr_azimuth_degrees=args.oscr_azimuth_degrees,
        blender_bin=args.blender_bin,
        blender_cache_dir=args.blender_cache_dir or (args.cache_dir / "blender_condition_cache"),
    )
    binding_prompts, call_ids, cuboids_segmasks = _build_binding_inputs(
        batch=batch,
        pipeline=pipeline,
        slot_mask=slot_mask,
        cuboid_masks=cuboid_masks,
        max_sequence_length=args.max_sequence_length,
        device=device,
        prompt_prefix=args.prompt_prefix,
    )
    encoder_device = _text_encoder_device(pipeline)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=binding_prompts,
        prompt_2=binding_prompts,
        device=encoder_device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=model_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=model_dtype)
    text_ids = text_ids.to(device=device)
    cond_grid = (int(cond_grid_tensor[0].item()), int(cond_grid_tensor[1].item()))
    cond_ids = _resize_condition_ids(
        cond_grid=cond_grid,
        image_grid=image_grid,
        device=device,
        dtype=image_ids.dtype,
    )
    image_and_condition_ids = torch.cat([image_ids, cond_ids], dim=0)

    examples: list[dict[str, Any]] = []
    batch_size = clean_latents.shape[0]
    for index in range(batch_size):
        sample_call_ids = [token_ids.detach().cpu().to(dtype=torch.long) for token_ids in call_ids[index]]
        sample = {
            "clean_latents": _to_cache_dtype(clean_latents[index], cache_dtype),
            "cond_latents": _to_cache_dtype(cond_latents[index], cache_dtype),
            "prompt_embeds": _to_cache_dtype(prompt_embeds[index], cache_dtype),
            "pooled_prompt_embeds": _to_cache_dtype(pooled_prompt_embeds[index], cache_dtype),
            "text_ids": _to_cache_dtype(text_ids, cache_dtype),
            "image_and_condition_ids": _to_cache_dtype(image_and_condition_ids, cache_dtype),
            "call_ids": sample_call_ids,
            "cuboids_segmasks": _to_cache_dtype(cuboids_segmasks[index], cache_dtype).to(dtype=torch.uint8),
            "predicted_centers": centers[index].detach().cpu().to(torch.float32),
            "predicted_log_sizes": log_sizes[index].detach().cpu().to(torch.float32),
            "pred_center_abs_mean": float(centers[index].float().abs().mean().item()),
            "pred_log_size_mean": float(log_sizes[index].float().mean().item()),
            "condition_latent_norm": float(cond_latents[index].float().pow(2).mean().sqrt().item()),
            "binding_token_count": float(sum(len(token_ids) for token_ids in sample_call_ids)),
            "binding_mask_pct": float(cuboids_segmasks[index].float().mean().mul(100.0).item()),
            "prompt": batch["prompts"][index],
            "binding_prompt": binding_prompts[index],
            "scene_graph": batch["scene_graphs"][index],
            "metadata": batch["metadata"][index],
        }
        examples.append(sample)
    return examples


def main() -> int:
    args = parse_args_with_config(make_parser(), section="precompute")
    _prepare_cache_dir(args.cache_dir, overwrite=args.overwrite)
    device = resolve_torch_device(args.device)
    model_dtype = choose_weight_dtype(device, args.mixed_precision)
    cache_dtype = _cache_dtype(args.cache_dtype)
    set_seed(args.seed)

    datasets = build_dataset_splits(
        args.dataset_dir,
        image_size=args.image_size,
        prompt_prefix=args.prompt_prefix,
        limit_rows=args.limit_rows,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        test_fraction=args.test_fraction,
    )
    (
        FluxPipeline,
        _FluxTransformer2DModel,
        _MultiDoubleStreamBlockLoraProcessor,
        _MultiSingleStreamBlockLoraProcessor,
        _FluxAttnProcessor2_0,
    ) = _import_seethrough3d_flux()
    pipeline = FluxPipeline.from_pretrained(args.model_id, transformer=None, torch_dtype=model_dtype)
    pipeline.to(device)
    pipeline._graph_device = device
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)

    graph_encoder = _load_graph_encoder(
        path=args.init_graph_encoder,
        text_hidden_dim=pipeline.text_encoder.config.hidden_size,
        slot_dim=args.slot_dim,
        gnn_layers=args.gnn_layers,
        device=device,
    )

    split_counts: dict[str, int] = {}
    for split, dataset in datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_training_items,
        )
        write_index = 0
        for batch in tqdm(dataloader, desc=f"PrecomputeFluxCache[{split}]"):
            examples = _precompute_batch(
                batch=batch,
                pipeline=pipeline,
                graph_encoder=graph_encoder,
                device=device,
                model_dtype=model_dtype,
                cache_dtype=cache_dtype,
                args=args,
            )
            for sample in examples:
                path = args.cache_dir / split / f"{write_index:08d}.pt"
                torch.save(sample, path)
                write_index += 1
        split_counts[split] = write_index

    graph_sha = file_sha256(args.init_graph_encoder)
    manifest = build_expected_manifest(
        args,
        graph_sha256=graph_sha,
        dtype_name=str(cache_dtype).replace("torch.", ""),
    )
    manifest.update(
        {
            "example_count": sum(split_counts.values()),
            "split_counts": split_counts,
            "model_compute_dtype": str(model_dtype).replace("torch.", ""),
            "cache_dtype": args.cache_dtype,
            "cache_files": "one .pt shard per example under train/eval/test",
        }
    )
    write_manifest(args.cache_dir, manifest)
    (args.cache_dir / "split_counts.json").write_text(json.dumps(split_counts, indent=2, sort_keys=True))
    print(f"Wrote FLUX training cache to {args.cache_dir}")
    print(json.dumps(split_counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
