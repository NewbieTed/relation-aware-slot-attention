"""Regenerate missing relation-aware FLUX sample PNGs from an existing run.

This is intentionally a repair utility, not a normal benchmark entrypoint. It
reads ``samples.jsonl`` from an output directory, checks which recorded PNG files
are missing, and regenerates only those records using the original generation
config. The common use case is recovering from a single missing image after a
long T2I-CompBench run, without spending another day regenerating every sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from evaluation.generate_flux_relation_t2i import (
    _load_graph_encoder,
    _load_pipeline,
    _predict_condition,
    _resolve_graph_path,
)
from training.config import _load_raw_config, _section_config
from training.flux_inference_runtime import pipeline_execution_device, text_encoder_device
from training.runtime import choose_weight_dtype, load_graph_label_encoder, resolve_torch_device, set_seed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate missing relation-aware FLUX sample PNGs.")
    parser.add_argument("--config", type=Path, required=True, help="Original relation eval config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output dir from config.")
    parser.add_argument("--max-repair", type=int, default=None, help="Optional cap on repaired missing files.")
    return parser


def _namespace_from_config(config_path: Path) -> argparse.Namespace:
    raw = _load_raw_config(config_path)
    generate = _section_config(raw, "generate")
    return argparse.Namespace(**generate)


def _load_records(samples_jsonl: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in samples_jsonl.read_text().splitlines() if line.strip()]


@torch.no_grad()
def main() -> int:
    args = make_parser().parse_args()
    gen_args = _namespace_from_config(args.config)
    if args.output_dir is not None:
        gen_args.output_dir = args.output_dir

    output_dir = Path(gen_args.output_dir)
    samples_dir = output_dir / "samples"
    records_path = output_dir / "samples.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing samples.jsonl: {records_path}")
    if not samples_dir.exists():
        raise FileNotFoundError(f"Missing samples directory: {samples_dir}")

    records = _load_records(records_path)
    missing_records = [record for record in records if not (samples_dir / record["file_name"]).exists()]
    if args.max_repair is not None:
        missing_records = missing_records[: args.max_repair]
    if not missing_records:
        print(f"No missing sample PNGs found under {samples_dir}")
        return 0

    print(f"Repairing {len(missing_records)} missing sample PNG(s) under {samples_dir}")
    for record in missing_records:
        print(f"  missing: {record['file_name']} prompt_index={record['prompt_index']} repeat={record['repeat_index']}")

    device = resolve_torch_device(gen_args.device)
    dtype = choose_weight_dtype(device, gen_args.mixed_precision)
    set_seed(gen_args.seed)

    pipeline, _quantization_config = _load_pipeline(gen_args, device=device, dtype=dtype)
    graph_path = _resolve_graph_path(gen_args)
    graph_encoder, text_encoder_type, _text_hidden_dim = _load_graph_encoder(path=graph_path, device=device)
    graph_tokenizer, graph_text_encoder = load_graph_label_encoder(
        gen_args.model_id,
        text_encoder_type=text_encoder_type,
        device=device,
        dtype=torch.float32,
    )

    condition_cache: dict[int, tuple[Any, Any, str, Any, Any]] = {}
    for record in missing_records:
        prompt = str(record["prompt"])
        prompt_index = int(record["prompt_index"])
        repeat_index = int(record["repeat_index"])
        seed = int(record["seed"])
        file_name = str(record["file_name"])

        if prompt_index not in condition_cache:
            condition_cache[prompt_index] = _predict_condition(
                prompt=prompt,
                pipeline=pipeline,
                graph_encoder=graph_encoder,
                graph_tokenizer=graph_tokenizer,
                graph_text_encoder=graph_text_encoder,
                device=device,
                oscr_size=gen_args.oscr_size,
                oscr_render_size=gen_args.oscr_render_size,
                max_sequence_length=gen_args.max_sequence_length,
                condition_renderer=gen_args.condition_renderer,
                oscr_face_alpha=gen_args.oscr_face_alpha,
                oscr_azimuth_degrees=gen_args.oscr_azimuth_degrees,
                blender_bin=gen_args.blender_bin,
                blender_cache_dir=gen_args.blender_cache_dir or (output_dir / "blender_condition_cache"),
                prompt_prefix=gen_args.prompt_prefix,
                gnn_layout_sample_mode=gen_args.gnn_layout_sample_mode,
                generation_scene_prefix=getattr(gen_args, "generation_scene_prefix", ""),
            )[:5]

        oscr_image, _oscr_viz_image, binding_prompt, call_ids, cuboids_segmasks = condition_cache[prompt_index]
        generation_prompt = " ".join([binding_prompt, getattr(gen_args, "generation_prompt_suffix", "")]).strip()
        encoder_device = text_encoder_device(pipeline)
        prompt_embeds, pooled_prompt_embeds, _text_ids = pipeline.encode_prompt(
            prompt=generation_prompt,
            prompt_2=generation_prompt,
            device=encoder_device,
            num_images_per_prompt=1,
            max_sequence_length=gen_args.max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

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
            height=gen_args.image_size,
            width=gen_args.image_size,
            num_inference_steps=gen_args.num_inference_steps,
            guidance_scale=gen_args.guidance_scale,
            generator=generator,
            max_sequence_length=gen_args.max_sequence_length,
            spatial_images=[oscr_image],
            subject_images=[],
            cond_size=gen_args.oscr_size,
            call_ids=call_ids,
            cuboids_segmasks=cuboids_segmasks,
        ).images[0]
        image.save(samples_dir / file_name)
        print(f"  repaired: {file_name} prompt_index={prompt_index} repeat={repeat_index}")

    still_missing = [record["file_name"] for record in records if not (samples_dir / record["file_name"]).exists()]
    if still_missing:
        raise RuntimeError(f"Repair incomplete. Still missing {len(still_missing)} sample(s): {still_missing[:5]}")
    print("Repair complete; all recorded sample PNGs are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
