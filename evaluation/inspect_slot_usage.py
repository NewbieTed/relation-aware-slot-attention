from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from evaluation.generate import (
    MODEL_REGISTRY,
    build_pipeline,
    build_relation_aware_conditioning,
    load_graph_encoder,
    load_prompt_lines,
    resolve_relation_aware_artifacts,
    resolve_torch_device,
)
from training.relation_attention import install_relation_aware_processors


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect how much cross-attention mass goes to text tokens versus appended slot tokens during inference."
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), default="sd15")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to inspect.")
    parser.add_argument("--prompts-file", type=Path, default=None, help="Optional prompt file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--unet-path", type=Path, default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--relation-aware-dir", type=Path, default=None)
    parser.add_argument("--graph-encoder-path", type=Path, default=None)
    parser.add_argument("--relation-attention-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit-prompts", type=int, default=None)
    return parser


def reset_processor_usage(processors: dict[str, torch.nn.Module]) -> None:
    for processor in processors.values():
        reset_fn = getattr(processor, "reset_usage_stats", None)
        if reset_fn is not None:
            reset_fn()
        clear_fn = getattr(processor, "clear_attention_cache", None)
        if clear_fn is not None:
            clear_fn()


def aggregate_usage(processors: dict[str, torch.nn.Module]) -> dict[str, Any]:
    by_layer: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    per_processor: dict[str, dict[str, float | int]] = {}

    for name, processor in processors.items():
        usage_fn = getattr(processor, "usage_stats", None)
        if usage_fn is None:
            continue
        stats = usage_fn()
        if int(stats["call_count"]) == 0:
            continue
        per_processor[name] = stats
        if name.startswith("down_blocks"):
            by_layer["down"].append(stats)
        elif name.startswith("mid_block"):
            by_layer["mid"].append(stats)
        elif name.startswith("up_blocks"):
            by_layer["up"].append(stats)
        else:
            by_layer["other"].append(stats)

    def reduce_stats(items: list[dict[str, float | int]]) -> dict[str, float | int]:
        if not items:
            return {
                "text_mass_mean": 0.0,
                "slot_mass_mean": 0.0,
                "query_count": 0,
                "call_count": 0,
                "slot_count_mean": 0.0,
            }
        total_queries = sum(int(item["query_count"]) for item in items)
        total_calls = sum(int(item["call_count"]) for item in items)
        if total_queries == 0 or total_calls == 0:
            return {
                "text_mass_mean": 0.0,
                "slot_mass_mean": 0.0,
                "query_count": total_queries,
                "call_count": total_calls,
                "slot_count_mean": 0.0,
            }
        return {
            "text_mass_mean": sum(float(item["text_mass_mean"]) * int(item["query_count"]) for item in items) / total_queries,
            "slot_mass_mean": sum(float(item["slot_mass_mean"]) * int(item["query_count"]) for item in items) / total_queries,
            "query_count": total_queries,
            "call_count": total_calls,
            "slot_count_mean": sum(float(item["slot_count_mean"]) * int(item["call_count"]) for item in items) / total_calls,
        }

    layer_summary = {layer: reduce_stats(items) for layer, items in by_layer.items()}
    overall = reduce_stats(list(per_processor.values()))
    return {
        "overall": overall,
        "by_layer_group": layer_summary,
        "per_processor": per_processor,
    }


def prompts_from_args(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt is not None:
        prompts.append(args.prompt)
    if args.prompts_file is not None:
        prompts.extend(load_prompt_lines(args.prompts_file))
    if not prompts:
        raise ValueError("Provide either --prompt or --prompts-file")
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    return prompts


def main() -> int:
    args = make_parser().parse_args()
    prompts = prompts_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_REGISTRY[args.model]
    device = resolve_torch_device(args.device)

    unet_path, lora_path, graph_encoder_path, relation_attention_path = resolve_relation_aware_artifacts(
        relation_aware_dir=args.relation_aware_dir,
        unet_path=args.unet_path,
        lora_path=args.lora_path,
        graph_encoder_path=args.graph_encoder_path,
        relation_attention_path=args.relation_attention_path,
    )
    relation_aware_enabled = graph_encoder_path is not None

    pipeline = build_pipeline(
        model_name,
        device,
        unet_path=unet_path,
        lora_path=lora_path,
        relation_attention_path=relation_attention_path if relation_aware_enabled else None,
        disable_progress_bar=True,
    )
    relation_attention_processors = install_relation_aware_processors(pipeline.unet)
    if relation_attention_path is not None and relation_attention_path.exists():
        processor_state = torch.load(relation_attention_path, map_location="cpu")
        for name, module in relation_attention_processors.items():
            state = processor_state.get(name)
            if state is not None:
                module.load_state_dict(state)

    graph_encoder = None
    if relation_aware_enabled:
        if not graph_encoder_path.exists():
            raise FileNotFoundError(f"Missing graph encoder checkpoint: {graph_encoder_path}")
        graph_encoder = load_graph_encoder(
            path=graph_encoder_path,
            text_hidden_dim=pipeline.text_encoder.config.hidden_size,
            device=device,
            dtype=pipeline.text_encoder.dtype,
        )

    results: list[dict[str, Any]] = []
    do_cfg = args.guidance_scale > 1.0
    for prompt_index, prompt in enumerate(prompts):
        reset_processor_usage(relation_attention_processors)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + prompt_index)

        if relation_aware_enabled:
            assert graph_encoder is not None
            prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=None,
            )
            text_token_count = int(prompt_embeds.shape[1])
            slot_embeddings, slot_positions, slot_count = build_relation_aware_conditioning(
                prompt=prompt,
                pipeline=pipeline,
                graph_encoder=graph_encoder,
                device=device,
            )
            prompt_embeds = torch.cat([prompt_embeds, slot_embeddings.to(prompt_embeds.dtype)], dim=1)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, torch.zeros_like(slot_embeddings, dtype=negative_prompt_embeds.dtype)],
                    dim=1,
                )
            cross_attention_kwargs = {
                "slot_positions": slot_positions,
                "slot_mask": torch.ones(1, slot_positions.shape[1], dtype=torch.bool, device=slot_positions.device),
                "text_token_count": text_token_count,
            }
            pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.image_size,
                width=args.image_size,
                generator=generator,
                output_type="pil",
            )
        else:
            pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.image_size,
                width=args.image_size,
                generator=generator,
                output_type="pil",
            )
            text_token_count = None
            slot_count = 0

        usage = aggregate_usage(relation_attention_processors)
        results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "text_token_count": text_token_count,
                "slot_count": slot_count,
                "usage": usage,
            }
        )

    output_json = args.output_dir / "slot_usage_summary.json"
    output_json.write_text(json.dumps(results, indent=2))

    lines = ["# Slot Usage Summary", ""]
    for item in results:
        overall = item["usage"]["overall"]
        lines.extend(
            [
                f"## Prompt {item['prompt_index']}",
                "",
                f"`{item['prompt']}`",
                "",
                f"- text tokens: {item['text_token_count']}",
                f"- slot count: {item['slot_count']}",
                f"- overall text attention mass: {overall['text_mass_mean']:.4f}",
                f"- overall slot attention mass: {overall['slot_mass_mean']:.4f}",
                f"- processor calls observed: {overall['call_count']}",
                "",
            ]
        )
        for layer_name, stats in sorted(item["usage"]["by_layer_group"].items()):
            lines.append(
                f"- {layer_name}: text={stats['text_mass_mean']:.4f}, slot={stats['slot_mass_mean']:.4f}, calls={stats['call_count']}"
            )
        lines.append("")
    (args.output_dir / "slot_usage_summary.md").write_text("\n".join(lines))
    print(f"Wrote slot usage summary to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
