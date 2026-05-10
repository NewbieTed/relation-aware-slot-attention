from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.graph_modules import GraphSlotEncoder, mean_pool_hidden
from training.runtime import DEFAULT_FLUX_MODEL_ID, load_graph_encoder, resolve_torch_device
from training.scene_graph import RELATION_VOCAB, build_batched_scene_graphs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace the relation-aware GNN on a single prompt and log every intermediate step."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--graph-encoder-path", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-vector-elements-print", type=int, default=3)
    parser.add_argument(
        "--layout-sample-mode",
        choices=("auto", "prior_mean", "prior_sample", "posterior"),
        default="auto",
        help="For CVAE checkpoints: prior_mean is deterministic, prior_sample is stochastic.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _tensor_to_list(tensor: torch.Tensor, *, decimals: int = 6) -> list[float] | list[list[float]]:
    rounded = tensor.detach().cpu().to(torch.float32)
    if rounded.ndim == 1:
        return [round(float(value), decimals) for value in rounded.tolist()]
    return [
        [round(float(value), decimals) for value in row]
        for row in rounded.tolist()
    ]


def _preview_vector(tensor: torch.Tensor, *, max_elements: int) -> str:
    flat = tensor.detach().cpu().to(torch.float32).flatten()
    values = [f"{float(value):+.4f}" for value in flat[:max_elements]]
    suffix = ", ..." if flat.numel() > max_elements else ""
    return "[" + ", ".join(values) + suffix + "]"


def _first_n_vector(tensor: torch.Tensor, *, n: int = 3) -> str:
    flat = tensor.detach().cpu().to(torch.float32).flatten()
    values = [f"{float(value):+.4f}" for value in flat[:n]]
    suffix = ", ..." if flat.numel() > n else ""
    return "[" + ", ".join(values) + suffix + "]"


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().to(torch.float32).norm().item())


def _cosine_and_angle(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a_flat = a.detach().cpu().to(torch.float32).flatten()
    b_flat = b.detach().cpu().to(torch.float32).flatten()
    cosine = float(torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item())
    cosine = max(-1.0, min(1.0, cosine))
    angle_deg = math.degrees(math.acos(cosine))
    return cosine, angle_deg


def _relation_name_from_index(index: int) -> str:
    inverse = {value: key for key, value in RELATION_VOCAB.items()}
    return inverse[index]


def _label_embeddings(
    *,
    labels: list[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    text_inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]
    pooled = mean_pool_hidden(hidden, text_inputs.attention_mask.to(device))

    details: list[dict[str, Any]] = []
    for label, pooled_embedding in zip(labels, pooled):
        details.append(
            {
                "label": label,
                "clip_embedding": _tensor_to_list(pooled_embedding),
                "clip_embedding_norm": _norm(pooled_embedding),
            }
        )
    return pooled, details


def _build_debug_payload(
    *,
    prompt: str,
    scene_graph: dict[str, Any],
    graph_encoder: GraphSlotEncoder,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: str,
    layout_sample_mode: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    node_labels = [str(node["label"]) for node in scene_graph["nodes"]]
    pooled_embeddings, label_details = _label_embeddings(
        labels=node_labels,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
    )
    pooled_embeddings = pooled_embeddings.to(dtype=graph_encoder.node_proj.weight.dtype)

    slot_targets = torch.zeros((1, len(node_labels), 3), device=device, dtype=pooled_embeddings.dtype)
    log_size_targets = torch.zeros((1, len(node_labels), 3), device=device, dtype=pooled_embeddings.dtype)
    slot_mask = torch.ones((1, len(node_labels)), device=device, dtype=torch.bool)
    batched_graph = build_batched_scene_graphs(
        [scene_graph],
        slot_targets=slot_targets,
        slot_mask=slot_mask,
        log_size_targets=log_size_targets,
    )

    node_states = graph_encoder.node_proj(pooled_embeddings).detach()
    relation_embeddings_table = graph_encoder.relation_embedding.weight.detach()

    relation_embedding_summary: list[dict[str, Any]] = []
    for relation_name, relation_index in RELATION_VOCAB.items():
        relation_vector = relation_embeddings_table[relation_index]
        relation_embedding_summary.append(
            {
                "relation": relation_name,
                "index": relation_index,
                "embedding": _tensor_to_list(relation_vector),
                "norm": _norm(relation_vector),
            }
        )

    active_edge_info: list[dict[str, Any]] = []
    active_relations: list[str] = []
    for (receiver_idx, sender_idx), relation_index in zip(
        batched_graph.edge_index[0].tolist(),
        batched_graph.edge_types[0].tolist(),
    ):
        relation_name = _relation_name_from_index(int(relation_index))
        relation_vector = relation_embeddings_table[int(relation_index)]
        active_edge_info.append(
            {
                "sender_label": node_labels[sender_idx],
                "receiver_label": node_labels[receiver_idx],
                "relation": relation_name,
                "relation_embedding": _tensor_to_list(relation_vector),
                "relation_embedding_norm": _norm(relation_vector),
            }
        )
        if relation_name not in active_relations:
            active_relations.append(relation_name)

    active_relation_angles: list[dict[str, Any]] = []
    for i, relation_a in enumerate(active_relations):
        for relation_b in active_relations[i + 1 :]:
            cosine, angle_deg = _cosine_and_angle(
                relation_embeddings_table[RELATION_VOCAB[relation_a]],
                relation_embeddings_table[RELATION_VOCAB[relation_b]],
            )
            active_relation_angles.append(
                {
                    "relation_a": relation_a,
                    "relation_b": relation_b,
                    "cosine_similarity": cosine,
                    "angle_degrees": angle_deg,
                }
            )

    layer_traces: list[dict[str, Any]] = []
    sample_edges = batched_graph.edge_index[0].to(device)
    sample_edge_types = batched_graph.edge_types[0].to(device)
    sample_states = node_states.clone()

    for layer_index, layer in enumerate(graph_encoder.layers):
        messages: list[dict[str, Any]] = []
        aggregated = torch.zeros_like(sample_states)
        if sample_edges.numel() > 0:
            src = sample_edges[:, 0]
            dst = sample_edges[:, 1]
            relation_vectors = graph_encoder.relation_embedding(sample_edge_types)
            sender_states = sample_states[dst]
            message_inputs = torch.cat([sender_states, relation_vectors], dim=-1)
            raw_messages = layer.message_mlp(message_inputs)
            aggregated.index_add_(0, src, raw_messages)

            for edge_position, (receiver_idx, sender_idx) in enumerate(sample_edges.tolist()):
                relation_index = int(sample_edge_types[edge_position].item())
                relation_name = _relation_name_from_index(relation_index)
                messages.append(
                    {
                        "sender_label": node_labels[sender_idx],
                        "receiver_label": node_labels[receiver_idx],
                        "relation": relation_name,
                        "sender_state": _tensor_to_list(sample_states[sender_idx]),
                        "relation_embedding": _tensor_to_list(relation_vectors[edge_position]),
                        "message": _tensor_to_list(raw_messages[edge_position]),
                        "message_norm": _norm(raw_messages[edge_position]),
                    }
                )

        update_input = torch.cat([sample_states, aggregated], dim=-1)
        residual_update = layer.update(update_input)
        new_states = sample_states + residual_update
        layer_traces.append(
            {
                "layer_index": layer_index,
                "states_before": {
                    node_labels[node_index]: _tensor_to_list(sample_states[node_index])
                    for node_index in range(len(node_labels))
                },
                "messages": messages,
                "aggregated_messages": {
                    node_labels[node_index]: _tensor_to_list(aggregated[node_index])
                    for node_index in range(len(node_labels))
                },
                "residual_updates": {
                    node_labels[node_index]: _tensor_to_list(residual_update[node_index])
                    for node_index in range(len(node_labels))
                },
                "states_after": {
                    node_labels[node_index]: _tensor_to_list(new_states[node_index])
                    for node_index in range(len(node_labels))
                },
            }
        )
        sample_states = new_states

    batched_node_states = sample_states.unsqueeze(0)
    if graph_encoder.layout_mode == "cvae":
        (
            batched_slot_positions,
            batched_position_mu,
            batched_position_logvar,
            batched_log_sizes_3d,
            batched_log_size_mu,
            batched_log_size_logvar,
            prior_mu,
            prior_logvar,
            posterior_mu,
            posterior_logvar,
            sampled_z,
        ) = graph_encoder._cvae_layout_outputs(
            batched_node_states,
            batched_graph,
            layout_sample_mode=layout_sample_mode,
        )
        slot_positions = batched_slot_positions[0]
        position_mu = batched_position_mu[0]
        position_logvar = batched_position_logvar[0]
        log_sizes_3d = batched_log_sizes_3d[0]
        log_size_mu = batched_log_size_mu[0]
        log_size_logvar = batched_log_size_logvar[0]
    else:
        prior_mu = prior_logvar = posterior_mu = posterior_logvar = sampled_z = None
        slot_positions = graph_encoder.position_head(sample_states)
        position_mu = slot_positions
        position_logvar = None
        log_sizes_3d = graph_encoder.log_size_3d_head(sample_states).clamp(min=-4.0, max=1.0)
        log_size_mu = log_sizes_3d
        log_size_logvar = None
    slot_embeddings = graph_encoder.slot_out(sample_states)
    slot_log_sigmas = graph_encoder.log_sigma_head(sample_states).clamp(min=-4.0, max=1.0)

    predicted_relations: list[dict[str, Any]] = []
    for source_idx, target_idx, relation_name in batched_graph.relation_triplets[0]:
        delta = slot_positions[target_idx] - slot_positions[source_idx]
        predicted_relations.append(
            {
                "source_label": node_labels[source_idx],
                "target_label": node_labels[target_idx],
                "relation": relation_name,
                "source_position": _tensor_to_list(slot_positions[source_idx]),
                "target_position": _tensor_to_list(slot_positions[target_idx]),
                "delta_target_minus_source": _tensor_to_list(delta),
            }
        )

    cvae_payload = None
    if graph_encoder.layout_mode == "cvae":
        assert prior_mu is not None and prior_logvar is not None
        assert posterior_mu is not None and posterior_logvar is not None and sampled_z is not None
        assert position_logvar is not None and log_size_logvar is not None
        cvae_payload = {
            "operation": "scene-level CVAE layout head",
            "layout_sample_mode": layout_sample_mode,
            "prior_mu": _tensor_to_list(prior_mu[0]),
            "prior_logvar": _tensor_to_list(prior_logvar[0]),
            "posterior_mu": _tensor_to_list(posterior_mu[0]),
            "posterior_logvar": _tensor_to_list(posterior_logvar[0]),
            "sampled_z": _tensor_to_list(sampled_z[0]),
            "position_mu": {
                node_labels[node_index]: _tensor_to_list(position_mu[node_index])
                for node_index in range(len(node_labels))
            },
            "position_logvar": {
                node_labels[node_index]: _tensor_to_list(position_logvar[node_index])
                for node_index in range(len(node_labels))
            },
            "log_size_3d_mu": {
                node_labels[node_index]: _tensor_to_list(log_size_mu[node_index])
                for node_index in range(len(node_labels))
            },
            "log_size_3d_logvar": {
                node_labels[node_index]: _tensor_to_list(log_size_logvar[node_index])
                for node_index in range(len(node_labels))
            },
        }

    return {
        "prompt": prompt,
        "layout_mode": graph_encoder.layout_mode,
        "layout_sample_mode": layout_sample_mode,
        "seed": seed,
        "scene_graph": scene_graph,
        "message_passing_edges": [
            {
                "receiver_label": node_labels[receiver_idx],
                "sender_label": node_labels[sender_idx],
                "relation": _relation_name_from_index(int(relation_index)),
            }
            for (receiver_idx, sender_idx), relation_index in zip(
                batched_graph.edge_index[0].tolist(),
                batched_graph.edge_types[0].tolist(),
            )
        ],
        "clip_label_embeddings": label_details,
        "projected_node_states": {
            node_labels[node_index]: _tensor_to_list(node_states[node_index])
            for node_index in range(len(node_labels))
        },
        "relation_embeddings": relation_embedding_summary,
        "active_relations": active_edge_info,
        "active_relation_angles": active_relation_angles,
        "message_passing_layers": layer_traces,
        "final_slot_embeddings": {
            node_labels[node_index]: _tensor_to_list(slot_embeddings[node_index])
            for node_index in range(len(node_labels))
        },
        "predicted_slot_positions": {
            node_labels[node_index]: _tensor_to_list(slot_positions[node_index])
            for node_index in range(len(node_labels))
        },
        "predicted_slot_log_sigmas": {
            node_labels[node_index]: _tensor_to_list(slot_log_sigmas[node_index])
            for node_index in range(len(node_labels))
        },
        "predicted_slot_sigmas": {
            node_labels[node_index]: _tensor_to_list(slot_log_sigmas[node_index].exp())
            for node_index in range(len(node_labels))
        },
        "predicted_log_sizes_3d": {
            node_labels[node_index]: _tensor_to_list(log_sizes_3d[node_index])
            for node_index in range(len(node_labels))
        },
        "predicted_sizes_3d": {
            node_labels[node_index]: _tensor_to_list(log_sizes_3d[node_index].exp())
            for node_index in range(len(node_labels))
        },
        "cvae": cvae_payload,
        "predicted_relation_deltas": predicted_relations,
    }


def _render_text_report(payload: dict[str, Any], *, max_vector_elements: int) -> str:
    lines: list[str] = []
    lines.append(f"Prompt: {payload['prompt']}")
    lines.append(f"Layout mode: {payload['layout_mode']}")
    lines.append(f"Layout sample mode: {payload['layout_sample_mode']}")
    lines.append(f"Seed: {payload['seed']}")
    lines.append("")
    lines.append("Scene graph:")
    for node in payload["scene_graph"]["nodes"]:
        lines.append(f"  Node {node['id']}: {node['label']}")
    for edge in payload["scene_graph"]["edges"]:
        lines.append(f"  Edge: {edge['source_id']} -> {edge['target_id']} ({edge['relation']})")
    lines.append("")
    lines.append("Expanded message-passing edges:")
    for edge in payload["message_passing_edges"]:
        lines.append(
            f"  {edge['sender_label']} -> {edge['receiver_label']} ({edge['relation']})"
        )
    lines.append("")

    lines.append("CLIP pooled label embeddings:")
    for item in payload["clip_label_embeddings"]:
        vector = torch.tensor(item["clip_embedding"])
        lines.append(
            f"  {item['label']}: norm={item['clip_embedding_norm']:.4f} "
            f"{_preview_vector(vector, max_elements=max_vector_elements)}"
        )
    lines.append("")

    lines.append("Active relation embeddings:")
    for item in payload["active_relations"]:
        vector = torch.tensor(item["relation_embedding"])
        lines.append(
            f"  {item['sender_label']} -> {item['receiver_label']} ({item['relation']}): "
            f"norm={item['relation_embedding_norm']:.4f} "
            f"{_preview_vector(vector, max_elements=max_vector_elements)}"
        )
    if payload["active_relation_angles"]:
        lines.append("  Angles between active relation embeddings:")
        for item in payload["active_relation_angles"]:
            lines.append(
                f"    {item['relation_a']} vs {item['relation_b']}: "
                f"cos={item['cosine_similarity']:.4f}, angle={item['angle_degrees']:.2f} deg"
            )
    lines.append("")

    lines.append("Projected node states before message passing:")
    for label, values in payload["projected_node_states"].items():
        lines.append(
            f"  {label}: {_preview_vector(torch.tensor(values), max_elements=max_vector_elements)}"
        )
    lines.append("")

    for layer in payload["message_passing_layers"]:
        lines.append(f"Layer {layer['layer_index']}:")
        for message in layer["messages"]:
            sender_state = torch.tensor(message["sender_state"])
            relation_embedding = torch.tensor(message["relation_embedding"])
            message_tensor = torch.tensor(message["message"])
            lines.append(
                f"  Message {message['sender_label']} => {message['receiver_label']} "
                f"via {message['relation']}:"
            )
            lines.append(
                "    message_input = concat(sender_state, relation_embedding)"
            )
            lines.append(
                f"      sender_state[{message['sender_label']}] = {_preview_vector(sender_state, max_elements=max_vector_elements)}"
            )
            lines.append(
                f"      relation_embedding[{message['relation']}] = {_preview_vector(relation_embedding, max_elements=max_vector_elements)}"
            )
            lines.append(
                "      message = MLP(message_input)"
                f" = {_preview_vector(message_tensor, max_elements=max_vector_elements)} (norm={message['message_norm']:.4f})"
            )
        lines.append("  Aggregation:")
        for label, values in layer["aggregated_messages"].items():
            lines.append(
                f"    aggregated[{label}] = "
                f"{_preview_vector(torch.tensor(values), max_elements=max_vector_elements)}"
            )
        lines.append("  Residual updates and new states:")
        for label, update_values in layer["residual_updates"].items():
            before_values = torch.tensor(layer["states_before"][label])
            aggregated_values = torch.tensor(layer["aggregated_messages"][label])
            update_tensor = torch.tensor(update_values)
            after_values = layer["states_after"][label]
            lines.append(
                f"    update_input[{label}] = concat(old_state[{label}], aggregated[{label}])"
            )
            lines.append(
                f"      old_state[{label}] = {_preview_vector(before_values, max_elements=max_vector_elements)}"
            )
            lines.append(
                f"      aggregated[{label}] = {_preview_vector(aggregated_values, max_elements=max_vector_elements)}"
            )
            lines.append(
                f"      residual_update[{label}] = MLP(update_input) = {_preview_vector(update_tensor, max_elements=max_vector_elements)}"
            )
            lines.append(
                f"      new_state[{label}] = old_state + residual_update"
                f" = {_preview_vector(before_values, max_elements=max_vector_elements)} + {_preview_vector(update_tensor, max_elements=max_vector_elements)}"
                f" = {_preview_vector(torch.tensor(after_values), max_elements=max_vector_elements)}"
            )
        lines.append("")

    lines.append("Final slot embeddings:")
    for label, values in payload["final_slot_embeddings"].items():
        lines.append(
            f"  {label}: {_preview_vector(torch.tensor(values), max_elements=max_vector_elements)}"
        )
    lines.append("")

    if payload["cvae"] is not None:
        cvae = payload["cvae"]
        lines.append("CVAE layout head:")
        lines.append("  graph_state = masked_mean(final_node_states)")
        lines.append("  prior_stats = prior_head(graph_state)")
        lines.append(
            f"    prior_mu = {_preview_vector(torch.tensor(cvae['prior_mu']), max_elements=max_vector_elements)}"
        )
        lines.append(
            f"    prior_logvar = {_preview_vector(torch.tensor(cvae['prior_logvar']), max_elements=max_vector_elements)}"
        )
        lines.append("  gt_layout_state = masked_mean(gt_layout_encoder(concat(gt_center, gt_log_size)))")
        lines.append("  posterior_stats = posterior_head(concat(graph_state, gt_layout_state))")
        lines.append(
            f"    posterior_mu = {_preview_vector(torch.tensor(cvae['posterior_mu']), max_elements=max_vector_elements)}"
        )
        lines.append(
            f"    posterior_logvar = {_preview_vector(torch.tensor(cvae['posterior_logvar']), max_elements=max_vector_elements)}"
        )
        lines.append("  z = mu + exp(0.5 * logvar) * epsilon")
        lines.append(
            f"    sampled_z = {_preview_vector(torch.tensor(cvae['sampled_z']), max_elements=max_vector_elements)}"
        )
        lines.append("  decoder_input[obj] = concat(final_node_state[obj], z)")
        for label, values in cvae["position_mu"].items():
            lines.append(
                f"    {label}: "
                f"position_mu={_preview_vector(torch.tensor(values), max_elements=max_vector_elements)}, "
                f"position_logvar={_preview_vector(torch.tensor(cvae['position_logvar'][label]), max_elements=max_vector_elements)}, "
                f"log_size_mu={_preview_vector(torch.tensor(cvae['log_size_3d_mu'][label]), max_elements=max_vector_elements)}, "
                f"log_size_logvar={_preview_vector(torch.tensor(cvae['log_size_3d_logvar'][label]), max_elements=max_vector_elements)}"
            )
        lines.append("")

    lines.append("Predicted (x, y, z):")
    for label, values in payload["predicted_slot_positions"].items():
        tensor_values = torch.tensor(values)
        lines.append(
            f"  {label}: ({tensor_values[0]:+.4f}, {tensor_values[1]:+.4f}, {tensor_values[2]:+.4f})"
        )
    lines.append("")

    lines.append("Predicted 3D log-size and size:")
    for label, values in payload["predicted_log_sizes_3d"].items():
        log_values = torch.tensor(values)
        size_values = torch.tensor(payload["predicted_sizes_3d"][label])
        lines.append(
            f"  {label}: log_size=({log_values[0]:+.4f}, {log_values[1]:+.4f}, {log_values[2]:+.4f}), "
            f"size=({size_values[0]:+.4f}, {size_values[1]:+.4f}, {size_values[2]:+.4f})"
        )
    lines.append("")

    lines.append("Predicted log-sigma and sigma spreads:")
    for label, values in payload["predicted_slot_log_sigmas"].items():
        log_values = torch.tensor(values)
        sigma_values = torch.tensor(payload["predicted_slot_sigmas"][label])
        lines.append(
            f"  {label}: log_sigma=({log_values[0]:+.4f}, {log_values[1]:+.4f}), "
            f"sigma=({sigma_values[0]:+.4f}, {sigma_values[1]:+.4f})"
        )
    lines.append("")

    lines.append("Predicted relation deltas (target - source):")
    for item in payload["predicted_relation_deltas"]:
        delta = torch.tensor(item["delta_target_minus_source"])
        lines.append(
            f"  {item['source_label']} -> {item['target_label']} ({item['relation']}): "
            f"({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f})"
        )
    lines.append("")
    lines.append("Full vectors are available in gnn_trace.json.")
    return "\n".join(lines)


def main() -> int:
    args = make_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    text_encoder.eval()

    graph_encoder = load_graph_encoder(
        path=args.graph_encoder_path,
        text_hidden_dim=text_encoder.config.hidden_size,
        device=device,
        dtype=text_encoder.dtype,
    )
    graph_encoder.eval()

    scene_graph = parse_prompt_to_scene_graph(args.prompt)
    payload = _build_debug_payload(
        prompt=args.prompt,
        scene_graph=scene_graph,
        graph_encoder=graph_encoder,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        layout_sample_mode=args.layout_sample_mode,
        seed=args.seed,
    )

    json_path = args.output_dir / "gnn_trace.json"
    report_path = args.output_dir / "gnn_trace.txt"
    json_path.write_text(json.dumps(payload, indent=2))
    report = _render_text_report(payload, max_vector_elements=args.max_vector_elements_print)
    report_path.write_text(report)
    print(report)
    print("")
    print(f"Saved full JSON trace to {json_path}")
    print(f"Saved text report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
