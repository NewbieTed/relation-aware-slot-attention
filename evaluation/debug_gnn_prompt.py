from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import CLIPTextModel, CLIPTokenizer

from evaluation.generate import MODEL_REGISTRY, load_graph_encoder, resolve_torch_device
from evaluation.prompt_parser import parse_prompt_to_scene_graph
from training.graph_modules import GraphSlotEncoder, mean_pool_hidden
from training.scene_graph import RELATION_VOCAB, build_batched_scene_graphs


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace the relation-aware GNN on a single prompt and log every intermediate step."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), default="sd15")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--graph-encoder-path", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-vector-elements-print", type=int, default=16)
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
) -> dict[str, Any]:
    node_labels = [str(node["label"]) for node in scene_graph["nodes"]]
    pooled_embeddings, label_details = _label_embeddings(
        labels=node_labels,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
    )
    pooled_embeddings = pooled_embeddings.to(dtype=graph_encoder.node_proj.weight.dtype)

    slot_targets = torch.zeros((1, len(node_labels), 3), device=device, dtype=pooled_embeddings.dtype)
    slot_mask = torch.ones((1, len(node_labels)), device=device, dtype=torch.bool)
    batched_graph = build_batched_scene_graphs([scene_graph], slot_targets=slot_targets, slot_mask=slot_mask)

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
    for edge in scene_graph["edges"]:
        source_index = next(i for i, node in enumerate(scene_graph["nodes"]) if node["id"] == edge["source_id"])
        target_index = next(i for i, node in enumerate(scene_graph["nodes"]) if node["id"] == edge["target_id"])
        relation_index = RELATION_VOCAB[str(edge["relation"])]
        relation_vector = relation_embeddings_table[relation_index]
        active_edge_info.append(
            {
                "source_label": scene_graph["nodes"][source_index]["label"],
                "target_label": scene_graph["nodes"][target_index]["label"],
                "relation": edge["relation"],
                "relation_embedding": _tensor_to_list(relation_vector),
                "relation_embedding_norm": _norm(relation_vector),
            }
        )

    active_relation_angles: list[dict[str, Any]] = []
    active_relations = list({str(edge["relation"]) for edge in scene_graph["edges"]})
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

    slot_embeddings = graph_encoder.slot_out(sample_states)
    slot_positions = graph_encoder.position_head(sample_states)

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

    return {
        "prompt": prompt,
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
        "predicted_relation_deltas": predicted_relations,
    }


def _render_text_report(payload: dict[str, Any], *, max_vector_elements: int) -> str:
    lines: list[str] = []
    lines.append(f"Prompt: {payload['prompt']}")
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
            f"  {item['source_label']} -> {item['target_label']} ({item['relation']}): "
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
            lines.append(
                f"  Message {message['sender_label']} => {message['receiver_label']} "
                f"via {message['relation']}: norm={message['message_norm']:.4f} "
                f"{_preview_vector(torch.tensor(message['message']), max_elements=max_vector_elements)}"
            )
        lines.append("  Aggregation:")
        for label, values in layer["aggregated_messages"].items():
            lines.append(
                f"    aggregated[{label}] = "
                f"{_preview_vector(torch.tensor(values), max_elements=max_vector_elements)}"
            )
        lines.append("  Residual updates and new states:")
        for label, update_values in layer["residual_updates"].items():
            after_values = layer["states_after"][label]
            lines.append(
                f"    update[{label}] = "
                f"{_preview_vector(torch.tensor(update_values), max_elements=max_vector_elements)}"
            )
            lines.append(
                f"    new_state[{label}] = "
                f"{_preview_vector(torch.tensor(after_values), max_elements=max_vector_elements)}"
            )
        lines.append("")

    lines.append("Final slot embeddings:")
    for label, values in payload["final_slot_embeddings"].items():
        lines.append(
            f"  {label}: {_preview_vector(torch.tensor(values), max_elements=max_vector_elements)}"
        )
    lines.append("")

    lines.append("Predicted (x, y, z):")
    for label, values in payload["predicted_slot_positions"].items():
        tensor_values = torch.tensor(values)
        lines.append(
            f"  {label}: ({tensor_values[0]:+.4f}, {tensor_values[1]:+.4f}, {tensor_values[2]:+.4f})"
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
    model_id = args.model_id or MODEL_REGISTRY[args.model]
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
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
