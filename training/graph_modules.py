from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene_graph import BatchedSceneGraphs, RELATION_VOCAB

INVERSE_RELATION_PAIRS = (
    ("left_of", "right_of"),
    ("above", "below"),
    ("in_front_of", "behind"),
)


def mean_pool_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


class GraphMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + relation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relations: torch.Tensor,
        relation_embedder: nn.Embedding,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_states

        num_nodes = node_states.shape[0]
        aggregated = torch.zeros_like(node_states)
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        relation_embeddings = relation_embedder(edge_relations)
        messages = self.message_mlp(
            torch.cat([node_states[dst], relation_embeddings], dim=-1)
        )
        aggregated.index_add_(0, src, messages)
        return node_states + self.update(torch.cat([node_states, aggregated], dim=-1))


@dataclass
class GraphConditioningOutput:
    slot_embeddings: torch.Tensor
    slot_positions: torch.Tensor
    slot_log_sigmas: torch.Tensor
    slot_mask: torch.Tensor
    relation_logits: list[torch.Tensor]


class GraphSlotEncoder(nn.Module):
    def __init__(
        self,
        text_hidden_dim: int,
        slot_dim: int,
        relation_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(text_hidden_dim, slot_dim)
        self.relation_embedding = nn.Embedding(len(RELATION_VOCAB), relation_dim)
        self.layers = nn.ModuleList(
            GraphMessagePassingLayer(slot_dim, relation_dim) for _ in range(num_layers)
        )
        self.slot_out = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, text_hidden_dim),
        )
        self.position_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.SiLU(),
            nn.Linear(slot_dim, 3),
            nn.Tanh(),
        )
        self.log_sigma_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.SiLU(),
            nn.Linear(slot_dim, 2),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.SiLU(),
            nn.Linear(slot_dim, 3),
        )
        self._initialize_relation_embeddings()

    def _initialize_relation_embeddings(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)
            base_directions = {
                "horizontal": F.normalize(torch.randn(self.relation_embedding.embedding_dim), dim=0),
                "vertical": F.normalize(torch.randn(self.relation_embedding.embedding_dim), dim=0),
                "depth": F.normalize(torch.randn(self.relation_embedding.embedding_dim), dim=0),
            }
            self.relation_embedding.weight[RELATION_VOCAB["left_of"]].copy_(-base_directions["horizontal"])
            self.relation_embedding.weight[RELATION_VOCAB["right_of"]].copy_(base_directions["horizontal"])
            self.relation_embedding.weight[RELATION_VOCAB["above"]].copy_(-base_directions["vertical"])
            self.relation_embedding.weight[RELATION_VOCAB["below"]].copy_(base_directions["vertical"])
            self.relation_embedding.weight[RELATION_VOCAB["in_front_of"]].copy_(base_directions["depth"])
            self.relation_embedding.weight[RELATION_VOCAB["behind"]].copy_(-base_directions["depth"])

    def forward(
        self,
        pooled_label_embeddings: torch.Tensor,
        scene_graph_batch: BatchedSceneGraphs,
    ) -> GraphConditioningOutput:
        batch_size, max_nodes, _ = pooled_label_embeddings.shape
        node_states = self.node_proj(pooled_label_embeddings)
        relation_logits: list[torch.Tensor] = []

        for batch_index in range(batch_size):
            valid_node_count = int(scene_graph_batch.position_mask[batch_index].sum().item())
            if valid_node_count == 0:
                relation_logits.append(torch.zeros((0, 3), device=node_states.device))
                continue

            sample_states = node_states[batch_index, :valid_node_count]
            sample_edges = scene_graph_batch.edge_index[batch_index].to(node_states.device)
            sample_edge_types = scene_graph_batch.edge_types[batch_index].to(node_states.device)
            for layer in self.layers:
                sample_states = layer(
                    sample_states,
                    sample_edges,
                    sample_edge_types,
                    self.relation_embedding,
                )
            node_states[batch_index, :valid_node_count] = sample_states

            logits_per_edge: list[torch.Tensor] = []
            for src, dst, _ in scene_graph_batch.relation_triplets[batch_index]:
                logits_per_edge.append(
                    self.relation_head(
                        torch.cat([sample_states[src], sample_states[dst]], dim=-1)
                    )
                )
            if logits_per_edge:
                relation_logits.append(torch.stack(logits_per_edge, dim=0))
            else:
                relation_logits.append(torch.zeros((0, 3), device=node_states.device))

        slot_embeddings = self.slot_out(node_states)
        slot_positions = self.position_head(node_states)
        slot_log_sigmas = self.log_sigma_head(node_states).clamp(min=-4.0, max=1.0)
        return GraphConditioningOutput(
            slot_embeddings=slot_embeddings,
            slot_positions=slot_positions,
            slot_log_sigmas=slot_log_sigmas,
            slot_mask=scene_graph_batch.position_mask.to(node_states.device),
            relation_logits=relation_logits,
        )


def pooled_label_embeddings(
    *,
    tokenizer: Any,
    text_encoder: Any,
    scene_graph_batch: BatchedSceneGraphs,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = len(scene_graph_batch.node_labels)
    max_nodes = scene_graph_batch.position_targets.shape[1]
    pooled = torch.zeros(
        batch_size,
        max_nodes,
        text_encoder.config.hidden_size,
        device=device,
        dtype=dtype,
    )

    for batch_index, labels in enumerate(scene_graph_batch.node_labels):
        if not labels:
            continue
        text_inputs = tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoded = text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )[0]
        pooled[batch_index, : len(labels)] = mean_pool_hidden(
            encoded,
            text_inputs.attention_mask.to(device),
        ).to(dtype=dtype)
    return pooled


def build_slot_conditioning(
    *,
    tokenizer: Any,
    text_encoder: Any,
    scene_graph_batch: BatchedSceneGraphs,
    graph_encoder: GraphSlotEncoder,
    device: str,
) -> GraphConditioningOutput:
    graph_dtype = graph_encoder.node_proj.weight.dtype
    pooled = pooled_label_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=scene_graph_batch,
        device=device,
        dtype=graph_dtype,
    )
    return graph_encoder(pooled, scene_graph_batch)


def embedding_alignment_loss(
    slot_embeddings: torch.Tensor,
    pooled_label_embeddings: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    if not slot_mask.any():
        return slot_embeddings.new_tensor(0.0)
    return 1.0 - F.cosine_similarity(
        slot_embeddings[slot_mask],
        pooled_label_embeddings[slot_mask].to(slot_embeddings.dtype),
        dim=-1,
    ).mean()


def log_sigma_loss(
    slot_log_sigmas: torch.Tensor,
    log_sigma_targets: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    if not slot_mask.any():
        return slot_log_sigmas.new_tensor(0.0)
    return F.smooth_l1_loss(
        slot_log_sigmas[slot_mask],
        log_sigma_targets[slot_mask].to(slot_log_sigmas.dtype),
    )


def relation_loss(
    relation_logits: list[torch.Tensor],
    slot_positions: torch.Tensor,
    scene_graph_batch: BatchedSceneGraphs,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for batch_index, triplets in enumerate(scene_graph_batch.relation_triplets):
        if not triplets:
            continue
        sample_positions = slot_positions[batch_index]
        sample_losses: list[torch.Tensor] = []
        for edge_index, (src, dst, relation) in enumerate(triplets):
            delta = sample_positions[dst] - sample_positions[src]
            if relation == "left_of":
                sample_losses.append(F.relu(0.1 - delta[0]))
            elif relation == "right_of":
                sample_losses.append(F.relu(0.1 + delta[0]))
            elif relation == "next_to":
                horizontal_proximity = F.relu(delta[0].abs() - 0.35)
                vertical_alignment = F.relu(delta[1].abs() - 0.2)
                depth_alignment = F.relu(delta[2].abs() - 0.2)
                sample_losses.append(
                    (horizontal_proximity + vertical_alignment + depth_alignment) / 3.0
                )
            elif relation == "above":
                sample_losses.append(F.relu(0.1 - delta[1]))
            elif relation == "below":
                sample_losses.append(F.relu(0.1 + delta[1]))
            elif relation in {"in_front_of", "hidden_by"}:
                sample_losses.append(F.relu(0.05 - delta[2]))
            elif relation == "behind":
                sample_losses.append(F.relu(0.05 + delta[2]))
            elif relation == "on":
                sample_losses.append(F.relu(delta[1].abs() - 0.2))
        if sample_losses:
            losses.append(torch.stack(sample_losses).mean())
    if not losses:
        return slot_positions.new_tensor(0.0)
    return torch.stack(losses).mean()


def inverse_relation_regularizer(graph_encoder: GraphSlotEncoder) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for relation_a, relation_b in INVERSE_RELATION_PAIRS:
        embedding_a = graph_encoder.relation_embedding.weight[RELATION_VOCAB[relation_a]]
        embedding_b = graph_encoder.relation_embedding.weight[RELATION_VOCAB[relation_b]]
        cosine = F.cosine_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0), dim=-1)
        penalties.append(1.0 + cosine.squeeze(0))
    if not penalties:
        return graph_encoder.relation_embedding.weight.new_tensor(0.0)
    return torch.stack(penalties).mean()
