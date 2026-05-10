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
    slot_position_mu: torch.Tensor
    slot_position_logvar: torch.Tensor | None
    slot_log_sigmas: torch.Tensor
    slot_log_sizes_3d: torch.Tensor
    slot_log_size_3d_mu: torch.Tensor
    slot_log_size_3d_logvar: torch.Tensor | None
    slot_mask: torch.Tensor
    relation_logits: list[torch.Tensor]
    prior_mu: torch.Tensor | None = None
    prior_logvar: torch.Tensor | None = None
    posterior_mu: torch.Tensor | None = None
    posterior_logvar: torch.Tensor | None = None
    sampled_z: torch.Tensor | None = None


class GraphSlotEncoder(nn.Module):
    def __init__(
        self,
        text_hidden_dim: int,
        slot_dim: int,
        relation_dim: int = 128,
        num_layers: int = 2,
        layout_mode: str = "deterministic",
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        if layout_mode not in {"deterministic", "cvae"}:
            raise ValueError(f"Unsupported layout_mode: {layout_mode}")
        self.layout_mode = layout_mode
        self.latent_dim = latent_dim
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
        self.log_size_3d_head = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim),
            nn.SiLU(),
            nn.Linear(slot_dim, 3),
        )
        if self.layout_mode == "cvae":
            self.position_mu_head = nn.Sequential(
                nn.LayerNorm(slot_dim + latent_dim),
                nn.Linear(slot_dim + latent_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 3),
                nn.Tanh(),
            )
            self.position_logvar_head = nn.Sequential(
                nn.LayerNorm(slot_dim + latent_dim),
                nn.Linear(slot_dim + latent_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 3),
            )
            self.log_size_3d_mu_head = nn.Sequential(
                nn.LayerNorm(slot_dim + latent_dim),
                nn.Linear(slot_dim + latent_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 3),
            )
            self.log_size_3d_logvar_head = nn.Sequential(
                nn.LayerNorm(slot_dim + latent_dim),
                nn.Linear(slot_dim + latent_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 3),
            )
            self.gt_layout_encoder = nn.Sequential(
                nn.Linear(6, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.prior_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
            )
            self.posterior_head = nn.Sequential(
                nn.LayerNorm(slot_dim * 2),
                nn.Linear(slot_dim * 2, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
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
        *,
        layout_sample_mode: str = "auto",
    ) -> GraphConditioningOutput:
        if layout_sample_mode not in {"auto", "posterior", "prior_sample", "prior_mean"}:
            raise ValueError(f"Unsupported layout_sample_mode: {layout_sample_mode}")
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
        slot_log_sigmas = self.log_sigma_head(node_states).clamp(min=-4.0, max=1.0)
        if self.layout_mode == "cvae":
            (
                slot_positions,
                position_mu,
                position_logvar,
                slot_log_sizes_3d,
                log_size_3d_mu,
                log_size_3d_logvar,
                prior_mu,
                prior_logvar,
                posterior_mu,
                posterior_logvar,
                sampled_z,
            ) = self._cvae_layout_outputs(
                node_states,
                scene_graph_batch,
                layout_sample_mode=layout_sample_mode,
            )
        else:
            slot_positions = self.position_head(node_states)
            slot_log_sizes_3d = self.log_size_3d_head(node_states).clamp(min=-4.0, max=1.0)
            position_mu = slot_positions
            position_logvar = None
            log_size_3d_mu = slot_log_sizes_3d
            log_size_3d_logvar = None
            prior_mu = None
            prior_logvar = None
            posterior_mu = None
            posterior_logvar = None
            sampled_z = None
        return GraphConditioningOutput(
            slot_embeddings=slot_embeddings,
            slot_positions=slot_positions,
            slot_position_mu=position_mu,
            slot_position_logvar=position_logvar,
            slot_log_sigmas=slot_log_sigmas,
            slot_log_sizes_3d=slot_log_sizes_3d,
            slot_log_size_3d_mu=log_size_3d_mu,
            slot_log_size_3d_logvar=log_size_3d_logvar,
            slot_mask=scene_graph_batch.position_mask.to(node_states.device),
            relation_logits=relation_logits,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            sampled_z=sampled_z,
        )

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(values.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (values * mask_f).sum(dim=1) / denom

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _cvae_layout_outputs(
        self,
        node_states: torch.Tensor,
        scene_graph_batch: BatchedSceneGraphs,
        *,
        layout_sample_mode: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        slot_mask = scene_graph_batch.position_mask.to(node_states.device)
        graph_state = self._masked_mean(node_states, slot_mask)
        prior_stats = self.prior_head(graph_state)
        prior_mu, prior_logvar = prior_stats.chunk(2, dim=-1)
        prior_logvar = prior_logvar.clamp(min=-8.0, max=4.0)

        gt_layout = torch.cat(
            [
                scene_graph_batch.position_targets.to(node_states.device, dtype=node_states.dtype),
                scene_graph_batch.log_size_targets.to(node_states.device, dtype=node_states.dtype),
            ],
            dim=-1,
        )
        gt_layout_state = self._masked_mean(self.gt_layout_encoder(gt_layout), slot_mask)
        posterior_stats = self.posterior_head(torch.cat([graph_state, gt_layout_state], dim=-1))
        posterior_mu, posterior_logvar = posterior_stats.chunk(2, dim=-1)
        posterior_logvar = posterior_logvar.clamp(min=-8.0, max=4.0)

        if layout_sample_mode == "posterior" or (layout_sample_mode == "auto" and self.training):
            z = self._reparameterize(posterior_mu, posterior_logvar)
        elif layout_sample_mode == "prior_sample":
            z = self._reparameterize(prior_mu, prior_logvar)
        else:
            z = prior_mu

        z_per_node = z.unsqueeze(1).expand(-1, node_states.shape[1], -1)
        decoder_input = torch.cat([node_states, z_per_node], dim=-1)
        position_mu = self.position_mu_head(decoder_input)
        position_logvar = self.position_logvar_head(decoder_input).clamp(min=-8.0, max=4.0)
        log_size_3d_mu = self.log_size_3d_mu_head(decoder_input).clamp(min=-4.0, max=1.0)
        log_size_3d_logvar = self.log_size_3d_logvar_head(decoder_input).clamp(min=-8.0, max=4.0)

        if self.training and layout_sample_mode in {"auto", "posterior"}:
            slot_positions = position_mu + torch.exp(0.5 * position_logvar) * torch.randn_like(position_mu)
            slot_positions = slot_positions.clamp(min=-1.0, max=1.0)
            slot_log_sizes_3d = (
                log_size_3d_mu + torch.exp(0.5 * log_size_3d_logvar) * torch.randn_like(log_size_3d_mu)
            ).clamp(min=-4.0, max=1.0)
        else:
            slot_positions = position_mu
            slot_log_sizes_3d = log_size_3d_mu

        return (
            slot_positions,
            position_mu,
            position_logvar,
            slot_log_sizes_3d,
            log_size_3d_mu,
            log_size_3d_logvar,
            prior_mu,
            prior_logvar,
            posterior_mu,
            posterior_logvar,
            z,
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
    layout_sample_mode: str = "auto",
) -> GraphConditioningOutput:
    graph_dtype = graph_encoder.node_proj.weight.dtype
    pooled = pooled_label_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=scene_graph_batch,
        device=device,
        dtype=graph_dtype,
    )
    return graph_encoder(pooled, scene_graph_batch, layout_sample_mode=layout_sample_mode)


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


def log_size_3d_loss(
    slot_log_sizes_3d: torch.Tensor,
    log_size_targets: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    if not slot_mask.any():
        return slot_log_sizes_3d.new_tensor(0.0)
    return F.smooth_l1_loss(
        slot_log_sizes_3d[slot_mask],
        log_size_targets[slot_mask].to(slot_log_sizes_3d.dtype),
    )


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor | None,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log likelihood for masked layout regression."""

    if logvar is None:
        if not mask.any():
            return mean.new_tensor(0.0)
        return F.smooth_l1_loss(mean[mask], target[mask].to(mean.dtype))
    if not mask.any():
        return mean.new_tensor(0.0)
    target = target.to(mean.dtype)
    error = target[mask] - mean[mask]
    selected_logvar = logvar[mask]
    return 0.5 * (selected_logvar + error.pow(2) * torch.exp(-selected_logvar)).mean()


def cvae_kl_loss(output: GraphConditioningOutput) -> torch.Tensor:
    """KL(q(z | graph, layout) || p(z | graph)) for the scene-level latent."""

    if (
        output.prior_mu is None
        or output.prior_logvar is None
        or output.posterior_mu is None
        or output.posterior_logvar is None
    ):
        return output.slot_positions.new_tensor(0.0)
    prior_var = torch.exp(output.prior_logvar)
    posterior_var = torch.exp(output.posterior_logvar)
    delta = output.posterior_mu - output.prior_mu
    kl = 0.5 * (
        output.prior_logvar
        - output.posterior_logvar
        + (posterior_var + delta.pow(2)) / prior_var.clamp_min(1e-8)
        - 1.0
    )
    return kl.sum(dim=-1).mean()


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
            elif relation == "in_front_of":
                sample_losses.append(F.relu(0.05 + delta[2]))
            elif relation == "hidden_by":
                sample_losses.append(F.relu(0.05 - delta[2]))
            elif relation == "behind":
                sample_losses.append(F.relu(0.05 - delta[2]))
            elif relation == "on":
                sample_losses.append(F.relu(0.1 - delta[1]))
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
