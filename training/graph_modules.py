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


class TripleGraphConvLayer(nn.Module):
    """3D_SLN-style triple update over subject, relation, and object states."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or node_dim * 2
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.triple_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim * 2 + edge_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return node_states, edge_states

        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        triple_input = torch.cat([node_states[src], edge_states, node_states[dst]], dim=-1)
        triple_output = self.triple_mlp(triple_input)
        src_messages = triple_output[:, : self.node_dim]
        edge_updates = triple_output[:, self.node_dim : self.node_dim + self.edge_dim]
        dst_messages = triple_output[:, self.node_dim + self.edge_dim :]

        aggregated = torch.zeros_like(node_states)
        counts = torch.zeros(node_states.shape[0], 1, device=node_states.device, dtype=node_states.dtype)
        aggregated.index_add_(0, src, src_messages)
        aggregated.index_add_(0, dst, dst_messages)
        ones = torch.ones(edge_index.shape[0], 1, device=node_states.device, dtype=node_states.dtype)
        counts.index_add_(0, src, ones)
        counts.index_add_(0, dst, ones)
        aggregated = aggregated / counts.clamp_min(1.0)

        new_node_states = node_states + self.node_update(torch.cat([node_states, aggregated], dim=-1))
        new_edge_states = edge_states + edge_updates
        return new_node_states, new_edge_states


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
    slot_boxes_3d: torch.Tensor | None
    slot_mask: torch.Tensor
    relation_logits: list[torch.Tensor]
    prior_mu: torch.Tensor | None = None
    prior_logvar: torch.Tensor | None = None
    posterior_mu: torch.Tensor | None = None
    posterior_logvar: torch.Tensor | None = None
    sampled_z: torch.Tensor | None = None
    object_prior_mu: torch.Tensor | None = None
    object_prior_logvar: torch.Tensor | None = None
    object_posterior_mu: torch.Tensor | None = None
    object_posterior_logvar: torch.Tensor | None = None
    sampled_object_z: torch.Tensor | None = None


class GraphSlotEncoder(nn.Module):
    def __init__(
        self,
        text_hidden_dim: int,
        slot_dim: int,
        relation_dim: int = 128,
        num_layers: int = 2,
        layout_mode: str = "deterministic",
        latent_dim: int = 64,
        decoder_node_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layout_mode not in {"deterministic", "cvae", "triple_cvae"}:
            raise ValueError(f"Unsupported layout_mode: {layout_mode}")
        self.layout_mode = layout_mode
        self.latent_dim = latent_dim
        self.decoder_node_dropout = float(decoder_node_dropout)
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
            self.graph_readout_score = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim // 2),
                nn.SiLU(),
                nn.Linear(slot_dim // 2, 1),
            )
            self.layout_readout_score = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim // 2),
                nn.SiLU(),
                nn.Linear(slot_dim // 2, 1),
            )
            self.prior_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
            )
        if self.layout_mode == "triple_cvae":
            self.edge_proj = nn.Linear(relation_dim, slot_dim)
            self.posterior_node_init = nn.Sequential(
                nn.LayerNorm(slot_dim * 2),
                nn.Linear(slot_dim * 2, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.gt_layout_encoder = nn.Sequential(
                nn.Linear(6, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.triple_prior_layers = nn.ModuleList(
                TripleGraphConvLayer(slot_dim, slot_dim) for _ in range(num_layers)
            )
            self.triple_posterior_layers = nn.ModuleList(
                TripleGraphConvLayer(slot_dim, slot_dim) for _ in range(num_layers)
            )
            self.triple_decoder_layers = nn.ModuleList(
                TripleGraphConvLayer(slot_dim, slot_dim) for _ in range(num_layers)
            )
            self.triple_node_readout_score = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim // 2),
                nn.SiLU(),
                nn.Linear(slot_dim // 2, 1),
            )
            self.triple_edge_readout_score = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim // 2),
                nn.SiLU(),
                nn.Linear(slot_dim // 2, 1),
            )
            self.triple_graph_fuse = nn.Sequential(
                nn.LayerNorm(slot_dim * 2),
                nn.Linear(slot_dim * 2, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.triple_prior_scene_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
            )
            self.triple_posterior_scene_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
            )
            self.triple_prior_object_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
            )
            self.triple_posterior_object_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, latent_dim * 2),
            )
            self.triple_decoder_node_in = nn.Sequential(
                nn.LayerNorm(slot_dim + latent_dim * 2),
                nn.Linear(slot_dim + latent_dim * 2, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.triple_decoder_film = nn.Sequential(
                nn.LayerNorm(latent_dim * 2),
                nn.Linear(latent_dim * 2, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, slot_dim * 2),
            )
            self.triple_position_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 3),
                nn.Tanh(),
            )
            self.triple_log_size_3d_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 3),
            )
            self.triple_box_3d_head = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, slot_dim),
                nn.SiLU(),
                nn.Linear(slot_dim, 6),
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
        layout_z_scale: float = 1.0,
    ) -> GraphConditioningOutput:
        if layout_sample_mode not in {"auto", "posterior", "prior_sample", "prior_mean"}:
            raise ValueError(f"Unsupported layout_sample_mode: {layout_sample_mode}")
        batch_size, max_nodes, _ = pooled_label_embeddings.shape
        node_states = self.node_proj(pooled_label_embeddings)
        if self.layout_mode == "triple_cvae":
            return self._triple_cvae_forward(
                node_states,
                scene_graph_batch,
                layout_sample_mode=layout_sample_mode,
                layout_z_scale=layout_z_scale,
            )
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
                layout_z_scale=layout_z_scale,
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
            slot_boxes_3d=None,
            slot_mask=scene_graph_batch.position_mask.to(node_states.device),
            relation_logits=relation_logits,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            sampled_z=sampled_z,
        )

    def _edge_readout(self, edge_states: torch.Tensor) -> torch.Tensor:
        if edge_states.numel() == 0:
            return edge_states.new_zeros(edge_states.shape[-1])
        edge_mask = torch.ones(1, edge_states.shape[0], device=edge_states.device, dtype=torch.bool)
        readout, _weights = self._attention_readout(
            edge_states.unsqueeze(0),
            edge_mask,
            self.triple_edge_readout_score,
        )
        return readout.squeeze(0)

    def _triple_graph_readout(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
    ) -> torch.Tensor:
        node_mask = torch.ones(1, node_states.shape[0], device=node_states.device, dtype=torch.bool)
        node_readout, _weights = self._attention_readout(
            node_states.unsqueeze(0),
            node_mask,
            self.triple_node_readout_score,
        )
        edge_readout = self._edge_readout(edge_states).unsqueeze(0)
        return self.triple_graph_fuse(torch.cat([node_readout, edge_readout], dim=-1)).squeeze(0)

    def _run_triple_layers(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        edge_index: torch.Tensor,
        layers: nn.ModuleList,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in layers:
            node_states, edge_states = layer(node_states, edge_states, edge_index)
        return node_states, edge_states

    def _split_stats(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-8.0, max=4.0)

    def _triple_cvae_forward(
        self,
        node_states: torch.Tensor,
        scene_graph_batch: BatchedSceneGraphs,
        *,
        layout_sample_mode: str,
        layout_z_scale: float,
    ) -> GraphConditioningOutput:
        batch_size, max_nodes, _ = node_states.shape
        device = node_states.device
        dtype = node_states.dtype
        slot_mask = scene_graph_batch.position_mask.to(device)

        slot_positions = torch.zeros(batch_size, max_nodes, 3, device=device, dtype=dtype)
        slot_log_sizes_3d = torch.zeros(batch_size, max_nodes, 3, device=device, dtype=dtype)
        slot_boxes_3d = torch.zeros(batch_size, max_nodes, 6, device=device, dtype=dtype)
        decoded_nodes = torch.zeros_like(node_states)
        prior_mu = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        prior_logvar = torch.zeros_like(prior_mu)
        posterior_mu = torch.zeros_like(prior_mu)
        posterior_logvar = torch.zeros_like(prior_mu)
        object_prior_mu = torch.zeros(batch_size, max_nodes, self.latent_dim, device=device, dtype=dtype)
        object_prior_logvar = torch.zeros_like(object_prior_mu)
        object_posterior_mu = torch.zeros_like(object_prior_mu)
        object_posterior_logvar = torch.zeros_like(object_prior_mu)
        sampled_scene_z = torch.zeros_like(prior_mu)
        sampled_object_z = torch.zeros_like(object_prior_mu)
        relation_logits: list[torch.Tensor] = []

        gt_layout = torch.cat(
            [
                scene_graph_batch.position_targets.to(device=device, dtype=dtype),
                scene_graph_batch.log_size_targets.to(device=device, dtype=dtype),
            ],
            dim=-1,
        )

        for batch_index in range(batch_size):
            valid_node_count = int(slot_mask[batch_index].sum().item())
            if valid_node_count == 0:
                relation_logits.append(torch.zeros((0, 3), device=device, dtype=dtype))
                continue

            sample_nodes = node_states[batch_index, :valid_node_count]
            sample_edges = scene_graph_batch.edge_index[batch_index].to(device)
            sample_edge_types = scene_graph_batch.edge_types[batch_index].to(device)
            relation_vectors = self.relation_embedding(sample_edge_types).to(dtype=dtype)
            edge_states = self.edge_proj(relation_vectors)

            prior_nodes, prior_edges = self._run_triple_layers(
                sample_nodes,
                edge_states,
                sample_edges,
                self.triple_prior_layers,
            )
            prior_graph_state = self._triple_graph_readout(prior_nodes, prior_edges)
            scene_prior_mu, scene_prior_logvar = self._split_stats(
                self.triple_prior_scene_head(prior_graph_state)
            )
            obj_prior_mu, obj_prior_logvar = self._split_stats(
                self.triple_prior_object_head(prior_nodes)
            )

            if scene_graph_batch.box_targets is not None:
                posterior_layout = scene_graph_batch.box_targets[batch_index, :valid_node_count].to(
                    device=device,
                    dtype=dtype,
                )
            else:
                posterior_layout = gt_layout[batch_index, :valid_node_count]
            layout_features = self.gt_layout_encoder(posterior_layout)
            posterior_input_nodes = self.posterior_node_init(
                torch.cat([sample_nodes, layout_features], dim=-1)
            )
            posterior_nodes, posterior_edges = self._run_triple_layers(
                posterior_input_nodes,
                edge_states,
                sample_edges,
                self.triple_posterior_layers,
            )
            posterior_graph_state = self._triple_graph_readout(posterior_nodes, posterior_edges)
            scene_posterior_mu, scene_posterior_logvar = self._split_stats(
                self.triple_posterior_scene_head(posterior_graph_state)
            )
            obj_posterior_mu, obj_posterior_logvar = self._split_stats(
                self.triple_posterior_object_head(posterior_nodes)
            )

            if layout_sample_mode == "posterior" or (layout_sample_mode == "auto" and self.training):
                scene_z = self._reparameterize(scene_posterior_mu, scene_posterior_logvar)
                object_z = self._reparameterize(obj_posterior_mu, obj_posterior_logvar)
                scene_z = scene_posterior_mu + layout_z_scale * (scene_z - scene_posterior_mu)
                object_z = obj_posterior_mu + layout_z_scale * (object_z - obj_posterior_mu)
            elif layout_sample_mode == "prior_sample":
                scene_z = self._reparameterize(scene_prior_mu, scene_prior_logvar)
                object_z = self._reparameterize(obj_prior_mu, obj_prior_logvar)
                scene_z = scene_prior_mu + layout_z_scale * (scene_z - scene_prior_mu)
                object_z = obj_prior_mu + layout_z_scale * (object_z - obj_prior_mu)
            else:
                scene_z = scene_prior_mu
                object_z = obj_prior_mu

            scene_z_nodes = scene_z.unsqueeze(0).expand(valid_node_count, -1)
            z_context = torch.cat([scene_z_nodes, object_z], dim=-1)
            film = self.triple_decoder_film(z_context)
            gamma, beta = film.chunk(2, dim=-1)
            decoder_prior_nodes = F.dropout(
                prior_nodes,
                p=self.decoder_node_dropout,
                training=self.training and self.decoder_node_dropout > 0.0,
            )
            modulated_prior_nodes = decoder_prior_nodes * (1.0 + 0.1 * gamma.tanh()) + 0.1 * beta
            decoder_nodes = self.triple_decoder_node_in(
                torch.cat([modulated_prior_nodes, z_context], dim=-1)
            )
            decoder_nodes, decoder_edges = self._run_triple_layers(
                decoder_nodes,
                prior_edges,
                sample_edges,
                self.triple_decoder_layers,
            )
            raw_boxes = self.triple_box_3d_head(decoder_nodes).sigmoid()
            box_mins = torch.minimum(raw_boxes[:, :3], raw_boxes[:, 3:])
            box_maxs = torch.maximum(raw_boxes[:, :3], raw_boxes[:, 3:])
            boxes = torch.cat([box_mins, box_maxs], dim=-1)
            centers_01 = (box_mins + box_maxs) * 0.5
            sizes = (box_maxs - box_mins).clamp(min=0.03)
            positions = centers_01.mul(2.0).sub(1.0)
            log_sizes = sizes.log()

            slot_positions[batch_index, :valid_node_count] = positions
            slot_log_sizes_3d[batch_index, :valid_node_count] = log_sizes
            decoded_nodes[batch_index, :valid_node_count] = decoder_nodes
            prior_mu[batch_index] = scene_prior_mu
            prior_logvar[batch_index] = scene_prior_logvar
            posterior_mu[batch_index] = scene_posterior_mu
            posterior_logvar[batch_index] = scene_posterior_logvar
            object_prior_mu[batch_index, :valid_node_count] = obj_prior_mu
            object_prior_logvar[batch_index, :valid_node_count] = obj_prior_logvar
            object_posterior_mu[batch_index, :valid_node_count] = obj_posterior_mu
            object_posterior_logvar[batch_index, :valid_node_count] = obj_posterior_logvar
            sampled_scene_z[batch_index] = scene_z
            sampled_object_z[batch_index, :valid_node_count] = object_z
            slot_boxes_3d[batch_index, :valid_node_count] = boxes

            logits_per_edge: list[torch.Tensor] = []
            for src, dst, _relation in scene_graph_batch.relation_triplets[batch_index]:
                logits_per_edge.append(
                    self.relation_head(torch.cat([decoder_nodes[src], decoder_nodes[dst]], dim=-1))
                )
            if logits_per_edge:
                relation_logits.append(torch.stack(logits_per_edge, dim=0))
            else:
                relation_logits.append(torch.zeros((0, 3), device=device, dtype=dtype))

        slot_embeddings = self.slot_out(decoded_nodes)
        slot_log_sigmas = self.log_sigma_head(decoded_nodes).clamp(min=-4.0, max=1.0)
        return GraphConditioningOutput(
            slot_embeddings=slot_embeddings,
            slot_positions=slot_positions,
            slot_position_mu=slot_positions,
            slot_position_logvar=None,
            slot_log_sigmas=slot_log_sigmas,
            slot_log_sizes_3d=slot_log_sizes_3d,
            slot_log_size_3d_mu=slot_log_sizes_3d,
            slot_log_size_3d_logvar=None,
            slot_boxes_3d=slot_boxes_3d,
            slot_mask=slot_mask,
            relation_logits=relation_logits,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            sampled_z=sampled_scene_z,
            object_prior_mu=object_prior_mu,
            object_prior_logvar=object_prior_logvar,
            object_posterior_mu=object_posterior_mu,
            object_posterior_logvar=object_posterior_logvar,
            sampled_object_z=sampled_object_z,
        )

    def _attention_readout(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        score_head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = score_head(values).squeeze(-1)
        scores = scores.masked_fill(~mask.to(torch.bool), torch.finfo(scores.dtype).min)
        empty = ~mask.any(dim=1)
        if empty.any():
            scores = scores.clone()
            scores[empty] = 0.0
        weights = torch.softmax(scores, dim=1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (values * weights.unsqueeze(-1)).sum(dim=1), weights

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
        layout_z_scale: float,
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
        graph_state, _graph_readout_weights = self._attention_readout(
            node_states,
            slot_mask,
            self.graph_readout_score,
        )
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
        gt_layout_state, _layout_readout_weights = self._attention_readout(
            self.gt_layout_encoder(gt_layout),
            slot_mask,
            self.layout_readout_score,
        )
        posterior_stats = self.posterior_head(torch.cat([graph_state, gt_layout_state], dim=-1))
        posterior_mu, posterior_logvar = posterior_stats.chunk(2, dim=-1)
        posterior_logvar = posterior_logvar.clamp(min=-8.0, max=4.0)

        if layout_sample_mode == "posterior" or (layout_sample_mode == "auto" and self.training):
            z = self._reparameterize(posterior_mu, posterior_logvar)
            z = posterior_mu + layout_z_scale * (z - posterior_mu)
        elif layout_sample_mode == "prior_sample":
            z = self._reparameterize(prior_mu, prior_logvar)
            z = prior_mu + layout_z_scale * (z - prior_mu)
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
    label_embedding_cache: dict[str, torch.Tensor] | None = None,
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

    cache = label_embedding_cache
    for batch_index, labels in enumerate(scene_graph_batch.node_labels):
        if not labels:
            continue
        missing_labels: list[str] = []
        missing_indices: list[int] = []
        if cache is not None:
            for label_index, label in enumerate(labels):
                cached = cache.get(label)
                if cached is None:
                    missing_labels.append(label)
                    missing_indices.append(label_index)
                else:
                    pooled[batch_index, label_index] = cached.to(device=device, dtype=dtype)
        else:
            missing_labels = labels
            missing_indices = list(range(len(labels)))

        if not missing_labels:
            continue

        text_inputs = tokenizer(missing_labels, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            encoded = text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
            )[0]
        encoded_pooled = mean_pool_hidden(
            encoded,
            text_inputs.attention_mask.to(device),
        ).to(dtype=dtype)
        for target_index, label, embedding in zip(missing_indices, missing_labels, encoded_pooled):
            pooled[batch_index, target_index] = embedding
            if cache is not None:
                cache[label] = embedding.detach().cpu()
    return pooled


def build_slot_conditioning(
    *,
    tokenizer: Any,
    text_encoder: Any,
    scene_graph_batch: BatchedSceneGraphs,
    graph_encoder: GraphSlotEncoder,
    device: str,
    layout_sample_mode: str = "auto",
    label_embedding_cache: dict[str, torch.Tensor] | None = None,
    layout_z_scale: float = 1.0,
) -> GraphConditioningOutput:
    graph_dtype = graph_encoder.node_proj.weight.dtype
    pooled = pooled_label_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scene_graph_batch=scene_graph_batch,
        device=device,
        dtype=graph_dtype,
        label_embedding_cache=label_embedding_cache,
    )
    return graph_encoder(
        pooled,
        scene_graph_batch,
        layout_sample_mode=layout_sample_mode,
        layout_z_scale=layout_z_scale,
    )


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


def box_3d_l1_loss(
    slot_boxes_3d: torch.Tensor | None,
    box_targets: torch.Tensor | None,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """3D_SLN-style L1 loss on normalized ``[x0,y0,z0,x1,y1,z1]`` boxes."""

    if slot_boxes_3d is None or box_targets is None or not slot_mask.any():
        fallback = slot_mask.new_tensor(0.0, dtype=torch.float32)
        if slot_boxes_3d is not None:
            fallback = slot_boxes_3d.new_tensor(0.0)
        return fallback
    return F.l1_loss(
        slot_boxes_3d[slot_mask],
        box_targets[slot_mask].to(slot_boxes_3d.dtype),
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


def _diagonal_gaussian_kl(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    posterior_mu = posterior_mu.float()
    posterior_logvar = posterior_logvar.float()
    prior_mu = prior_mu.float()
    prior_logvar = prior_logvar.float()
    prior_var = torch.exp(prior_logvar)
    posterior_var = torch.exp(posterior_logvar)
    delta = posterior_mu - prior_mu
    kl = 0.5 * (
        prior_logvar
        - posterior_logvar
        + (posterior_var + delta.pow(2)) / prior_var.clamp_min(1e-8)
        - 1.0
    )
    return kl.clamp_min(0.0)


def cvae_kl_loss(output: GraphConditioningOutput) -> torch.Tensor:
    """Conditional CVAE KL between posterior and graph-conditioned prior."""

    if (
        output.prior_mu is None
        or output.prior_logvar is None
        or output.posterior_mu is None
        or output.posterior_logvar is None
    ):
        return output.slot_positions.new_tensor(0.0)
    scene_kl = _diagonal_gaussian_kl(
        output.posterior_mu,
        output.posterior_logvar,
        output.prior_mu,
        output.prior_logvar,
    ).sum(dim=-1).mean()
    if (
        output.object_prior_mu is None
        or output.object_prior_logvar is None
        or output.object_posterior_mu is None
        or output.object_posterior_logvar is None
    ):
        return scene_kl.to(output.slot_positions.dtype)
    object_kl = _diagonal_gaussian_kl(
        output.object_posterior_mu,
        output.object_posterior_logvar,
        output.object_prior_mu,
        output.object_prior_logvar,
    ).sum(dim=-1)
    if not output.slot_mask.any():
        return scene_kl.to(output.slot_positions.dtype)
    object_kl = object_kl[output.slot_mask].mean()
    return (scene_kl + object_kl).to(output.slot_positions.dtype)


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
