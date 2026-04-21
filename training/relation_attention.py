from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _query_grid(
    *,
    batch_size: int,
    query_length: int,
    hidden_states: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if hidden_states.ndim == 4:
        _, _, height, width = hidden_states.shape
    else:
        side = int(math.sqrt(query_length))
        if side * side == query_length:
            height = side
            width = side
        else:
            height = 1
            width = query_length

    ys = torch.linspace(-1.0, 1.0, steps=height, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
    return grid.expand(batch_size, -1, -1)


def _repeat_batch(tensor: torch.Tensor, target_batch: int) -> torch.Tensor:
    if tensor.shape[0] == target_batch:
        return tensor
    if target_batch % tensor.shape[0] != 0:
        raise ValueError(
            f"Cannot broadcast batch of size {tensor.shape[0]} to {target_batch} in relation-aware attention"
        )
    repeat_factor = target_batch // tensor.shape[0]
    return tensor.repeat_interleave(repeat_factor, dim=0)


class RelationAwareAttnProcessor2_0(nn.Module):
    def __init__(self, enable_bias: bool, capture_attention: bool = False) -> None:
        super().__init__()
        self.enable_bias = enable_bias
        self.capture_attention = capture_attention
        self.spatial_scale = nn.Parameter(torch.tensor(2.0))
        self.slot_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.latest_slot_attention_map: torch.Tensor | None = None
        self.latest_query_hw: tuple[int, int] | None = None

    def clear_attention_cache(self) -> None:
        self.latest_slot_attention_map = None
        self.latest_query_hw = None

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        slot_positions: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        text_token_count: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            slot_positions=slot_positions,
            slot_mask=slot_mask,
            text_token_count=text_token_count,
            *args,
            **kwargs,
        )

    def forward(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        slot_positions: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        text_token_count: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        height = 1
        width = hidden_states.shape[1]
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        batch_size, key_length, _ = encoder_hidden_states.shape
        query_length = hidden_states.shape[1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self.enable_bias and slot_positions is not None and slot_mask is not None and text_token_count is not None:
            slot_positions = _repeat_batch(slot_positions.to(query.device), batch_size)
            slot_mask = _repeat_batch(slot_mask.to(query.device), batch_size)
            slot_count = slot_positions.shape[1]
            grid = _query_grid(
                batch_size=batch_size,
                query_length=query_length,
                hidden_states=residual if input_ndim == 4 else hidden_states,
                device=query.device,
            )
            xy_positions = slot_positions[..., :2].to(query.device)
            dist2 = ((grid.unsqueeze(2) - xy_positions.unsqueeze(1)) ** 2).sum(dim=-1)
            slot_bias = -self.spatial_scale.exp() * dist2
            slot_bias = slot_bias.masked_fill(~slot_mask.unsqueeze(1), -1e4)
            full_bias = torch.zeros(
                batch_size,
                query_length,
                text_token_count + slot_count,
                device=query.device,
                dtype=query.dtype,
            )
            full_bias[:, :, text_token_count:] = slot_bias.to(query.dtype) * self.slot_logit_scale
            full_bias = full_bias.unsqueeze(1).expand(-1, attn.heads, -1, -1)
            attention_mask = full_bias if attention_mask is None else attention_mask + full_bias

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).to(query.dtype)

        if self.capture_attention and text_token_count is not None:
            self.latest_slot_attention_map = attention_probs.mean(dim=1)[..., text_token_count:]
            self.latest_query_hw = (height, width)
        else:
            self.clear_attention_cache()

        hidden_states = torch.matmul(attention_probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_relation_aware_processors(unet: Any) -> dict[str, nn.Module]:
    processors: dict[str, nn.Module] = {}
    for name in unet.attn_processors.keys():
        enable_bias = not name.endswith("attn1.processor")
        capture_attention = enable_bias
        processor = RelationAwareAttnProcessor2_0(
            enable_bias=enable_bias,
            capture_attention=capture_attention,
        )
        processors[name] = processor
    unet.set_attn_processor(processors.copy())
    return processors
