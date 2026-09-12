from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from data_processing import (
    lightgcn_propagate,
    segment_softmax,
    segment_sum,
)


class PGRL(nn.Module):

    def __init__(
        self,
        num_mashups: int,
        num_apis: int,
        num_categories: int,
        text_embeddings: torch.Tensor,
        tag_ids: torch.Tensor,
        tag_offsets: torch.Tensor,
        tag_available: torch.Tensor,
        text_available: torch.Tensor,
        dimension: int = 64,
        structure_layers: int = 2,
        semantic_layers: int = 1,
        include_layer0: bool = True,
        tag_pooling: str = "attention",
        dropout: float = 0.1,
        use_tag_prompt: bool = True,
        use_text_prompt: bool = True,
        neighbor_aggregation: str = "attention",
        attention_dim: int = 512,
        gate_mode: str = "learned",
        fixed_gate: float = 0.5,
        fixed_tag_gate: float | None = None,
        fixed_text_gate: float | None = None,
    ) -> None:
        super().__init__()
        self.num_mashups = num_mashups
        self.num_apis = num_apis
        self.num_nodes = num_mashups + num_apis
        self.dimension = dimension
        self.structure_layers = structure_layers
        self.semantic_layers = semantic_layers
        self.include_layer0 = include_layer0
        self.tag_pooling = tag_pooling
        if neighbor_aggregation not in {"attention", "cross_attention", "mean"}:
            raise ValueError(
                "neighbor_aggregation must be 'attention', 'cross_attention', or 'mean'"
            )
        if isinstance(attention_dim, bool) or int(attention_dim) <= 0:
            raise ValueError("attention_dim must be a positive integer")
        if gate_mode not in {"learned", "fixed"}:
            raise ValueError("gate_mode must be either 'learned' or 'fixed'")
        if not 0.0 <= float(fixed_gate) <= 1.0:
            raise ValueError("fixed_gate must be between 0 and 1")
        if (fixed_tag_gate is None) != (fixed_text_gate is None):
            raise ValueError(
                "fixed_tag_gate and fixed_text_gate must be provided together"
            )
        resolved_tag_gate = fixed_gate if fixed_tag_gate is None else fixed_tag_gate
        resolved_text_gate = fixed_gate if fixed_text_gate is None else fixed_text_gate
        if not 0.0 <= float(resolved_tag_gate) <= 1.0:
            raise ValueError("fixed_tag_gate must be between 0 and 1")
        if not 0.0 <= float(resolved_text_gate) <= 1.0:
            raise ValueError("fixed_text_gate must be between 0 and 1")
        self.neighbor_aggregation = neighbor_aggregation
        self.attention_dim = int(attention_dim)
        self.gate_mode = gate_mode
        self.fixed_gate = float(fixed_gate)
        self.fixed_tag_gate = float(resolved_tag_gate)
        self.fixed_text_gate = float(resolved_text_gate)
        self.fixed_source_gates = {
            "tag": self.fixed_tag_gate,
            "text": self.fixed_text_gate,
        }
        sources = ("tag", "text")
        self.prompt_sources = sources
        self.prompt_source_enabled = (bool(use_tag_prompt), bool(use_text_prompt))

        self.structure_embedding = nn.Embedding(self.num_nodes, dimension)
        self.category_embedding = nn.Embedding(max(num_categories, 1), dimension)
        self.tag_key = nn.Linear(dimension, dimension)
        self.tag_query = nn.Parameter(torch.empty(dimension))
        self.text_projection = nn.Linear(text_embeddings.shape[1], dimension)
        self.prompt_queries = nn.ModuleDict(
            {
                source: nn.Linear(dimension, dimension, bias=False)
                for source in self.prompt_sources
                if self.neighbor_aggregation == "attention"
            }
        )
        self.structure_prompt_query = (
            nn.Linear(dimension, self.attention_dim, bias=False)
            if self.neighbor_aggregation == "cross_attention"
            else None
        )
        self.prompt_keys = nn.ModuleDict(
            {
                source: nn.Linear(
                    dimension,
                    self.attention_dim
                    if self.neighbor_aggregation == "cross_attention"
                    else dimension,
                    bias=False,
                )
                for source in self.prompt_sources
                if self.neighbor_aggregation in {"attention", "cross_attention"}
            }
        )
        self.prompt_values = nn.ModuleDict(
            {
                source: nn.Linear(dimension, dimension, bias=False)
                for source in self.prompt_sources
            }
        )
        self.prompt_projections = nn.ModuleDict(
            {
                source: nn.Linear(dimension, dimension)
                for source in self.prompt_sources
            }
        )
        self.prompt_gates = nn.ModuleDict(
            {
                source: nn.Linear(2 * dimension, dimension)
                for source in self.prompt_sources
                if self.gate_mode == "learned"
            }
        )
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.structure_embedding.weight, std=0.1)
        nn.init.normal_(self.category_embedding.weight, std=0.1)
        nn.init.normal_(self.tag_query, std=0.1)
        for projection in self.prompt_projections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.register_buffer("text_features", text_embeddings.float())
        self.register_buffer("tag_ids", tag_ids.long())
        self.register_buffer("tag_offsets", tag_offsets.long())
        self.register_buffer("tag_available", tag_available.bool())
        self.register_buffer("text_available", text_available.bool())

    def load_structure_pretrain(self, weight: torch.Tensor) -> None:
        if weight.shape != self.structure_embedding.weight.shape:
            raise ValueError("Pretrained structure embedding has an incompatible shape")
        with torch.no_grad():
            self.structure_embedding.weight.copy_(weight)

    def _tag_initial(self) -> torch.Tensor:
        if self.tag_ids.numel() == 0:
            return torch.zeros(
                (self.num_nodes, self.dimension),
                device=self.structure_embedding.weight.device,
            )
        counts = self.tag_offsets[1:] - self.tag_offsets[:-1]
        node_index = torch.repeat_interleave(
            torch.arange(self.num_nodes, device=self.tag_ids.device), counts
        )
        embeddings = self.category_embedding(self.tag_ids)
        if self.tag_pooling == "mean":
            pooled = segment_sum(embeddings, node_index, self.num_nodes)
            return pooled / counts.clamp_min(1).to(embeddings.dtype)[:, None]
        if self.tag_pooling != "attention":
            raise ValueError(f"Unknown tag pooling: {self.tag_pooling}")
        scores = torch.tanh(self.tag_key(embeddings)) @ self.tag_query
        weights = segment_softmax(scores, node_index, self.num_nodes)
        return segment_sum(weights[:, None] * embeddings, node_index, self.num_nodes)

    def view_embeddings(
        self, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structure = lightgcn_propagate(
            self.structure_embedding.weight,
            adjacency,
            self.structure_layers,
            include_layer0=True,
        )
        tag = lightgcn_propagate(
            self._tag_initial(), adjacency, self.semantic_layers, self.include_layer0
        )
        text = lightgcn_propagate(
            self.text_projection(self.text_features),
            adjacency,
            self.semantic_layers,
            self.include_layer0,
        )
        return structure, tag, text

    def _prompt_source_mask(self) -> torch.Tensor:
        availability = {
            "structure": torch.ones(
                self.num_nodes, dtype=torch.bool, device=self.tag_available.device
            ),
            "tag": self.tag_available,
            "text": self.text_available,
        }
        available = torch.stack(
            [availability[source] for source in self.prompt_sources], dim=1
        )
        enabled = torch.tensor(
            self.prompt_source_enabled, dtype=torch.bool, device=available.device
        )
        return available & enabled[None, :]

    def _encode_independent_prompts(
        self,
        structure: torch.Tensor,
        tag: torch.Tensor,
        text: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        representations = {"tag": tag, "text": text}
        available = self._prompt_source_mask()
        semantic_attention_scale = math.sqrt(self.dimension)
        cross_attention_scale = math.sqrt(self.attention_dim)
        cross_queries = (
            self.structure_prompt_query(structure)
            if self.structure_prompt_query is not None
            else None
        )
        prompted = structure
        diagnostics: dict[str, torch.Tensor] = {}
        prompts: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []

        for index, source in enumerate(self.prompt_sources):
            representation = representations[source]
            values = self.prompt_values[source](representation[edge_target])
            if self.neighbor_aggregation == "attention":
                queries = self.prompt_queries[source](representation)
                keys = self.prompt_keys[source](representation)
                score = (
                    queries[edge_source] * keys[edge_target]
                ).sum(dim=-1) / semantic_attention_scale
                score = F.leaky_relu(score, negative_slope=0.2)
                alpha = segment_softmax(score, edge_source, self.num_nodes)
            elif self.neighbor_aggregation == "cross_attention":
                assert cross_queries is not None
                keys = self.prompt_keys[source](representation)
                score = (
                    cross_queries[edge_source] * keys[edge_target]
                ).sum(dim=-1) / cross_attention_scale
                score = F.leaky_relu(score, negative_slope=0.2)
                alpha = segment_softmax(score, edge_source, self.num_nodes)
            else:
                degree = segment_sum(
                    torch.ones(
                        (edge_source.numel(), 1),
                        dtype=values.dtype,
                        device=values.device,
                    ),
                    edge_source,
                    self.num_nodes,
                ).squeeze(-1).clamp_min(1.0)
                alpha = degree[edge_source].reciprocal()
            message = segment_sum(
                alpha[:, None] * values, edge_source, self.num_nodes
            )
            prompt = torch.tanh(
                self.prompt_projections[source](self.dropout(message))
            )
            source_available = available[:, index, None]
            prompt = prompt.masked_fill(~source_available, 0.0)
            if self.gate_mode == "learned":
                gamma = torch.sigmoid(
                    self.prompt_gates[source](
                        torch.cat([structure, prompt], dim=-1)
                    )
                )
            else:
                gamma = torch.full_like(prompt, self.fixed_source_gates[source])
            gamma = gamma.masked_fill(~source_available, 0.0)
            residual = gamma * prompt
            prompted = prompted + residual
            prompts.append(prompt)
            residuals.append(residual)
            diagnostics[f"alpha_{source}"] = alpha
            diagnostics[f"prompt_{source}"] = prompt
            diagnostics[f"gamma_{source}"] = gamma
            diagnostics[f"residual_{source}"] = residual

        zero = torch.zeros_like(structure)
        diagnostics["delta"] = sum(prompts, zero)
        diagnostics["residual"] = sum(residuals, zero)
        return prompted, diagnostics

    def encode_all(
        self,
        adjacency: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        structure, tag, text = self.view_embeddings(adjacency)
        prompted, diagnostics = self._encode_independent_prompts(
            structure, tag, text, edge_source, edge_target
        )
        return structure, prompted, diagnostics

    def encode(
        self,
        adjacency: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, enhanced, _ = self.encode_all(adjacency, edge_source, edge_target)
        return enhanced[: self.num_mashups], enhanced[self.num_mashups :]


def bpr_loss(
    mashup_embeddings: torch.Tensor,
    api_embeddings: torch.Tensor,
    mashups: torch.Tensor,
    positive_apis: torch.Tensor,
    negative_apis: torch.Tensor,
) -> torch.Tensor:
    users = mashup_embeddings[mashups]
    positives = api_embeddings[positive_apis]
    if negative_apis.ndim == 1:
        negative_apis = negative_apis[:, None]
    negatives = api_embeddings[negative_apis]
    positive_scores = (users * positives).sum(dim=-1, keepdim=True)
    negative_scores = (users[:, None, :] * negatives).sum(dim=-1)
    difference = positive_scores - negative_scores
    return -F.logsigmoid(difference).mean()
