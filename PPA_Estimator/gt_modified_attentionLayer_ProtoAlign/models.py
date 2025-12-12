"""Model components for the circuit graph transformer."""

from __future__ import annotations

from typing import Dict, Sequence

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import GraphBatch, NodeFeatureDict, TypeAssignment


class HeteroNodeMapper(nn.Module):
    """Map heterogeneous node features to a unified hidden dimension."""

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        alignment_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temp = alignment_temperature
        self.types = list(input_dims.keys())
        self.type_to_index = {node_type: idx for idx, node_type in enumerate(self.types)}

        self.type_mlps = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for node_type, dim in input_dims.items()
        })

        self.residual_proj = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim) for node_type, dim in input_dims.items()
        })
        self.alignment_logits = nn.Parameter(torch.full((len(self.types),), -1.0))

        # Attention-based alignment toward learnable type centers
        self.type_centers = nn.Parameter(torch.randn(len(self.types), hidden_dim))
        self.center_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.center_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.center_value = nn.Linear(hidden_dim, hidden_dim)
        self.align_scale = hidden_dim ** -0.5

    def forward(
        self,
        node_features: NodeFeatureDict,
        type_assignments: Sequence[TypeAssignment],
    ) -> torch.Tensor:
        processed: Dict[str, torch.Tensor] = {}

        for node_type in self.types:
            feats = node_features.get(node_type)
            if feats is None or feats.numel() == 0:
                processed[node_type] = torch.zeros(
                    (0, self.hidden_dim), device=self.alignment_logits.device
                )
                continue

            projected = self.type_mlps[node_type](feats)
            residual = self.residual_proj[node_type](feats)
            weight = torch.sigmoid(self.alignment_logits[self.type_to_index[node_type]])
            mixed = weight * projected + (1.0 - weight) * residual

            type_idx = self.type_to_index[node_type]
            center = self.type_centers[type_idx]
            query = self.center_query(center)
            keys = self.center_key(mixed)
            values = self.center_value(mixed)

            scores = (keys @ query) * self.align_scale
            attn = F.softmax(scores / self.temp, dim=0)
            context = torch.sum(attn.unsqueeze(-1) * values, dim=0, keepdim=True)
            processed[node_type] = mixed + 0.1 * (context - mixed)

        if not type_assignments:
            return torch.empty(0, self.hidden_dim, device=self.alignment_logits.device)

        first_type = type_assignments[0][0]
        dtype = processed[first_type].dtype
        device = processed[first_type].device
        unified = torch.empty(len(type_assignments), self.hidden_dim, device=device, dtype=dtype)

        for idx, (node_type, local_idx) in enumerate(type_assignments):
            unified[idx] = processed[node_type][local_idx]

        return unified


class EdgeFeatureMapper(nn.Module):
    """Project edge features with type-aware refinement."""

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_edge_types: int = 9,
    ) -> None:
        super().__init__()
        self.base_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embed = nn.Embedding(num_edge_types, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_features: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        base = self.base_proj(edge_features)
        type_features = self.type_embed(edge_types)
        return self.fuse(torch.cat([base, type_features], dim=-1))


def _segment_sum(values: torch.Tensor, indices: torch.Tensor, num_segments: int) -> torch.Tensor:
    output = torch.zeros(num_segments, *values.shape[1:], device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return output
    return output.index_add_(0, indices, values)


def _segment_max(values: torch.Tensor, indices: torch.Tensor, num_segments: int) -> torch.Tensor:
    output = torch.full(
        (num_segments, *values.shape[1:]),
        float("-inf"),
        device=values.device,
        dtype=values.dtype,
    )
    if values.numel() == 0:
        return output

    if values.dim() == 1:
        expand_indices = indices
    else:
        expand_indices = indices.unsqueeze(-1).expand_as(values)
    if hasattr(output, "scatter_reduce_"):
        output.scatter_reduce_(0, expand_indices, values, reduce="amax", include_self=True)
        return output

    for seg_id in torch.unique(indices):
        mask = indices == seg_id
        segment_vals = values[mask]
        if segment_vals.numel() == 0:
            continue
        output[seg_id] = torch.maximum(output[seg_id], segment_vals.max(dim=0).values)
    return output


def _edge_softmax(scores: torch.Tensor, src_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    max_per_src = _segment_max(scores, src_index, num_nodes)
    norm_scores = scores - max_per_src[src_index]
    exp_scores = torch.exp(norm_scores)
    denom = _segment_sum(exp_scores, src_index, num_nodes)
    return exp_scores / (denom[src_index] + 1e-9)


def _segment_softmax(values: torch.Tensor, indices: torch.Tensor, num_segments: int) -> torch.Tensor:
    max_per_segment = _segment_max(values, indices, num_segments)
    normalized = values - max_per_segment[indices]
    exp_scores = torch.exp(normalized)
    denom = _segment_sum(exp_scores, indices, num_segments)
    return exp_scores / (denom[indices] + 1e-9)


def _segment_mean(values: torch.Tensor, indices: torch.Tensor, num_segments: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(num_segments, *values.shape[1:])
    summed = _segment_sum(values, indices, num_segments)
    counts = torch.bincount(indices, minlength=num_segments).clamp_min(1)
    counts = counts.to(device=values.device)
    view_shape = (num_segments,) + (1,) * (values.dim() - 1)
    counts = counts.view(view_shape).to(values.dtype)
    return summed / counts


class PrototypeContext(nn.Module):
    """Graph-level prototype attention with learnable temperature and gating."""

    def __init__(self, hidden_dim: int, num_prototypes: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.register_buffer("prototypes", torch.randn(num_prototypes, hidden_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, graph_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if graph_emb.numel() == 0:
            assignments = graph_emb.new_zeros(graph_emb.size(0), self.num_prototypes)
            return graph_emb, assignments
        proto = self.prototypes.to(graph_emb.device)
        sims = F.cosine_similarity(graph_emb.unsqueeze(1), proto.unsqueeze(0), dim=-1)
        sims = sims * torch.exp(self.temperature)
        assignments = F.softmax(sims, dim=-1)
        context = assignments @ proto
        alpha = torch.sigmoid(self.gate(graph_emb))
        fused = alpha * graph_emb + (1 - alpha) * context
        return fused, assignments

    def update_prototypes(self, new_proto: torch.Tensor) -> None:
        if new_proto.shape != self.prototypes.shape:
            raise ValueError(
                f"Prototype shape mismatch: expected {self.prototypes.shape}, got {new_proto.shape}"
            )
        self.prototypes.data.copy_(new_proto)


class NodePrototypeInjector(nn.Module):
    """Inject prototype-aware context back into node embeddings via gating."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,
        fused_graph_emb: torch.Tensor,
        node_graph_index: torch.Tensor,
    ) -> torch.Tensor:
        if node_states.numel() == 0:
            return node_states
        context = self.context_proj(fused_graph_emb)
        context_nodes = context[node_graph_index]
        gate_input = torch.cat([node_states, context_nodes], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * node_states + (1 - gate) * context_nodes


class GraphAttentionLayer(nn.Module):
    """Graph attention layer with edge-conditioned biases."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        hidden_edge_dim = max(edge_dim // 2, num_heads)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_edge_dim),
            nn.GELU(),
            nn.Linear(hidden_edge_dim, num_heads),
        )
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_states: torch.Tensor, edge_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = node_states.size(0)
        src, dst = edge_index

        Q = self.q_proj(node_states).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(node_states).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(node_states).view(num_nodes, self.num_heads, self.head_dim)

        q_src = Q[src]
        k_dst = K[dst]
        scores = (q_src * k_dst).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.edge_proj(edge_feat)
        attn = _segment_softmax(scores, dst, num_nodes)
        attn = self.dropout(attn).unsqueeze(-1)

        messages = V[src] * attn
        aggregated = _segment_sum(messages, dst, num_nodes)
        aggregated = aggregated.reshape(num_nodes, self.num_heads * self.head_dim)

        return self.out_proj(aggregated)


class GraphTransformerEncoder(nn.Module):
    """Stack of graph attention layers with residual connections."""

    def __init__(
        self,
        num_layers: int = 6,
        node_dim: int = 256,
        edge_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GraphAttentionLayer(node_dim, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        ffn_hidden = node_dim * 4
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(node_dim, ffn_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_hidden, node_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])

    def forward(
        self, node_states: torch.Tensor, edge_feat: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = node_states
        for attn_layer, attn_norm, ffn, ffn_norm in zip(
            self.layers, self.attn_norms, self.ffns, self.ffn_norms
        ):
            attn_out = attn_layer(x, edge_feat, edge_index)
            x = attn_norm(x + attn_out)
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)
        return x


class TaskDecoder(nn.Module):
    """Multi-task decoder with per-task pooling over full node embeddings."""

    def __init__(self, node_dim: int = 256) -> None:
        super().__init__()
        self.node_dim = node_dim

        self.task_output_dims = {
            "leakage": 1,
            "internal_power": 7,
            "cell_rise": 7,
            "cell_fall": 7,
            "rise_transition": 7,
            "fall_transition": 7,
        }

        self.task_queries = nn.ParameterDict(
            {
                task: nn.Parameter(torch.randn(node_dim))
                for task in self.task_output_dims
            }
        )

        self.projections = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(node_dim, node_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(node_dim * 2, output_dim),
                )
                for task, output_dim in self.task_output_dims.items()
            }
        )

    def forward(
        self,
        node_states: torch.Tensor,
        node_graph_index: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for task, projection in self.projections.items():
            query = self.task_queries[task]
            scores = (node_states * query).sum(dim=-1) / math.sqrt(self.node_dim)
            attn = _segment_softmax(scores, node_graph_index, batch_size).unsqueeze(-1)
            weighted = node_states * attn
            pooled = _segment_sum(weighted, node_graph_index, batch_size)
            logits = projection(pooled)
            outputs[task] = logits
        return outputs


class GraphTransformerModel(nn.Module):
    """Full graph transformer model encapsulating mapping, encoder, and decoder."""

    def __init__(
        self,
        node_input_dims: Dict[str, int] | None = None,
        edge_input_dim: int = 17,
        node_hidden_dim: int = 256,
        edge_hidden_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_prototypes: int = 32,
    ) -> None:
        super().__init__()
        if node_input_dims is None:
            node_input_dims = {"routing": 25, "fet": 29, "supply": 21}

        self.node_mapper = HeteroNodeMapper(
            node_input_dims, hidden_dim=node_hidden_dim
        )
        self.edge_mapper = EdgeFeatureMapper(
            input_dim=edge_input_dim, hidden_dim=edge_hidden_dim
        )
        self.encoder = GraphTransformerEncoder(
            num_layers=num_encoder_layers,
            node_dim=node_hidden_dim,
            edge_dim=edge_hidden_dim,
            num_heads=num_heads,
        )
        self.prototype_module = PrototypeContext(node_hidden_dim, num_prototypes=num_prototypes)
        self.node_injector = NodePrototypeInjector(node_hidden_dim)
        self.decoder = TaskDecoder(node_dim=node_hidden_dim)

    def update_prototypes(self, new_proto: torch.Tensor) -> None:
        self.prototype_module.update_prototypes(new_proto)

    @property
    def prototypes(self) -> torch.Tensor:
        return self.prototype_module.prototypes

    def forward(
        self,
        batch: GraphBatch,
        return_details: bool = False,
    ):
        node_states = self.node_mapper(batch.node_features, batch.type_assignments)
        edge_states = self.edge_mapper(batch.edge_features, batch.edge_types)
        encoded = self.encoder(node_states, edge_states, batch.edge_index)
        graph_emb = _segment_mean(encoded, batch.node_graph_index, batch.batch_size)
        fused_graph_emb, assignments = self.prototype_module(graph_emb)
        injected = self.node_injector(encoded, fused_graph_emb, batch.node_graph_index)
        predictions = self.decoder(injected, batch.node_graph_index, batch.batch_size)
        if return_details:
            return {
                "predictions": predictions,
                "graph_emb": graph_emb,
                "assignments": assignments,
                "fused_graph_emb": fused_graph_emb,
            }
        return predictions
