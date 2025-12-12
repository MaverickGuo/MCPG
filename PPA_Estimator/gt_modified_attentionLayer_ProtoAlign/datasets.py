"""Dataset utilities for graph transformer training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json

import torch
from torch.utils.data import Dataset


NodeFeatureDict = Dict[str, torch.Tensor]
TargetDict = Dict[str, torch.Tensor]
TypeAssignment = Tuple[str, int]


class TargetNormalizationError(RuntimeError):
    """Raised when normalization statistics are inconsistent with dataset targets."""


def _log1p_safe(tensor: torch.Tensor) -> torch.Tensor:
    """Apply log1p with clipping to keep the domain valid (>= -1)."""

    min_input = torch.full((), -0.999_999, dtype=tensor.dtype, device=tensor.device)
    return torch.log1p(torch.clamp(tensor, min=min_input))


@dataclass
class TargetNormalizationEntry:
    """Holds statistics and transform metadata for a single prediction task."""

    mean: torch.Tensor
    std: torch.Tensor
    transform: Optional[str] = None
    eps: float = 1e-8

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)

        transformed = value
        if self.transform == "log1p":
            transformed = _log1p_safe(transformed)

        mask = std > self.eps
        safe_std = torch.where(mask, std, torch.ones_like(std))
        normalized = (transformed - mean) / safe_std
        if not torch.all(mask):
            normalized = normalized * mask.to(dtype=value.dtype)

        return normalized

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)

        mask = std > self.eps
        safe_std = torch.where(mask, std, torch.ones_like(std))
        transformed = value * safe_std + mean
        if not torch.all(mask):
            mask_f = mask.to(dtype=value.dtype)
            transformed = transformed * mask_f + mean * (1 - mask_f)

        if self.transform == "log1p":
            return torch.expm1(transformed)
        return transformed


class TargetNormalizer:
    """Stores normalization entries for each task and provides helpers."""

    def __init__(self, entries: Dict[str, TargetNormalizationEntry]) -> None:
        self._entries = entries

    @classmethod
    def from_stats_dict(cls, stats: Dict) -> "TargetNormalizer":
        overall = stats.get("overall")
        if not overall:
            raise TargetNormalizationError("Statistics JSON missing 'overall' section")

        entries: Dict[str, TargetNormalizationEntry] = {}
        for key, payload in overall.items():
            if not {"mean", "std"}.issubset(payload.keys()):
                raise TargetNormalizationError(
                    f"Statistics entry for '{key}' must contain 'mean' and 'std'"
                )

            mean_tensor = torch.tensor(payload["mean"], dtype=torch.float32)
            std_tensor = torch.tensor(payload["std"], dtype=torch.float32)
            transform = payload.get("transform")

            entries[key] = TargetNormalizationEntry(
                mean=mean_tensor,
                std=std_tensor,
                transform=transform,
            )

        return cls(entries)

    def normalize(self, targets: TargetDict) -> TargetDict:
        normalized: TargetDict = {}
        for key, value in targets.items():
            entry = self._entries.get(key)
            if entry is None:
                normalized[key] = value
                continue
            normalized[key] = entry.normalize(value)
        return normalized

    def denormalize(self, targets: TargetDict) -> TargetDict:
        if not self._entries:
            raise TargetNormalizationError("No entries available for denormalization")

        restored: TargetDict = {}
        for key, value in targets.items():
            entry = self._entries.get(key)
            if entry is None:
                restored[key] = value
                continue
            restored[key] = entry.denormalize(value)
        return restored

    def has_entry(self, key: str) -> bool:
        return key in self._entries


def _infer_node_type(node_name: str) -> str:
    """Infer node type from node identifier in the embedding file."""

    lowered = node_name.lower()
    if lowered.startswith("routing"):
        return "routing"
    if lowered.startswith("supply"):
        return "supply"
    # Remaining instances correspond to transistor devices (e.g., MM0)
    return "fet"


@dataclass
class GraphSample:
    """Single graph sample before batching."""

    name: str
    node_features: NodeFeatureDict
    type_assignments: List[TypeAssignment]
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    targets: TargetDict
    sample_id: int

    @property
    def num_nodes(self) -> int:
        return len(self.type_assignments)

def _parse_values_string(values_str: str) -> List[float]:
    return [float(x.strip()) for x in values_str.split(',')]


def _derive_target_stem(embedding_stem: str) -> str:
    """Translate an embedding filename stem to the expected target stem."""

    try:
        design_part, tag, index = embedding_stem.rsplit("_", 2)
    except ValueError as exc:  # pragma: no cover - sanity guard for unexpected names
        raise ValueError(
            f"Embedding stem '{embedding_stem}' must follow '<design>_embedding_<idx>' pattern"
        ) from exc

    if tag != "embedding":
        raise ValueError(
            f"Embedding stem '{embedding_stem}' must contain '_embedding_' before the index"
        )

    return f"{design_part}_lib_{index}"

@dataclass
class GraphBatch:
    """Batch of graph samples merged for training."""

    node_features: NodeFeatureDict
    type_assignments: List[TypeAssignment]
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    targets: TargetDict
    graph_ptr: torch.Tensor
    node_graph_index: torch.Tensor
    sample_names: List[str]
    sample_ids: torch.Tensor

    def to(self, device: torch.device) -> "GraphBatch":
        node_features = {k: v.to(device) for k, v in self.node_features.items()}
        edge_features = self.edge_features.to(device)
        edge_index = self.edge_index.to(device)
        edge_types = self.edge_types.to(device)
        targets = {k: v.to(device) for k, v in self.targets.items()}
        graph_ptr = self.graph_ptr.to(device)
        node_graph_index = self.node_graph_index.to(device)
        return GraphBatch(
            node_features=node_features,
            type_assignments=self.type_assignments,
            edge_features=edge_features,
            edge_index=edge_index,
            edge_types=edge_types,
            targets=targets,
            graph_ptr=graph_ptr,
            node_graph_index=node_graph_index,
            sample_names=self.sample_names,
            sample_ids=self.sample_ids,
        )

    @property
    def batch_size(self) -> int:
        return len(self.sample_names)


class LibertyGraphDataset(Dataset):
    """Dataset pairing embedding JSONs with Liberty-derived targets."""

    def __init__(
        self,
        embedding_dir: Path,
        target_dir: Path,
        selection: Optional[Sequence[str]] = None,
        target_stats_path: Optional[Path] = None,
        normalize_targets: bool = False,
    ) -> None:
        self.embedding_dir = Path(embedding_dir)
        self.target_dir = Path(target_dir)
        self.normalize_targets = normalize_targets

        if self.normalize_targets and target_stats_path is None:
            raise ValueError("target_stats_path must be provided when normalize_targets=True")

        self._target_normalizer: Optional[TargetNormalizer] = None
        if target_stats_path is not None:
            stats = _load_json(Path(target_stats_path))
            self._target_normalizer = TargetNormalizer.from_stats_dict(stats)
            if self.normalize_targets:
                expected_keys = {"leakage", "internal_power", "cell_rise", "cell_fall", "rise_transition", "fall_transition"}
                missing = [key for key in expected_keys if not self._target_normalizer.has_entry(key)]
                if missing:
                    raise TargetNormalizationError(
                        f"Normalization stats missing entries for tasks: {', '.join(missing)}"
                    )

        if selection is not None:
            stems = list(selection)
        else:
            stems = sorted(p.stem for p in self.embedding_dir.glob("*.json"))

        self.embedding_paths = []
        self.target_paths = []

        for stem in stems:
            embed_path = self.embedding_dir / f"{stem}.json"
            if not embed_path.exists():
                raise FileNotFoundError(embed_path)

            target_stem = _derive_target_stem(stem)
            target_path = self.target_dir / f"{target_stem}.json"
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Derived target file not found: {target_path} (from embedding stem '{stem}')"
                )

            self.embedding_paths.append(embed_path)
            self.target_paths.append(target_path)

    def __len__(self) -> int:
        return len(self.embedding_paths)


    def __getitem__(self, idx: int) -> GraphSample:
        embed_path = self.embedding_paths[idx]
        target_path = self.target_paths[idx]

        embedding = _load_json(embed_path)
        targets = _load_json(target_path)

        node_feats: NodeFeatureDict = {
            node_type: torch.tensor(features, dtype=torch.float32)
            for node_type, features in embedding["node_embeddings"]["node_types"].items()
        }

        edge_features = torch.tensor(
            embedding["edge_embeddings"]["edge_features"], dtype=torch.float32
        )
        edge_types = torch.tensor(
            embedding["edge_embeddings"]["edge_types"], dtype=torch.long
        )
        edge_index = torch.tensor(
            embedding["coo_format"]["edge_index"], dtype=torch.long
        )

        type_assignments = _build_type_assignments(
            embedding["coo_format"], node_feats
        )

        try:
            formatted_targets: TargetDict = {
                "leakage": torch.tensor(
                    [targets["leakage_power"]["cell_leakage_power"]],
                    dtype=torch.float32,
                ),
                "internal_power": torch.tensor(
                    _parse_values_string(targets["internal_power"]["values"][0]),
                    dtype=torch.float32,
                ),
                "cell_rise": torch.tensor(
                    _parse_values_string(targets["timing"]["cell_rise"]["values"][0]),
                    dtype=torch.float32,
                ),
                "cell_fall": torch.tensor(
                    _parse_values_string(targets["timing"]["cell_fall"]["values"][0]),
                    dtype=torch.float32,
                ),
                "rise_transition": torch.tensor(
                    _parse_values_string(targets["timing"]["rise_transition"]["values"][0]),
                    dtype=torch.float32,
                ),
                "fall_transition": torch.tensor(
                    _parse_values_string(targets["timing"]["fall_transition"]["values"][0]),
                    dtype=torch.float32,
                ),
            }
        except KeyError as exc:
            missing_key = exc.args[0]
            raise KeyError(
                f"Target file '{target_path}' missing required key '{missing_key}'"
            ) from exc

        sample_name = embed_path.stem

        if self.normalize_targets and self._target_normalizer is not None:
            normalized_targets = self._target_normalizer.normalize(formatted_targets)
        else:
            normalized_targets = formatted_targets

        return GraphSample(
            name=sample_name,
            node_features=node_feats,
            type_assignments=type_assignments,
            edge_features=edge_features,
            edge_index=edge_index,
            edge_types=edge_types,
            targets=normalized_targets,
            sample_id=idx,
        )

    @property
    def target_normalizer(self) -> Optional[TargetNormalizer]:
        return self._target_normalizer

    def denormalize_targets(self, targets: TargetDict) -> TargetDict:
        if self._target_normalizer is None:
            raise TargetNormalizationError("No normalization statistics loaded")
        return self._target_normalizer.denormalize(targets)


def _load_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def _build_type_assignments(
    coo_section: Dict,
    node_features: NodeFeatureDict,
) -> List[TypeAssignment]:
    id_to_node = coo_section["node_mapping"]["id_to_node"]
    max_id = len(id_to_node)
    type_offsets = {node_type: 0 for node_type in node_features.keys()}
    assignments: List[TypeAssignment] = []

    for idx in range(max_id):
        node_name = id_to_node[str(idx)] if isinstance(id_to_node, dict) else id_to_node[idx]
        node_type = _infer_node_type(node_name)
        local_idx = type_offsets[node_type]
        type_offsets[node_type] += 1
        assignments.append((node_type, local_idx))

    for node_type, tensor in node_features.items():
        if type_offsets[node_type] != tensor.size(0):
            raise ValueError(
                f"Node count mismatch for {node_type}: mapping={type_offsets[node_type]} vs features={tensor.size(0)}"
            )

    return assignments


def collate_graphs(batch: Sequence[GraphSample]) -> GraphBatch:
    if not batch:
        raise ValueError("Empty batch")

    sample_names = [sample.name for sample in batch]
    sample_ids = torch.tensor([sample.sample_id for sample in batch], dtype=torch.long)

    node_features: NodeFeatureDict = {
        node_type: [] for node_type in batch[0].node_features.keys()
    }

    type_assignments: List[TypeAssignment] = []
    edge_features = []
    edge_types = []
    edge_indices = []
    targets_accum: Dict[str, List[torch.Tensor]] = {
        "leakage": [],
        "internal_power": [],
        "cell_rise": [],
        "cell_fall": [],
        "rise_transition": [],
        "fall_transition": [],
    }

    graph_ptr = [0]
    node_graph_index = []

    type_offset_totals = {node_type: 0 for node_type in node_features.keys()}
    node_offset = 0

    for graph_idx, sample in enumerate(batch):
        for node_type, feats in sample.node_features.items():
            node_features[node_type].append(feats)

        for node_type, local_idx in sample.type_assignments:
            adjusted_idx = local_idx + type_offset_totals[node_type]
            type_assignments.append((node_type, adjusted_idx))
            node_graph_index.append(graph_idx)

        for node_type in type_offset_totals.keys():
            type_offset_totals[node_type] += sample.node_features[node_type].size(0)

        edge_features.append(sample.edge_features)
        edge_types.append(sample.edge_types)
        edge_indices.append(sample.edge_index + node_offset)

        node_count = sample.num_nodes
        node_offset += node_count
        graph_ptr.append(node_offset)

        for key in targets_accum.keys():
            targets_accum[key].append(sample.targets[key])

    stacked_node_features = {
        node_type: torch.cat(feat_list, dim=0)
        for node_type, feat_list in node_features.items()
    }

    batched_targets = {
        key: torch.stack(values, dim=0)
        for key, values in targets_accum.items()
    }

    return GraphBatch(
        node_features=stacked_node_features,
        type_assignments=type_assignments,
        edge_features=torch.cat(edge_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_types=torch.cat(edge_types, dim=0),
        targets=batched_targets,
        graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
        node_graph_index=torch.tensor(node_graph_index, dtype=torch.long),
        sample_names=sample_names,
        sample_ids=sample_ids,
    )
