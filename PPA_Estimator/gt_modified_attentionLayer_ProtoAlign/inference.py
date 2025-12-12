"""
Inference script for Graph Transformer
"""
import os
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse

import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.serialization import add_safe_globals

try:
    from .models import GraphTransformerModel
    from .datasets import GraphBatch, GraphSample, LibertyGraphDataset, TargetDict, TargetNormalizer, collate_graphs
except ImportError:
    from models import GraphTransformerModel
    from datasets import GraphBatch, GraphSample, LibertyGraphDataset, TargetDict, TargetNormalizer, collate_graphs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _tensor_to_python(value: torch.Tensor):
    if value.ndim == 0:
        return float(value.item())
    return value.tolist()


TASK_KEYS = [
    "leakage",
    "internal_power",
    "cell_rise",
    "cell_fall",
    "rise_transition",
    "fall_transition",
]


def _looks_like_state_dict(obj: Any) -> bool:
    """Return True when the object behaves like a PyTorch state_dict."""

    if not isinstance(obj, Mapping):
        return False
    return any(torch.is_tensor(value) or isinstance(value, nn.Parameter) for value in obj.values())


def _extract_state_dict(state: Any) -> Mapping[str, torch.Tensor]:
    """Best-effort extraction of the model weights from different checkpoint formats."""

    if isinstance(state, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = state.get(key)
            if _looks_like_state_dict(candidate):
                return candidate  # type: ignore[return-value]
        if _looks_like_state_dict(state):
            return state  # type: ignore[return-value]

    if hasattr(state, "state_dict"):
        candidate = state.state_dict()
        if _looks_like_state_dict(candidate):
            return candidate  # type: ignore[return-value]

    raise KeyError(
        "Checkpoint missing model state dict. Expected keys 'model', 'model_state_dict', 'state_dict', "
        "or a raw state_dict/object with .state_dict()."
    )


def load_model(checkpoint: Path, device: str = "cpu") -> GraphTransformerModel:
    add_safe_globals([Path])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(state)

    model = GraphTransformerModel()

    log_task_vars = nn.Parameter(torch.zeros(len(TASK_KEYS), device=device))
    model.register_parameter("log_task_vars", log_task_vars)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class Predictor:
    """Predictor class for inference"""
    
    def __init__(
        self,
        model: GraphTransformerModel,
        device: str = 'cpu'
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained Graph Transformer model
            device: Device to use for inference
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_single(self, sample: GraphSample) -> Dict[str, torch.Tensor]:
        """
        Make prediction for a single graph.

        Args:
            sample: GraphSample containing node/edge features and indices

        Returns:
            Dictionary of predictions as numpy arrays
        """
        batch = collate_graphs([sample]).to(self.device)

        with torch.no_grad():
            predictions = self.model(batch)

        return {task: pred.detach().cpu().squeeze(0) for task, pred in predictions.items()}

    def predict_dataset(
        self,
        dataset: LibertyGraphDataset,
        output_dir: Optional[Path] = None,
        target_normalizer: Optional[TargetNormalizer] = None,
        denormalize_output: bool = False,
    ) -> List[Dict]:
        """
        Make predictions for entire dataset.
        
        Args:
            dataset: Dataset to predict
            output_dir: Optional directory to save predictions
        
        Returns:
            List of prediction dictionaries
        """
        all_predictions: List[Dict] = []
        skipped_samples: List[str] = []

        logger.info(f"Making predictions for {len(dataset)} samples...")

        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]

            edge_shape = tuple(sample.edge_features.shape)
            if sample.edge_features.numel() == 0 or (len(edge_shape) > 1 and edge_shape[-1] == 0):
                logger.error(
                    "Sample '%s' has empty edge features (shape=%s). Skipping prediction.",
                    sample.name,
                    edge_shape,
                )
                skipped_samples.append(sample.name)
                continue

            try:
                predictions = self.predict_single(sample)
            except RuntimeError as exc:
                logger.error("Failed to process sample %s: %s", sample.name, exc)
                continue

            if denormalize_output and target_normalizer is not None:
                predictions = target_normalizer.denormalize(predictions)

            pred_scalar = float(predictions['leakage'].view(-1).item())
            pred_vector = _tensor_to_python(predictions['internal_power'])
            pred_cell_rise = _tensor_to_python(predictions['cell_rise'])
            pred_cell_fall = _tensor_to_python(predictions['cell_fall'])
            pred_rise_transition = _tensor_to_python(predictions['rise_transition'])
            pred_fall_transition = _tensor_to_python(predictions['fall_transition'])
            
            # Format results
            embed_path = dataset.embedding_paths[idx]
            result = {
                'file': embed_path.name,
                'predictions': {
                    'scalar': {
                        'leakage': pred_scalar
                    },
                    'vector': {
                        'internal_power': pred_vector
                    },
                    'matrices': {
                        'cell_rise': pred_cell_rise,
                        'cell_fall': pred_cell_fall,
                        'rise_transition': pred_rise_transition,
                        'fall_transition': pred_fall_transition
                    }
                }
            }
            
            # Add ground truth if available
            targets = sample.targets
            if targets:
                target_tensors: TargetDict = {
                    key: value.detach().cpu() for key, value in targets.items()
                }

                if denormalize_output and target_normalizer is not None:
                    target_tensors = target_normalizer.denormalize(target_tensors)

                result['ground_truth'] = {
                    'scalar': {
                        'leakage': float(target_tensors['leakage'].view(-1).item())
                    },
                    'vector': {
                        'internal_power': _tensor_to_python(target_tensors['internal_power'])
                    },
                    'matrices': {
                        'cell_rise': _tensor_to_python(target_tensors['cell_rise']),
                        'cell_fall': _tensor_to_python(target_tensors['cell_fall']),
                        'rise_transition': _tensor_to_python(target_tensors['rise_transition']),
                        'fall_transition': _tensor_to_python(target_tensors['fall_transition'])
                    }
                }
            
            all_predictions.append(result)
            
            # Save individual prediction if output directory specified
            if output_dir:
                output_file = output_dir / f"{embed_path.stem}_pred.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        if skipped_samples:
            logger.warning(
                "Skipped %d samples due to missing edge features: %s",
                len(skipped_samples),
                ", ".join(skipped_samples),
            )

        return all_predictions


def compute_metrics(predictions: List[Dict]) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries with ground truth
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': {},
        'mae': {},
        'mape': {},
        'std': {},
        'r2': {}
    }
    
    # Collect errors for each task
    errors = {
        'leakage': [],
        'internal_power': [],
        'cell_rise': [],
        'cell_fall': [],
        'rise_transition': [],
        'fall_transition': []
    }
    pred_gt_values = {
        task: {'pred': [], 'gt': []}
        for task in errors.keys()
    }
    
    for pred_dict in predictions:
        if 'ground_truth' not in pred_dict:
            continue
        
        pred = pred_dict['predictions']
        gt = pred_dict['ground_truth']
        
        # Scalar metrics
        if 'scalar' in pred and 'scalar' in gt:
            pred_leak = pred['scalar']['leakage']
            gt_leak = gt['scalar']['leakage']
            errors['leakage'].append({
                'mse': (pred_leak - gt_leak) ** 2,
                'mae': abs(pred_leak - gt_leak),
                'mape': abs((pred_leak - gt_leak) / (gt_leak + 1e-8))
            })
            pred_gt_values['leakage']['pred'].append(float(pred_leak))
            pred_gt_values['leakage']['gt'].append(float(gt_leak))
        
        # Vector metrics
        if 'vector' in pred and 'vector' in gt:
            pred_int = np.array(pred['vector']['internal_power'])
            gt_int = np.array(gt['vector']['internal_power'])
            errors['internal_power'].append({
                'mse': np.mean((pred_int - gt_int) ** 2),
                'mae': np.mean(np.abs(pred_int - gt_int)),
                'mape': np.mean(np.abs((pred_int - gt_int) / (gt_int + 1e-8)))
            })
            pred_gt_values['internal_power']['pred'].extend(pred_int.reshape(-1).tolist())
            pred_gt_values['internal_power']['gt'].extend(gt_int.reshape(-1).tolist())
        
        # Matrix metrics
        if 'matrices' in pred and 'matrices' in gt:
            for task in ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']:
                if task in pred['matrices'] and task in gt['matrices']:
                    pred_mat = np.array(pred['matrices'][task])
                    gt_mat = np.array(gt['matrices'][task])
                    errors[task].append({
                        'mse': np.mean((pred_mat - gt_mat) ** 2),
                        'mae': np.mean(np.abs(pred_mat - gt_mat)),
                        'mape': np.mean(np.abs((pred_mat - gt_mat) / (gt_mat + 1e-8)))
                    })
                    pred_gt_values[task]['pred'].extend(pred_mat.reshape(-1).tolist())
                    pred_gt_values[task]['gt'].extend(gt_mat.reshape(-1).tolist())
    
    # Aggregate metrics
    for task, task_errors in errors.items():
        if task_errors:
            metrics['mse'][task] = np.mean([e['mse'] for e in task_errors])
            metrics['mae'][task] = np.mean([e['mae'] for e in task_errors])
            metrics['mape'][task] = np.mean([e['mape'] for e in task_errors])

    # Compute std and R^2 using per-element predictions/targets
    for task, values in pred_gt_values.items():
        if not values['pred'] or not values['gt']:
            continue
        preds = np.array(values['pred'], dtype=np.float64)
        targets = np.array(values['gt'], dtype=np.float64)
        residuals = preds - targets
        metrics['std'][task] = float(np.std(residuals))
        denom = np.sum((targets - targets.mean()) ** 2)
        if denom < 1e-12:
            metrics['r2'][task] = float(1.0 if np.allclose(residuals, 0.0) else 0.0)
        else:
            metrics['r2'][task] = float(1 - np.sum(residuals ** 2) / denom)

    return metrics


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Inference with Graph Transformer')
    
    # Data arguments
    parser.add_argument('--emb_dir', type=Path, required=True,
                        help='Path to embeddings directory')
    parser.add_argument('--target_dir', type=Path, required=True,
                        help='Path to targets directory')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use for inference')
    
    # Output arguments
    parser.add_argument('--output_dir', type=Path, default=Path('predictions'),
                        help='Directory to save predictions')
    parser.add_argument('--save_metrics', action='store_true',
                        help='Save evaluation metrics')
    parser.add_argument('--target-stats', type=Path, default=None,
                        help='Path to target normalization statistics JSON')
    parser.add_argument('--no-target-normalization', action='store_true',
                        help='Disable applying normalization when statistics are provided')
    parser.add_argument('--denormalize-output', action='store_true',
                        help='Convert predictions (and ground truth if available) back to original scale when stats are provided')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    logger.info("Loading dataset...")
    normalize_targets = args.target_stats is not None and not args.no_target_normalization
    dataset = LibertyGraphDataset(
        args.emb_dir,
        args.target_dir,
        target_stats_path=args.target_stats,
        normalize_targets=normalize_targets,
    )
    target_normalizer = dataset.target_normalizer if args.target_stats is not None else None

    denormalize_output = args.denormalize_output
    if denormalize_output and target_normalizer is None:
        logger.warning(
            "--denormalize-output requested but --target-stats not provided; skipping denormalization"
        )
        denormalize_output = False
    elif denormalize_output and not normalize_targets:
        logger.warning(
            "--denormalize-output requested but target normalization is disabled; outputs already in the original scale"
        )
        denormalize_output = False
    elif not denormalize_output and normalize_targets:
        logger.info(
            "Target normalization (including log1p for leakage/internal_power when present) is enabled; predictions will remain in the normalized domain unless --denormalize-output is set"
        )

    # Create model
    logger.info("Loading model...")
    model = load_model(args.checkpoint, device=args.device)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create predictor
    predictor = Predictor(model, args.device)
    
    # Make predictions
    predictions = predictor.predict_dataset(
        dataset,
        output_dir,
        target_normalizer=target_normalizer,
        denormalize_output=denormalize_output,
    )
    
    # Save all predictions
    all_predictions_file = output_dir / 'all_predictions.json'
    with open(all_predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Saved all predictions to {all_predictions_file}")
    
    # Compute and save metrics if ground truth availaxble
    if args.target_dir and args.save_metrics:
        logger.info("Computing evaluation metrics...")
        metrics = compute_metrics(predictions)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print("=" * 50)
        for metric_type in ['mse', 'mae', 'mape', 'std', 'r2']:
            print(f"\n{metric_type.upper()}:")
            for task, value in metrics[metric_type].items():
                print(f"  {task}: {value:.6f}")
        
        # Save metrics
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()
