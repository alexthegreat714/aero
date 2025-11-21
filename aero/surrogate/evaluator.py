"""
Surrogate Model Evaluator for Aero Agent.

Provides evaluation metrics and prediction utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def evaluate_model(
    model: "nn.Module",
    dataset: "Dataset",
    device: str = "cpu",
    batch_size: int = 64,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.

    Computes MSE, MAE, max error, and R² metrics.

    Args:
        model: PyTorch model to evaluate
        dataset: Dataset to evaluate on
        device: Device for computation
        batch_size: Batch size for evaluation
        max_batches: Maximum batches to evaluate (None = all)

    Returns:
        Dictionary with metrics: mse, mae, max_err, r2, num_samples
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")

    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    if not all_preds:
        return {
            "mse": None,
            "mae": None,
            "max_err": None,
            "r2": None,
            "num_samples": 0,
        }

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    errors = preds - targets
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))

    # R² (coefficient of determination)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "max_err": max_err,
        "r2": r2,
        "num_samples": len(preds),
    }


def predict_single(
    model: "nn.Module",
    x: Union[np.ndarray, "torch.Tensor", List],
    device: str = "cpu",
    normalization: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Make a prediction for a single input.

    Args:
        model: PyTorch model
        x: Input array/tensor of shape (input_dim,) or (1, input_dim)
        device: Device for computation
        normalization: Optional normalization parameters

    Returns:
        Output array of shape (output_dim,)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")

    model = model.to(device)
    model.eval()

    # Convert to tensor
    if isinstance(x, list):
        x = np.array(x, dtype=np.float32)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))

    # Ensure 2D
    if x.dim() == 1:
        x = x.unsqueeze(0)

    # Apply input normalization
    if normalization:
        input_mean = normalization.get("input_mean")
        input_std = normalization.get("input_std")
        if input_mean is not None and input_std is not None:
            input_mean = torch.tensor(input_mean, dtype=torch.float32)
            input_std = torch.tensor(input_std, dtype=torch.float32)
            x = (x - input_mean) / input_std

    x = x.to(device)

    with torch.no_grad():
        pred = model(x)

    result = pred.cpu().numpy().squeeze()

    # Apply output denormalization
    if normalization:
        output_mean = normalization.get("output_mean")
        output_std = normalization.get("output_std")
        if output_mean is not None and output_std is not None:
            output_mean = np.array(output_mean)
            output_std = np.array(output_std)
            result = result * output_std + output_mean

    return result


def predict_batch(
    model: "nn.Module",
    inputs: Union[np.ndarray, "torch.Tensor", List[List]],
    device: str = "cpu",
    normalization: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Make predictions for a batch of inputs.

    Args:
        model: PyTorch model
        inputs: Input array of shape (N, input_dim)
        device: Device for computation
        normalization: Optional normalization parameters
        batch_size: Batch size for processing

    Returns:
        Output array of shape (N, output_dim)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")

    model = model.to(device)
    model.eval()

    # Convert to tensor
    if isinstance(inputs, list):
        inputs = np.array(inputs, dtype=np.float32)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs.astype(np.float32))

    # Apply input normalization
    if normalization:
        input_mean = normalization.get("input_mean")
        input_std = normalization.get("input_std")
        if input_mean is not None and input_std is not None:
            input_mean = torch.tensor(input_mean, dtype=torch.float32)
            input_std = torch.tensor(input_std, dtype=torch.float32)
            inputs = (inputs - input_mean) / input_std

    # Process in batches
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size].to(device)
            preds = model(batch)
            all_preds.append(preds.cpu().numpy())

    result = np.concatenate(all_preds, axis=0)

    # Apply output denormalization
    if normalization:
        output_mean = normalization.get("output_mean")
        output_std = normalization.get("output_std")
        if output_mean is not None and output_std is not None:
            output_mean = np.array(output_mean)
            output_std = np.array(output_std)
            result = result * output_std + output_mean

    return result


def compare_with_ground_truth(
    model: "nn.Module",
    inputs: np.ndarray,
    ground_truth: np.ndarray,
    device: str = "cpu",
    normalization: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare model predictions with ground truth.

    Args:
        model: PyTorch model
        inputs: Input array (N, input_dim)
        ground_truth: Ground truth outputs (N, output_dim)
        device: Device for computation
        normalization: Normalization parameters

    Returns:
        Dictionary with comparison metrics and predictions
    """
    predictions = predict_batch(model, inputs, device, normalization)

    errors = predictions - ground_truth
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))

    # Per-sample errors
    sample_mse = np.mean(errors ** 2, axis=-1) if errors.ndim > 1 else errors ** 2

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "max_err": max_err,
        "r2": r2,
        "num_samples": len(inputs),
        "predictions": predictions,
        "errors": errors,
        "sample_mse": sample_mse,
    }


def model_summary(model: "nn.Module") -> Dict[str, Any]:
    """
    Get a summary of model architecture.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get layer info
    layers = []
    for name, module in model.named_modules():
        if name:  # Skip the root module
            layer_info = {
                "name": name,
                "type": type(module).__name__,
            }
            if hasattr(module, "in_features"):
                layer_info["in_features"] = module.in_features
            if hasattr(module, "out_features"):
                layer_info["out_features"] = module.out_features
            layers.append(layer_info)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layers": layers,
        "model_class": type(model).__name__,
    }


def estimate_inference_time(
    model: "nn.Module",
    input_dim: int,
    device: str = "cpu",
    num_runs: int = 100,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Estimate model inference time.

    Args:
        model: PyTorch model
        input_dim: Input dimension
        device: Device for computation
        num_runs: Number of inference runs
        batch_size: Batch size for testing

    Returns:
        Dictionary with timing statistics (in milliseconds)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")

    import time

    model = model.to(device)
    model.eval()

    # Warmup
    dummy_input = torch.randn(batch_size, input_dim).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Time inference
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "batch_size": batch_size,
        "device": device,
    }
