"""
Surrogate Model API routes for Aero Agent.

Provides REST API endpoints for training, managing, and using
surrogate models.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Request body for training a surrogate model."""

    name: str = Field(..., description="Model name")
    sim_type: str = Field(..., description="Simulation type to train on")
    model_type: str = Field("mlp", description="Model type: mlp or pinn")
    hidden_dims: List[int] = Field([64, 64], description="Hidden layer dimensions")
    epochs: int = Field(100, ge=1, le=10000, description="Training epochs")
    batch_size: int = Field(64, ge=1, le=1024, description="Batch size")
    lr: float = Field(0.001, gt=0, description="Learning rate")
    max_records: int = Field(50, ge=1, le=500, description="Max simulation records to use")
    num_samples: int = Field(1000, ge=100, description="Samples per simulation")
    activation: str = Field("relu", description="Activation function")


class TrainResponse(BaseModel):
    """Response from training a model."""

    name: str
    model_type: str
    input_dim: int
    output_dim: int
    path: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]


class PredictRequest(BaseModel):
    """Request body for predictions."""

    name: str = Field(..., description="Model name")
    inputs: List[List[float]] = Field(..., description="Input vectors")


class PredictResponse(BaseModel):
    """Response from predictions."""

    name: str
    predictions: List[List[float]]
    num_inputs: int


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    model_type: str
    input_dim: int
    output_dim: int
    path: str
    metadata: Dict[str, Any]


class ModelListResponse(BaseModel):
    """List of models response."""

    models: List[ModelInfo]
    count: int


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _get_registry():
    """Get the surrogate registry."""
    try:
        from aero.surrogate import get_default_registry
        return get_default_registry()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registry not initialized: {e}")


def _get_trainer():
    """Get a surrogate trainer."""
    from aero.surrogate import SurrogateTrainer
    registry = _get_registry()
    return SurrogateTrainer(registry)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """
    List all registered surrogate models.

    Returns list of model info including name, type, dimensions, and metadata.
    """
    try:
        registry = _get_registry()
        models = registry.list_models()

        return ModelListResponse(
            models=[
                ModelInfo(
                    name=m.name,
                    model_type=m.model_type,
                    input_dim=m.input_dim,
                    output_dim=m.output_dim,
                    path=m.path,
                    metadata=m.metadata,
                )
                for m in models
            ],
            count=len(models),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{name}", response_model=ModelInfo)
async def get_model(name: str):
    """
    Get information about a specific model.

    Args:
        name: Model name

    Returns:
        Model information including metadata
    """
    try:
        registry = _get_registry()
        info = registry.get(name)

        if info is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        return ModelInfo(
            name=info.name,
            model_type=info.model_type,
            input_dim=info.input_dim,
            output_dim=info.output_dim,
            path=info.path,
            metadata=info.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{name}")
async def delete_model(name: str):
    """
    Delete a model from the registry.

    Note: This removes the model from tracking but does not delete the file.
    """
    try:
        registry = _get_registry()
        success = registry.remove(name)

        if not success:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        return {"status": "deleted", "name": name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Train a new surrogate model.

    Trains on simulation data from the DataStore for the specified
    simulation type. Returns trained model info and metrics.
    """
    try:
        from aero.surrogate import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PyTorch not available. Install pytorch to train models."
            )

        from aero.data import get_default_store
        from aero.surrogate.datasets import build_dataset_from_simulations

        # Get data store
        try:
            store = get_default_store()
        except Exception:
            raise HTTPException(status_code=503, detail="DataStore not initialized")

        # Build dataset
        dataset = build_dataset_from_simulations(
            store=store,
            sim_type=request.sim_type,
            num_samples=request.num_samples,
            max_records=request.max_records,
        )

        if len(dataset) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No data found for simulation type '{request.sim_type}'"
            )

        # Determine dimensions from dataset
        sample_x, sample_y = dataset[0]
        input_dim = len(sample_x)
        output_dim = len(sample_y) if hasattr(sample_y, '__len__') else 1

        # Train model
        trainer = _get_trainer()

        metadata = {
            "sim_type": request.sim_type,
            "num_samples": request.num_samples,
            "max_records": request.max_records,
        }

        if request.model_type.lower() == "mlp":
            info = trainer.train_mlp(
                name=request.name,
                dataset=dataset,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=request.hidden_dims,
                epochs=request.epochs,
                batch_size=request.batch_size,
                lr=request.lr,
                activation=request.activation,
                metadata=metadata,
            )
        elif request.model_type.lower() == "pinn":
            info = trainer.train_pinn(
                name=request.name,
                dataset=dataset,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=request.hidden_dims,
                epochs=request.epochs,
                batch_size=request.batch_size,
                lr=request.lr,
                activation=request.activation if request.activation != "relu" else "tanh",
                metadata=metadata,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model type: {request.model_type}"
            )

        return TrainResponse(
            name=info.name,
            model_type=info.model_type,
            input_dim=info.input_dim,
            output_dim=info.output_dim,
            path=info.path,
            metrics={
                "final_train_loss": info.metadata.get("final_train_loss"),
                "final_val_loss": info.metadata.get("final_val_loss"),
                "best_val_loss": info.metadata.get("best_val_loss"),
                "training_time_seconds": info.metadata.get("training_time_seconds"),
            },
            metadata=info.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make predictions using a trained surrogate model.

    Args:
        name: Model name
        inputs: List of input vectors

    Returns:
        List of prediction vectors
    """
    try:
        from aero.surrogate import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available")

        import torch
        from pathlib import Path
        import numpy as np
        from aero.surrogate import predict_batch
        from aero.surrogate.models import create_model

        registry = _get_registry()
        info = registry.get(request.name)

        if info is None:
            raise HTTPException(status_code=404, detail=f"Model '{request.name}' not found")

        # Load model
        model_path = Path(info.path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found")

        checkpoint = torch.load(model_path, map_location="cpu")

        model = create_model(
            model_type=checkpoint["model_type"],
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_dims=checkpoint["hidden_dims"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Get normalization
        normalization = checkpoint.get("normalization")

        # Make predictions
        inputs = np.array(request.inputs, dtype=np.float32)
        predictions = predict_batch(model, inputs, device="cpu", normalization=normalization)

        # Convert to list
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        return PredictResponse(
            name=request.name,
            predictions=predictions.tolist(),
            num_inputs=len(request.inputs),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{name}/evaluate")
async def evaluate_model_endpoint(
    name: str,
    sim_type: Optional[str] = Query(None, description="Simulation type for evaluation data"),
    max_samples: int = Query(500, ge=10, le=5000, description="Max samples to evaluate"),
):
    """
    Evaluate a model on held-out data.

    Args:
        name: Model name
        sim_type: Simulation type (defaults to model's training type)
        max_samples: Maximum samples to use

    Returns:
        Evaluation metrics
    """
    try:
        from aero.surrogate import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            raise HTTPException(status_code=503, detail="PyTorch not available")

        import torch
        from pathlib import Path
        from aero.data import get_default_store
        from aero.surrogate.datasets import build_dataset_from_simulations
        from aero.surrogate.evaluator import evaluate_model
        from aero.surrogate.models import create_model

        registry = _get_registry()
        info = registry.get(name)

        if info is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        # Get simulation type
        if sim_type is None:
            sim_type = info.metadata.get("sim_type")
        if sim_type is None:
            raise HTTPException(
                status_code=400,
                detail="sim_type required for evaluation"
            )

        # Load model
        model_path = Path(info.path)
        checkpoint = torch.load(model_path, map_location="cpu")

        model = create_model(
            model_type=checkpoint["model_type"],
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            hidden_dims=checkpoint["hidden_dims"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Build evaluation dataset
        store = get_default_store()
        dataset = build_dataset_from_simulations(
            store=store,
            sim_type=sim_type,
            num_samples=max_samples,
            max_records=20,
            normalize=True,  # Same as training
        )

        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="No evaluation data found")

        # Evaluate
        metrics = evaluate_model(model, dataset, device="cpu")

        return {
            "name": name,
            "sim_type": sim_type,
            "metrics": metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def list_available_types():
    """
    List simulation types that have data available for training.

    Returns types with enough data to train surrogates.
    """
    try:
        from aero.data import get_default_store, count_by_type

        store = get_default_store()
        counts = count_by_type(store, "simulations")

        # Filter types with enough data
        min_records = 1
        available = {k: v for k, v in counts.items() if v >= min_records}

        return {
            "available_types": list(available.keys()),
            "counts": available,
        }

    except Exception as e:
        logger.error(f"Error listing types: {e}")
        raise HTTPException(status_code=500, detail=str(e))
