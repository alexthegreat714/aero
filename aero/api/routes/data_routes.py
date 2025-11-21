"""
Data Lake API routes for Aero Agent.

Provides REST API endpoints for querying and managing simulation
and experiment results in the data lake.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from aero.data import (
    DataStore,
    get_default_store,
    list_recent_simulations,
    list_recent_experiments,
    find_simulations_by_type,
    find_experiments_by_type,
    find_simulations_by_tag,
    find_experiments_by_tag,
    find_by_hypothesis,
    get_simulation_with_timeseries,
    get_experiment_with_timeseries,
    count_by_type,
    search_metadata,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class SimulationQueryParams(BaseModel):
    """Parameters for querying simulations."""

    sim_type: Optional[str] = Field(None, description="Filter by simulation type")
    tag: Optional[str] = Field(None, description="Filter by tag")
    hypothesis_id: Optional[str] = Field(None, description="Filter by hypothesis ID")
    limit: int = Field(20, ge=1, le=100, description="Maximum results to return")


class ExperimentQueryParams(BaseModel):
    """Parameters for querying experiments."""

    exp_type: Optional[str] = Field(None, description="Filter by experiment type")
    tag: Optional[str] = Field(None, description="Filter by tag")
    hypothesis_id: Optional[str] = Field(None, description="Filter by hypothesis ID")
    limit: int = Field(20, ge=1, le=100, description="Maximum results to return")


class MetadataSearchParams(BaseModel):
    """Parameters for metadata search."""

    key: str = Field(..., description="Metadata key to search")
    value: str = Field(..., description="Value to search for (partial match)")
    table: str = Field("simulations", description="Table to search: simulations or experiments")
    limit: int = Field(20, ge=1, le=100, description="Maximum results to return")


class CountResponse(BaseModel):
    """Response for count queries."""

    table: str
    counts: Dict[str, int]


class StatsResponse(BaseModel):
    """Response for database statistics."""

    simulation_count: int
    experiment_count: int
    timeseries_count: int
    field_count: int
    type_breakdown: Dict[str, Dict[str, int]]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.get("/simulations", response_model=List[Dict[str, Any]])
async def get_simulations(
    sim_type: Optional[str] = Query(None, description="Filter by simulation type"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    hypothesis_id: Optional[str] = Query(None, description="Filter by hypothesis ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """
    List simulations with optional filtering.

    Returns recent simulations, optionally filtered by type, tag, or hypothesis.
    """
    try:
        store = get_default_store()

        if hypothesis_id:
            results = find_by_hypothesis(store, hypothesis_id)
            return results.get("simulations", [])[:limit]
        elif tag:
            return find_simulations_by_tag(store, tag, limit=limit)
        elif sim_type:
            return find_simulations_by_type(store, sim_type, limit=limit)
        else:
            return list_recent_simulations(store, limit=limit)

    except Exception as e:
        logger.error(f"Error querying simulations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulations/{sim_id}")
async def get_simulation(
    sim_id: str,
    include_timeseries: bool = Query(False, description="Include associated timeseries data"),
):
    """
    Get a specific simulation by ID.

    Optionally includes associated timeseries data.
    """
    try:
        store = get_default_store()

        if include_timeseries:
            result = get_simulation_with_timeseries(store, sim_id)
        else:
            result = store.get_simulation(sim_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation {sim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/simulations/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a simulation by ID."""
    try:
        store = get_default_store()
        success = store.delete_simulation(sim_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")

        return {"status": "deleted", "id": sim_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting simulation {sim_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def get_experiments(
    exp_type: Optional[str] = Query(None, description="Filter by experiment type"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    hypothesis_id: Optional[str] = Query(None, description="Filter by hypothesis ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """
    List experiments with optional filtering.

    Returns recent experiments, optionally filtered by type, tag, or hypothesis.
    """
    try:
        store = get_default_store()

        if hypothesis_id:
            results = find_by_hypothesis(store, hypothesis_id)
            return results.get("experiments", [])[:limit]
        elif tag:
            return find_experiments_by_tag(store, tag, limit=limit)
        elif exp_type:
            return find_experiments_by_type(store, exp_type, limit=limit)
        else:
            return list_recent_experiments(store, limit=limit)

    except Exception as e:
        logger.error(f"Error querying experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{exp_id}")
async def get_experiment(
    exp_id: str,
    include_timeseries: bool = Query(False, description="Include associated timeseries data"),
):
    """
    Get a specific experiment by ID.

    Optionally includes associated timeseries data.
    """
    try:
        store = get_default_store()

        if include_timeseries:
            result = get_experiment_with_timeseries(store, exp_id)
        else:
            result = store.get_experiment(exp_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment {exp_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/experiments/{exp_id}")
async def delete_experiment(exp_id: str):
    """Delete an experiment by ID."""
    try:
        store = get_default_store()
        success = store.delete_experiment(exp_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")

        return {"status": "deleted", "id": exp_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment {exp_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeseries/{parent_id}")
async def get_timeseries(
    parent_id: str,
    parent_type: str = Query("simulation", description="Parent type: simulation or experiment"),
):
    """
    Get all timeseries data for a simulation or experiment.
    """
    try:
        store = get_default_store()
        results = store.list_timeseries(parent_id, parent_type)

        if not results:
            return []

        return results

    except Exception as e:
        logger.error(f"Error getting timeseries for {parent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fields/{parent_id}")
async def get_fields(
    parent_id: str,
    parent_type: str = Query("simulation", description="Parent type: simulation or experiment"),
):
    """
    Get metadata for all field arrays associated with a simulation or experiment.
    """
    try:
        store = get_default_store()
        results = store.list_fields(parent_id, parent_type)

        if not results:
            return []

        return results

    except Exception as e:
        logger.error(f"Error getting fields for {parent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fields/{parent_id}/{field_name}")
async def get_field_array(
    parent_id: str,
    field_name: str,
    parent_type: str = Query("simulation", description="Parent type: simulation or experiment"),
):
    """
    Get a specific field array by name.

    Note: Returns field metadata and file path. For large arrays,
    load the .npz file directly for better performance.
    """
    try:
        store = get_default_store()

        # Find the field
        fields = store.list_fields(parent_id, parent_type)
        field_info = None
        for f in fields:
            if f.get("name") == field_name:
                field_info = f
                break

        if not field_info:
            raise HTTPException(
                status_code=404,
                detail=f"Field '{field_name}' not found for {parent_type} {parent_id}",
            )

        # Load the array
        array = store.load_field_array(field_info["id"])
        if array is None:
            raise HTTPException(status_code=404, detail="Field array file not found")

        return {
            "id": field_info["id"],
            "name": field_name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "data": array.tolist() if array.size < 10000 else None,
            "file_path": field_info.get("file_path"),
            "note": "For large arrays, access the file directly" if array.size >= 10000 else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field {field_name} for {parent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search(
    key: str = Query(..., description="Metadata key to search"),
    value: str = Query(..., description="Value to search for"),
    table: str = Query("simulations", description="Table: simulations or experiments"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """
    Search for records by metadata key/value.

    Performs partial matching on the specified metadata field.
    """
    try:
        store = get_default_store()
        results = search_metadata(store, key, value, table=table, limit=limit)
        return results

    except Exception as e:
        logger.error(f"Error searching metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database statistics.

    Returns counts for each table and type breakdowns.
    """
    try:
        store = get_default_store()

        # Get counts
        sim_counts = count_by_type(store, "simulations")
        exp_counts = count_by_type(store, "experiments")

        # Get total counts from store
        stats = store.get_stats()

        return StatsResponse(
            simulation_count=stats.get("simulations", 0),
            experiment_count=stats.get("experiments", 0),
            timeseries_count=stats.get("timeseries", 0),
            field_count=stats.get("fields", 0),
            type_breakdown={
                "simulations": sim_counts,
                "experiments": exp_counts,
            },
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hypothesis/{hypothesis_id}")
async def get_by_hypothesis(hypothesis_id: str):
    """
    Get all simulations and experiments associated with a hypothesis.

    Returns both simulations and experiments linked to the hypothesis.
    """
    try:
        store = get_default_store()
        results = find_by_hypothesis(store, hypothesis_id)
        return results

    except Exception as e:
        logger.error(f"Error getting hypothesis {hypothesis_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
