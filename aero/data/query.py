"""
Query utilities for Aero Agent data lake.

Provides high-level functions for querying simulations,
experiments, and other data from the store.
"""

import json
from typing import Any, Dict, List, Optional

from aero.data.store import DataStore


def list_recent_simulations(
    store: DataStore,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    List most recent simulations.

    Args:
        store: DataStore instance
        limit: Maximum number of results

    Returns:
        List of simulation records
    """
    return store.list_simulations(limit=limit)


def list_recent_experiments(
    store: DataStore,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    List most recent experiments.

    Args:
        store: DataStore instance
        limit: Maximum number of results

    Returns:
        List of experiment records
    """
    return store.list_experiments(limit=limit)


def find_simulations_by_type(
    store: DataStore,
    sim_type: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Find simulations by type.

    Args:
        store: DataStore instance
        sim_type: Simulation type (e.g., "heat_1d", "laplace_2d")
        limit: Maximum number of results

    Returns:
        List of matching simulation records
    """
    return store.list_simulations(limit=limit, filters={"type": sim_type})


def find_experiments_by_type(
    store: DataStore,
    exp_type: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Find experiments by type.

    Args:
        store: DataStore instance
        exp_type: Experiment type (e.g., "synthetic_flow")
        limit: Maximum number of results

    Returns:
        List of matching experiment records
    """
    return store.list_experiments(limit=limit, filters={"type": exp_type})


def find_simulations_by_tag(
    store: DataStore,
    tag: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Find simulations containing a specific tag.

    Args:
        store: DataStore instance
        tag: Tag to search for
        limit: Maximum number of results

    Returns:
        List of matching simulation records
    """
    # Get all simulations and filter by tag
    all_sims = store.list_simulations(limit=limit * 2)  # Get extra for filtering

    matching = []
    for sim in all_sims:
        tags = sim.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []

        if tag in tags:
            matching.append(sim)
            if len(matching) >= limit:
                break

    return matching


def find_experiments_by_tag(
    store: DataStore,
    tag: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Find experiments containing a specific tag.

    Args:
        store: DataStore instance
        tag: Tag to search for
        limit: Maximum number of results

    Returns:
        List of matching experiment records
    """
    # Get all experiments and filter by tag
    all_exps = store.list_experiments(limit=limit * 2)

    matching = []
    for exp in all_exps:
        tags = exp.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []

        if tag in tags:
            matching.append(exp)
            if len(matching) >= limit:
                break

    return matching


def find_by_hypothesis(
    store: DataStore,
    hypothesis_id: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find all simulations and experiments for a hypothesis.

    Args:
        store: DataStore instance
        hypothesis_id: Hypothesis ID to search for

    Returns:
        Dictionary with "simulations" and "experiments" lists
    """
    simulations = store.list_simulations(
        limit=100,
        filters={"hypothesis_id": hypothesis_id}
    )

    experiments = store.list_experiments(
        limit=100,
        filters={"hypothesis_id": hypothesis_id}
    )

    return {
        "simulations": simulations,
        "experiments": experiments,
    }


def get_simulation_with_timeseries(
    store: DataStore,
    sim_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a simulation with its associated timeseries data.

    Args:
        store: DataStore instance
        sim_id: Simulation ID

    Returns:
        Simulation record with timeseries data included
    """
    sim = store.get_simulation(sim_id)
    if sim is None:
        return None

    sim["timeseries"] = store.list_timeseries_for_parent(sim_id, "simulation")
    return sim


def get_experiment_with_timeseries(
    store: DataStore,
    exp_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get an experiment with its associated timeseries data.

    Args:
        store: DataStore instance
        exp_id: Experiment ID

    Returns:
        Experiment record with timeseries data included
    """
    exp = store.get_experiment(exp_id)
    if exp is None:
        return None

    exp["timeseries"] = store.list_timeseries_for_parent(exp_id, "experiment")
    return exp


def count_by_type(store: DataStore) -> Dict[str, Dict[str, int]]:
    """
    Count simulations and experiments by type.

    Args:
        store: DataStore instance

    Returns:
        Dictionary with counts per type
    """
    conn = store._get_connection()

    result = {
        "simulations": {},
        "experiments": {},
    }

    # Count simulations by type
    try:
        cursor = conn.execute(
            "SELECT type, COUNT(*) FROM simulations GROUP BY type"
        )
        for row in cursor.fetchall():
            result["simulations"][row[0]] = row[1]
    except Exception:
        pass

    # Count experiments by type
    try:
        cursor = conn.execute(
            "SELECT type, COUNT(*) FROM experiments GROUP BY type"
        )
        for row in cursor.fetchall():
            result["experiments"][row[0]] = row[1]
    except Exception:
        pass

    return result


def search_metadata(
    store: DataStore,
    key: str,
    value: Any,
    table: str = "simulations",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Search for records with matching metadata value.

    Note: This performs a LIKE search on JSON metadata,
    which may be slow for large datasets.

    Args:
        store: DataStore instance
        key: Metadata key to search
        value: Value to match
        table: Table to search ("simulations" or "experiments")
        limit: Maximum number of results

    Returns:
        List of matching records
    """
    conn = store._get_connection()

    # Build search pattern
    search_pattern = f'%"{key}":%{value}%'

    if table == "simulations":
        sql = "SELECT * FROM simulations WHERE metadata LIKE ? LIMIT ?"
    else:
        sql = "SELECT * FROM experiments WHERE metadata LIKE ? LIMIT ?"

    cursor = conn.execute(sql, (search_pattern, limit))
    rows = cursor.fetchall()

    return [store._row_to_dict(row) for row in rows]


# -----------------------------------------------------------------------------
# Helper functions for Surrogate Model training
# -----------------------------------------------------------------------------


def get_simulation_records_for_type(
    store: DataStore,
    sim_type: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get simulation records suitable for surrogate training.

    Returns records with their associated fields metadata for
    loading field data.

    Args:
        store: DataStore instance
        sim_type: Simulation type (e.g., "heat_1d", "laplace_2d")
        limit: Maximum number of records

    Returns:
        List of simulation records with field metadata
    """
    # Get simulations of the specified type
    records = find_simulations_by_type(store, sim_type, limit=limit)

    # Enrich with fields metadata
    for record in records:
        try:
            fields_meta = store.list_fields(record["id"], "simulation")
            record["fields_meta"] = fields_meta
        except Exception:
            record["fields_meta"] = []

    return records


def get_timeseries_for_tag(
    store: DataStore,
    tag: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get timeseries records that belong to simulations/experiments with a tag.

    Args:
        store: DataStore instance
        tag: Tag to filter by
        limit: Maximum number of records

    Returns:
        List of timeseries records
    """
    # Find simulations with the tag
    tagged_sims = find_simulations_by_tag(store, tag, limit=limit)

    all_timeseries = []
    for sim in tagged_sims:
        ts_list = store.list_timeseries(sim["id"], "simulation")
        all_timeseries.extend(ts_list)
        if len(all_timeseries) >= limit:
            break

    return all_timeseries[:limit]


def get_simulation_field_data(
    store: DataStore,
    sim_id: str,
    field_name: str = "solution",
) -> Optional[Any]:
    """
    Get field data from a simulation.

    Args:
        store: DataStore instance
        sim_id: Simulation ID
        field_name: Name of the field to retrieve

    Returns:
        Field data (numpy array) or None if not found
    """
    # First check field table
    fields = store.list_fields(sim_id, "simulation")
    for field in fields:
        if field.get("name") == field_name:
            return store.load_field_array(field["id"])

    # Fallback: check if stored in simulation record
    sim = store.get_simulation(sim_id)
    if sim and "fields" in sim:
        fields_data = sim["fields"]
        if isinstance(fields_data, str):
            try:
                fields_data = json.loads(fields_data)
            except json.JSONDecodeError:
                return None

        if field_name in fields_data:
            import numpy as np
            data = fields_data[field_name]
            if isinstance(data, list):
                return np.array(data)
            return data

    return None


def count_simulations_by_type(
    store: DataStore,
) -> Dict[str, int]:
    """
    Count simulations grouped by type.

    Args:
        store: DataStore instance

    Returns:
        Dictionary mapping type -> count
    """
    conn = store._get_connection()
    counts = {}

    try:
        cursor = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM simulations GROUP BY type"
        )
        for row in cursor.fetchall():
            counts[row[0]] = row[1]
    except Exception:
        pass

    return counts


def count_experiments_by_type(
    store: DataStore,
) -> Dict[str, int]:
    """
    Count experiments grouped by type.

    Args:
        store: DataStore instance

    Returns:
        Dictionary mapping type -> count
    """
    conn = store._get_connection()
    counts = {}

    try:
        cursor = conn.execute(
            "SELECT type, COUNT(*) as cnt FROM experiments GROUP BY type"
        )
        for row in cursor.fetchall():
            counts[row[0]] = row[1]
    except Exception:
        pass

    return counts
