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
