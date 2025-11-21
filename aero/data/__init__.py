"""
Data lake module for Aero Agent.

Provides persistent storage for simulation and experiment results
using DuckDB (preferred) or SQLite (fallback).
"""

from aero.data.store import (
    DataStore,
    get_default_store,
    set_default_store,
    DUCKDB_AVAILABLE,
)

from aero.data.schema import (
    get_all_schema_sql,
    get_table_names,
)

from aero.data.timeseries import (
    pack_timeseries,
    unpack_timeseries,
    compute_basic_ts_stats,
    resample_timeseries,
    detect_anomalies,
    compute_fft_features,
    smooth_timeseries,
)

from aero.data.query import (
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

__all__ = [
    # Store
    "DataStore",
    "get_default_store",
    "set_default_store",
    "DUCKDB_AVAILABLE",
    # Schema
    "get_all_schema_sql",
    "get_table_names",
    # Timeseries
    "pack_timeseries",
    "unpack_timeseries",
    "compute_basic_ts_stats",
    "resample_timeseries",
    "detect_anomalies",
    "compute_fft_features",
    "smooth_timeseries",
    # Query
    "list_recent_simulations",
    "list_recent_experiments",
    "find_simulations_by_type",
    "find_experiments_by_type",
    "find_simulations_by_tag",
    "find_experiments_by_tag",
    "find_by_hypothesis",
    "get_simulation_with_timeseries",
    "get_experiment_with_timeseries",
    "count_by_type",
    "search_metadata",
]
