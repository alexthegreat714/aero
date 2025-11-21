"""
Database schema definitions for Aero Agent data lake.

Provides SQL schema for simulations, experiments, timeseries, and fields tables.
Compatible with both DuckDB and SQLite.
"""

# =============================================================================
# Table Schemas
# =============================================================================

SIMULATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL,
    config TEXT,
    metadata TEXT,
    status TEXT DEFAULT 'completed',
    tags TEXT,
    hypothesis_id TEXT
)
"""

EXPERIMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    type TEXT NOT NULL,
    metadata TEXT,
    features TEXT,
    tags TEXT,
    hypothesis_id TEXT
)
"""

TIMESERIES_TABLE = """
CREATE TABLE IF NOT EXISTS timeseries (
    id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    parent_type TEXT NOT NULL,
    name TEXT NOT NULL,
    t TEXT,
    values TEXT,
    metadata TEXT,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

FIELDS_TABLE = """
CREATE TABLE IF NOT EXISTS fields (
    id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    parent_type TEXT NOT NULL,
    name TEXT NOT NULL,
    shape TEXT,
    dtype TEXT,
    storage_path TEXT NOT NULL,
    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Index definitions for faster queries
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_simulations_type ON simulations(type)",
    "CREATE INDEX IF NOT EXISTS idx_simulations_created ON simulations(created)",
    "CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(type)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created)",
    "CREATE INDEX IF NOT EXISTS idx_timeseries_parent ON timeseries(parent_id, parent_type)",
    "CREATE INDEX IF NOT EXISTS idx_fields_parent ON fields(parent_id, parent_type)",
]

ALL_TABLES = [
    SIMULATIONS_TABLE,
    EXPERIMENTS_TABLE,
    TIMESERIES_TABLE,
    FIELDS_TABLE,
]

ALL_SCHEMA = ALL_TABLES + INDEXES


def get_all_schema_sql() -> list[str]:
    """Get all schema SQL statements."""
    return ALL_SCHEMA


def get_table_names() -> list[str]:
    """Get list of table names."""
    return ["simulations", "experiments", "timeseries", "fields"]
