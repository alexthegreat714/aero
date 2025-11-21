"""
Data store for Aero Agent.

Provides persistent storage for simulation and experiment results
using DuckDB (preferred) or SQLite (fallback).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from aero.data.schema import get_all_schema_sql, get_table_names

logger = logging.getLogger(__name__)

# Try to import DuckDB
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.info("DuckDB not available, will use SQLite")

# SQLite is always available
import sqlite3


class DataStore:
    """
    Data store for Aero Agent results.

    Uses DuckDB if available, otherwise falls back to SQLite.
    Provides methods for storing and querying simulations,
    experiments, timeseries, and field data.

    Example:
        store = DataStore("./data/aero.db", "./data/fields")
        store.init_schema()

        result = {"id": "sim_001", "type": "heat_1d", ...}
        store.save_simulation_result(result)

        sims = store.list_simulations(limit=10)
    """

    def __init__(
        self,
        db_path: str = "./data/aero.db",
        fields_dir: str = "./data/fields",
    ):
        """
        Initialize the data store.

        Args:
            db_path: Path to database file
            fields_dir: Directory for storing large field arrays
        """
        self.db_path = Path(db_path)
        self.fields_dir = Path(fields_dir)
        self._conn = None
        self._backend = "duckdb" if DUCKDB_AVAILABLE else "sqlite"

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fields_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataStore initialized: backend={self._backend}, path={self.db_path}")

    @property
    def backend(self) -> str:
        """Get the database backend name."""
        return self._backend

    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None:
            if self._backend == "duckdb":
                self._conn = duckdb.connect(str(self.db_path))
            else:
                self._conn = sqlite3.connect(str(self.db_path))
                self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        """Initialize database schema (create tables if missing)."""
        conn = self._get_connection()

        for sql in get_all_schema_sql():
            try:
                conn.execute(sql)
            except Exception as e:
                logger.warning(f"Schema statement warning: {e}")

        if self._backend == "sqlite":
            conn.commit()

        logger.info(f"Schema initialized: tables={get_table_names()}")

    # =========================================================================
    # Simulation Methods
    # =========================================================================

    def save_simulation_result(
        self,
        result: Dict[str, Any],
        hypothesis_id: Optional[str] = None,
    ) -> str:
        """
        Save a simulation result to the database.

        Args:
            result: Simulation result dictionary with keys:
                    id, type, config, metadata, status, tags
            hypothesis_id: Optional associated hypothesis ID

        Returns:
            The simulation ID
        """
        conn = self._get_connection()

        sim_id = result.get("id") or str(uuid4())
        sim_type = result.get("type", result.get("sim_type", "unknown"))
        config = json.dumps(result.get("config", {}))
        metadata = json.dumps(result.get("metadata", {}))
        status = result.get("status", "completed")
        tags = json.dumps(result.get("tags", []))

        sql = """
            INSERT INTO simulations (id, type, config, metadata, status, tags, hypothesis_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        try:
            conn.execute(sql, (sim_id, sim_type, config, metadata, status, tags, hypothesis_id))
            if self._backend == "sqlite":
                conn.commit()
            logger.debug(f"Saved simulation: {sim_id}")
        except Exception as e:
            logger.error(f"Error saving simulation: {e}")
            raise

        # Save any field arrays
        fields = result.get("fields", {})
        for name, array in fields.items():
            if isinstance(array, np.ndarray):
                self.save_field_array(sim_id, "simulation", name, array)
            elif isinstance(array, list):
                try:
                    arr = np.array(array)
                    self.save_field_array(sim_id, "simulation", name, arr)
                except Exception:
                    pass  # Skip non-convertible data

        return sim_id

    def list_simulations(
        self,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List simulation results.

        Args:
            limit: Maximum number of results
            filters: Optional filters (type, status, etc.)

        Returns:
            List of simulation records
        """
        conn = self._get_connection()

        sql = "SELECT * FROM simulations"
        params = []

        if filters:
            conditions = []
            if "type" in filters:
                conditions.append("type = ?")
                params.append(filters["type"])
            if "status" in filters:
                conditions.append("status = ?")
                params.append(filters["status"])
            if "hypothesis_id" in filters:
                conditions.append("hypothesis_id = ?")
                params.append(filters["hypothesis_id"])

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY created DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_simulation(self, sim_id: str) -> Optional[Dict[str, Any]]:
        """Get a simulation by ID."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM simulations WHERE id = ?",
            (sim_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        result = self._row_to_dict(row)

        # Load associated fields
        result["fields"] = self._get_fields_for_parent(sim_id, "simulation")

        return result

    # =========================================================================
    # Experiment Methods
    # =========================================================================

    def save_experiment_result(
        self,
        result: Dict[str, Any],
        hypothesis_id: Optional[str] = None,
    ) -> str:
        """
        Save an experiment result to the database.

        Args:
            result: Experiment result dictionary with keys:
                    id, experiment_type, data, metadata, tags
            hypothesis_id: Optional associated hypothesis ID

        Returns:
            The experiment ID
        """
        conn = self._get_connection()

        exp_id = result.get("id") or str(uuid4())
        exp_type = result.get("experiment_type", result.get("type", "unknown"))
        metadata = json.dumps(result.get("metadata", {}))
        features = json.dumps(result.get("data", result.get("features", {})))
        tags = json.dumps(result.get("tags", []))

        sql = """
            INSERT INTO experiments (id, type, metadata, features, tags, hypothesis_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        try:
            conn.execute(sql, (exp_id, exp_type, metadata, features, tags, hypothesis_id))
            if self._backend == "sqlite":
                conn.commit()
            logger.debug(f"Saved experiment: {exp_id}")
        except Exception as e:
            logger.error(f"Error saving experiment: {e}")
            raise

        return exp_id

    def list_experiments(
        self,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List experiment results.

        Args:
            limit: Maximum number of results
            filters: Optional filters (type, etc.)

        Returns:
            List of experiment records
        """
        conn = self._get_connection()

        sql = "SELECT * FROM experiments"
        params = []

        if filters:
            conditions = []
            if "type" in filters:
                conditions.append("type = ?")
                params.append(filters["type"])
            if "hypothesis_id" in filters:
                conditions.append("hypothesis_id = ?")
                params.append(filters["hypothesis_id"])

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY created DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get an experiment by ID."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (exp_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    # =========================================================================
    # Timeseries Methods
    # =========================================================================

    def save_timeseries(
        self,
        parent_id: str,
        parent_type: str,
        name: str,
        t: List[float],
        values: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a time series to the database.

        Args:
            parent_id: ID of parent simulation or experiment
            parent_type: "simulation" or "experiment"
            name: Name of the time series
            t: Time values
            values: Data values
            metadata: Optional metadata

        Returns:
            The timeseries ID
        """
        conn = self._get_connection()

        ts_id = str(uuid4())
        t_json = json.dumps(t)
        values_json = json.dumps(values)
        metadata_json = json.dumps(metadata or {})

        sql = """
            INSERT INTO timeseries (id, parent_id, parent_type, name, t, values, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        conn.execute(sql, (ts_id, parent_id, parent_type, name, t_json, values_json, metadata_json))
        if self._backend == "sqlite":
            conn.commit()

        logger.debug(f"Saved timeseries: {ts_id} ({name})")
        return ts_id

    def get_timeseries(self, ts_id: str) -> Optional[Dict[str, Any]]:
        """Get a timeseries by ID."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM timeseries WHERE id = ?",
            (ts_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list_timeseries_for_parent(
        self,
        parent_id: str,
        parent_type: str,
    ) -> List[Dict[str, Any]]:
        """List all timeseries for a parent."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM timeseries WHERE parent_id = ? AND parent_type = ?",
            (parent_id, parent_type)
        )
        rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # Field Array Methods
    # =========================================================================

    def save_field_array(
        self,
        parent_id: str,
        parent_type: str,
        name: str,
        array: np.ndarray,
    ) -> str:
        """
        Save a field array to disk and register in database.

        Args:
            parent_id: ID of parent simulation or experiment
            parent_type: "simulation" or "experiment"
            name: Name of the field
            array: NumPy array to save

        Returns:
            The field ID
        """
        conn = self._get_connection()

        field_id = str(uuid4())
        storage_path = f"{parent_type}_{parent_id}_{name}_{field_id[:8]}.npz"
        full_path = self.fields_dir / storage_path

        # Save array to disk
        np.savez_compressed(full_path, data=array)

        # Register in database
        shape_json = json.dumps(list(array.shape))
        dtype_str = str(array.dtype)

        sql = """
            INSERT INTO fields (id, parent_id, parent_type, name, shape, dtype, storage_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        conn.execute(sql, (field_id, parent_id, parent_type, name, shape_json, dtype_str, storage_path))
        if self._backend == "sqlite":
            conn.commit()

        logger.debug(f"Saved field array: {field_id} ({name}, shape={array.shape})")
        return field_id

    def load_field_array(self, field_id: str) -> Optional[np.ndarray]:
        """
        Load a field array from disk.

        Args:
            field_id: Field ID

        Returns:
            NumPy array or None if not found
        """
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT storage_path FROM fields WHERE id = ?",
            (field_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        storage_path = row[0] if isinstance(row, tuple) else row["storage_path"]
        full_path = self.fields_dir / storage_path

        if not full_path.exists():
            logger.warning(f"Field file not found: {full_path}")
            return None

        data = np.load(full_path)
        return data["data"]

    def get_field(self, field_id: str) -> Optional[Dict[str, Any]]:
        """Get field metadata by ID."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM fields WHERE id = ?",
            (field_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def _get_fields_for_parent(
        self,
        parent_id: str,
        parent_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Get all field metadata for a parent."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM fields WHERE parent_id = ? AND parent_type = ?",
            (parent_id, parent_type)
        )
        rows = cursor.fetchall()

        fields = {}
        for row in rows:
            field = self._row_to_dict(row)
            fields[field["name"]] = {
                "id": field["id"],
                "shape": json.loads(field["shape"]) if field["shape"] else None,
                "dtype": field["dtype"],
                "storage_path": field["storage_path"],
            }

        return fields

    # =========================================================================
    # Summary and Stats
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        conn = self._get_connection()

        summary = {
            "backend": self._backend,
            "db_path": str(self.db_path),
            "fields_dir": str(self.fields_dir),
        }

        # Count tables
        for table in ["simulations", "experiments", "timeseries", "fields"]:
            try:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                summary[f"{table}_count"] = count
            except Exception:
                summary[f"{table}_count"] = 0

        # Get recent items
        try:
            cursor = conn.execute(
                "SELECT id, type, created FROM simulations ORDER BY created DESC LIMIT 5"
            )
            summary["recent_simulations"] = [
                {"id": r[0], "type": r[1], "created": str(r[2])}
                for r in cursor.fetchall()
            ]
        except Exception:
            summary["recent_simulations"] = []

        try:
            cursor = conn.execute(
                "SELECT id, type, created FROM experiments ORDER BY created DESC LIMIT 5"
            )
            summary["recent_experiments"] = [
                {"id": r[0], "type": r[1], "created": str(r[2])}
                for r in cursor.fetchall()
            ]
        except Exception:
            summary["recent_experiments"] = []

        return summary

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to dictionary."""
        if row is None:
            return {}

        if self._backend == "duckdb":
            # DuckDB returns tuples with column names accessible via description
            # We need to handle this differently
            if hasattr(row, "keys"):
                return dict(row)
            else:
                # Fallback for tuple rows
                return dict(zip(
                    ["id", "created", "type", "config", "metadata", "status", "tags", "hypothesis_id"],
                    row
                ))
        else:
            # SQLite Row objects
            return dict(row)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# Module-level singleton
# =============================================================================

_default_store: Optional[DataStore] = None


def get_default_store() -> DataStore:
    """Get or create the default data store."""
    global _default_store

    if _default_store is None:
        from aero.config.loader import get_config
        config = get_config()

        db_path = config.get("data.db_path", "./data/aero.db")
        fields_dir = config.get("data.fields_dir", "./data/fields")

        _default_store = DataStore(db_path, fields_dir)
        _default_store.init_schema()

    return _default_store


def set_default_store(store: DataStore) -> None:
    """Set the default data store."""
    global _default_store
    _default_store = store
