"""
Agent API routes for Aero Agent.

Provides endpoints for agent management and task execution.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class TaskRequest(BaseModel):
    """Request for submitting a task."""

    name: str
    type: str
    parameters: Optional[dict] = None


class AgentConfig(BaseModel):
    """Configuration for agent settings."""

    max_workers: int = 4
    timeout: int = 300


@router.get("/")
async def agent_status():
    """Get agent system status."""
    return {
        "status": "ok",
        "module": "agent",
        "running": True,
        "workers": 4,
        "queue_size": 0,
        "message": "Agent system is operational (stub)",
    }


@router.get("/info")
async def agent_info():
    """Get detailed agent information."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "name": "Aero Agent",
        "capabilities": [
            "rag",
            "simulation",
            "ocr",
            "web_search",
            "experiments",
        ],
        "message": "Agent info (stub)",
    }


@router.post("/tasks")
async def submit_task(task: TaskRequest):
    """
    Submit a task to the agent.

    Returns task ID for tracking.
    """
    logger.info(f"Submitting task: {task.name} ({task.type})")

    return {
        "status": "submitted",
        "task_id": "task_001",
        "name": task.name,
        "type": task.type,
        "message": "Task submitted (stub)",
    }


@router.get("/tasks")
async def list_tasks(status: Optional[str] = None):
    """List all tasks with optional status filter."""
    return {
        "status": "ok",
        "tasks": [],
        "total": 0,
        "message": "Task listing (stub)",
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details and status."""
    return {
        "status": "ok",
        "task_id": task_id,
        "state": "pending",
        "progress": 0.0,
        "result": None,
        "message": "Task details (stub)",
    }


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running task."""
    return {
        "status": "cancelled",
        "task_id": task_id,
        "message": "Task cancelled (stub)",
    }


@router.post("/config")
async def update_config(config: AgentConfig):
    """Update agent configuration."""
    logger.info(f"Updating config: workers={config.max_workers}")

    return {
        "status": "updated",
        "config": config.dict(),
        "message": "Configuration updated (stub)",
    }


@router.get("/registries")
async def list_registries():
    """List all component registries."""
    try:
        from aero.core.registry import get_registry_manager
        manager = get_registry_manager()
        return {
            "status": "ok",
            "registries": manager.get_summary(),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/events")
async def list_events(limit: int = 100):
    """List recent events from the event bus."""
    return {
        "status": "ok",
        "events": [],
        "total": 0,
        "message": "Event listing (stub)",
    }
