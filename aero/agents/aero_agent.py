"""
Aero Agent implementation for inter-agent communication.

This module wraps the Aero system capabilities (reasoning loop, RAG,
simulations, surrogates) and exposes them through the agent message interface.
"""

import logging
from typing import Any, Dict, List, Optional

from aero.agents.base_agent_api import BaseAgentAPI
from aero.agents.messages import (
    AgentMessage,
    ACTION_HEALTH_CHECK,
    ACTION_HYPOTHESIS_LOOP,
    ACTION_RUN_SIMULATION,
    ACTION_RAG_SEARCH,
    ACTION_LIST_SURROGATES,
    ACTION_CHECK_CONSTRAINTS,
)

logger = logging.getLogger(__name__)


class AeroAgent(BaseAgentAPI):
    """
    Aero Agent wrapper for inter-agent communication.

    Exposes Aero's capabilities to other agents:
    - Scientific reasoning loop
    - RAG search
    - Simulation execution
    - Surrogate model listing
    - Constraint checking
    - Health check

    Example:
        aero_agent = AeroAgent(
            loop=reasoning_loop,
            rag_store=rag_store,
            data_store=data_store,
            surrogate_registry=surrogate_registry,
        )

        response = aero_agent.handle_message(
            AgentMessage.new(
                sender="Sky",
                recipient="Aero",
                kind="request",
                action="hypothesis_loop",
                payload={"query": "Analyze heat diffusion"},
            )
        )
    """

    def __init__(
        self,
        loop=None,
        rag_store=None,
        data_store=None,
        surrogate_registry=None,
    ):
        """
        Initialize the Aero Agent.

        Args:
            loop: ScientificReasoningLoop instance
            rag_store: RAG vector store instance
            data_store: DataStore instance
            surrogate_registry: SurrogateRegistry instance
        """
        super().__init__(name="Aero", version="0.1.0")

        self.loop = loop
        self.rag_store = rag_store
        self.data_store = data_store
        self.surrogate_registry = surrogate_registry

        # Set metadata
        self.set_metadata("description", "Aerodynamics Research Agent")
        self.set_metadata("domain", "scientific_reasoning")

        logger.info("AeroAgent initialized")

    def handle_message(self, msg: AgentMessage) -> dict:
        """
        Handle incoming agent messages.

        Supported actions:
        - hypothesis_loop: Run the scientific reasoning loop
        - run_simulation: Execute a single simulation
        - rag_search: Search the RAG store
        - list_surrogates: List available surrogate models
        - check_constraints: Run constraint checks on results
        - health_check: Return system health status

        Args:
            msg: AgentMessage to process

        Returns:
            Response dictionary with status and data
        """
        action = msg.action
        payload = msg.payload

        logger.info(f"AeroAgent handling action: {action} from {msg.sender}")

        try:
            if action == ACTION_HEALTH_CHECK:
                return self._handle_health_check(payload)

            elif action == ACTION_HYPOTHESIS_LOOP:
                return self._handle_hypothesis_loop(payload)

            elif action == ACTION_RUN_SIMULATION:
                return self._handle_run_simulation(payload)

            elif action == ACTION_RAG_SEARCH:
                return self._handle_rag_search(payload)

            elif action == ACTION_LIST_SURROGATES:
                return self._handle_list_surrogates(payload)

            elif action == ACTION_CHECK_CONSTRAINTS:
                return self._handle_check_constraints(payload)

            elif action == "describe":
                return self._make_ok_response(data=self.describe())

            else:
                return self._unknown_action_response(action)

        except Exception as e:
            logger.exception(f"Error handling action {action}: {e}")
            return self._make_error_response(
                message=f"Error processing action '{action}': {str(e)}",
                error_code="HANDLER_ERROR",
            )

    def get_capabilities(self) -> List[str]:
        """Return list of supported actions."""
        return [
            ACTION_HEALTH_CHECK,
            ACTION_HYPOTHESIS_LOOP,
            ACTION_RUN_SIMULATION,
            ACTION_RAG_SEARCH,
            ACTION_LIST_SURROGATES,
            ACTION_CHECK_CONSTRAINTS,
            "describe",
        ]

    def _handle_health_check(self, payload: Dict[str, Any]) -> dict:
        """Handle health check request."""
        health = {
            "agent": "Aero",
            "status": "healthy",
            "components": {},
        }

        # Check loop
        if self.loop is not None:
            health["components"]["reasoning_loop"] = "available"
        else:
            health["components"]["reasoning_loop"] = "not_initialized"

        # Check RAG
        if self.rag_store is not None:
            try:
                doc_count = self.rag_store.count()
                health["components"]["rag_store"] = {
                    "status": "available",
                    "documents": doc_count,
                }
            except Exception as e:
                health["components"]["rag_store"] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            health["components"]["rag_store"] = "not_initialized"

        # Check data store
        if self.data_store is not None:
            try:
                stats = self.data_store.get_stats()
                health["components"]["data_store"] = {
                    "status": "available",
                    "stats": stats,
                }
            except Exception as e:
                health["components"]["data_store"] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            health["components"]["data_store"] = "not_initialized"

        # Check surrogates
        if self.surrogate_registry is not None:
            try:
                model_count = len(self.surrogate_registry)
                health["components"]["surrogate_registry"] = {
                    "status": "available",
                    "models": model_count,
                }
            except Exception as e:
                health["components"]["surrogate_registry"] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            health["components"]["surrogate_registry"] = "not_initialized"

        return self._make_ok_response(data=health)

    def _handle_hypothesis_loop(self, payload: Dict[str, Any]) -> dict:
        """Handle hypothesis loop request."""
        if self.loop is None:
            return self._make_error_response(
                message="Reasoning loop not initialized",
                error_code="LOOP_NOT_INITIALIZED",
            )

        query = payload.get("query")
        if not query:
            return self._make_error_response(
                message="Missing 'query' in payload",
                error_code="MISSING_QUERY",
            )

        try:
            # Run a single step of the loop
            if payload.get("full_loop", False):
                # Run full loop until convergence
                result = self.loop.run(query)
                return self._make_ok_response(
                    data=result.to_dict(),
                    message="Reasoning loop completed",
                )
            else:
                # Run single step
                result = self.loop.step(query)
                return self._make_ok_response(
                    data=result,
                    message="Loop step completed",
                )

        except Exception as e:
            logger.exception(f"Hypothesis loop error: {e}")
            return self._make_error_response(
                message=f"Loop execution failed: {str(e)}",
                error_code="LOOP_ERROR",
            )

    def _handle_run_simulation(self, payload: Dict[str, Any]) -> dict:
        """Handle simulation request."""
        sim_type = payload.get("sim_type", "heat_1d")
        params = payload.get("parameters", {})

        try:
            from aero.pipeline.planner import SimulationJobConfig, SimulationType

            # Parse simulation type
            try:
                sim_type_enum = SimulationType(sim_type)
            except ValueError:
                return self._make_error_response(
                    message=f"Unknown simulation type: {sim_type}",
                    error_code="INVALID_SIM_TYPE",
                    data={"valid_types": [t.value for t in SimulationType]},
                )

            # Create simulation config
            config = SimulationJobConfig(
                sim_type=sim_type_enum,
                hypothesis_id=payload.get("hypothesis_id", "agent_request"),
                grid_size=params.get("grid_size", 50),
                grid_size_y=params.get("grid_size_y"),
                dx=params.get("dx", 0.02),
                dt=params.get("dt", 0.001),
                max_steps=params.get("max_steps", 1000),
                max_iterations=params.get("max_iterations", 10000),
                tolerance=params.get("tolerance", 1e-6),
                initial_condition=params.get("initial_condition", "gaussian"),
                boundary_conditions=params.get("boundary_conditions", {}),
                solver_params=params.get("solver_params", {}),
            )

            # Run simulation if loop is available
            if self.loop is not None:
                result = self.loop._run_simulation(config)
            else:
                # Fallback to direct solver call
                from aero.sim.numerics.fd_solver import solve_heat_1d
                import numpy as np

                nx = config.grid_size
                x = np.linspace(0, 1, nx)
                u0 = np.exp(-100 * (x - 0.5) ** 2)
                solution, metadata = solve_heat_1d(
                    u0, 0.01, config.dx, config.dt, config.max_steps
                )
                result = {
                    "solution": solution.tolist(),
                    "converged": True,
                    "metadata": metadata,
                }

            return self._make_ok_response(
                data=result,
                message=f"Simulation {sim_type} completed",
            )

        except Exception as e:
            logger.exception(f"Simulation error: {e}")
            return self._make_error_response(
                message=f"Simulation failed: {str(e)}",
                error_code="SIMULATION_ERROR",
            )

    def _handle_rag_search(self, payload: Dict[str, Any]) -> dict:
        """Handle RAG search request."""
        if self.rag_store is None:
            return self._make_error_response(
                message="RAG store not initialized",
                error_code="RAG_NOT_INITIALIZED",
            )

        query = payload.get("query")
        if not query:
            return self._make_error_response(
                message="Missing 'query' in payload",
                error_code="MISSING_QUERY",
            )

        try:
            top_k = payload.get("top_k", 5)
            results = self.rag_store.search(query, top_k=top_k)

            # Convert results to serializable format
            search_results = []
            for doc, score in results:
                search_results.append({
                    "content": doc.content[:500],  # Truncate for response
                    "score": float(score),
                    "metadata": doc.metadata,
                })

            return self._make_ok_response(
                data={
                    "query": query,
                    "results": search_results,
                    "count": len(search_results),
                },
                message=f"Found {len(search_results)} results",
            )

        except Exception as e:
            logger.exception(f"RAG search error: {e}")
            return self._make_error_response(
                message=f"RAG search failed: {str(e)}",
                error_code="RAG_ERROR",
            )

    def _handle_list_surrogates(self, payload: Dict[str, Any]) -> dict:
        """Handle list surrogates request."""
        if self.surrogate_registry is None:
            return self._make_error_response(
                message="Surrogate registry not initialized",
                error_code="SURROGATES_NOT_INITIALIZED",
            )

        try:
            models = self.surrogate_registry.list_models()
            model_list = []

            for model in models:
                model_list.append({
                    "name": model.name,
                    "model_type": model.model_type,
                    "sim_type": model.sim_type,
                    "created": model.created.isoformat() if model.created else None,
                    "metrics": model.metrics,
                })

            return self._make_ok_response(
                data={
                    "models": model_list,
                    "count": len(model_list),
                },
                message=f"Found {len(model_list)} surrogate models",
            )

        except Exception as e:
            logger.exception(f"List surrogates error: {e}")
            return self._make_error_response(
                message=f"Failed to list surrogates: {str(e)}",
                error_code="SURROGATE_ERROR",
            )

    def _handle_check_constraints(self, payload: Dict[str, Any]) -> dict:
        """Handle constraint check request."""
        try:
            from aero.symbolic.checks import (
                check_simulation_constraints,
                check_experiment_constraints,
            )

            result_type = payload.get("result_type", "simulation")
            result_data = payload.get("result", {})

            if result_type == "simulation":
                check_result = check_simulation_constraints(sim_result=result_data)
            else:
                check_result = check_experiment_constraints(exp_result=result_data)

            return self._make_ok_response(
                data=check_result,
                message="Constraint check completed",
            )

        except ImportError:
            return self._make_error_response(
                message="Symbolic module not available",
                error_code="SYMBOLIC_NOT_AVAILABLE",
            )
        except Exception as e:
            logger.exception(f"Constraint check error: {e}")
            return self._make_error_response(
                message=f"Constraint check failed: {str(e)}",
                error_code="CONSTRAINT_ERROR",
            )
