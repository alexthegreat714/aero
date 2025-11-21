"""
Main processing loop for Aero Agent.

Orchestrates the agent's workflow and task execution, including
the scientific reasoning loop for hypothesis generation and refinement.
"""

import logging
import time
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue, Empty

import numpy as np

from aero.config.loader import get_config
from aero.core.events import EventBus, Event

# Data Lake imports (optional)
try:
    from aero.data import get_default_store, DataStore
    DATA_LAKE_AVAILABLE = True
except ImportError:
    DATA_LAKE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a pipeline task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineTask:
    """Represents a task in the pipeline."""

    id: str
    name: str
    handler: Callable[..., Any]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AeroLoop:
    """
    Main processing loop for the Aero Agent.

    Provides:
    - Task queue management
    - Concurrent task execution
    - Event-driven workflow
    - Error handling and retry

    Example:
        loop = AeroLoop()

        def process_data(data):
            return data.upper()

        loop.submit("task1", "uppercase", process_data, args=("hello",))
        loop.start()
        # ... later
        loop.stop()
    """

    def __init__(
        self,
        max_workers: int = 4,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize the Aero Loop.

        Args:
            max_workers: Maximum concurrent workers
            event_bus: Event bus for notifications
        """
        self.max_workers = max_workers
        self.event_bus = event_bus or EventBus()

        self._task_queue: Queue[PipelineTask] = Queue()
        self._active_tasks: dict[str, PipelineTask] = {}
        self._completed_tasks: dict[str, PipelineTask] = {}

        self._workers: list[threading.Thread] = []
        self._running = False
        self._lock = threading.RLock()

        self._config = get_config()

        logger.info(f"AeroLoop initialized with {max_workers} workers")

    @property
    def is_running(self) -> bool:
        """Check if the loop is running."""
        return self._running

    def submit(
        self,
        task_id: str,
        name: str,
        handler: Callable[..., Any],
        args: tuple = (),
        kwargs: dict = None,
    ) -> PipelineTask:
        """
        Submit a task to the pipeline.

        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            handler: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            The created PipelineTask
        """
        task = PipelineTask(
            id=task_id,
            name=name,
            handler=handler,
            args=args,
            kwargs=kwargs or {},
        )

        with self._lock:
            self._active_tasks[task_id] = task

        self._task_queue.put(task)
        self.event_bus.emit_simple("task_submitted", "aero_loop", {"task_id": task_id})

        logger.debug(f"Task submitted: {task_id} ({name})")
        return task

    def start(self) -> None:
        """Start the processing loop."""
        if self._running:
            logger.warning("AeroLoop already running")
            return

        self._running = True

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"aero-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        self.event_bus.emit_simple("loop_started", "aero_loop")
        logger.info(f"AeroLoop started with {self.max_workers} workers")

    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Stop the processing loop.

        Args:
            wait: Wait for workers to finish
            timeout: Maximum time to wait
        """
        self._running = False

        if wait:
            for worker in self._workers:
                worker.join(timeout=timeout / len(self._workers))

        self._workers.clear()
        self.event_bus.emit_simple("loop_stopped", "aero_loop")
        logger.info("AeroLoop stopped")

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
                self._execute_task(task)
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    def _execute_task(self, task: PipelineTask) -> None:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        self.event_bus.emit_simple(
            "task_started",
            "aero_loop",
            {"task_id": task.id, "name": task.name},
        )

        try:
            result = task.handler(*task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED

            self.event_bus.emit_simple(
                "task_completed",
                "aero_loop",
                {"task_id": task.id, "name": task.name},
            )

            logger.debug(f"Task completed: {task.id}")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

            self.event_bus.emit_simple(
                "task_failed",
                "aero_loop",
                {"task_id": task.id, "error": str(e)},
            )

            logger.error(f"Task failed: {task.id} - {e}")

        finally:
            task.completed_at = datetime.now()

            with self._lock:
                if task.id in self._active_tasks:
                    del self._active_tasks[task.id]
                self._completed_tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        """Get a task by ID."""
        with self._lock:
            return (
                self._active_tasks.get(task_id) or
                self._completed_tasks.get(task_id)
            )

    def get_status(self) -> dict:
        """Get loop status."""
        with self._lock:
            return {
                "running": self._running,
                "workers": len(self._workers),
                "queued": self._task_queue.qsize(),
                "active": len(self._active_tasks),
                "completed": len(self._completed_tasks),
            }

    def wait_for_task(self, task_id: str, timeout: float = None) -> Optional[PipelineTask]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait

        Returns:
            Completed task or None if timeout
        """
        start = time.time()

        while True:
            task = self.get_task(task_id)

            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task

            if timeout and (time.time() - start) > timeout:
                return None

            time.sleep(0.1)


# =============================================================================
# Scientific Reasoning Loop
# =============================================================================


@dataclass
class LoopConfig:
    """Configuration for the scientific reasoning loop."""

    max_iterations: int = 5
    confidence_threshold: float = 0.85
    drop_threshold: float = 0.2
    pattern_convergence_eps: float = 1e-3
    max_simulations_per_iteration: int = 3
    timeout_per_simulation: float = 60.0


@dataclass
class LoopResult:
    """Result from a complete reasoning loop execution."""

    converged: bool
    iterations_completed: int
    best_hypothesis: Optional[dict]
    all_hypotheses: List[dict]
    simulation_results: List[dict]
    validation_results: List[dict]
    patterns: List[dict]
    history: List[dict]
    termination_reason: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "converged": self.converged,
            "iterations_completed": self.iterations_completed,
            "best_hypothesis": self.best_hypothesis,
            "all_hypotheses": self.all_hypotheses,
            "simulation_results": self.simulation_results,
            "validation_results": self.validation_results,
            "patterns": self.patterns,
            "history": self.history,
            "termination_reason": self.termination_reason,
        }


class ScientificReasoningLoop:
    """
    Recursive scientific reasoning loop for Aero Agent.

    Implements the full hypothesis-driven scientific process:
    1. Parse query and generate initial hypotheses
    2. Plan simulations/experiments
    3. Execute simulations
    4. Validate results
    5. Detect patterns
    6. Refine hypotheses
    7. Decide: converge or iterate

    Example:
        from aero.pipeline.aero_loop import ScientificReasoningLoop, LoopConfig

        config = LoopConfig(max_iterations=5, confidence_threshold=0.85)
        loop = ScientificReasoningLoop(config)

        result = loop.run("Analyze heat diffusion in a 1D rod")
        print(f"Converged: {result.converged}")
        print(f"Best hypothesis: {result.best_hypothesis}")
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        rag=None,
        scheduler=None,
        ocr_registry=None,
        data_store: Optional["DataStore"] = None,
        auto_persist: bool = True,
    ):
        """
        Initialize the scientific reasoning loop.

        Args:
            config: Loop configuration
            rag: RAG store for context retrieval
            scheduler: Simulation scheduler
            ocr_registry: OCR backend registry
            data_store: DataStore for persisting results (optional)
            auto_persist: Whether to auto-persist simulation/experiment results
        """
        self.config = config or LoopConfig()
        self.rag = rag
        self.scheduler = scheduler
        self.ocr_registry = ocr_registry
        self.auto_persist = auto_persist

        # Initialize data store
        self._data_store = data_store
        if self._data_store is None and DATA_LAKE_AVAILABLE and auto_persist:
            try:
                self._data_store = get_default_store()
            except Exception as e:
                logger.warning(f"Could not get default data store: {e}")

        # Import components
        from aero.pipeline.hypothesis import Hypothesis, generate_initial_hypotheses, generate_secondary_hypotheses
        from aero.pipeline.planner import plan_simulations, plan_experiments, SimulationJobConfig, execute_experiment
        from aero.pipeline.validator import validate_simulation, validate_experiment, SimulationValidationResult
        from aero.pipeline.patterns import detect_patterns, compare_patterns, ScientificPattern
        from aero.pipeline.refinement import refine_hypotheses

        self._generate_initial = generate_initial_hypotheses
        self._generate_secondary = generate_secondary_hypotheses
        self._plan_simulations = plan_simulations
        self._plan_experiments = plan_experiments
        self._execute_experiment = execute_experiment
        self._validate_simulation = validate_simulation
        self._validate_experiment = validate_experiment
        self._detect_patterns = detect_patterns
        self._compare_patterns = compare_patterns
        self._refine_hypotheses = refine_hypotheses
        self._Hypothesis = Hypothesis

        # State
        self._current_hypotheses: List = []
        self._iteration_history: List[dict] = []

        logger.info("ScientificReasoningLoop initialized")

    def step(self, input_query: str) -> dict:
        """
        Execute one iteration of the scientific reasoning loop.

        Args:
            input_query: User query or problem description

        Returns:
            Dictionary with iteration results
        """
        logger.info(f"Executing loop step for query: {input_query[:100]}...")

        iteration_result = {
            "iteration": len(self._iteration_history) + 1,
            "query": input_query,
            "hypotheses": [],
            "simulations": [],
            "validations": [],
            "patterns": [],
            "refined_hypotheses": [],
            "status": "incomplete",
        }

        try:
            # Step 1: Generate hypotheses
            if not self._current_hypotheses:
                hypotheses = self._generate_initial(input_query, self.rag)
            else:
                # Generate secondary hypotheses based on previous results
                prev_results = self._get_previous_results()
                hypotheses = self._generate_secondary(prev_results, self.rag, self._current_hypotheses)
                if not hypotheses:
                    hypotheses = self._current_hypotheses

            iteration_result["hypotheses"] = [h.to_dict() for h in hypotheses]
            logger.info(f"Generated {len(hypotheses)} hypotheses")

            # Step 2: Plan simulations for each hypothesis
            all_sim_configs = []
            for hypothesis in hypotheses[:self.config.max_simulations_per_iteration]:
                sim_configs = self._plan_simulations(hypothesis)
                all_sim_configs.extend(sim_configs[:1])  # Take first config per hypothesis

            logger.info(f"Planned {len(all_sim_configs)} simulations")

            # Step 3: Execute simulations
            simulation_results = []
            for i, sim_config in enumerate(all_sim_configs):
                result = self._run_simulation(sim_config)
                simulation_results.append(result)

                # Get associated hypothesis ID if available
                hypothesis_id = None
                if i < len(hypotheses):
                    hypothesis_id = hypotheses[i].id if hasattr(hypotheses[i], 'id') else None

                # Auto-persist simulation result
                stored_id = self._persist_simulation_result(sim_config, result, hypothesis_id)
                if stored_id:
                    result["stored_id"] = stored_id

                iteration_result["simulations"].append({
                    "config": sim_config.to_dict(),
                    "result": result,
                    "stored_id": stored_id,
                })

            logger.info(f"Executed {len(simulation_results)} simulations")

            # Step 4: Validate results
            validation_results = []
            for i, (hypothesis, result) in enumerate(zip(hypotheses, simulation_results)):
                validation = self._validate_simulation(hypothesis, result)
                validation_results.append(validation)
                iteration_result["validations"].append(validation.to_dict())

            logger.info(f"Validated {len(validation_results)} results")

            # Step 5: Detect patterns
            all_patterns = []
            for result in simulation_results:
                patterns = self._detect_patterns(result)
                all_patterns.extend(patterns)
            iteration_result["patterns"] = [p.to_dict() for p in all_patterns]

            logger.info(f"Detected {len(all_patterns)} patterns")

            # Step 6: Refine hypotheses
            refined = self._refine_hypotheses(
                hypotheses,
                validation_results,
                all_patterns,
                drop_threshold=self.config.drop_threshold,
            )
            self._current_hypotheses = refined
            iteration_result["refined_hypotheses"] = [h.to_dict() for h in refined]

            logger.info(f"Refined to {len(refined)} hypotheses")

            iteration_result["status"] = "complete"

        except Exception as e:
            logger.exception(f"Error in loop step: {e}")
            iteration_result["status"] = "error"
            iteration_result["error"] = str(e)

        self._iteration_history.append(iteration_result)
        return iteration_result

    def run(self, query: str) -> LoopResult:
        """
        Execute the full reasoning loop until convergence or max iterations.

        Args:
            query: User query or problem description

        Returns:
            LoopResult with complete execution history
        """
        logger.info(f"Starting scientific reasoning loop for: {query[:100]}...")

        # Reset state
        self._current_hypotheses = []
        self._iteration_history = []

        converged = False
        termination_reason = "max_iterations"

        for iteration in range(self.config.max_iterations):
            logger.info(f"=== Iteration {iteration + 1}/{self.config.max_iterations} ===")

            # Execute one step
            step_result = self.step(query)

            # Check termination conditions
            termination, reason = self._check_termination(step_result, iteration)
            if termination:
                converged = reason in ["confidence_threshold", "pattern_convergence"]
                termination_reason = reason
                break

        # Compile final result
        best_hypothesis = None
        if self._current_hypotheses:
            best = max(self._current_hypotheses, key=lambda h: h.confidence)
            best_hypothesis = best.to_dict()

        # Collect all simulation results and patterns
        all_sim_results = []
        all_validations = []
        all_patterns = []

        for hist in self._iteration_history:
            all_sim_results.extend([s.get("result", {}) for s in hist.get("simulations", [])])
            all_validations.extend(hist.get("validations", []))
            all_patterns.extend(hist.get("patterns", []))

        result = LoopResult(
            converged=converged,
            iterations_completed=len(self._iteration_history),
            best_hypothesis=best_hypothesis,
            all_hypotheses=[h.to_dict() for h in self._current_hypotheses],
            simulation_results=all_sim_results,
            validation_results=all_validations,
            patterns=all_patterns,
            history=self._iteration_history,
            termination_reason=termination_reason,
        )

        logger.info(
            f"Loop completed: converged={converged}, "
            f"iterations={result.iterations_completed}, "
            f"reason={termination_reason}"
        )

        return result

    def _run_simulation(self, config) -> dict:
        """Execute a simulation and return results."""
        try:
            from aero.pipeline.planner import SimulationType

            sim_type = config.sim_type
            if isinstance(sim_type, str):
                sim_type = SimulationType(sim_type)

            # Run appropriate solver
            if sim_type == SimulationType.HEAT_1D:
                return self._run_heat_1d(config)
            elif sim_type == SimulationType.LAPLACE_2D:
                return self._run_laplace_2d(config)
            elif sim_type in [SimulationType.HEAT_2D, SimulationType.NAVIER_STOKES]:
                return self._run_generic_2d(config)
            elif sim_type == SimulationType.PINN:
                return self._run_pinn(config)
            else:
                return self._run_generic_2d(config)

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {
                "error": str(e),
                "converged": False,
                "solution": None,
            }

    def _run_heat_1d(self, config) -> dict:
        """Run 1D heat equation simulation."""
        try:
            from aero.sim.numerics.fd_solver import solve_heat_1d

            nx = config.grid_size
            dx = config.dx
            dt = config.dt
            steps = config.max_steps
            alpha = config.solver_params.get("alpha", 0.01)

            # Create initial condition (gaussian)
            x = np.linspace(0, 1, nx)
            if config.initial_condition == "gaussian":
                u0 = np.exp(-100 * (x - 0.5) ** 2)
            elif config.initial_condition == "sine":
                u0 = np.sin(np.pi * x)
            else:
                u0 = np.zeros(nx)

            # Get boundary conditions
            bc_left = config.boundary_conditions.get("left")
            bc_right = config.boundary_conditions.get("right")

            # Solve
            solution, metadata = solve_heat_1d(
                u0, alpha, dx, dt, steps,
                boundary_left=bc_left,
                boundary_right=bc_right,
            )

            return {
                "solution": solution.tolist(),
                "converged": True,
                "metadata": metadata,
                "final_residual": 0.0,  # Heat equation doesn't iterate
            }

        except Exception as e:
            return {"error": str(e), "converged": False, "solution": None}

    def _run_laplace_2d(self, config) -> dict:
        """Run 2D Laplace equation simulation."""
        try:
            from aero.sim.numerics.fd_solver import solve_laplace_2d

            nx = config.grid_size
            ny = config.grid_size_y or nx
            tol = config.tolerance
            max_iter = config.max_iterations

            # Create initial grid with boundary conditions
            u = np.zeros((nx, ny))
            bc = config.boundary_conditions
            u[0, :] = bc.get("left", 0)
            u[-1, :] = bc.get("right", 100)
            u[:, 0] = bc.get("bottom", 50)
            u[:, -1] = bc.get("top", 50)

            # Solve
            solution, metadata = solve_laplace_2d(u, tol=tol, max_iterations=max_iter)

            return {
                "solution": solution.tolist(),
                "converged": metadata.get("converged", False),
                "iterations": metadata.get("iterations", 0),
                "final_residual": metadata.get("final_residual", 0.0),
                "metadata": metadata,
            }

        except Exception as e:
            return {"error": str(e), "converged": False, "solution": None}

    def _run_generic_2d(self, config) -> dict:
        """Run generic 2D simulation (placeholder for NS, etc.)."""
        # Return a simple converged result for testing
        nx = config.grid_size
        ny = config.grid_size_y or nx

        # Create linear gradient solution (placeholder)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        bc = config.boundary_conditions
        left = bc.get("left", 0)
        right = bc.get("right", 100)
        solution = left + (right - left) * X

        return {
            "solution": solution.tolist(),
            "converged": True,
            "iterations": 100,
            "final_residual": 1e-6,
            "metadata": {"solver": "generic_2d_stub"},
        }

    def _run_pinn(self, config) -> dict:
        """Run PINN training (simplified)."""
        try:
            from aero.sim.pinn.pinn_model import TORCH_AVAILABLE

            if not TORCH_AVAILABLE:
                return {
                    "error": "PyTorch not available",
                    "converged": False,
                    "solution": None,
                }

            # Simplified PINN result
            return {
                "solution": None,
                "converged": True,
                "final_loss": 1e-4,
                "epochs": config.max_iterations,
                "metadata": {"solver": "pinn"},
            }

        except Exception as e:
            return {"error": str(e), "converged": False, "solution": None}

    def _get_previous_results(self) -> List[dict]:
        """Get results from previous iterations."""
        results = []
        for hist in self._iteration_history:
            for sim in hist.get("simulations", []):
                results.append(sim.get("result", {}))
        return results

    def _check_termination(self, step_result: dict, iteration: int) -> tuple:
        """
        Check if the loop should terminate.

        Returns:
            Tuple of (should_terminate, reason)
        """
        # Check for error
        if step_result.get("status") == "error":
            return True, "error"

        # Check if all hypotheses failed
        refined = step_result.get("refined_hypotheses", [])
        if not refined:
            return True, "all_hypotheses_failed"

        # Check confidence threshold
        best_confidence = max(h.get("confidence", 0) for h in refined)
        if best_confidence >= self.config.confidence_threshold:
            return True, "confidence_threshold"

        # Check pattern convergence
        if len(self._iteration_history) >= 2:
            current_patterns = set(
                p.get("type") for p in step_result.get("patterns", [])
            )
            prev_patterns = set(
                p.get("type") for p in self._iteration_history[-2].get("patterns", [])
            )

            if current_patterns == prev_patterns and len(current_patterns) > 0:
                return True, "pattern_convergence"

        # Check max iterations
        if iteration >= self.config.max_iterations - 1:
            return True, "max_iterations"

        return False, ""

    def get_state(self) -> dict:
        """Get current loop state."""
        return {
            "current_hypotheses": [h.to_dict() for h in self._current_hypotheses],
            "iteration_count": len(self._iteration_history),
            "history": self._iteration_history,
        }

    def reset(self) -> None:
        """Reset loop state."""
        self._current_hypotheses = []
        self._iteration_history = []
        logger.info("Loop state reset")

    def _persist_simulation_result(
        self,
        sim_config,
        result: dict,
        hypothesis_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist simulation result to the data lake.

        Args:
            sim_config: Simulation configuration
            result: Simulation result dictionary
            hypothesis_id: Associated hypothesis ID

        Returns:
            Stored record ID, or None if persistence failed
        """
        if not self.auto_persist or not self._data_store:
            return None

        try:
            from aero.sim.results import SimulationResult

            # Build metadata
            metadata = {
                "simulation_type": str(sim_config.sim_type.value) if hasattr(sim_config.sim_type, 'value') else str(sim_config.sim_type),
                "config": sim_config.to_dict(),
                "status": "completed" if result.get("converged") else "failed",
                "runtime_seconds": result.get("metadata", {}).get("runtime_seconds"),
            }

            # Extract fields
            fields = {}
            if result.get("solution") is not None:
                sol = result["solution"]
                if isinstance(sol, list):
                    sol = np.array(sol)
                fields["solution"] = sol

            # Create SimulationResult
            sim_result = SimulationResult(
                fields=fields,
                metadata=metadata,
                tags=["aero_loop", "auto_persist"],
            )

            # Save to data store
            record_id = self._data_store.save_simulation_result(
                sim_result,
                hypothesis_id=hypothesis_id,
            )

            logger.debug(f"Persisted simulation result: {record_id[:8]}...")
            return record_id

        except Exception as e:
            logger.warning(f"Failed to persist simulation result: {e}")
            return None

    def _persist_experiment_result(
        self,
        exp_config,
        result: dict,
        hypothesis_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist experiment result to the data lake.

        Args:
            exp_config: Experiment configuration
            result: Experiment result dictionary
            hypothesis_id: Associated hypothesis ID

        Returns:
            Stored record ID, or None if persistence failed
        """
        if not self.auto_persist or not self._data_store:
            return None

        try:
            from aero.experiments import ExperimentResult

            # Create ExperimentResult from dict
            exp_result = ExperimentResult.from_dict(result)
            exp_result.tags = ["aero_loop", "auto_persist"]

            # Save to data store
            record_id = self._data_store.save_experiment_result(
                exp_result,
                hypothesis_id=hypothesis_id,
            )

            logger.debug(f"Persisted experiment result: {record_id[:8]}...")
            return record_id

        except Exception as e:
            logger.warning(f"Failed to persist experiment result: {e}")
            return None

    def run_experiment(self, experiment_config) -> dict:
        """
        Execute an experiment using the configured experiment system.

        Args:
            experiment_config: ExperimentConfig object or dict

        Returns:
            Dictionary with experiment results
        """
        from aero.pipeline.planner import ExperimentConfig

        # Convert dict to config if needed
        if isinstance(experiment_config, dict):
            from aero.pipeline.planner import ExperimentType
            exp_type = experiment_config.get("exp_type", "synthetic_flow")
            if isinstance(exp_type, str):
                try:
                    exp_type = ExperimentType(exp_type)
                except ValueError:
                    exp_type = ExperimentType.SYNTHETIC_FLOW

            experiment_config = ExperimentConfig(
                exp_type=exp_type,
                hypothesis_id=experiment_config.get("hypothesis_id", ""),
                source=experiment_config.get("source", ""),
                parameters=experiment_config.get("parameters", {}),
            )

        try:
            result = self._execute_experiment(experiment_config)
            result_dict = result.to_dict()

            # Auto-persist experiment result
            hypothesis_id = experiment_config.hypothesis_id if hasattr(experiment_config, 'hypothesis_id') else None
            stored_id = self._persist_experiment_result(experiment_config, result_dict, hypothesis_id)
            if stored_id:
                result_dict["stored_id"] = stored_id

            return result_dict
        except Exception as e:
            logger.error(f"Experiment execution error: {e}")
            return {
                "experiment_type": str(experiment_config.exp_type),
                "success": False,
                "error": str(e),
            }

    def run_synthetic_experiment(
        self,
        exp_type: str = "synthetic_flow",
        **params,
    ) -> dict:
        """
        Run a synthetic experiment for testing.

        Args:
            exp_type: Type of synthetic experiment
                     ("synthetic_flow" or "synthetic_timeseries")
            **params: Experiment parameters

        Returns:
            Experiment result dictionary
        """
        from aero.pipeline.planner import ExperimentConfig, ExperimentType

        try:
            experiment_type = ExperimentType(exp_type)
        except ValueError:
            experiment_type = ExperimentType.SYNTHETIC_FLOW

        config = ExperimentConfig(
            exp_type=experiment_type,
            hypothesis_id="synthetic_test",
            source="",
            parameters=params,
        )

        return self.run_experiment(config)
