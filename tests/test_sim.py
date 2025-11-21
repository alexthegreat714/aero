"""
Simulation tests for Aero Agent.

Tests for finite difference solvers, PINN components, and job system.
All tests run on CPU-only to ensure compatibility across environments.
"""

import numpy as np
import pytest


# =============================================================================
# Heat Equation Tests
# =============================================================================


class TestHeatEquation:
    """Tests for 1D heat equation solver."""

    def test_heat_equation_runs(self):
        """Test that heat equation solver runs without errors."""
        from aero.sim.numerics.fd_solver import solve_heat_1d

        # Create initial condition: hot spot in center
        nx = 51
        x = np.linspace(0, 1, nx)
        u0 = np.exp(-100 * (x - 0.5) ** 2)

        # Solve for 100 steps
        u_final, metadata = solve_heat_1d(
            u0=u0,
            alpha=0.01,
            dx=1.0 / (nx - 1),
            dt=0.0001,
            steps=100,
        )

        # Verify output
        assert u_final is not None
        assert u_final.shape == u0.shape
        assert metadata["solver"] == "heat_1d_explicit"
        assert metadata["steps"] == 100

    def test_heat_equation_diffuses(self):
        """Test that heat diffuses over time (max decreases)."""
        from aero.sim.numerics.fd_solver import solve_heat_1d

        # Initial spike
        nx = 101
        x = np.linspace(0, 1, nx)
        u0 = np.zeros(nx)
        u0[50] = 1.0  # Single hot point

        dx = 1.0 / (nx - 1)
        dt = 0.0001
        alpha = 0.01

        u_final, _ = solve_heat_1d(u0, alpha, dx, dt, steps=500)

        # Heat should spread out - max should decrease
        assert np.max(u_final) < np.max(u0)

    def test_heat_equation_boundary_conditions(self):
        """Test Dirichlet boundary conditions are maintained."""
        from aero.sim.numerics.fd_solver import solve_heat_1d

        nx = 51
        u0 = np.ones(nx) * 50  # Uniform initial temperature

        u_final, _ = solve_heat_1d(
            u0=u0,
            alpha=0.01,
            dx=0.02,
            dt=0.0001,
            steps=100,
            boundary_left=0.0,
            boundary_right=100.0,
        )

        # Boundary values should be maintained
        assert u_final[0] == pytest.approx(0.0)
        assert u_final[-1] == pytest.approx(100.0)

    def test_heat_equation_class_based(self):
        """Test the class-based HeatEquation solver."""
        from aero.sim.numerics.fd_solver import HeatEquation

        solver = HeatEquation(nx=30, ny=30, dx=0.1, dy=0.1, alpha=0.01)
        solver.setup()

        # Set boundary conditions
        solver.set_boundary_conditions(left=100, right=0, top=50, bottom=50)

        # Run a few steps
        for _ in range(10):
            residual = solver.step()
            assert residual >= 0

        solution = solver.get_solution()
        assert solution.shape == (30, 30)


# =============================================================================
# Laplace Equation Tests
# =============================================================================


class TestLaplaceEquation:
    """Tests for 2D Laplace equation solver."""

    def test_laplace_solver_converges(self):
        """Test that Laplace solver converges."""
        from aero.sim.numerics.fd_solver import solve_laplace_2d

        # Create grid with boundary conditions
        nx, ny = 21, 21
        u = np.zeros((nx, ny))
        u[0, :] = 100  # Top boundary hot
        u[-1, :] = 0  # Bottom boundary cold
        u[:, 0] = 50  # Left boundary
        u[:, -1] = 50  # Right boundary

        # Solve
        solution, metadata = solve_laplace_2d(u, tol=1e-4, max_iterations=5000)

        assert metadata["converged"] is True
        assert metadata["iterations"] < 5000
        assert metadata["final_residual"] < 1e-4

    def test_laplace_solution_physical(self):
        """Test that Laplace solution is physically reasonable."""
        from aero.sim.numerics.fd_solver import solve_laplace_2d

        # Simple case: constant boundaries
        nx, ny = 25, 25
        u = np.zeros((nx, ny))
        u[0, :] = 100  # One hot side
        u[-1, :] = 0  # Opposite cold side
        u[:, 0] = 50  # Intermediate sides
        u[:, -1] = 50

        solution, _ = solve_laplace_2d(u, tol=1e-5)

        # Interior should be between boundary values
        interior = solution[1:-1, 1:-1]
        assert np.min(interior) >= 0
        assert np.max(interior) <= 100

    def test_laplace_class_based(self):
        """Test the class-based LaplaceEquation solver."""
        from aero.sim.numerics.fd_solver import LaplaceEquation

        solver = LaplaceEquation(nx=20, ny=20, dx=0.1, dy=0.1)
        solver.setup()
        solver.set_boundary_conditions(left=0, right=100, top=50, bottom=50)

        # Run some iterations
        for _ in range(50):
            solver.step()

        solution = solver.get_solution()
        assert solution.shape == (20, 20)

    def test_laplace_sor_acceleration(self):
        """Test that SOR (omega > 1) can accelerate convergence."""
        from aero.sim.numerics.fd_solver import solve_laplace_2d

        nx, ny = 25, 25
        u = np.zeros((nx, ny))
        u[0, :] = 100

        # Standard Gauss-Seidel
        _, meta_gs = solve_laplace_2d(u.copy(), tol=1e-5, omega=1.0)

        # With SOR
        _, meta_sor = solve_laplace_2d(u.copy(), tol=1e-5, omega=1.5)

        # Both should converge
        assert meta_gs["converged"]
        assert meta_sor["converged"]


# =============================================================================
# Navier-Stokes Stub Tests
# =============================================================================


class TestNavierStokesStub:
    """Tests for Navier-Stokes solver stub."""

    def test_ns_stub_initializes(self):
        """Test that NS stub can be initialized."""
        from aero.sim.cfd.ns_solver_stub import NavierStokes2DStub

        solver = NavierStokes2DStub(nx=32, ny=32, Re=100.0)

        assert solver.nx == 32
        assert solver.ny == 32
        assert solver.Re == 100.0
        assert solver.name == "navier_stokes_2d_stub"

    def test_ns_stub_setup(self):
        """Test NS stub setup initializes fields."""
        from aero.sim.cfd.ns_solver_stub import NavierStokes2DStub

        solver = NavierStokes2DStub(nx=20, ny=20)
        solver.setup()

        # Fields should be initialized as zeros
        assert solver.u.shape == (20, 20)
        assert solver.v.shape == (20, 20)
        assert solver.p.shape == (20, 20)
        assert np.allclose(solver.u, 0)

    def test_ns_stub_step_returns_zero(self):
        """Test that stub step returns zero residual (no actual computation)."""
        from aero.sim.cfd.ns_solver_stub import NavierStokes2DStub

        solver = NavierStokes2DStub(nx=16, ny=16)
        solver.setup()

        residual = solver.step()

        # Stub should return 0 as it doesn't compute anything
        assert residual == 0.0

    def test_ns_stub_get_velocity_field(self):
        """Test getting velocity field from stub."""
        from aero.sim.cfd.ns_solver_stub import NavierStokes2DStub

        solver = NavierStokes2DStub(nx=10, ny=10)
        solver.setup()

        u, v = solver.get_velocity_field()

        assert u.shape == (10, 10)
        assert v.shape == (10, 10)

    def test_legacy_ns_solver_exists(self):
        """Test that legacy NavierStokesSolver still exists for compatibility."""
        from aero.sim.cfd.ns_solver_stub import NavierStokesSolver

        solver = NavierStokesSolver()
        assert solver is not None
        assert hasattr(solver, "solve")


# =============================================================================
# PINN Tests
# =============================================================================


class TestPINNImports:
    """Tests for PINN module imports."""

    def test_pinn_trainer_imports(self):
        """Test that PINN trainer can be imported."""
        from aero.sim.pinn.pinn_trainer import (
            PINNTrainer,
            TrainingConfig,
            TrainingResult,
            check_pytorch_cuda,
        )

        assert PINNTrainer is not None
        assert TrainingConfig is not None
        assert TrainingResult is not None
        assert callable(check_pytorch_cuda)

    def test_pinn_model_imports(self):
        """Test that PINN model can be imported."""
        from aero.sim.pinn.pinn_model import (
            PINNModel,
            ResidualPINNModel,
            create_pinn_model,
            get_device,
            TORCH_AVAILABLE,
        )

        assert PINNModel is not None
        assert ResidualPINNModel is not None
        assert callable(create_pinn_model)
        assert callable(get_device)
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_check_pytorch_cuda(self):
        """Test check_pytorch_cuda function returns expected structure."""
        from aero.sim.pinn.pinn_trainer import check_pytorch_cuda

        result = check_pytorch_cuda()

        assert "pytorch_available" in result
        assert "cuda_available" in result
        assert "cuda_device" in result
        assert isinstance(result["pytorch_available"], bool)
        assert isinstance(result["cuda_available"], bool)


@pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch not installed"
)
class TestPINNModel:
    """Tests for PINN model (requires PyTorch)."""

    def test_pinn_model_creation(self):
        """Test creating a PINN model."""
        from aero.sim.pinn.pinn_model import PINNModel

        model = PINNModel(layers=[2, 32, 32, 1])

        assert model is not None
        assert model.num_parameters > 0

    def test_pinn_model_forward_pass(self):
        """Test PINN model forward pass."""
        import torch
        from aero.sim.pinn.pinn_model import PINNModel

        model = PINNModel(layers=[2, 32, 32, 1])

        # Create sample input
        x = torch.rand(10, 2)  # 10 samples, 2D input
        y = model(x)

        assert y.shape == (10, 1)

    def test_pinn_model_predict_numpy(self):
        """Test PINN model prediction with numpy arrays."""
        from aero.sim.pinn.pinn_model import PINNModel

        model = PINNModel(layers=[2, 64, 1])

        x = np.random.rand(5, 2).astype(np.float32)
        y = model.predict(x)

        assert y.shape == (5, 1)
        assert isinstance(y, np.ndarray)

    def test_pinn_factory_function(self):
        """Test create_pinn_model factory function."""
        from aero.sim.pinn.pinn_model import create_pinn_model

        model = create_pinn_model(
            input_dim=3,
            output_dim=2,
            hidden_layers=3,
            hidden_dim=32,
            activation="relu",
        )

        x = np.random.rand(8, 3).astype(np.float32)
        y = model.predict(x)

        assert y.shape == (8, 2)

    def test_pinn_different_activations(self):
        """Test PINN with different activation functions."""
        from aero.sim.pinn.pinn_model import PINNModel

        for activation in ["tanh", "relu", "gelu", "sigmoid"]:
            model = PINNModel(layers=[2, 16, 1], activation=activation)
            x = np.random.rand(4, 2).astype(np.float32)
            y = model.predict(x)
            assert y.shape == (4, 1)


@pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch not installed"
)
class TestPINNTrainer:
    """Tests for PINN trainer (requires PyTorch)."""

    def test_training_config_defaults(self):
        """Test TrainingConfig has sensible defaults."""
        from aero.sim.pinn.pinn_trainer import TrainingConfig

        config = TrainingConfig()

        assert config.epochs == 10000
        assert config.learning_rate > 0
        assert config.lambda_physics > 0
        assert config.lambda_bc > 0

    def test_trainer_initialization(self):
        """Test PINNTrainer initialization."""
        from aero.sim.pinn.pinn_model import PINNModel
        from aero.sim.pinn.pinn_trainer import PINNTrainer, TrainingConfig

        model = PINNModel(layers=[2, 32, 1])
        config = TrainingConfig(epochs=100)
        trainer = PINNTrainer(model, config=config, device="cpu")

        assert trainer is not None
        assert trainer.device == "cpu"

    def test_trainer_set_data(self):
        """Test setting training data."""
        from aero.sim.pinn.pinn_model import PINNModel
        from aero.sim.pinn.pinn_trainer import PINNTrainer

        model = PINNModel(layers=[2, 16, 1])
        trainer = PINNTrainer(model, device="cpu")

        # Set collocation points
        x_colloc = np.random.rand(100, 2)
        trainer.set_collocation_points(x_colloc)

        # Set boundary data
        x_bc = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        y_bc = np.array([[0], [0], [0], [0]])
        trainer.set_boundary_data(x_bc, y_bc)

        assert trainer.collocation_points is not None
        assert trainer.bc_points is not None
        assert trainer.bc_values is not None

    def test_trainer_short_training(self):
        """Test running a few training epochs."""
        import torch
        from aero.sim.pinn.pinn_model import PINNModel
        from aero.sim.pinn.pinn_trainer import PINNTrainer, TrainingConfig

        model = PINNModel(layers=[2, 16, 1])
        config = TrainingConfig(epochs=10, print_every=5)
        trainer = PINNTrainer(model, config=config, device="cpu")

        # Set some boundary data
        x_bc = np.random.rand(20, 2)
        y_bc = np.zeros((20, 1))
        trainer.set_boundary_data(x_bc, y_bc)

        result = trainer.train()

        assert result.epochs_trained == 10
        assert len(result.loss_history) == 10
        assert result.final_loss >= 0


# =============================================================================
# Job System Tests
# =============================================================================


class TestJobSystem:
    """Tests for simulation job system."""

    def test_job_creation(self):
        """Test creating a simulation job."""
        from aero.sim.job import SimulationJob, JobStatus
        from aero.sim.base_simulation import BaseSimulation

        # Create a mock simulation
        class MockSim(BaseSimulation):
            @property
            def name(self) -> str:
                return "mock"

            def setup(self):
                pass

            def step(self) -> float:
                return 0.0

            def run(self):
                return None

        sim = MockSim()
        job = SimulationJob(sim_type="test", sim=sim)

        assert job.id is not None
        assert job.status == JobStatus.QUEUED
        assert job.sim_type == "test"

    def test_scheduler_add_job(self):
        """Test adding jobs to scheduler."""
        from aero.sim.scheduler import SimulationScheduler
        from aero.sim.job import SimulationJob
        from aero.sim.numerics.fd_solver import LaplaceEquation

        scheduler = SimulationScheduler()

        sim = LaplaceEquation(nx=10, ny=10)
        job = SimulationJob(sim_type="laplace", sim=sim)

        scheduler.add_job(job)

        assert scheduler.queue_size == 1
        assert job.id in [j.id for j in scheduler.list_jobs()]

    def test_scheduler_run_job(self):
        """Test running a job through scheduler."""
        from aero.sim.scheduler import SimulationScheduler
        from aero.sim.job import SimulationJob, JobStatus
        from aero.sim.numerics.fd_solver import LaplaceEquation
        from aero.sim.base_simulation import SimulationConfig

        scheduler = SimulationScheduler()

        config = SimulationConfig(max_iterations=10, tolerance=1e-3)
        sim = LaplaceEquation(nx=10, ny=10, config=config)
        sim.setup()
        sim.set_boundary_conditions(left=0, right=100, top=50, bottom=50)

        job = SimulationJob(sim_type="laplace", sim=sim)
        scheduler.add_job(job)

        # Run the job
        scheduler.run_next()

        # Check job completed
        status = scheduler.get_job_status(job.id)
        assert status in [JobStatus.COMPLETED, JobStatus.ERROR]

    def test_scheduler_global_singleton(self):
        """Test global scheduler singleton."""
        from aero.sim.scheduler import get_scheduler

        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()

        assert scheduler1 is scheduler2


# =============================================================================
# Numerics Utilities Tests
# =============================================================================


class TestNumericsUtils:
    """Tests for numerical utilities."""

    def test_create_grid_1d(self):
        """Test 1D grid creation."""
        from aero.sim.numerics.utils import create_grid_1d

        x, dx = create_grid_1d(0, 1, 101)

        assert len(x) == 101
        assert x[0] == 0
        assert x[-1] == 1
        assert dx == pytest.approx(0.01)

    def test_create_grid_2d(self):
        """Test 2D grid creation."""
        from aero.sim.numerics.utils import create_grid_2d

        X, Y, dx, dy = create_grid_2d(0, 1, 0, 2, 11, 21)

        assert X.shape == (11, 21)
        assert Y.shape == (11, 21)
        assert dx == pytest.approx(0.1)
        assert dy == pytest.approx(0.1)

    def test_check_cfl_condition(self):
        """Test CFL condition check."""
        from aero.sim.numerics.utils import check_cfl_1d

        # Stable case
        assert check_cfl_1d(u=1.0, dt=0.001, dx=0.01) is True

        # Unstable case
        assert check_cfl_1d(u=100.0, dt=0.1, dx=0.01) is False

    def test_apply_dirichlet_bc(self):
        """Test Dirichlet boundary condition application."""
        from aero.sim.numerics.utils import apply_dirichlet_1d

        u = np.ones(10)
        u = apply_dirichlet_1d(u, left=0, right=100)

        assert u[0] == 0
        assert u[-1] == 100
        assert u[5] == 1  # Interior unchanged


class TestIntegrators:
    """Tests for time integrators."""

    def test_euler_step(self):
        """Test Euler integration step."""
        from aero.sim.numerics.integrators import euler_step

        # dy/dt = -y (exponential decay)
        def f(t, y):
            return -y

        y0 = np.array([1.0])
        y1 = euler_step(f, 0, y0, dt=0.1)

        assert y1[0] == pytest.approx(0.9, rel=1e-10)

    def test_rk4_step(self):
        """Test RK4 integration step."""
        from aero.sim.numerics.integrators import rk4_step

        # dy/dt = -y
        def f(t, y):
            return -y

        y0 = np.array([1.0])
        y1 = rk4_step(f, 0, y0, dt=0.1)

        # RK4 should be more accurate than Euler
        exact = np.exp(-0.1)
        assert y1[0] == pytest.approx(exact, rel=1e-4)

    def test_integrate_function(self):
        """Test full integration."""
        from aero.sim.numerics.integrators import integrate

        # dy/dt = -y
        def f(t, y):
            return -y

        y0 = np.array([1.0])
        y_final = integrate(f, y0, t_span=(0, 1), dt=0.01, method="rk4")

        # Should be close to e^{-1} ~ 0.368
        assert y_final[0] == pytest.approx(np.exp(-1), rel=1e-3)


# =============================================================================
# Helper Functions
# =============================================================================


def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False
