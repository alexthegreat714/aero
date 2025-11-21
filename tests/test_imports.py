"""
Import tests for Aero Agent.

Verifies that all modules can be imported without errors.
"""

import pytest


class TestConfigImports:
    """Test config module imports."""

    def test_import_config_module(self):
        """Test importing config module."""
        from aero.config import loader
        assert hasattr(loader, 'load_config')
        assert hasattr(loader, 'get_config')
        assert hasattr(loader, 'AeroConfig')

    def test_import_config_functions(self):
        """Test importing config functions directly."""
        from aero.config.loader import load_config, get_config, AeroConfig
        assert callable(load_config)
        assert callable(get_config)

    def test_load_default_config(self):
        """Test loading default configuration."""
        from aero.config.loader import load_config
        config = load_config()
        assert config is not None
        assert config.get("app.name") == "Aero Agent"


class TestCoreImports:
    """Test core module imports."""

    def test_import_base_agent(self):
        """Test importing BaseAgent."""
        from aero.core.base_agent import BaseAgent
        assert BaseAgent is not None

    def test_import_base_module(self):
        """Test importing BaseModule."""
        from aero.core.base_module import BaseModule
        assert BaseModule is not None

    def test_import_events(self):
        """Test importing event system."""
        from aero.core.events import EventBus, Event
        assert EventBus is not None
        assert Event is not None

    def test_import_registry(self):
        """Test importing registry system."""
        from aero.core.registry import Registry, RegistryManager
        assert Registry is not None
        assert RegistryManager is not None


class TestRAGImports:
    """Test RAG module imports."""

    def test_import_rag_store(self):
        """Test importing RAGStore."""
        from aero.rag.store import RAGStore, Document
        assert RAGStore is not None
        assert Document is not None

    def test_import_embedder(self):
        """Test importing Embedder."""
        from aero.rag.embedder import Embedder, BaseEmbedder
        assert Embedder is not None
        assert BaseEmbedder is not None

    def test_import_reader(self):
        """Test importing DocumentReader."""
        from aero.rag.reader import DocumentReader, ParsedDocument
        assert DocumentReader is not None
        assert ParsedDocument is not None


class TestSimImports:
    """Test simulation module imports."""

    def test_import_base_simulation(self):
        """Test importing BaseSimulation."""
        from aero.sim.base_simulation import BaseSimulation, SimulationResult
        assert BaseSimulation is not None
        assert SimulationResult is not None

    def test_import_fd_solver(self):
        """Test importing FD solver."""
        from aero.sim.numerics.fd_solver import FDSolver, LaplaceEquation, HeatEquation
        assert FDSolver is not None
        assert LaplaceEquation is not None
        assert HeatEquation is not None

    def test_import_ns_solver(self):
        """Test importing Navier-Stokes solver."""
        from aero.sim.cfd.ns_solver_stub import NavierStokesSolver
        assert NavierStokesSolver is not None

    def test_import_pinn_solver(self):
        """Test importing PINN solver."""
        from aero.sim.pinn.pinn_stub import PINNSolver
        assert PINNSolver is not None


class TestExperimentsImports:
    """Test experiments module imports."""

    def test_import_camera(self):
        """Test importing camera utilities."""
        from aero.experiments.camera import Camera, CameraCapture
        assert Camera is not None
        assert CameraCapture is not None

    def test_import_sensors(self):
        """Test importing sensor utilities."""
        from aero.experiments.sensors import SensorReader, SensorData
        assert SensorReader is not None
        assert SensorData is not None

    def test_import_analyzer(self):
        """Test importing analyzer."""
        from aero.experiments.analyzer import DataAnalyzer
        assert DataAnalyzer is not None


class TestWebImports:
    """Test web module imports."""

    def test_import_search(self):
        """Test importing web search."""
        from aero.web.search import WebSearcher, SearchResult
        assert WebSearcher is not None
        assert SearchResult is not None

    def test_import_scraper(self):
        """Test importing web scraper."""
        from aero.web.scraper import WebScraper, ScrapedPage
        assert WebScraper is not None
        assert ScrapedPage is not None

    def test_import_parser(self):
        """Test importing content parser."""
        from aero.web.parser import ContentParser, ParsedContent
        assert ContentParser is not None
        assert ParsedContent is not None


class TestOCRImports:
    """Test OCR module imports."""

    def test_import_registry(self):
        """Test importing OCR registry."""
        from aero.ocr.registry import (
            detect_available_backends,
            detect_tesseract,
            detect_deepseek,
            detect_easyocr,
        )
        assert callable(detect_available_backends)
        assert callable(detect_tesseract)
        assert callable(detect_deepseek)
        assert callable(detect_easyocr)

    def test_import_tesseract_backend(self):
        """Test importing Tesseract backend."""
        from aero.ocr.tesseract_backend import TesseractBackend
        assert TesseractBackend is not None

    def test_import_deepseek_backend(self):
        """Test importing DeepSeek backend."""
        from aero.ocr.deepseek_backend import DeepSeekBackend
        assert DeepSeekBackend is not None

    def test_import_easyocr_backend(self):
        """Test importing EasyOCR backend."""
        from aero.ocr.easyocr_backend import EasyOCRBackend
        assert EasyOCRBackend is not None


class TestPipelineImports:
    """Test pipeline module imports."""

    def test_import_aero_loop(self):
        """Test importing AeroLoop."""
        from aero.pipeline.aero_loop import AeroLoop, PipelineTask
        assert AeroLoop is not None
        assert PipelineTask is not None

    def test_import_validator(self):
        """Test importing Validator."""
        from aero.pipeline.validator import Validator, ValidationResult
        assert Validator is not None
        assert ValidationResult is not None

    def test_import_patterns(self):
        """Test importing pattern matcher."""
        from aero.pipeline.patterns import Pattern, PatternMatcher
        assert Pattern is not None
        assert PatternMatcher is not None


class TestAPIImports:
    """Test API module imports."""

    def test_import_server(self):
        """Test importing API server."""
        from aero.api.server import create_app, run_server
        assert callable(create_app)
        assert callable(run_server)

    def test_import_routes(self):
        """Test importing API routes."""
        from aero.api.routes import (
            rag_routes,
            sim_routes,
            ocr_routes,
            web_routes,
            agent_routes,
        )
        assert rag_routes is not None
        assert sim_routes is not None
        assert ocr_routes is not None
        assert web_routes is not None
        assert agent_routes is not None


class TestUIImports:
    """Test UI module imports."""

    def test_import_dashboard(self):
        """Test importing Dashboard."""
        from aero.ui.dashboard_stub import Dashboard
        assert Dashboard is not None


class TestMainPackage:
    """Test main package imports."""

    def test_import_aero(self):
        """Test importing main aero package."""
        import aero
        assert hasattr(aero, '__version__')
        assert hasattr(aero, 'load_config')
        assert hasattr(aero, 'BaseAgent')
        assert hasattr(aero, 'Registry')

    def test_version(self):
        """Test version is set."""
        from aero import __version__
        assert __version__ == "0.1.0"
