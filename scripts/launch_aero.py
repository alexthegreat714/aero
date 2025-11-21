#!/usr/bin/env python3
"""
Aero Agent Launcher

Windows-friendly launcher that:
1. Loads configuration
2. Prints system diagnostics
3. Initializes registries
4. Starts the API server

Usage:
    python scripts/launch_aero.py
    python scripts/launch_aero.py --host 0.0.0.0 --port 9000
    python scripts/launch_aero.py --diagnostics-only
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the launcher."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def print_header() -> None:
    """Print the Aero Agent header."""
    header = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ███████╗██████╗  ██████╗                          ║
    ║    ██╔══██╗██╔════╝██╔══██╗██╔═══██╗                         ║
    ║    ███████║█████╗  ██████╔╝██║   ██║                         ║
    ║    ██╔══██║██╔══╝  ██╔══██╗██║   ██║                         ║
    ║    ██║  ██║███████╗██║  ██║╚██████╔╝                         ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝                          ║
    ║                                                               ║
    ║    Aerodynamics Research Agent v0.1.0                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(header)


def get_python_info() -> dict:
    """Get Python environment information."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
    }


def get_system_info() -> dict:
    """Get system information."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count() or 0,
    }


def check_cuda() -> dict:
    """Check CUDA/GPU availability."""
    result = {
        "available": False,
        "version": None,
        "device_name": None,
        "device_count": 0,
    }

    try:
        import torch
        result["available"] = torch.cuda.is_available()
        if result["available"]:
            result["version"] = torch.version.cuda
            result["device_count"] = torch.cuda.device_count()
            if result["device_count"] > 0:
                result["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["error"] = "PyTorch not installed"
    except Exception as e:
        result["error"] = str(e)

    return result


def check_tesseract() -> dict:
    """Check Tesseract OCR availability."""
    result = {
        "available": False,
        "version": None,
        "path": None,
    }

    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        result["path"] = tesseract_path
        result["available"] = True

        try:
            proc = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = proc.stdout or proc.stderr
            if output:
                result["version"] = output.strip().split("\n")[0]
        except Exception as e:
            result["error"] = str(e)

    return result


def check_ollama() -> dict:
    """Check Ollama availability and models."""
    result = {
        "installed": False,
        "running": False,
        "models": [],
    }

    ollama_path = shutil.which("ollama")
    if not ollama_path:
        return result

    result["installed"] = True

    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if proc.returncode == 0:
            result["running"] = True
            lines = proc.stdout.strip().split("\n")
            # Skip header line
            for line in lines[1:]:
                if line.strip():
                    model_name = line.split()[0]
                    result["models"].append(model_name)
    except subprocess.TimeoutExpired:
        result["error"] = "Ollama command timed out"
    except Exception as e:
        result["error"] = str(e)

    return result


def check_webcam() -> dict:
    """Check webcam availability."""
    result = {
        "available": False,
        "index": None,
    }

    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            result["available"] = True
            result["index"] = 0
            cap.release()
    except ImportError:
        result["error"] = "OpenCV not installed"
    except Exception as e:
        result["error"] = str(e)

    return result


def print_diagnostics() -> dict:
    """Print system diagnostics and return results."""
    print("\n" + "=" * 60)
    print("SYSTEM DIAGNOSTICS")
    print("=" * 60)

    diagnostics = {}

    # Python info
    print("\n[Python Environment]")
    python_info = get_python_info()
    diagnostics["python"] = python_info
    print(f"  Version:    {python_info['version']}")
    print(f"  Impl:       {python_info['implementation']}")
    print(f"  Executable: {python_info['executable']}")

    # System info
    print("\n[System Information]")
    sys_info = get_system_info()
    diagnostics["system"] = sys_info
    print(f"  OS:         {sys_info['os']} {sys_info['os_version']}")
    print(f"  Arch:       {sys_info['architecture']}")
    print(f"  CPU Cores:  {sys_info['cpu_count']}")

    # CUDA/GPU
    print("\n[GPU/CUDA]")
    cuda_info = check_cuda()
    diagnostics["cuda"] = cuda_info
    if cuda_info["available"]:
        print(f"  Available:  Yes")
        print(f"  CUDA Ver:   {cuda_info['version']}")
        print(f"  GPU:        {cuda_info['device_name']}")
        print(f"  Devices:    {cuda_info['device_count']}")
    else:
        print(f"  Available:  No")
        if "error" in cuda_info:
            print(f"  Note:       {cuda_info['error']}")

    # Tesseract
    print("\n[Tesseract OCR]")
    tess_info = check_tesseract()
    diagnostics["tesseract"] = tess_info
    if tess_info["available"]:
        print(f"  Available:  Yes")
        print(f"  Version:    {tess_info['version']}")
        print(f"  Path:       {tess_info['path']}")
    else:
        print(f"  Available:  No (not found in PATH)")

    # Ollama
    print("\n[Ollama]")
    ollama_info = check_ollama()
    diagnostics["ollama"] = ollama_info
    if ollama_info["installed"]:
        print(f"  Installed:  Yes")
        print(f"  Running:    {'Yes' if ollama_info['running'] else 'No'}")
        if ollama_info["models"]:
            print(f"  Models:     {', '.join(ollama_info['models'][:5])}")
            if len(ollama_info["models"]) > 5:
                print(f"              ... and {len(ollama_info['models']) - 5} more")
        else:
            print(f"  Models:     None found")
    else:
        print(f"  Installed:  No")

    # Webcam
    print("\n[Webcam]")
    webcam_info = check_webcam()
    diagnostics["webcam"] = webcam_info
    if webcam_info["available"]:
        print(f"  Available:  Yes (index {webcam_info['index']})")
    else:
        print(f"  Available:  No")
        if "error" in webcam_info:
            print(f"  Note:       {webcam_info['error']}")

    # DeepSeek OCR check
    print("\n[DeepSeek OCR]")
    deepseek_available = any(
        "deepseek" in m.lower()
        for m in ollama_info.get("models", [])
    )
    diagnostics["deepseek_ocr"] = {"available": deepseek_available}
    if deepseek_available:
        print(f"  Available:  Yes (via Ollama)")
    else:
        print(f"  Available:  No (model not found in Ollama)")

    print("\n" + "=" * 60)

    return diagnostics


def initialize_registries() -> None:
    """Initialize component registries."""
    print("\n[Initializing Registries]")

    try:
        from aero.core.registry import get_registry_manager, register_rag_components
        manager = get_registry_manager()
        print(f"  Created registries: {', '.join(manager.list_registries())}")

        # Register RAG components
        print("  Registering RAG components...")
        register_rag_components()
        print(f"  RAG registries: rag_stores, rag_readers, embedders")

    except Exception as e:
        print(f"  Error: {e}")


def initialize_rag() -> dict:
    """Initialize RAG system and return status."""
    print("\n[Initializing RAG System]")

    result = {
        "store": None,
        "embedder": None,
        "status": "not_initialized",
    }

    try:
        from aero.rag.store import create_vector_store, get_default_store
        from aero.rag.embedder import get_default_embedder, detect_embedding_backends

        # Check embedding backends
        backends = detect_embedding_backends()
        print(f"  Embedding backends:")
        print(f"    Ollama:              {'Yes' if backends['ollama'] else 'No'}")
        print(f"    SentenceTransformers: {'Yes' if backends['sentence_transformers'] else 'No'}")
        print(f"    Hash fallback:       Yes")

        # Initialize default embedder
        embedder = get_default_embedder()
        result["embedder"] = embedder.name
        print(f"  Active embedder:       {embedder.name} (dim={embedder.dimension})")

        # Initialize default store
        store = get_default_store()
        result["store"] = store.__class__.__name__
        doc_count = store.count()
        print(f"  Vector store:          {store.__class__.__name__}")
        print(f"  Documents loaded:      {doc_count}")

        result["status"] = "ready"

    except Exception as e:
        print(f"  Error initializing RAG: {e}")
        result["status"] = f"error: {e}"

    return result


def initialize_simulation() -> dict:
    """Initialize simulation system and return status."""
    print("\n[Initializing Simulation Engine]")

    result = {
        "scheduler": None,
        "pytorch": False,
        "cuda": False,
        "cuda_device": None,
        "numerics_backend": "numpy",
    }

    try:
        # Initialize scheduler
        from aero.sim.scheduler import get_scheduler
        scheduler = get_scheduler()
        result["scheduler"] = "ready"
        print(f"  Scheduler:            Initialized")

        # Check PyTorch and CUDA
        try:
            from aero.sim.pinn.pinn_trainer import check_pytorch_cuda
            cuda_info = check_pytorch_cuda()

            result["pytorch"] = cuda_info["pytorch_available"]
            result["cuda"] = cuda_info["cuda_available"]
            result["cuda_device"] = cuda_info.get("cuda_device")

            print(f"  PyTorch:              {'Yes' if result['pytorch'] else 'No'}")
            print(f"  CUDA available:       {'Yes' if result['cuda'] else 'No'}")
            if result["cuda_device"]:
                print(f"  CUDA device:          {result['cuda_device']}")
        except ImportError:
            print(f"  PyTorch:              Not installed")

        print(f"  Numerics backend:     numpy (CPU)")
        print(f"  Available solvers:    heat_1d, laplace_2d, ns_stub, pinn")

    except Exception as e:
        print(f"  Error initializing simulation: {e}")
        result["error"] = str(e)

    return result


def check_experiments() -> dict:
    """Check experiment system capabilities."""
    result = {
        "opencv": False,
        "webcam": False,
        "synthetic": True,  # Always available
        "sensor_file": True,  # Always available
    }

    # Check OpenCV
    try:
        import cv2
        result["opencv"] = True
        result["opencv_version"] = cv2.__version__
    except ImportError:
        result["opencv_error"] = "OpenCV not installed"

    # Check webcam availability (if OpenCV available)
    if result["opencv"]:
        try:
            from aero.experiments import is_webcam_available, list_cameras
            result["webcam"] = is_webcam_available(0)
            available_cameras = list_cameras(max_index=5)
            result["available_cameras"] = available_cameras
        except ImportError:
            result["webcam_error"] = "Experiments module not available"
        except Exception as e:
            result["webcam_error"] = str(e)

    return result


def initialize_experiments() -> dict:
    """Initialize experiment system and return status."""
    print("\n[Initializing Experiment System]")

    result = check_experiments()

    # Print results
    print(f"  OpenCV:              {'Yes (v' + result.get('opencv_version', '?') + ')' if result['opencv'] else 'No'}")
    if not result["opencv"]:
        print(f"    Note: Install opencv-python for camera/video features")

    print(f"  Webcam:              {'Yes' if result['webcam'] else 'No'}")
    if result.get("available_cameras"):
        print(f"    Available indices: {result['available_cameras']}")

    print(f"  Synthetic experiments: Yes (always available)")
    print(f"  File-based sensors:    Yes (CSV, JSON, TXT)")

    # Check for experiment data directory
    from pathlib import Path
    data_dir = Path("./data/experiments")
    if not data_dir.exists():
        print(f"  Data directory:      Creating ./data/experiments")
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  Data directory:      ./data/experiments")

    return result


def initialize_data_store() -> dict:
    """Initialize the data lake and return status."""
    print("\n[Initializing Data Lake]")

    result = {
        "backend": None,
        "db_path": None,
        "fields_dir": None,
        "status": "not_initialized",
    }

    try:
        from aero.data import DataStore, get_default_store, set_default_store, DUCKDB_AVAILABLE

        # Get config
        from aero.config.loader import get_config
        config = get_config()

        db_path = config.get("data.db_path", "./data/aero.db")
        fields_dir = config.get("data.fields_dir", "./data/fields")

        # Initialize store
        store = DataStore(db_path=db_path, fields_dir=fields_dir)
        set_default_store(store)

        # Get stats
        stats = store.get_stats()

        result["backend"] = "DuckDB" if DUCKDB_AVAILABLE else "SQLite"
        result["db_path"] = db_path
        result["fields_dir"] = fields_dir
        result["status"] = "ready"
        result["stats"] = stats

        print(f"  Backend:             {result['backend']}")
        print(f"  Database path:       {db_path}")
        print(f"  Fields directory:    {fields_dir}")
        print(f"  Simulations stored:  {stats.get('simulations', 0)}")
        print(f"  Experiments stored:  {stats.get('experiments', 0)}")
        print(f"  Timeseries records:  {stats.get('timeseries', 0)}")
        print(f"  Field arrays:        {stats.get('fields', 0)}")

    except Exception as e:
        print(f"  Error initializing data lake: {e}")
        result["status"] = f"error: {e}"

    return result


def load_configuration() -> None:
    """Load configuration."""
    print("\n[Loading Configuration]")

    try:
        from aero.config.loader import load_config
        config = load_config()
        print(f"  Config loaded successfully")
        print(f"  Debug mode: {config.debug}")
        print(f"  Log level: {config.log_level}")
    except Exception as e:
        print(f"  Error: {e}")


def start_server(host: str, port: int) -> None:
    """Start the API server."""
    print("\n[Starting API Server]")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print("\n" + "=" * 60)
    print("Server starting... Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        from aero.api.server import run_server
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nServer error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aero Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Run diagnostics only, don't start server",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    print_header()

    # Run diagnostics
    print_diagnostics()

    if args.diagnostics_only:
        print("\nDiagnostics complete. Exiting.")
        return

    # Initialize
    load_configuration()
    initialize_registries()
    initialize_simulation()
    initialize_rag()
    initialize_experiments()
    initialize_data_store()

    # Start server
    start_server(args.host, args.port)


if __name__ == "__main__":
    main()
