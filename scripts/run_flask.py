#!/usr/bin/env python3
"""
Aero Flask UI Launcher

Starts the Flask web interface for chatting with Aero.

Usage:
    python scripts/run_flask.py
    python scripts/run_flask.py --host 0.0.0.0 --port 5000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def check_ollama() -> dict:
    """Check Ollama availability and models."""
    result = {
        "available": False,
        "models": [],
        "has_gemma": False,
        "has_deepcoder": False,
    }

    try:
        from aero.llm.ollama_client import OllamaClient

        client = OllamaClient()
        result["available"] = client.is_available()

        if result["available"]:
            models = client.list_models()
            result["models"] = [m.get("name", "") for m in models]
            result["has_gemma"] = any("gemma" in m.lower() for m in result["models"])
            result["has_deepcoder"] = any("deep" in m.lower() for m in result["models"])

    except Exception as e:
        result["error"] = str(e)

    return result


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ███████╗██████╗  ██████╗                          ║
    ║    ██╔══██╗██╔════╝██╔══██╗██╔═══██╗                         ║
    ║    ███████║█████╗  ██████╔╝██║   ██║                         ║
    ║    ██╔══██║██╔══╝  ██╔══██╗██║   ██║                         ║
    ║    ██║  ██║███████╗██║  ██║╚██████╔╝                         ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝                          ║
    ║                                                               ║
    ║    Chat Interface                                             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aero Flask UI",
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
        default=5000,
        help="Server port (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    debug = args.debug and not args.no_debug

    # Setup
    setup_logging(args.log_level)
    print_banner()

    # Check Ollama
    print("\n[Checking LLM Backend]")
    ollama_status = check_ollama()

    if ollama_status["available"]:
        print(f"  Ollama:      Connected")
        print(f"  Models:      {len(ollama_status['models'])} available")
        print(f"  Gemma 3:     {'Yes' if ollama_status['has_gemma'] else 'No - install with: ollama pull gemma3'}")
        print(f"  DeepCoder:   {'Yes' if ollama_status['has_deepcoder'] else 'No - install with: ollama pull deepcoder'}")
    else:
        print(f"  Ollama:      Not available")
        print(f"  Note:        Start Ollama with: ollama serve")
        if "error" in ollama_status:
            print(f"  Error:       {ollama_status['error']}")

    # Load config
    print("\n[Loading Configuration]")
    try:
        from aero.config.loader import load_config
        config = load_config()
        print(f"  Main model:      {config.get('llm.main_model', 'gemma3')}")
        print(f"  Reasoning model: {config.get('llm.reasoning_model', 'deepcoder')}")
    except Exception as e:
        print(f"  Using defaults: {e}")

    # Initialize components
    print("\n[Initializing Components]")
    try:
        from aero.llm.reasoning import ReasoningOrchestrator, set_orchestrator
        from aero.llm.ollama_client import OllamaClient

        client = OllamaClient()
        orchestrator = ReasoningOrchestrator(client=client)
        set_orchestrator(orchestrator)
        print(f"  Orchestrator:    Ready")
    except Exception as e:
        print(f"  Warning:         {e}")

    # Start Flask
    print(f"\n{'=' * 60}")
    print(f"  Starting Flask server...")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Debug: {debug}")
    print(f"{'=' * 60}\n")

    try:
        from aero.ui.flask_app import app
        app.run(host=args.host, port=args.port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nServer error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
