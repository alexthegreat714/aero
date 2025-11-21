"""
Flask application for Aero chat interface.

Provides web UI for chatting with Aero via Gemma 3 and DeepCoder.
"""

import json
import logging
import os
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from typing import Optional

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "aero-dev-secret-key-change-in-prod")


def get_config():
    """Get Aero config."""
    try:
        from aero.config.loader import get_config as _get_config
        return _get_config()
    except Exception:
        return {}


def init_orchestrator():
    """Initialize the reasoning orchestrator."""
    from aero.llm.reasoning import ReasoningOrchestrator, set_orchestrator
    from aero.llm.ollama_client import OllamaClient

    config = get_config()

    client = OllamaClient(
        base_url=config.get("llm.ollama_url", "http://localhost:11434"),
        timeout=config.get("llm.timeout", 120.0),
    )

    orchestrator = ReasoningOrchestrator(
        client=client,
        main_model=config.get("llm.main_model", "gemma3"),
        reasoning_model=config.get("llm.reasoning_model", "deepcoder"),
        reasoning_threshold=config.get("llm.reasoning_threshold", 2),
    )

    set_orchestrator(orchestrator)
    return orchestrator


# Initialize on first request
_orchestrator: Optional["ReasoningOrchestrator"] = None


def get_orchestrator():
    """Get or create orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = init_orchestrator()
    return _orchestrator


# =============================================================================
# Routes
# =============================================================================


@app.route("/")
def index():
    """Main chat interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat message (non-streaming)."""
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        orchestrator = get_orchestrator()
        result = orchestrator.chat(
            message=message,
            include_rag=data.get("include_rag", True),
        )

        return jsonify({
            "status": "ok",
            "response": result.content,
            "reasoning_used": result.reasoning_used,
            "deep_reasoning": result.deep_reasoning,
            "complexity": result.complexity,
            "reasoning_reason": result.reasoning_reason,
        })

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Handle streaming chat message."""
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    def generate():
        try:
            orchestrator = get_orchestrator()
            for chunk in orchestrator.chat_stream(
                message=message,
                include_rag=data.get("include_rag", True),
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/history")
def get_history():
    """Get conversation history."""
    try:
        orchestrator = get_orchestrator()
        history = orchestrator.get_history()
        return jsonify({"status": "ok", "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear_history():
    """Clear conversation history."""
    try:
        orchestrator = get_orchestrator()
        orchestrator.clear_history()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status")
def status():
    """Get system status."""
    try:
        from aero.llm.ollama_client import get_ollama_client

        client = get_ollama_client()
        available = client.is_available()
        models = client.list_models() if available else []

        config = get_config()

        return jsonify({
            "status": "ok",
            "ollama_available": available,
            "models": [m.get("name", "") for m in models],
            "main_model": config.get("llm.main_model", "gemma3"),
            "reasoning_model": config.get("llm.reasoning_model", "deepcoder"),
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/rag/search", methods=["POST"])
def rag_search():
    """Search RAG store."""
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        from aero.rag.store import get_default_store
        store = get_default_store()
        results = store.search(query, top_k=data.get("top_k", 5))

        return jsonify({
            "status": "ok",
            "results": [
                {
                    "content": doc.content[:500],
                    "score": float(score),
                    "metadata": doc.metadata,
                }
                for doc, score in results
            ],
        })

    except Exception as e:
        logger.exception(f"RAG search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """Run a simulation and return results for graphing."""
    data = request.get_json()
    sim_type = data.get("sim_type", "heat_1d")
    params = data.get("parameters", {})

    try:
        import numpy as np
        from aero.sim.numerics.fd_solver import solve_heat_1d, solve_laplace_2d

        if sim_type == "heat_1d":
            nx = params.get("grid_size", 50)
            x = np.linspace(0, 1, nx)
            u0 = np.exp(-100 * (x - 0.5) ** 2)

            solution, metadata = solve_heat_1d(
                u0=u0,
                alpha=params.get("alpha", 0.01),
                dx=params.get("dx", 0.02),
                dt=params.get("dt", 0.001),
                num_steps=params.get("steps", 500),
            )

            return jsonify({
                "status": "ok",
                "sim_type": sim_type,
                "x": x.tolist(),
                "initial": u0.tolist(),
                "solution": solution.tolist(),
                "metadata": metadata,
            })

        elif sim_type == "laplace_2d":
            nx = params.get("grid_size", 30)
            ny = params.get("grid_size", 30)

            solution, metadata = solve_laplace_2d(
                nx=nx,
                ny=ny,
                tolerance=params.get("tolerance", 1e-6),
                max_iterations=params.get("max_iter", 10000),
            )

            return jsonify({
                "status": "ok",
                "sim_type": sim_type,
                "solution": solution.tolist(),
                "metadata": metadata,
            })

        else:
            return jsonify({"error": f"Unknown simulation type: {sim_type}"}), 400

    except Exception as e:
        logger.exception(f"Simulation error: {e}")
        return jsonify({"error": str(e)}), 500


def create_app(config=None):
    """Create Flask app with optional config."""
    if config:
        app.config.update(config)
    return app


def run_dev_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = True):
    """Run development server."""
    print(f"\n{'=' * 60}")
    print("AERO Chat Interface")
    print(f"{'=' * 60}")
    print(f"  URL: http://{host}:{port}")
    print(f"  Debug: {debug}")
    print(f"{'=' * 60}\n")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_dev_server()
