# Aero Agent

**Intelligent Agent Framework for Aerodynamics Research**

> **WARNING**: This is an alpha skeleton release. Many features are stubs or placeholders.
> Do not use in production environments.

## Overview

Aero Agent is an intelligent assistant framework designed to support aerodynamics research and experimentation. It provides:

- **RAG System**: Document storage and retrieval with semantic search
- **Simulation Modules**: Finite difference, CFD, and PINN solvers
- **OCR Capabilities**: Multi-backend OCR (Tesseract, DeepSeek, EasyOCR)
- **Web Integration**: Search and scraping utilities
- **Experiment Support**: Camera capture and sensor data acquisition
- **REST API**: FastAPI-based interface for all features

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aero-team/aero.git
cd aero

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Running the Agent

```bash
# Run with system diagnostics
python scripts/launch_aero.py

# Diagnostics only (no server)
python scripts/launch_aero.py --diagnostics-only

# Custom host/port
python scripts/launch_aero.py --host 0.0.0.0 --port 9000
```

### API Documentation

Once running, access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
aero/
├── aero/                   # Main package
│   ├── config/             # Configuration management
│   ├── core/               # Base classes and infrastructure
│   ├── rag/                # RAG document store
│   ├── sim/                # Simulation modules
│   │   ├── numerics/       # Finite difference solvers
│   │   ├── cfd/            # CFD solvers
│   │   └── pinn/           # Physics-informed neural networks
│   ├── experiments/        # Camera and sensor utilities
│   ├── web/                # Web search and scraping
│   ├── ocr/                # OCR backends
│   ├── pipeline/           # Task processing loop
│   ├── api/                # REST API server
│   │   └── routes/         # API route handlers
│   ├── ui/                 # Dashboard (stub)
│   └── data/               # Data storage
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
└── pyproject.toml          # Project configuration
```

## Features

### Configuration

Configuration is loaded from `aero/config/defaults.yaml` and can be overridden via environment variables:

```bash
# Override server port
export AERO_SERVER_PORT=9000

# Enable debug mode
export AERO_APP_DEBUG=true
```

### RAG System

```python
from aero.rag import RAGStore

store = RAGStore()
store.add_document("Document content here", {"source": "manual"})
results = store.search("query text", top_k=5)
```

### Simulations

```python
from aero.sim.numerics.fd_solver import LaplaceEquation

solver = LaplaceEquation(nx=100, ny=100)
solver.set_boundary_conditions(left=0, right=100, top=0, bottom=0)
result = solver.run()
solution = solver.get_solution()
```

### OCR

```python
from aero.ocr import detect_available_backends, TesseractBackend

# Check available backends
backends = detect_available_backends()
print(backends)  # {'tesseract': True, 'deepseek_ocr': False, 'easyocr': True}

# Use Tesseract
ocr = TesseractBackend()
result = ocr.ocr("image.png")
print(result.text)
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `POST /api/rag/documents` | Add document |
| `POST /api/rag/search` | Search documents |
| `POST /api/sim/run` | Run simulation |
| `POST /api/ocr/process` | Process image with OCR |
| `POST /api/web/search` | Web search |
| `GET /api/agent/status` | Agent status |

## System Requirements

- Python 3.10+
- 4GB RAM minimum (8GB recommended for ML features)
- GPU optional but recommended for PINN and large simulations

### Optional Dependencies

- **Tesseract OCR**: For Tesseract backend
- **Ollama**: For DeepSeek OCR
- **CUDA**: For GPU acceleration

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=aero
```

### Code Style

```bash
# Format code
black aero tests
isort aero tests

# Lint
ruff check aero
mypy aero
```

## Roadmap

### Phase 1 (Current) - Skeleton
- [x] Repository structure
- [x] Configuration system
- [x] Core infrastructure
- [x] API server framework
- [x] Module stubs

### Phase 2 - Core Features
- [ ] Functional RAG with embeddings
- [ ] Complete FD solvers
- [ ] Working OCR pipeline
- [ ] Web scraping implementation

### Phase 3 - Advanced Features
- [ ] Navier-Stokes solver
- [ ] PINN training loop
- [ ] Dashboard UI
- [ ] Agent orchestration

### Phase 4 - Production
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Deployment guides

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is provided as-is for research purposes. The simulation results should be validated independently before use in critical applications. No warranty is provided for accuracy or fitness for any particular purpose.
