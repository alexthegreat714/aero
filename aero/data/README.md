# Aero Data Directory

This directory is used for storing data files used by the Aero Agent.

## Storage Conventions

### Directory Structure

```
data/
├── rag_store/          # RAG document storage
│   ├── documents.json  # Document metadata
│   └── embeddings.npz  # Document embeddings
├── experiments/        # Experiment data output
│   ├── captures/       # Camera captures
│   ├── sensor_logs/    # Sensor data logs
│   └── analysis/       # Analysis results
├── simulations/        # Simulation outputs
│   ├── results/        # Result files (JSON, NPY)
│   └── checkpoints/    # Solver checkpoints
├── cache/              # Temporary cache files
│   ├── ocr/            # OCR cache
│   └── web/            # Web scraping cache
└── models/             # Model weights and configs
    ├── embeddings/     # Embedding model files
    └── pinn/           # PINN model checkpoints
```

### File Naming Conventions

- Use lowercase with underscores: `experiment_001.json`
- Include timestamps in ISO format: `2024_01_15_143022_capture.png`
- Use descriptive prefixes: `sim_laplace_100x100.npy`

### Data Formats

- **JSON**: Configuration, metadata, small structured data
- **NPY/NPZ**: Numpy arrays (simulation results, embeddings)
- **CSV**: Tabular data, sensor logs
- **PNG/JPG**: Images, visualizations
- **PDF**: Documents for RAG ingestion

### Git Ignore

Large data files should not be committed to git. The following patterns
are typically ignored:

- `*.npy`, `*.npz` (large numerical arrays)
- `*.h5`, `*.hdf5` (HDF5 data files)
- `*.pkl`, `*.pickle` (Python pickle files)
- `cache/` (temporary files)
- `models/` (model weights)

### Backup Recommendations

Important data should be backed up regularly:

1. RAG document store (contains indexed documents)
2. Experiment results (irreplaceable measurement data)
3. Trained PINN model checkpoints

### Size Limits

- Individual files: < 100 MB recommended
- Total data directory: Monitor growth, archive old data
- Cache: Automatically cleaned after 7 days of inactivity
