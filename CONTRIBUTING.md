# Contributing to Lokum Engine

Thank you for considering contributing to Lokum Engine. We welcome contributions that improve the functionality and stability of the library.

## Development Setup

1. Fork the repository and clone it locally.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: `source .venv/bin/activate`
4. Install the package in editable mode with development dependencies: `pip install -e ".[dev]"`
5. Install the testing framework: `pip install pytest`

## Testing

Before submitting a Pull Request, ensure all tests pass:

```bash
PYTHONPATH=src pytest tests/
```

If you are adding a new feature or fixing a bug, please include a corresponding test in the `tests/` directory to maintain test coverage.

## Pull Request Process

1. Ensure your code adheres to standard Python PEP-8 formatting. We recommend using `black` and `ruff`.
2. Update `README.md` or `docs/USER_GUIDE.md` if your changes affect the public interface or configuration variables.
3. Your PR will be reviewed by maintainers before merging.

## Design Philosophy

- **Apple Silicon Focus:** We prioritize optimization for Apple M-series chips and the MLX framework.
- **Simplicity:** We aim to abstract complex operations (like multi-index RAG or ChatML-aware splitting) into simple, high-level API calls.
- **Explicit Failure:** We prefer raising explicit errors over silent failures (e.g., when a document fails to parse or a file is missing).
