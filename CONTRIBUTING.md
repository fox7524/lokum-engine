# Contributing to Lokum Engine

First off, thank you for considering contributing to Lokum Engine! It's people like you that make Lokum Engine such a great tool for the MLX and Local AI community.

## Development Setup

1. Fork the repo and clone it locally.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Install in editable mode with test dependencies: `pip install -e ".[dev]"`
5. Install test framework: `pip install pytest`

## Testing

Before submitting a Pull Request, please ensure all tests pass:

```bash
PYTHONPATH=src pytest tests/
```

Lokum Engine takes pride in having zero broken tests. If you are adding a new feature, please add a corresponding test in the `tests/` directory.

## Pull Request Process

1. Ensure your code strictly adheres to standard Python PEP-8 formatting (we recommend `black` and `ruff`).
2. Update the `README.md` or `docs/USER_GUIDE.md` with details of changes to the interface.
3. The PR will be merged once you have the sign-off of at least one core maintainer.

## Philosophy

- **Apple Silicon First:** We prioritize performance on M-series chips.
- **Developer Experience:** 3 lines of code is better than 30. We abstract the complexity.
- **No Silent Failures:** If something is wrong (e.g., a missing file in RAG), we throw an explicit error rather than failing silently.
