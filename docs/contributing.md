# Contributing

Contributions are welcome! Follow these guidelines to participate.

## Development Standards

The project adheres to strict development rules documented in [`DEVELOPMENT_RULES.md`](../DEVELOPMENT_RULES.md). In summary:

- Use type hints and docstrings.
- Keep dependencies in `pyproject.toml`.
- Format code with Black and import-sort with isort.
- Add tests for new features using `pytest`.

## Workflow

1. Fork the repository and create a feature branch.
2. Install dev dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run formatting and tests before committing:
   ```bash
   black .
   isort .
   flake8
   pytest
   ```
4. Submit a pull request describing your changes.

## TODOs

- [ ] Add GitHub Actions workflow for automated testing.
- [ ] Provide code of conduct and issue templates.
