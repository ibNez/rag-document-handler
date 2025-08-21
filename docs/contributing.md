# Contributing

Contributions are welcome! Follow these guidelines to participate in the refactored architecture.

## Development Standards

The project adheres to strict development rules documented in [`DEVELOPMENT_RULES.md`](../DEVELOPMENT_RULES.md). Key requirements:

- **Always source environment**: `source .venv/bin/activate` before any Python commands
- Use type hints and comprehensive docstrings
- Follow modular architecture patterns in `ingestion/` and `rag_manager/`
- Keep dependencies in `pyproject.toml`
- Format code with Black and import-sort with isort
- Add tests for new features using `pytest`
- Implement comprehensive logging per DEVELOPMENT_RULES.md

## Refactored Architecture

Understanding the new structure is essential for contributions:

```
ingestion/           # Data ingestion modules
├── core/           # Database abstractions
├── email/          # Email processing pipeline  
├── url/            # URL crawling and processing
├── document/       # Document extraction and chunking
└── utils/          # Shared utilities

rag_manager/        # RAG functionality
├── managers/       # Vector operations and search
└── web/           # Web interface and routes
```

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone <your-fork-url>
   cd rag-document-handler
   ```

2. **Set up development environment**
   ```bash
   ./setup.sh --dev    # Infrastructure only
   ```
   This will:
   - Create virtual environment
   - Install all dependencies
   - Start infrastructure containers (PostgreSQL + Milvus)
   - Keep webui container stopped for local development

3. **Start development server**
   ```bash
   source .venv/bin/activate  # ALWAYS source first
   ./start.sh                 # Use proper startup script
   ```

4. **Install dev dependencies**
   ```bash
   source .venv/bin/activate  # Always source environment
   pip install -e .[dev]
   ```

## Workflow

1. Create a feature branch from main
2. Make your changes with the development server running
3. Run formatting and tests before committing:
   ```bash
   black .
   isort .
   flake8
   pytest
   ```
4. Test your changes thoroughly
5. Submit a pull request describing your changes

## Testing

### Database Testing
```bash
python test_postgres.py    # Test PostgreSQL connectivity
./status.sh                # Check system status
```

### Safe Cleanup
```bash
./uninstall.sh --dry-run   # Preview what would be removed
./uninstall.sh             # Clean removal when done
```

## TODOs

- [ ] Add GitHub Actions workflow for automated testing.
- [ ] Provide code of conduct and issue templates.
