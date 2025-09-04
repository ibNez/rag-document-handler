# Contributing

Contributions are welcome! Follow these guidelines to participate in the refactored architecture.

## Development Standards

The project adheres to strict development rules documented in [`DEVELOPMENT_RULES.md`](../DEVELOPMENT_RULES.md). Key requirements:

- **Always source environment**: `source .venv/bin/activate` before any Python commands
- Use type hints and comprehensive docstrings
- Follow modular architecture patterns in `ingestion/` and `rag_manager/`
- Keep dependencies in `pyproject.toml`
- Format code with Black and import-sort with isort
- Add tests for features using `pytest`
- Implement comprehensive logging per DEVELOPMENT_RULES.md

## Refactored Architecture

Understanding the structure is essential for contributions:

```
ingestion/           # Data ingestion modules
├── core/           # Database abstractions
├── email/          # Email processing pipeline  
├── url/            # URL crawling and processing
├── document/       # Document extraction and chunking
└── utils/          # Shared utilities

rag_manager/        # RAG functionality
├── managers/       # Vector operations and search
└── web/           # Web interface, routes, and statistics
    ├── panels/     # Panel-specific statistics providers
    └── stats.py    # Statistics coordinator

templates/          # Web interface templates
├── partials/       # Modular template components
└── *.html         # Main templates using partials
```

## Working with Template Partials

The web interface uses a modular partials system for better maintainability:

### Adding Partials

1. **Create the partial** in `templates/partials/` with underscore prefix:
   ```bash
   templates/partials/_feature.html
   ```

2. **Add descriptive comment** at the top:
   ```html
   <!-- Feature Description - Specific purpose of this partial -->
   ```

3. **Include in main template**:
   ```html
   {% include 'partials/_feature.html' %}
   ```

4. **Test responsiveness** across screen sizes

### Modifying Existing Partials

1. **Identify the specific partial** that needs changes
2. **Make targeted modifications** without affecting other components
3. **Maintain Bootstrap consistency** and responsive design
4. **Test integration** with main templates

### Guidelines for Partials

- **Single Responsibility**: Each partial should handle one UI section
- **Descriptive Names**: Use clear, descriptive names with underscore prefix
- **Comprehensive Comments**: Document the purpose and any complex logic
- **Bootstrap Consistency**: Use consistent Bootstrap classes and patterns
- **Responsive Design**: Ensure all partials work across device sizes

## Working with Panel Statistics

The statistics system uses dedicated providers for each dashboard panel:

### Adding Statistics Panels

1. **Create panel provider** in `rag_manager/web/panels/`:
   ```python
   # my_panel.py
   class MyPanelStats:
       def __init__(self, rag_manager):
           self.rag_manager = rag_manager
           
       def get_stats(self) -> Dict[str, Any]:
           # Implementation with error handling
           pass
   ```

2. **Update coordinator** in `stats.py`:
   ```python
   from .panels import MyPanelStats
   
   def __init__(self, rag_manager):
       self.my_panel = MyPanelStats(rag_manager)
   ```

3. **Create template partial** for display
4. **Add comprehensive documentation**

### Modifying Existing Panels

1. **Identify the specific panel** in `rag_manager/web/panels/`
2. **Make targeted changes** with proper error handling
3. **Update type hints** and documentation
4. **Test error conditions** and edge cases

### Guidelines for Statistics Panels

- **Error Handling**: Always include comprehensive try/catch blocks
- **Type Hints**: Use proper type annotations for all methods
- **Logging**: Include meaningful log messages for debugging
- **Consistent Returns**: Return consistent data structures across panels
- **Database Safety**: Handle connection failures gracefully

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
