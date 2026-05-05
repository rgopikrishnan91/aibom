# Migration Guide

## Updating from Old Structure to New Structure

If you have existing code using the old structure, here's how to update your imports:

### Old Import Style

```python
from Demo_UI import app
from processors import AIBOMProcessor, DATABOMProcessor
from agentic_rag_core import AgenticRAG, DirectLLM
from metadata_utils import MetadataFetcher
from link_fallback import LinkFallbackFinder
from spdx_validator import SPDXValidator
```

### New Import Style

```python
from aikaboom.web.app import app
from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor
from aikaboom.core.agentic_rag import AgenticRAG, DirectLLM
from aikaboom.utils.metadata_fetcher import MetadataFetcher
from aikaboom.utils.link_fallback import LinkFallbackFinder
from aikaboom.utils.spdx_validator import SPDXValidator
```

### Running the Application

#### Old Way
```bash
python Demo_UI.py
```

#### New Way
```bash
# Option 1: Run a Python helper that starts Flask
python run.py

# Option 2: Subcommand on the installed CLI (after pip install -e .)
aikaboom serve --port 5000

# Generate a BOM directly
aikaboom generate --type ai --repo microsoft/DialoGPT-medium --output result.json

# Plain `aikaboom` prints the available subcommands.
```

### File Locations

| Old Location | New Location |
|-------------|--------------|
| `Demo_UI.py` | `src/aikaboom/web/app.py` |
| `processors.py` | `src/aikaboom/core/processors.py` |
| `agentic_rag_core.py` | `src/aikaboom/core/agentic_rag.py` |
| `source_handling.py` | `src/aikaboom/core/source_handler.py` |
| `metadata_utils.py` | `src/aikaboom/utils/metadata_fetcher.py` |
| `link_fallback.py` | `src/aikaboom/utils/link_fallback.py` |
| `spdx_validator.py` | `src/aikaboom/utils/spdx_validator.py` |
| `templates/` | `src/aikaboom/web/templates/` |

## Key Changes

1. **Package Structure**: Code is now organized in a Python package under `src/aikaboom/` (src layout)
2. **Installation**: Can now be installed via `pip install -e .`
3. **Testing**: Tests are in a dedicated `tests/` directory
4. **Documentation**: Organized in `docs/` directory
5. **Examples**: Sample scripts in `examples/` directory
6. **Configuration**: Proper `setup.py`, `pyproject.toml`, and `requirements.txt`

## Benefits

✅ Professional package structure  
✅ Easier installation and distribution  
✅ Better code organization  
✅ Testable modules  
✅ IDE-friendly  
✅ Ready for PyPI publication  
