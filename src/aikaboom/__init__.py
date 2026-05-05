"""AIkaBoOM - AI Bills of Materials"""
__version__ = "1.0.0"

from aikaboom.utils.spdx_validator import validate_bom_to_spdx, validate_spdx_export
from aikaboom.utils.recursive_bom import (
    discover_recursive_targets,
    generate_recursive_boms,
    build_linked_spdx_bundle,
    linked_bundle_summary,
)
from aikaboom.utils.source_priority import (
    load_source_priority,
    get_direct_priority,
    get_rag_priority,
    set_source_priority_path,
)

# Optional dependencies that may legitimately be missing in slim install
# profiles (HF Spaces base images, schema-only consumers, etc). We only
# silence ModuleNotFoundError when the missing module is one of these — real
# bugs in the wrapped modules still raise.
_OPTIONAL_DEPS = {
    "pandas", "flask", "langchain", "langchain_core", "langchain_openai",
    "langchain_community", "langchain_huggingface", "langgraph", "openai",
    "sentence_transformers", "huggingface_hub", "faiss", "chromadb",
    "google", "httpx", "requests", "PyGithub", "github", "bs4",
    "beautifulsoup4", "pymupdf", "fitz", "dotenv",
}


def _missing_optional(exc: ModuleNotFoundError) -> bool:
    name = (exc.name or "").split(".")[0]
    return name in _OPTIONAL_DEPS


try:
    from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor
except ModuleNotFoundError as e:
    if not _missing_optional(e):
        raise
    AIBOMProcessor = None
    DATABOMProcessor = None

try:
    from aikaboom.core.agentic_rag import AgenticRAG, DirectLLM
except ModuleNotFoundError as e:
    if not _missing_optional(e):
        raise
    AgenticRAG = None
    DirectLLM = None

try:
    from aikaboom.utils.openrouter_models import (
        list_openrouter_models,
        list_free_openrouter_models,
        pick_free_openrouter_model,
    )
except ModuleNotFoundError as e:
    if not _missing_optional(e):
        raise
    list_openrouter_models = None
    list_free_openrouter_models = None
    pick_free_openrouter_model = None

try:
    from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter, bom_to_cyclonedx
except ModuleNotFoundError as e:
    if not _missing_optional(e):
        raise
    CycloneDXExporter = None
    bom_to_cyclonedx = None

__all__ = [
    "AIBOMProcessor",
    "DATABOMProcessor",
    "AgenticRAG",
    "DirectLLM",
    "list_openrouter_models",
    "list_free_openrouter_models",
    "pick_free_openrouter_model",
    "CycloneDXExporter",
    "bom_to_cyclonedx",
    "validate_bom_to_spdx",
    "validate_spdx_export",
    "discover_recursive_targets",
    "generate_recursive_boms",
    "build_linked_spdx_bundle",
    "linked_bundle_summary",
    "load_source_priority",
    "get_direct_priority",
    "get_rag_priority",
    "set_source_priority_path",
]
