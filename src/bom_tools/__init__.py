"""BOM Tools - Core Package"""
__version__ = "1.0.0"

from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor
from bom_tools.core.agentic_rag import AgenticRAG, DirectLLM
from bom_tools.utils.openrouter_models import (
    list_openrouter_models,
    list_free_openrouter_models,
    pick_free_openrouter_model,
)

__all__ = [
    "AIBOMProcessor",
    "DATABOMProcessor",
    "AgenticRAG",
    "DirectLLM",
    "list_openrouter_models",
    "list_free_openrouter_models",
    "pick_free_openrouter_model",
]
