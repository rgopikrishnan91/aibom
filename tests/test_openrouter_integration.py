"""
Integration tests — real OpenRouter API calls with content fetched directly
from a HuggingFace model page.

These tests are SKIPPED automatically when OPENROUTER_API_KEY is not set.

What is being tested
--------------------
1. test_conflict_detection_bert_domain
   Fetches the bert-base-uncased README from HuggingFace, splits it into two
   chunks, and calls _detect_conflicts() via a real OpenRouter LLM.
   Checks that the conflict fields are valid strings ("No" or "Yes: ...").

2. test_answer_generation_bert_domain
   Continues from test 1: calls _generate_answer_node() after conflict
   detection and checks that the answer is non-empty and NLP-relevant.

Model under test  : google-bert/bert-base-uncased  (HuggingFace)
OpenRouter model  : qwen/qwen-2.5-72b-instruct
BOM question      : domain — "What is the domain in which the AI package can be used?"
"""

import os
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()   # reads OPENROUTER_API_KEY from .env

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from aikaboom.core.agentic_rag import AgenticRAG, AgentState, create_llm, FIXED_QUESTIONS_AI

# ---------------------------------------------------------------------------
# Skip the whole module if no API key is available
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not (os.getenv("OPENROUTER_API_KEY") or os.getenv("My_OPENROUTER_API_KEY")),
    reason="OPENROUTER_API_KEY not set — skipping OpenRouter integration tests",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_MODEL = "qwen/qwen-2.5-72b-instruct"
HF_MODEL_ID      = "google-bert/bert-base-uncased"
HF_README_URL    = f"https://huggingface.co/{HF_MODEL_ID}/raw/main/README.md"

# Use only the domain question — copied verbatim from FIXED_QUESTIONS_AI
_DOMAIN_QUESTION = {"domain": FIXED_QUESTIONS_AI["domain"]}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_hf_readme(model_id: str) -> str:
    """
    Fetch the raw README.md from a public HuggingFace model page.
    Skips the test automatically if the network request fails.
    """
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        pytest.skip(f"Could not fetch HuggingFace README for {model_id}: {e}")


def _make_real_rag() -> AgenticRAG:
    """
    Build an AgenticRAG backed by a real OpenRouter LLM.
    Uses object.__new__ to skip __init__ (which would try to download
    embeddings and connect to GitHub / HuggingFace APIs).
    Only rag.llm and rag.questions are needed for the two methods under test.
    """
    rag = object.__new__(AgenticRAG)
    rag.llm = create_llm(
        model=OPENROUTER_MODEL,
        temperature=0,
        llm_provider="openrouter",
    )
    rag.questions = _DOMAIN_QUESTION
    return rag


def _readme_to_documents(readme_text: str) -> list:
    """
    Turn the raw HuggingFace README text into two Document chunks so that
    _detect_conflicts has multiple chunks to compare, which is the realistic
    scenario in a production RAG run.

    Both chunks are tagged source='huggingface' (single-source scenario).
    The first chunk covers the model card / overview; the second covers the
    longer evaluation / usage section.
    """
    halfway = len(readme_text) // 2
    return [
        Document(
            page_content=readme_text[:halfway],
            metadata={"source": "huggingface", "chunk_index": 0},
        ),
        Document(
            page_content=readme_text[halfway:],
            metadata={"source": "huggingface", "chunk_index": 1},
        ),
    ]


def _base_state(documents) -> AgentState:
    return {
        "question":          _DOMAIN_QUESTION["domain"]["question"],
        "question_type":     "domain",
        "documents":         documents,
        "answer":            "",
        "source_priority":   [],
        "sources_used":      [],
        "row_index":         0,
        "row_retrievers":    {},
        "all_results":       [],
        "chunks_per_source": {},
        "internal_conflict": "No",
        "external_conflict": "No",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_conflict_detection_bert_domain():
    """
    Pipeline step 1 — _detect_conflicts()

    Flow
    ----
    1. Fetch the real bert-base-uncased README from HuggingFace.
    2. Split it into two chunks (both from source='huggingface').
    3. Call _detect_conflicts() — this sends a real prompt to OpenRouter
       (qwen/qwen-2.5-72b-instruct) and parses the INTERNAL_CONFLICT /
       EXTERNAL_CONFLICT fields from the response.

    Assertions
    ----------
    - internal_conflict is a non-empty string starting with "No" or "Yes"
    - external_conflict is a non-empty string starting with "No" or "Yes"
    - external_conflict is "No" (both chunks are from the same source)
    """
    readme = _fetch_hf_readme(HF_MODEL_ID)
    docs   = _readme_to_documents(readme)
    rag    = _make_real_rag()

    print(f"\n  Model  : {OPENROUTER_MODEL}")
    print(f"  HF page: {HF_MODEL_ID}")
    print(f"  Chunks : {len(docs)} (source=huggingface)")
    print(f"  Question: {_DOMAIN_QUESTION['domain']['question']}")

    state = rag._detect_conflicts(_base_state(docs))

    internal = state["internal_conflict"]
    external = state["external_conflict"]

    print(f"\n  [OpenRouter response]")
    print(f"  INTERNAL_CONFLICT: {internal}")
    print(f"  EXTERNAL_CONFLICT: {external}")

    assert isinstance(internal, str) and internal, "internal_conflict must be a non-empty string"
    assert isinstance(external, str) and external, "external_conflict must be a non-empty string"

    # The LLM may wrap the value in brackets e.g. "[No]" or "[Yes: ...]"
    # Strip them before checking the prefix.
    internal_clean = internal.strip("[] ")
    external_clean = external.strip("[] ")
    assert internal_clean.startswith(("No", "Yes")), f"Unexpected format: {internal}"
    assert external_clean.startswith(("No", "Yes")), f"Unexpected format: {external}"

    # Both chunks are from the same source so a cross-source conflict is impossible
    assert external_clean.startswith("No"), (
        f"Both chunks come from 'huggingface' — external conflict should be 'No', got: {external}"
    )


def test_answer_generation_bert_domain():
    """
    Pipeline steps 1 + 2 — _detect_conflicts() then _generate_answer_node()

    Flow
    ----
    1. Fetch the real bert-base-uncased README from HuggingFace.
    2. Split it into two chunks.
    3. Run _detect_conflicts() (OpenRouter call 1).
    4. Run _generate_answer_node() (OpenRouter call 2).

    Assertions
    ----------
    - The answer is a non-empty string.
    - It does not just say "Not found."
    - It mentions at least one expected NLP keyword, confirming the LLM
      actually read the bert-base-uncased model card.
    """
    readme = _fetch_hf_readme(HF_MODEL_ID)
    docs   = _readme_to_documents(readme)
    rag    = _make_real_rag()

    print(f"\n  Model  : {OPENROUTER_MODEL}")
    print(f"  HF page: {HF_MODEL_ID}")

    # Step 1: conflict detection
    state = rag._detect_conflicts(_base_state(docs))
    print(f"\n  INTERNAL_CONFLICT : {state['internal_conflict']}")
    print(f"  EXTERNAL_CONFLICT : {state['external_conflict']}")

    # Step 2: answer generation
    state = rag._generate_answer_node(state)
    answer = state["answer"]

    print(f"\n  [Final answer from OpenRouter]")
    print(f"  ANSWER: {answer}")

    assert isinstance(answer, str) and answer.strip(), "Answer should be a non-empty string"
    assert "not found" not in answer.lower(), f"Answer was 'Not found': {answer}"

    nlp_keywords = {
        "nlp", "natural language", "language model", "text", "bert",
        "transformer", "classification", "masked", "pretrained", "fine-tun",
    }
    matched = [kw for kw in nlp_keywords if kw in answer.lower()]
    assert matched, (
        f"Answer does not mention any NLP-related keyword ({sorted(nlp_keywords)}).\n"
        f"Got: {answer}"
    )
    print(f"\n  Matched NLP keywords: {matched}")
