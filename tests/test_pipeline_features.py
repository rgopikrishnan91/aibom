import json

from bom_tools.core.processors import AIBOMProcessor, _build_triplet_payload

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from bom_tools.core.agentic_rag import AgenticRAG, AgentState


# ---------------------------------------------------------------------------
# Helpers shared by ALL tests
# ---------------------------------------------------------------------------

class _StubRagProcessor:
    def __init__(self, results):
        self._results = results

    def process_ai_model(self, repo_id, arxiv_url, github_url, huggingface_url):
        return list(self._results)


def _make_ai_processor(rag_results, direct_metadata, github_readme="", hf_readme=""):
    processor = object.__new__(AIBOMProcessor)
    processor.use_case = "complete"
    processor.questions_config = {"license": {"question": "What is the license?"}}
    processor.processor = _StubRagProcessor(rag_results)
    processor.generate_model_id = lambda repo_id, github_url: "owner_model"
    processor.fetch_direct_metadata = lambda github_url, hf_repo_id=None: dict(direct_metadata)
    processor._fetch_github_readme = lambda github_url: github_readme
    processor._fetch_hf_readme = lambda hf_repo_id: hf_readme
    return processor


# ---------------------------------------------------------------------------
# Helpers for the AgenticRAG conflict tests (Tests 2 & 3)
# ---------------------------------------------------------------------------

# Question config copied verbatim from FIXED_QUESTIONS_AI['domain'].
# Priority order: arxiv > huggingface > github — so arxiv wins when an
# external conflict is detected in test 3.
_DOMAIN_QUESTION = {
    "domain": {
        "question": "What is the domain in which the AI package can be used?",
        "priority": ["arxiv", "huggingface", "github"],
        "keywords": (
            "domain application area field sector industry vertical computer vision "
            "natural language processing NLP machine learning classification regression "
            "detection segmentation speech recognition image processing text analysis "
            "audio processing video analysis time series forecasting recommendation "
            "systems robotics healthcare finance automotive agriculture education "
            "retail manufacturing cybersecurity gaming entertainment"
        ),
        "description": (
            "A free-form text that describes the domain where the AI model contained "
            "in the AI software can be expected to operate successfully. Examples "
            "include computer vision, natural language processing, etc."
        ),
    }
}


class _FakeResponse:
    """Stand-in for a LangChain AIMessage — only .content is needed."""
    def __init__(self, content: str):
        self.content = content


class _SequentialLLM:
    """
    Replaces ChatOpenAI (OpenRouter).  Returns pre-canned strings one per
    .invoke() call — no network I/O, no API key required.

    The response strings must use the exact markers that _detect_conflicts
    and _generate_answer_node parse:
        INTERNAL_CONFLICT: ...
        EXTERNAL_CONFLICT: ...
        ANSWER: ...
    """
    def __init__(self, *responses: str):
        self._responses = responses
        self._idx = 0

    def invoke(self, prompt):
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return _FakeResponse(text)


def _make_rag(llm_stub) -> AgenticRAG:
    """Build an AgenticRAG without __init__ (no embedding downloads, no API keys)."""
    rag = object.__new__(AgenticRAG)
    rag.llm = llm_stub
    rag.questions = _DOMAIN_QUESTION
    return rag


def _base_state(documents) -> AgentState:
    """Minimal valid AgentState for the domain question."""
    return {
        "question": _DOMAIN_QUESTION["domain"]["question"],
        "question_type": "domain",
        "documents": documents,
        "answer": "",
        "source_priority": [],
        "sources_used": [],
        "row_index": 0,
        "row_retrievers": {},
        "all_results": [],
        "chunks_per_source": {},
        "internal_conflict": "No",
        "external_conflict": "No",
    }


def test_json_output_is_triplet_based_and_serializable():
    processor = _make_ai_processor(
        rag_results=[
            {
                "question_type": "license",
                "answer": "MIT",
                "sources_used": ["huggingface"],
                "conflict": {"internal": "No", "external": "No"},
            }
        ],
        direct_metadata={
            "license": "MIT",
            "license_source": "huggingface",
            "license_conflicts": None,
        },
    )

    result = processor.process_ai_model(
        repo_id="owner/model",
        arxiv_url=None,
        github_url="https://github.com/owner/model",
    )

    assert result["model_id"] == "owner_model"
    assert result["direct_fields"]["license"] == {
        "value": "MIT",
        "source": "huggingface",
        "conflict": None,
    }
    assert result["rag_fields"]["license"] == {
        "value": "MIT",
        "source": "huggingface",
        "conflict": {"internal": "No", "external": "No"},
    }

    # Final payload should be JSON serializable without any custom encoder.
    json.dumps(result)


def test_internal_domain_conflict_same_source():
    """
    Scenario
    --------
    Both chunks come from the SAME source (huggingface) but describe different
    domains — Chunk 1: "computer vision", Chunk 2: "natural language processing".
    This is an intra-source (internal) conflict: the same model card is
    inconsistent about what domain the model targets.

    Question used: FIXED_QUESTIONS_AI['domain']
        "What is the domain in which the AI package can be used?"
        priority: ['arxiv', 'huggingface', 'github']

    The stub OpenRouter (meta-llama/llama-3.3-70b-instruct) returns the
    pre-formatted conflict-detection output, then an answer.  We call:
        1. _detect_conflicts  → state["internal_conflict"] starts with "Yes"
        2. _generate_answer_node → state["answer"] is populated
    """
    docs = [
        Document(
            page_content=(
                "This model is designed for computer vision tasks such as "
                "image classification, object detection, and segmentation."
            ),
            metadata={"source": "huggingface"},
        ),
        Document(
            page_content=(
                "The model targets natural language processing applications "
                "including text classification, sentiment analysis, and NER."
            ),
            metadata={"source": "huggingface"},
        ),
    ]

    # --- Call 1: conflict-detection response (what OpenRouter would return) ---
    conflict_response = (
        "INTERNAL_CONFLICT: Yes: huggingface — "
        'Chunk 1 says "computer vision" vs Chunk 2 says "natural language processing"\n'
        "EXTERNAL_CONFLICT: No"
    )
    # --- Call 2: answer-generation response ---
    answer_response = "ANSWER: Computer vision and natural language processing"

    rag = _make_rag(_SequentialLLM(conflict_response, answer_response))

    # Step 1 – run conflict detection
    state = rag._detect_conflicts(_base_state(docs))

    assert state["internal_conflict"].startswith("Yes"), (
        f"Expected internal conflict, got: {state['internal_conflict']}"
    )
    assert state["external_conflict"] == "No"

    # Step 2 – run answer generation (no source filtering since conflict is internal)
    state = rag._generate_answer_node(state)

    assert state["answer"] == "Computer vision and natural language processing"


def test_external_domain_conflict_arxiv_wins():
    """
    Scenario
    --------
    Two DIFFERENT sources disagree about the model's domain:
      - arxiv paper says: "speech recognition"
      - huggingface card says: "audio classification"
    This is a cross-source (external) conflict.

    Question used: FIXED_QUESTIONS_AI['domain']
        "What is the domain in which the AI package can be used?"
        priority: ['arxiv', 'huggingface', 'github']

    After _detect_conflicts flags the conflict, _generate_answer_node filters
    to the highest-priority source — 'arxiv' — and produces the final answer
    from that chunk alone.  The stub OpenRouter model is called twice:
        Call 1 (conflict detection)  → EXTERNAL_CONFLICT: Yes: ...
        Call 2 (answer generation)   → ANSWER: Speech recognition
    """
    docs = [
        Document(
            page_content=(
                "We present a large-scale speech recognition model trained on "
                "100,000 hours of multilingual audio data."
            ),
            metadata={"source": "arxiv"},
        ),
        Document(
            page_content=(
                "This model can be used for audio classification tasks such as "
                "environmental sound detection and music genre recognition."
            ),
            metadata={"source": "huggingface"},
        ),
    ]

    conflict_response = (
        "INTERNAL_CONFLICT: No\n"
        'EXTERNAL_CONFLICT: Yes: arxiv (Chunk 1) says "speech recognition" '
        'while huggingface (Chunk 2) says "audio classification"'
    )
    answer_response = "ANSWER: Speech recognition"

    rag = _make_rag(_SequentialLLM(conflict_response, answer_response))

    # Step 1 – run conflict detection
    state = rag._detect_conflicts(_base_state(docs))

    assert state["external_conflict"].startswith("Yes"), (
        f"Expected external conflict, got: {state['external_conflict']}"
    )
    assert state["internal_conflict"] == "No"

    # Step 2 – run answer generation.
    # _generate_answer_node detects the "Yes" external conflict, walks the
    # priority list ['arxiv', 'huggingface', 'github'], finds 'arxiv' first,
    # and filters to only the arxiv chunk before calling the stub LLM.
    state = rag._generate_answer_node(state)

    assert state["answer"] == "Speech recognition"