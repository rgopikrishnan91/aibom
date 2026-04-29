"""
Regression tests — ensure previously-broken behavior stays fixed.
Includes mock-based processor tests that work without API keys.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestProcessorInitWithoutAPIKey:
    """
    Regression: test_processors.py tests fail because AIBOMProcessor.__init__
    creates a real LLM client (requires OPENAI_API_KEY). These tests mock
    the LLM creation so we can test processor logic without API keys.
    """

    @patch("aikaboom.core.agentic_rag.create_llm")
    @patch("aikaboom.core.agentic_rag.HuggingFaceEmbeddings")
    def test_ai_processor_init_rag(self, mock_embeddings, mock_llm):
        """AIBOMProcessor initializes in RAG mode without API key."""
        mock_llm.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()

        from aikaboom.core.processors import AIBOMProcessor
        proc = AIBOMProcessor(model="gpt-4o", mode="rag", use_case="complete")
        assert proc.mode == "rag"
        assert proc.model == "gpt-4o"
        assert proc.use_case == "complete"

    @patch("aikaboom.core.agentic_rag.create_llm")
    def test_ai_processor_init_direct(self, mock_llm):
        """AIBOMProcessor initializes in direct mode without API key."""
        mock_llm.return_value = MagicMock()

        from aikaboom.core.processors import AIBOMProcessor
        proc = AIBOMProcessor(model="gpt-4o", mode="direct", use_case="safety")
        assert proc.mode == "direct"
        assert proc.use_case == "safety"

    @patch("aikaboom.core.agentic_rag.create_llm")
    @patch("aikaboom.core.agentic_rag.HuggingFaceEmbeddings")
    def test_data_processor_init_rag(self, mock_embeddings, mock_llm):
        """DATABOMProcessor initializes in RAG mode without API key."""
        mock_llm.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()

        from aikaboom.core.processors import DATABOMProcessor
        proc = DATABOMProcessor(model="gpt-4o", mode="rag", use_case="complete")
        assert proc.mode == "rag"

    @patch("aikaboom.core.agentic_rag.create_llm")
    def test_data_processor_init_direct(self, mock_llm):
        """DATABOMProcessor initializes in direct mode without API key."""
        mock_llm.return_value = MagicMock()

        from aikaboom.core.processors import DATABOMProcessor
        proc = DATABOMProcessor(model="gpt-4o", mode="direct", use_case="lineage")
        assert proc.mode == "direct"
        assert proc.use_case == "lineage"


class TestProcessorIDGeneration:
    """
    Regression: generate_model_id and generate_dataset_id were failing
    because the old tests constructed a full processor (which needs LLM).
    These mock the LLM so we can test ID generation.
    """

    @patch("aikaboom.core.agentic_rag.create_llm")
    @patch("aikaboom.core.agentic_rag.HuggingFaceEmbeddings")
    def test_model_id_from_repo(self, mock_emb, mock_llm):
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()

        from aikaboom.core.processors import AIBOMProcessor
        proc = AIBOMProcessor()
        model_id = proc.generate_model_id(
            repo_id="microsoft/DialoGPT-medium", github_url=None
        )
        assert model_id == "microsoft_DialoGPT-medium"

    @patch("aikaboom.core.agentic_rag.create_llm")
    @patch("aikaboom.core.agentic_rag.HuggingFaceEmbeddings")
    def test_model_id_from_github(self, mock_emb, mock_llm):
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()

        from aikaboom.core.processors import AIBOMProcessor
        proc = AIBOMProcessor()
        model_id = proc.generate_model_id(
            repo_id=None, github_url="https://github.com/microsoft/DialoGPT"
        )
        assert model_id == "microsoft_DialoGPT"

    @patch("aikaboom.core.agentic_rag.create_llm")
    @patch("aikaboom.core.agentic_rag.HuggingFaceEmbeddings")
    def test_dataset_id_from_hf(self, mock_emb, mock_llm):
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()

        from aikaboom.core.processors import DATABOMProcessor
        proc = DATABOMProcessor()
        dataset_id = proc.generate_dataset_id(
            arxiv_url=None, github_url=None,
            hf_url="https://huggingface.co/datasets/rajpurkar/squad"
        )
        assert "squad" in dataset_id.lower()


class TestSampleOutputValid:
    """Regression: sample-output.json was previously malformed."""

    def test_sample_output_is_valid_json(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "sample-output.json")
        with open(path) as f:
            data = json.load(f)
        assert "repo_id" in data
        assert "model_id" in data
        assert "direct_fields" in data
        assert "rag_fields" in data

    def test_sample_output_triplet_structure(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "sample-output.json")
        with open(path) as f:
            data = json.load(f)

        # Every field in direct_fields should have value/source/conflict
        for field, triplet in data["direct_fields"].items():
            assert "value" in triplet, f"direct_fields.{field} missing 'value'"
            assert "source" in triplet, f"direct_fields.{field} missing 'source'"
            assert "conflict" in triplet, f"direct_fields.{field} missing 'conflict'"

        for field, triplet in data["rag_fields"].items():
            assert "value" in triplet, f"rag_fields.{field} missing 'value'"
            assert "source" in triplet, f"rag_fields.{field} missing 'source'"
            assert "conflict" in triplet, f"rag_fields.{field} missing 'conflict'"

    def test_sample_output_conflict_format(self):
        """The license conflict in sample output should have correct structure."""
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "sample-output.json")
        with open(path) as f:
            data = json.load(f)

        license_field = data["direct_fields"]["license"]
        assert license_field["conflict"] is not None
        assert license_field["conflict"]["type"] in ("inter", "intra")
        assert "value" in license_field["conflict"]


class TestNoSecurityRegressions:
    """Regression: ensure previously-fixed security issues don't return."""

    def test_metadata_fetcher_no_module_level_token(self):
        """GITHUB_TOKEN should not be a module-level constant anymore."""
        import aikaboom.utils.metadata_fetcher as mf
        assert not hasattr(mf, "GITHUB_TOKEN"), "GITHUB_TOKEN should not be a module-level variable"
        assert not hasattr(mf, "GITHUB_HEADERS"), "GITHUB_HEADERS should not be a module-level variable"
        assert hasattr(mf, "_get_github_headers"), "_get_github_headers function should exist"

    def test_web_app_uses_secure_filename(self):
        """Download endpoint should sanitize filenames."""
        from aikaboom.web.app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/download/../../../etc/passwd")
            # Should not return 200 with file contents
            assert resp.status_code in (400, 404)

    def test_web_app_binds_localhost_by_default(self):
        """run.py should default to 127.0.0.1, not 0.0.0.0."""
        import os
        with open(os.path.join(os.path.dirname(__file__), "..", "run.py")) as f:
            content = f.read()
        assert "0.0.0.0" not in content
        assert "BOM_HOST" in content
