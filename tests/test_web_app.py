"""
Baseline regression tests for web app utility functions.
Tests pure logic functions + Flask integration tests.
"""
import pytest
from bom_tools.web.app import normalize_use_case, get_use_case_label, count_fields, app


class TestNormalizeUseCase:
    """Tests for normalize_use_case."""

    def test_valid_ai_use_cases(self):
        assert normalize_use_case("complete", "ai") == "complete"
        assert normalize_use_case("safety", "ai") == "safety"
        assert normalize_use_case("security", "ai") == "security"
        assert normalize_use_case("lineage", "ai") == "lineage"
        assert normalize_use_case("license", "ai") == "license"

    def test_valid_data_use_cases(self):
        assert normalize_use_case("complete", "data") == "complete"
        assert normalize_use_case("safety", "data") == "safety"

    def test_case_insensitive(self):
        assert normalize_use_case("COMPLETE", "ai") == "complete"
        assert normalize_use_case("Safety", "ai") == "safety"

    def test_whitespace_handling(self):
        assert normalize_use_case("  complete  ", "ai") == "complete"

    def test_unknown_defaults_to_complete(self):
        assert normalize_use_case("unknown_preset", "ai") == "complete"

    def test_none_defaults_to_complete(self):
        assert normalize_use_case(None, "ai") == "complete"

    def test_empty_string_defaults_to_complete(self):
        assert normalize_use_case("", "ai") == "complete"


class TestGetUseCaseLabel:
    """Tests for get_use_case_label."""

    def test_complete_ai(self):
        assert get_use_case_label("complete", "ai") == "Complete AI BOM"

    def test_safety_ai(self):
        assert get_use_case_label("safety", "ai") == "Safety & Compliance"

    def test_complete_data(self):
        assert get_use_case_label("complete", "data") == "Complete Data BOM"

    def test_unknown_returns_complete_label(self):
        assert get_use_case_label("nonexistent", "ai") == "Complete AI BOM"


class TestCountFields:
    """Tests for count_fields."""

    def test_triplet_structure(self):
        metadata = {
            "license": {"value": "MIT", "source": "hf", "conflict": None},
            "model_type": {"value": "transformer", "source": "hf", "conflict": None},
            "domain": {"value": None, "source": None, "conflict": None},
        }
        assert count_fields(metadata) == 2

    def test_direct_values(self):
        metadata = {"name": "MyModel", "version": "1.0", "empty": "", "null": None}
        assert count_fields(metadata) == 2

    def test_empty_dict(self):
        assert count_fields({}) == 0

    def test_none_input(self):
        assert count_fields(None) == 0

    def test_mixed_structure(self):
        metadata = {
            "license": {"value": "MIT", "source": "hf", "conflict": None},
            "name": "MyModel",
            "empty_triplet": {"value": "", "source": None, "conflict": None},
            "empty_direct": "",
        }
        assert count_fields(metadata) == 2


class TestFlaskApp:
    """Integration tests using Flask test client."""

    @pytest.fixture
    def client(self):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_home_page(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_process_no_json(self, client):
        """Test that /process rejects non-JSON requests."""
        response = client.post('/process', data="not json", content_type='text/plain')
        assert response.status_code == 400

    def test_find_links_no_json(self, client):
        """Test that /find_links rejects non-JSON requests."""
        response = client.post('/find_links', data="not json", content_type='text/plain')
        assert response.status_code == 400

    def test_download_path_traversal_blocked(self, client):
        """Test that path traversal in /download is blocked."""
        response = client.get('/download/../../etc/passwd')
        # secure_filename strips the ../ so it becomes 'etcpasswd' or similar
        # The file won't exist so we get 404, but NOT the contents of /etc/passwd
        assert response.status_code in (400, 404)

    def test_download_nonexistent_file(self, client):
        """Test that downloading a non-existent file returns 404."""
        response = client.get('/download/nonexistent_file_12345.json')
        assert response.status_code == 404
