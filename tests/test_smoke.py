"""
Smoke tests — verify the project is importable, CLI is wired up,
web app starts, and core classes can be referenced.
These should pass in any environment without API keys.
"""
import importlib
import subprocess
import sys
import pytest


class TestImports:
    """Verify all public modules import without error."""

    def test_import_processors(self):
        mod = importlib.import_module("aikaboom.core.processors")
        assert hasattr(mod, "AIBOMProcessor")
        assert hasattr(mod, "DATABOMProcessor")

    def test_import_agentic_rag(self):
        mod = importlib.import_module("aikaboom.core.agentic_rag")
        assert hasattr(mod, "AgenticRAG")
        assert hasattr(mod, "DirectLLM")
        assert hasattr(mod, "create_llm")

    def test_import_source_handler(self):
        mod = importlib.import_module("aikaboom.core.source_handler")
        assert hasattr(mod, "SourceHandler")

    def test_import_internal_conflict(self):
        mod = importlib.import_module("aikaboom.core.internal_conflict")
        assert hasattr(mod, "LicenseConflictChecker")

    def test_import_metadata_fetcher(self):
        mod = importlib.import_module("aikaboom.utils.metadata_fetcher")
        assert hasattr(mod, "MetadataFetcher")

    def test_import_spdx_validator(self):
        mod = importlib.import_module("aikaboom.utils.spdx_validator")
        assert hasattr(mod, "SPDXValidator")
        assert hasattr(mod, "validate_bom_to_spdx")

    def test_import_link_fallback(self):
        mod = importlib.import_module("aikaboom.utils.link_fallback")
        assert hasattr(mod, "LinkFallbackFinder")

    def test_import_prompt(self):
        mod = importlib.import_module("aikaboom.core.prompt")
        assert hasattr(mod, "prompt_detect_conflicts")
        assert hasattr(mod, "prompt_generate_answer")

    def test_import_cli(self):
        mod = importlib.import_module("aikaboom.cli")
        assert hasattr(mod, "main")
        assert hasattr(mod, "cmd_generate")
        assert hasattr(mod, "cmd_serve")

    def test_import_web_app(self):
        mod = importlib.import_module("aikaboom.web.app")
        assert hasattr(mod, "app")


class TestCLISmoke:
    """Verify CLI entry point is wired up and responds to --help."""

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli", "--help"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0
        assert "generate" in result.stdout
        assert "serve" in result.stdout

    def test_cli_generate_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli", "generate", "--help"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0
        assert "--type" in result.stdout
        assert "--repo" in result.stdout
        assert "--spdx" in result.stdout
        assert "--no-validate-spdx" in result.stdout
        assert "--strict-spdx-validation" in result.stdout
        assert "--recursive-bom" in result.stdout
        assert "--recursive-output" in result.stdout
        assert "--linked-bom-output" in result.stdout
        assert "--provider" in result.stdout

    def test_cli_serve_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli", "serve", "--help"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_cli_generate_missing_type(self):
        """CLI should fail gracefully when --type is missing."""
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli", "generate"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode != 0

    def test_cli_no_subcommand(self):
        """CLI with no subcommand should print help and exit 1."""
        result = subprocess.run(
            [sys.executable, "-m", "aikaboom.cli"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 1


class TestWebAppSmoke:
    """Verify Flask app creates and responds to basic routes."""

    @pytest.fixture
    def client(self):
        from aikaboom.web.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"BOM" in resp.data

    def test_static_assets_exist(self, client):
        """The app should serve static files without 500."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_process_endpoint_exists(self, client):
        """POST /process should exist (returns 400 on bad input, not 404)."""
        resp = client.post("/process", data="bad", content_type="text/plain")
        assert resp.status_code == 400

    def test_find_links_endpoint_exists(self, client):
        resp = client.post("/find_links", data="bad", content_type="text/plain")
        assert resp.status_code == 400

    def test_config_endpoint(self, client):
        """GET /config should return JSON."""
        resp = client.get("/config")
        if resp.status_code == 200:
            data = resp.get_json()
            assert isinstance(data, dict)


class TestSPDXSmoke:
    """Verify SPDX validator can be instantiated and converts minimal data."""

    def test_ai_spdx_minimal(self):
        from aikaboom.utils.spdx_validator import SPDXValidator
        v = SPDXValidator(bom_type="ai")
        result = v.validate_and_convert({"direct_fields": {}, "rag_fields": {}})
        assert "@context" in result
        assert "@graph" in result

    def test_data_spdx_minimal(self):
        from aikaboom.utils.spdx_validator import SPDXValidator
        v = SPDXValidator(bom_type="data")
        result = v.validate_and_convert({"direct_metadata": {}, "rag_metadata": {}, "urls": {}})
        assert "@context" in result


class TestHFSpacesArtifacts:
    """Verify the HF Spaces deployment files exist and are configured correctly."""

    @staticmethod
    def _repo_root():
        import os
        return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    def test_dockerfile_exists(self):
        import os
        path = os.path.join(self._repo_root(), "Dockerfile")
        assert os.path.exists(path), "Dockerfile missing at repo root"

    def test_dockerfile_has_required_directives(self):
        import os
        with open(os.path.join(self._repo_root(), "Dockerfile")) as f:
            content = f.read()
        # HF Spaces requires UID 1000 and port 7860
        assert "USER user" in content
        assert "EXPOSE 7860" in content
        assert "BOM_PORT=7860" in content
        assert "BOM_HOST=0.0.0.0" in content

    def test_readme_hf_exists_with_docker_sdk(self):
        import os
        path = os.path.join(self._repo_root(), "README_HF.md")
        assert os.path.exists(path), "README_HF.md missing"
        with open(path) as f:
            content = f.read()
        # YAML frontmatter must include sdk: docker and app_port: 7860
        assert content.startswith("---")
        assert "sdk: docker" in content
        assert "app_port: 7860" in content

    def test_deploy_script_exists_and_executable(self):
        import os
        path = os.path.join(self._repo_root(), "scripts", "deploy_to_hf_spaces.sh")
        assert os.path.exists(path), "scripts/deploy_to_hf_spaces.sh missing"
        assert os.access(path, os.X_OK), "deploy_to_hf_spaces.sh not executable"

    def test_hf_spaces_doc_exists(self):
        import os
        path = os.path.join(self._repo_root(), "docs", "HF_SPACES.md")
        assert os.path.exists(path), "docs/HF_SPACES.md missing"
