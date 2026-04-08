"""
Security tests to verify no hardcoded secrets or unsafe patterns remain.
"""
import os
import pytest


def _read_source_files():
    """Read all Python source files and return (filepath, content) pairs."""
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    files = []
    for root, dirs, filenames in os.walk(src_dir):
        for f in filenames:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r') as fh:
                    files.append((path, fh.read()))
    return files


class TestNoHardcodedSecrets:
    """Ensure no credentials are hardcoded in source."""

    def test_no_github_tokens(self):
        for path, content in _read_source_files():
            assert 'ghp_' not in content, f"Hardcoded GitHub token found in {path}"

    def test_no_openai_keys(self):
        for path, content in _read_source_files():
            assert 'sk-' not in content or 'sk-' in content.split('#')[0] is False, \
                f"Possible OpenAI key in {path}"

    def test_no_verify_false(self):
        for path, content in _read_source_files():
            assert 'verify=False' not in content, f"verify=False found in {path}"

    def test_no_hardcoded_internal_ips(self):
        for path, content in _read_source_files():
            assert '10.218.163.118' not in content, f"Hardcoded internal IP in {path}"

    def test_github_token_from_env(self):
        """Verify metadata_fetcher reads token from environment."""
        from bom_tools.utils.metadata_fetcher import GITHUB_TOKEN
        # Should be None if env var not set, not a hardcoded string
        expected = os.getenv("GITHUB_TOKEN")
        assert GITHUB_TOKEN == expected
