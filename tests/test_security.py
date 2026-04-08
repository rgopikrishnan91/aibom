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
        import re
        # Match ghp_ followed by 36+ alphanumeric chars (real token pattern)
        token_pattern = re.compile(r'ghp_[A-Za-z0-9]{36,}')
        for path, content in _read_source_files():
            assert not token_pattern.search(content), f"Hardcoded GitHub token found in {path}"

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
        """Verify metadata_fetcher reads token from environment at call time."""
        from bom_tools.utils.metadata_fetcher import _get_github_headers
        headers = _get_github_headers()
        if os.getenv("GITHUB_TOKEN"):
            assert "Authorization" in headers
        else:
            assert "Authorization" not in headers
