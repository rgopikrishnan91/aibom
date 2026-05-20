"""Shared fixtures for license_compat tests."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_matrix_paths():
    return {
        "matrix": FIXTURES / "tiny_matrix.json",
        "allowed": FIXTURES / "tiny_allowed.json",
        "missing": FIXTURES / "tiny_missing.json",
    }


@pytest.fixture
def tiny_matrix(tiny_matrix_paths):
    from aikaboom.plugins.license_compat.matrix import load_matrix
    return load_matrix(
        matrix_path=tiny_matrix_paths["matrix"],
        allowed_path=tiny_matrix_paths["allowed"],
        missing_path=tiny_matrix_paths["missing"],
    )
