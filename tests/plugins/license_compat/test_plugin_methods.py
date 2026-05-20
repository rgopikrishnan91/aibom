"""Direct tests for ``LicenseCompatPlugin`` methods that the contract suite
exercises only indirectly: ``register_cli`` and ``spdx_annotations``.
"""
from __future__ import annotations

import argparse

from aikaboom.plugins.license_compat.engine import (
    CompatVerdict,
    Finding,
    Findings,
    Recommendation,
)
from aikaboom.plugins.license_compat.plugin import LicenseCompatPlugin


def test_plugin_register_cli_attaches_subcommands(tiny_matrix):
    plugin = LicenseCompatPlugin(matrix=tiny_matrix)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    plugin.register_cli(sub)
    assert "license-check" in sub.choices
    assert "license-audit" in sub.choices


def test_plugin_spdx_annotations_returns_list(tiny_matrix):
    plugin = LicenseCompatPlugin(matrix=tiny_matrix)
    finding = Finding(
        downstream_iri="https://example.org/Down",
        downstream_label="Down",
        upstream_iri="https://example.org/Up",
        upstream_label="Up",
        predicate="https://aikaboom.dev/aibom#trainedOn",
        downstream_license="gpl-3.0",
        upstream_licenses=frozenset({"apache-2.0"}),
        verdict=CompatVerdict(
            downstream="gpl-3.0",
            upstreams=frozenset({"apache-2.0"}),
            status="violation",
            incompatible_with=frozenset({"apache-2.0"}),
        ),
        recommendation=Recommendation(by_category={"PERMISSIVE": ["mit"]}, is_solvable=True),
    )
    anns = plugin.spdx_annotations("https://example.org/Claim", Findings([finding]))
    assert isinstance(anns, list)
    assert len(anns) >= 1
    assert anns[0]["type"] == "Annotation"
