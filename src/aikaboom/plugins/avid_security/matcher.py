from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any

from aikaboom.plugins.avid_security.snapshot import AvidIndex, family_prefixes
from aikaboom.plugins.avid_security.walker import Component

Tier = Literal[1, 2, 3]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Match:
    component: Component
    avid_report: dict
    tier: Tier
    confidence: Confidence
    evidence: dict[str, Any]


@dataclass
class ComponentMatcher:
    index: AvidIndex

    def match(self, component: Component) -> list[Match]:
        results: list[Match] = []
        results.extend(self._tier1(component))
        results.extend(self._tier2(component))
        results.extend(self._tier3(component))
        return self._dedup(results)

    def _tier1(self, c: Component) -> list[Match]:
        out = []
        for r in self.index.find_exact(c.bare_name, artifact_kind=c.kind):
            out.append(Match(
                component=c, avid_report=r, tier=1, confidence="high",
                evidence={"matched_via": "exact_bare_name", "bare_name": c.bare_name},
            ))
        return out

    def _tier2(self, c: Component) -> list[Match]:
        out = []
        for base_path in c.base_models:
            base_bare = base_path.split("/")[-1].lower()
            for r in self.index.find_exact(base_bare, artifact_kind="Model"):
                out.append(Match(
                    component=c, avid_report=r, tier=2, confidence="medium",
                    evidence={"matched_via": "base_model_lineage", "base_model": base_path},
                ))
        return out

    def _tier3(self, c: Component) -> list[Match]:
        if c.developer is None:
            return []
        prefixes = family_prefixes(c.bare_name)
        out = []
        seen_reports: set[str] = set()
        for prefix in prefixes:
            for r in self.index.find_by_family_prefix(
                prefix=prefix, developer=c.developer, artifact_kind=c.kind,
            ):
                rid = r["report_id"]
                if rid in seen_reports:
                    continue
                seen_reports.add(rid)
                out.append(Match(
                    component=c, avid_report=r, tier=3, confidence="low",
                    evidence={
                        "matched_via": "family_prefix_developer",
                        "family_prefix": prefix,
                        "developer": c.developer,
                    },
                ))
        return out

    @staticmethod
    def _dedup(matches: list[Match]) -> list[Match]:
        # Keep the highest-confidence (lowest tier) match per (component, report) pair.
        by_key: dict[tuple[str, str], Match] = {}
        for m in matches:
            key = (m.component.spdx_id, m.avid_report["report_id"])
            if key not in by_key or m.tier < by_key[key].tier:
                by_key[key] = m
        return list(by_key.values())
