"""Pure license-compatibility engine.

No I/O, no graph. Inputs are LicenseMatrix values; outputs are dataclass
verdicts and recommendations. Mirrors LicenseRec.py's analytical primitives.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from aikaboom.plugins.license_compat.matrix import LicenseMatrix

Status = Literal["compatible", "violation", "unknown_upstream", "unknown_downstream", "missing_data"]


@dataclass(frozen=True)
class CompatVerdict:
    downstream: Optional[str]
    upstreams: frozenset[str]
    status: Status
    incompatible_with: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Recommendation:
    by_category: dict[str, list[str]]
    is_solvable: bool


_CC_VERSION_RE = re.compile(r"^(cc-[a-z\-]+)-(\d\.\d)$", re.IGNORECASE)


def check_compat(downstream: Optional[str], upstreams: frozenset[str], matrix: LicenseMatrix) -> CompatVerdict:
    if downstream is None:
        return CompatVerdict(None, upstreams, "unknown_downstream")
    if "UNKNOWN" in upstreams:
        return CompatVerdict(downstream, upstreams, "unknown_upstream")
    if not upstreams:
        return CompatVerdict(downstream, upstreams, "compatible")
    if downstream not in matrix.details:
        return CompatVerdict(downstream, upstreams, "missing_data")
    blocked = frozenset(
        up for up in upstreams
        if downstream not in matrix.upstream_compat_index.get(up, frozenset())
    )
    if blocked:
        return CompatVerdict(downstream, upstreams, "violation", incompatible_with=blocked)
    return CompatVerdict(downstream, upstreams, "compatible")


def recommend(
    upstreams: frozenset[str],
    matrix: LicenseMatrix,
    frequencies: Counter,
    top_k_per_category: int = 5,
) -> Recommendation:
    if not upstreams:
        return Recommendation(by_category={}, is_solvable=False)

    compat_sets = [matrix.upstream_compat_index.get(up, frozenset()) for up in upstreams]
    if not compat_sets:
        return Recommendation(by_category={}, is_solvable=False)

    candidate_names = frozenset.intersection(*compat_sets) if len(compat_sets) > 1 else compat_sets[0]
    is_solvable = len(candidate_names) > 0
    if not is_solvable:
        return Recommendation(by_category={}, is_solvable=False)

    filtered = [matrix.details[n] for n in candidate_names if n.lower() in matrix.allowed_licenses and n in matrix.details]
    if not filtered:
        return Recommendation(by_category={}, is_solvable=is_solvable)

    grouped: dict[str, list[str]] = defaultdict(list)
    for entry in filtered:
        grouped[entry.get("category", "UNKNOWN")].append(entry["name"])

    for cat, lic_list in list(grouped.items()):
        processed = lic_list
        cc_versions = [m for l in lic_list if (m := _CC_VERSION_RE.match(l))]
        if cc_versions:
            bases_with_4_0 = {m.group(1).lower() for l in lic_list if (m := _CC_VERSION_RE.match(l)) and m.group(2) == "4.0"}
            processed = [
                l for l in lic_list
                if not (m := _CC_VERSION_RE.match(l)) or m.group(2) == "4.0" or m.group(1).lower() not in bases_with_4_0
            ]
        grouped[cat] = sorted(processed, key=lambda l: (-frequencies.get(l, 0), l.lower()))[:top_k_per_category]

    return Recommendation(by_category=dict(grouped), is_solvable=is_solvable)
