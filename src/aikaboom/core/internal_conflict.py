"""License-specific intra-source consistency check.

Compares the structured license string a source emits (e.g. HuggingFace
``cardData.license``, GitHub repo-license API) against the license
mention regex-extracted from that same source's free text (README /
LICENSE file). When the two disagree for a given source, the
disagreement is recorded as an intra-source conflict.

This is deliberately license-only: license is the one direct field
where a source reliably carries both a structured value and a
redundantly-stated unstructured one (the README usually mentions the
license too). The other direct fields don't have an analogous pair, so
no general intra-source machinery is needed.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from aikaboom.utils.normalise import (
    extract_license_from_text,
    license_similarity,
    normalize_license,
)

_SIMILARITY_THRESHOLD = 0.8


def check_license_intra_source(
    named_sources: Mapping[str, Optional[Mapping]],
    named_readmes: Mapping[str, Optional[str]],
) -> Dict[str, str]:
    """Per-source check: structured ``source["license"]`` vs the license
    extracted from ``readmes[source]`` text.

    Returns a ``{source_name: narrative}`` dict listing only the sources
    where the two values disagree (similarity below the threshold).
    Sources missing either side (no structured license, or no
    extractable license in the README) are silently skipped — there's
    nothing to compare.
    """
    conflicts: Dict[str, str] = {}
    for name, source in (named_sources or {}).items():
        if not isinstance(source, Mapping):
            continue
        structured = source.get("license")
        if not structured:
            continue
        readme = (named_readmes or {}).get(name) or ""
        readme_license = extract_license_from_text(readme)
        if not readme_license:
            continue
        norm_structured = normalize_license(structured)
        norm_readme = normalize_license(readme_license)
        if not norm_structured or not norm_readme:
            continue
        if license_similarity(norm_structured, norm_readme) >= _SIMILARITY_THRESHOLD:
            continue
        conflicts[name] = f"cardData={norm_structured}; readme={norm_readme}"
    return conflicts
