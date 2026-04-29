"""
Internal Conflict Checker for License Fields

Compares structured license metadata (HuggingFace card data, GitHub API)
with license mentions extracted from unstructured text (README, LICENSE files)
to detect internal inconsistencies within the same artifact.
"""

import re
import difflib
from typing import Optional, Dict


class LicenseConflictChecker:

    # Lowercased alias → canonical SPDX form
    SPDX_ALIASES: Dict[str, str] = {
        "mit": "MIT",
        "mit license": "MIT",
        "the mit license": "MIT",
        "apache-2.0": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "apache license 2.0": "Apache-2.0",
        "apache license, version 2.0": "Apache-2.0",
        "apache2": "Apache-2.0",
        "apache v2": "Apache-2.0",
        "gpl-2.0": "GPL-2.0",
        "gpl-2.0-only": "GPL-2.0",
        "gpl v2": "GPL-2.0",
        "gplv2": "GPL-2.0",
        "gnu general public license v2": "GPL-2.0",
        "gnu general public license version 2": "GPL-2.0",
        "gpl-3.0": "GPL-3.0",
        "gpl-3.0-only": "GPL-3.0",
        "gpl v3": "GPL-3.0",
        "gplv3": "GPL-3.0",
        "gnu general public license v3": "GPL-3.0",
        "gnu general public license version 3": "GPL-3.0",
        "agpl-3.0": "AGPL-3.0",
        "agpl v3": "AGPL-3.0",
        "agplv3": "AGPL-3.0",
        "gnu affero general public license v3": "AGPL-3.0",
        "lgpl-2.1": "LGPL-2.1",
        "lgpl-3.0": "LGPL-3.0",
        "lgpl v2.1": "LGPL-2.1",
        "lgpl v3": "LGPL-3.0",
        "bsd-2-clause": "BSD-2-Clause",
        "bsd 2-clause": "BSD-2-Clause",
        "simplified bsd": "BSD-2-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "bsd 3-clause": "BSD-3-Clause",
        "new bsd": "BSD-3-Clause",
        "bsd": "BSD-3-Clause",
        "cc-by-4.0": "CC-BY-4.0",
        "cc by 4.0": "CC-BY-4.0",
        "creative commons attribution 4.0": "CC-BY-4.0",
        "cc-by-sa-4.0": "CC-BY-SA-4.0",
        "cc-by-nc-4.0": "CC-BY-NC-4.0",
        "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
        "cc0-1.0": "CC0-1.0",
        "cc0": "CC0-1.0",
        "public domain": "CC0-1.0",
        "unlicense": "Unlicense",
        "the unlicense": "Unlicense",
        "isc": "ISC",
        "isc license": "ISC",
        "mpl-2.0": "MPL-2.0",
        "mozilla public license 2.0": "MPL-2.0",
        "openrail": "OpenRAIL",
        "openrail++": "OpenRAIL++",
        "llama 2": "Llama-2",
        "llama2": "Llama-2",
        "gemma": "Gemma",
        "other": "other",
        "proprietary": "proprietary",
    }

    # Patterns ordered from most specific to most general
    LICENSE_PATTERNS = [
        re.compile(
            r'licensed\s+under\s+(?:the\s+)?([A-Za-z0-9][^.\n,;(]{2,60}?)(?:\s+license)?[.\n,;]',
            re.IGNORECASE,
        ),
        re.compile(
            r'released\s+under\s+(?:the\s+)?([A-Za-z0-9][^.\n,;(]{2,60}?)(?:\s+license)?[.\n,;]',
            re.IGNORECASE,
        ),
        re.compile(
            r'distributed\s+under\s+(?:the\s+)?([A-Za-z0-9][^.\n,;(]{2,60}?)(?:\s+license)?(?:[.\n,;]|$)',
            re.IGNORECASE,
        ),
        # YAML-style header: "license: MIT" or "- license: Apache-2.0"
        re.compile(
            r'^[-*\s]*license\s*[:\-]\s*([A-Za-z0-9][^\n,;(]{1,60})',
            re.IGNORECASE | re.MULTILINE,
        ),
        # Markdown section: "## License\n\nMIT"
        re.compile(
            r'##\s*license\s*\n+\s*([A-Za-z0-9][^\n]{1,60})',
            re.IGNORECASE,
        ),
        # Well-known names at line start
        re.compile(
            r'(?:^|\n)\s*(MIT License|Apache License[\s,]*2\.0'
            r'|GNU (?:General|Affero|Lesser) Public License[\s\w.]*'
            r'|BSD[\s\-\w]*License|Creative Commons[^\n,;]{2,50}'
            r'|Unlicense|ISC License|Mozilla Public License[\s\w]*)',
            re.IGNORECASE,
        ),
    ]

    @classmethod
    def normalize_license(cls, license_str: str) -> str:
        """Map a license string to canonical SPDX form, or return the cleaned input."""
        if not license_str:
            return ""
        cleaned = license_str.strip().lower()
        if cleaned in cls.SPDX_ALIASES:
            return cls.SPDX_ALIASES[cleaned]
        stripped = re.sub(r'\s+license$', '', cleaned).strip()
        if stripped in cls.SPDX_ALIASES:
            return cls.SPDX_ALIASES[stripped]
        return cleaned

    @classmethod
    def extract_license_from_text(cls, text: str) -> Optional[str]:
        """Extract the most likely license name from free text (README / LICENSE file)."""
        if not text:
            return None
        for pattern in cls.LICENSE_PATTERNS:
            match = pattern.search(text)
            if match:
                candidate = match.group(1).strip()
                if 3 <= len(candidate) <= 80:
                    return candidate
        return None

    @classmethod
    def compute_similarity(cls, a: str, b: str) -> float:
        """Similarity [0.0, 1.0] between two license strings after normalization."""
        norm_a = cls.normalize_license(a)
        norm_b = cls.normalize_license(b)
        if not norm_a or not norm_b:
            return 0.0
        if norm_a == norm_b:
            return 1.0
        return difflib.SequenceMatcher(None, norm_a.lower(), norm_b.lower()).ratio()

    @classmethod
    def check(
        cls,
        structured_license: Optional[str],
        unstructured_text: Optional[str],
        source_name: str = "readme",
        similarity_threshold: float = 0.8,
    ) -> Dict:
        """
        Compare a structured metadata license against the license extracted
        from one unstructured text source.

        Returns a dict with keys:
            has_conflict, similarity_score, structured_license, extracted_license,
            normalized_structured, normalized_extracted, source, conflict_description.
        """
        extracted = cls.extract_license_from_text(unstructured_text)

        result = {
            "has_conflict": False,
            "similarity_score": None,
            "structured_license": structured_license,
            "extracted_license": extracted,
            "normalized_structured": cls.normalize_license(structured_license) if structured_license else "",
            "normalized_extracted": cls.normalize_license(extracted) if extracted else "",
            "source": source_name,
            "conflict_description": None,
        }

        if not structured_license or not extracted:
            missing = (
                "structured license missing" if not structured_license
                else "no license found in unstructured text"
            )
            result["conflict_description"] = f"Cannot compare: {missing}"
            return result

        score = cls.compute_similarity(structured_license, extracted)
        result["similarity_score"] = round(score, 4)

        if score < similarity_threshold:
            result["has_conflict"] = True
            result["conflict_description"] = (
                f"License mismatch between metadata ({structured_license!r}) "
                f"and {source_name} ({extracted!r}). "
                f"Similarity: {score:.2%}"
            )

        return result

    @classmethod
    def check_all_sources(
        cls,
        structured_license: Optional[str],
        readme_texts: Dict[str, Optional[str]],
        similarity_threshold: float = 0.8,
    ) -> Dict:
        """
        Run check() against multiple unstructured sources and aggregate results.

        Args:
            structured_license: License from structured metadata (HF card / GitHub API).
            readme_texts: Mapping of source_name → raw text.
                          e.g. {"github_readme": "...", "hf_readme": "..."}
            similarity_threshold: Score below which a conflict is flagged.

        Returns:
            {
              "has_conflict": bool,
              "per_source": dict,
              "conflict_description": str | None,
            }
        """
        per_source = {}
        conflict_descriptions = []

        for source_name, text in readme_texts.items():
            result = cls.check(
                structured_license=structured_license,
                unstructured_text=text,
                source_name=source_name,
                similarity_threshold=similarity_threshold,
            )
            per_source[source_name] = result
            if result["has_conflict"]:
                conflict_descriptions.append(result["conflict_description"])

        return {
            "has_conflict": bool(conflict_descriptions),
            "per_source": per_source,
            "conflict_description": "; ".join(conflict_descriptions) if conflict_descriptions else None,
        }
