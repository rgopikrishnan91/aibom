"""Direct-pipeline tests covering the agreed strategies in
docs/FIELD_STRATEGIES.md.

Each test feeds synthetic HF/GH source dicts through
``SourceHandler.get_field_conflict_with_priority`` /
``get_date_field_with_window_conflict`` to verify (value, source, conflict)
without spinning up the full processor.
"""
from aikaboom.core.source_handler import SourceHandler
from aikaboom.utils.normalise import normalize_org, normalize_url, normalize_version


# ---------------------------------------------------------------------------
# Date fields: releaseTime / builtTime
# ---------------------------------------------------------------------------

class TestDateFieldsWithWindow:
    def test_release_time_picks_latest_no_conflict_within_7_days(self):
        hf = {"releaseTime": "2024-08-15"}
        gh = {"releaseTime": "2024-08-13"}
        v, s, c = SourceHandler.get_date_field_with_window_conflict(
            "releaseTime", {"huggingface": hf, "github": gh},
            mode="latest", window_days=7,
        )
        assert v == "2024-08-15" and s == "huggingface"
        assert c is None

    def test_release_time_flags_conflict_outside_window(self):
        hf = {"releaseTime": "2024-08-15"}
        gh = {"releaseTime": "2024-01-01"}
        v, s, c = SourceHandler.get_date_field_with_window_conflict(
            "releaseTime", {"huggingface": hf, "github": gh},
            mode="latest", window_days=7,
        )
        assert v == "2024-08-15"
        assert c is not None and c["source"] == "github"
        assert c["delta_days"] > 200

    def test_built_time_picks_earliest(self):
        hf = {"builtTime": "2024-08-15"}
        gh = {"builtTime": "2024-08-01"}
        v, s, c = SourceHandler.get_date_field_with_window_conflict(
            "builtTime", {"huggingface": hf, "github": gh},
            mode="earliest", window_days=7,
        )
        assert v == "2024-08-01" and s == "github"
        assert c is not None and c["source"] == "huggingface"
        assert c["delta_days"] == 14

    def test_one_source_only_no_conflict(self):
        v, s, c = SourceHandler.get_date_field_with_window_conflict(
            "releaseTime", {"huggingface": {"releaseTime": "2024-08-15"}},
            mode="latest", window_days=7,
        )
        assert v == "2024-08-15" and s == "huggingface" and c is None

    def test_no_dates_returns_none_tuple(self):
        v, s, c = SourceHandler.get_date_field_with_window_conflict(
            "releaseTime", {"huggingface": {}, "github": {}},
            mode="latest", window_days=7,
        )
        assert v is None and s is None and c is None


# ---------------------------------------------------------------------------
# Org name normalisation: suppliedBy / originatedBy
# ---------------------------------------------------------------------------

class TestOrgFields:
    def test_same_org_no_conflict(self):
        hf = {"suppliedBy": "Meta"}
        gh = {"suppliedBy": "meta"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "suppliedBy", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_org,
        )
        assert v == "Meta" and s == "huggingface" and c is None

    def test_different_org_priority_wins(self):
        hf = {"suppliedBy": "Org-A"}
        gh = {"suppliedBy": "Org-B"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "suppliedBy", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_org,
        )
        assert v == "Org-A" and s == "huggingface"
        assert c is not None
        assert "github" in (c if isinstance(c, str) else c.get("source", ""))

    def test_alias_treats_orgs_as_same(self):
        hf = {"suppliedBy": "Meta"}
        gh = {"suppliedBy": "facebookresearch"}
        custom = lambda v: normalize_org(v, aliases={"meta": "fb", "facebookresearch": "fb"})
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "suppliedBy", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=custom,
        )
        assert v == "Meta" and c is None


# ---------------------------------------------------------------------------
# URL normalisation: downloadLocation
# ---------------------------------------------------------------------------

class TestDownloadLocationField:
    def test_trailing_slash_no_conflict(self):
        hf = {"software_downloadLocation": "https://huggingface.co/m/x"}
        gh = {"software_downloadLocation": "https://huggingface.co/m/x/"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "software_downloadLocation", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_url,
        )
        assert v == "https://huggingface.co/m/x" and c is None

    def test_www_no_conflict(self):
        hf = {"software_downloadLocation": "https://huggingface.co/m/x"}
        gh = {"software_downloadLocation": "https://www.huggingface.co/m/x"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "software_downloadLocation", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_url,
        )
        assert c is None

    def test_different_repos_conflict(self):
        hf = {"software_downloadLocation": "https://huggingface.co/A/x"}
        gh = {"software_downloadLocation": "https://github.com/B/x"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "software_downloadLocation", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_url,
        )
        assert v.startswith("https://huggingface.co")
        assert c is not None


# ---------------------------------------------------------------------------
# Version normalisation: packageVersion
# ---------------------------------------------------------------------------

class TestPackageVersionField:
    def test_v_prefix_no_conflict(self):
        hf = {"packageVersion": "v1.2.3"}
        gh = {"packageVersion": "1.2.3"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "packageVersion", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_version,
        )
        assert v == "v1.2.3" and c is None

    def test_different_versions_conflict(self):
        hf = {"packageVersion": "1.2.3"}
        gh = {"packageVersion": "1.2.4"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "packageVersion", {"huggingface": hf, "github": gh},
            priority=["huggingface", "github"], normaliser=normalize_version,
        )
        assert v == "1.2.3"
        assert c is not None


# ---------------------------------------------------------------------------
# contentIdentifier (Dataset): GH > HF
# ---------------------------------------------------------------------------

class TestContentIdentifierField:
    def test_gh_wins_priority_when_both_present(self):
        hf = {"contentIdentifier": "abc123"}
        gh = {"contentIdentifier": "def456"}
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "contentIdentifier", {"huggingface": hf, "github": gh},
            priority=["github", "huggingface"],
        )
        assert v == "def456" and s == "github"
        assert c is not None

    def test_hf_only_no_conflict(self):
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "contentIdentifier", {"huggingface": {"contentIdentifier": "abc"}, "github": {}},
            priority=["github", "huggingface"],
        )
        assert v == "abc" and s == "huggingface" and c is None

    def test_neither_returns_none(self):
        v, s, c = SourceHandler.get_field_conflict_with_priority(
            "contentIdentifier", {"huggingface": {}, "github": {}},
            priority=["github", "huggingface"],
        )
        assert v is None and s is None and c is None
