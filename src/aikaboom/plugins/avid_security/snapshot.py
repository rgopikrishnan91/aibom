from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sqlite3
import subprocess

AVID_REPO = "https://github.com/avidml/avid-db.git"


def _git_clone(target: Path) -> str:
    """Shallow-clone avid-db, return the HEAD SHA. Overridden in tests."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        subprocess.run(["rm", "-rf", str(target)], check=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", AVID_REPO, str(target)],
        check=True, capture_output=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return sha


@dataclass
class AvidSnapshot:
    cache_dir: Path
    ttl_days: int = 10

    @property
    def repo_dir(self) -> Path:
        return self.cache_dir / "avid-db"

    @property
    def marker_path(self) -> Path:
        return self.cache_dir / "snapshot.json"

    def _is_stale(self) -> bool:
        marker = json.loads(self.marker_path.read_text())
        fetched_at = datetime.fromisoformat(marker["fetched_at"].replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - fetched_at
        return age > timedelta(days=self.ttl_days)

    def ensure_fresh(self) -> None:
        if not self.marker_path.exists() or self._is_stale():
            sha = _git_clone(self.repo_dir)
            self._write_marker(sha)

    def force_refresh(self) -> None:
        sha = _git_clone(self.repo_dir)
        self._write_marker(sha)

    def _write_marker(self, sha: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(json.dumps({
            "sha": sha,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ttl_days": self.ttl_days,
        }, indent=2))


def family_prefixes(bare_name: str) -> list[str]:
    """Every hyphenated prefix of bare_name with >=2 tokens, excluding bare_name itself.
    Longest prefix first."""
    tokens = bare_name.lower().split("-")
    if len(tokens) < 3:
        if len(tokens) == 2:
            return []  # bare_name itself is the only 2-token prefix; excluded
        return []
    return ["-".join(tokens[:i]) for i in range(len(tokens) - 1, 1, -1)]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS avid_report (
  report_id TEXT PRIMARY KEY,
  bare_name TEXT NOT NULL,
  developer TEXT,
  artifact_kind TEXT,
  sep_view TEXT,
  risk_domain TEXT,
  lifecycle_view TEXT,
  published_date TEXT,
  source_path TEXT NOT NULL,
  raw_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bare_name ON avid_report(bare_name);
CREATE INDEX IF NOT EXISTS idx_developer_kind ON avid_report(developer, artifact_kind);

CREATE TABLE IF NOT EXISTS avid_report_family_prefix (
  report_id TEXT NOT NULL,
  family_prefix TEXT NOT NULL,
  PRIMARY KEY (report_id, family_prefix),
  FOREIGN KEY (report_id) REFERENCES avid_report(report_id)
);
CREATE INDEX IF NOT EXISTS idx_family_prefix ON avid_report_family_prefix(family_prefix);
"""


@dataclass
class AvidIndex:
    db_path: Path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def build(self, repo_dir: Path) -> int:
        """Rebuild the index from a freshly-cloned avid-db repo dir. Returns count."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()
        count = 0
        with self._conn() as conn:
            for json_path in (repo_dir / "reports").rglob("*.json"):
                doc = json.loads(json_path.read_text())
                artifact = (doc.get("affects", {}).get("artifacts") or [{}])[0]
                if "name" not in artifact:
                    continue
                bare = artifact["name"].lower()
                developer = (doc.get("affects", {}).get("developer") or [None])[0]
                impact = doc.get("impact", {}).get("avid", {})
                conn.execute(
                    "INSERT OR REPLACE INTO avid_report "
                    "(report_id, bare_name, developer, artifact_kind, sep_view, "
                    " risk_domain, lifecycle_view, published_date, source_path, raw_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc["metadata"]["report_id"], bare, developer,
                        artifact.get("type"),
                        json.dumps(impact.get("sep_view", [])),
                        json.dumps(impact.get("risk_domain", [])),
                        json.dumps(impact.get("lifecycle_view", [])),
                        doc.get("reported_date") or doc.get("published_date"),
                        str(json_path.relative_to(repo_dir)),
                        json.dumps(doc),
                    ),
                )
                for fp in family_prefixes(bare):
                    conn.execute(
                        "INSERT OR REPLACE INTO avid_report_family_prefix "
                        "(report_id, family_prefix) VALUES (?, ?)",
                        (doc["metadata"]["report_id"], fp),
                    )
                count += 1
        return count

    def find_exact(self, bare_name: str, artifact_kind: str) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM avid_report WHERE bare_name = ? AND artifact_kind = ?",
                (bare_name.lower(), artifact_kind),
            )
            return [dict(r) for r in cur.fetchall()]

    def find_by_family_prefix(
        self, prefix: str, developer: str | None, artifact_kind: str
    ) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT r.* FROM avid_report r "
                "JOIN avid_report_family_prefix p ON r.report_id = p.report_id "
                "WHERE p.family_prefix = ? AND r.artifact_kind = ? "
                "  AND (r.developer = ? OR (? IS NULL AND r.developer IS NULL))",
                (prefix.lower(), artifact_kind, developer, developer),
            )
            return [dict(r) for r in cur.fetchall()]
