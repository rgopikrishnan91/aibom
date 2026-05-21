from datetime import datetime, timezone
from unittest.mock import patch
import json

from aikaboom.plugins.avid_security.snapshot import AvidSnapshot


def test_first_run_clones_and_writes_marker(tmp_path):
    cache_dir = tmp_path / "avid"
    with patch("aikaboom.plugins.avid_security.snapshot._git_clone") as gc:
        gc.return_value = "3f2a91c"
        snap = AvidSnapshot(cache_dir=cache_dir, ttl_days=10)
        snap.ensure_fresh()
    assert (cache_dir / "snapshot.json").exists()
    marker = json.loads((cache_dir / "snapshot.json").read_text())
    assert marker["sha"] == "3f2a91c"
    assert marker["ttl_days"] == 10
    assert gc.call_count == 1


from freezegun import freeze_time


def _seed_marker(cache_dir, sha, when):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "snapshot.json").write_text(json.dumps({
        "sha": sha,
        "fetched_at": when,
        "ttl_days": 10,
    }))


@freeze_time("2026-05-20T12:00:00Z", ignore=["transformers"])
def test_within_ttl_does_not_refresh(tmp_path):
    cache_dir = tmp_path / "avid"
    _seed_marker(cache_dir, "old-sha", "2026-05-15T12:00:00Z")  # 5 days old
    with patch("aikaboom.plugins.avid_security.snapshot._git_clone") as gc:
        AvidSnapshot(cache_dir=cache_dir, ttl_days=10).ensure_fresh()
    assert gc.call_count == 0


@freeze_time("2026-05-20T12:00:00Z", ignore=["transformers"])
def test_beyond_ttl_refreshes(tmp_path):
    cache_dir = tmp_path / "avid"
    _seed_marker(cache_dir, "old-sha", "2026-05-05T12:00:00Z")  # 15 days old
    with patch("aikaboom.plugins.avid_security.snapshot._git_clone") as gc:
        gc.return_value = "new-sha"
        AvidSnapshot(cache_dir=cache_dir, ttl_days=10).ensure_fresh()
    assert gc.call_count == 1
    marker = json.loads((cache_dir / "snapshot.json").read_text())
    assert marker["sha"] == "new-sha"


@freeze_time("2026-05-20T12:00:00Z", ignore=["transformers"])
def test_force_refresh_resets_ttl(tmp_path):
    cache_dir = tmp_path / "avid"
    _seed_marker(cache_dir, "old-sha", "2026-05-19T12:00:00Z")  # 1 day old, fresh
    with patch("aikaboom.plugins.avid_security.snapshot._git_clone") as gc:
        gc.return_value = "new-sha"
        AvidSnapshot(cache_dir=cache_dir, ttl_days=10).force_refresh()
    assert gc.call_count == 1
    marker = json.loads((cache_dir / "snapshot.json").read_text())
    assert marker["sha"] == "new-sha"
