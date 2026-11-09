"""The shared item model can point at an OpenCode session."""

import json
from pathlib import Path

from scripts.ingest.catalog import Catalog


def test_opencode_item_carries_source(tmp_path: Path):
    path = tmp_path / "sessions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "ses_a",
                "messages": [{"role": "user", "content": "fix the failing test"}],
                "score": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = Catalog(path, tmp_path / "missing-gitlab")
    rows = catalog.opencode_records()
    assert rows[0]["source"] == "opencode"
    assert catalog.gitlab_mrs() == []
    item = catalog.resolve("ses_a")
    assert item is not None
    assert item.source == "opencode"
    assert item.locator["session_id"] == "ses_a"
    assert item.as_dict()["source"] == "opencode"
    assert catalog.resolve("missing") is None
