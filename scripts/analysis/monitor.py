"""Compare session metrics before and after an adopted rules/skills version, and roll back."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.analysis.store import PatternStore
from scripts.ingest.opencode_sessions import load_review_decisions


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _in_window(created_at: str, start: str | None, end: str | None) -> bool:
    if not created_at:
        return False
    if start and created_at < start:
        return False
    if end and created_at >= end:
        return False
    return True


def snapshot_version(store: PatternStore, version_id: str, files: list[Path], note: str = "") -> Path:
    """Copy exported files into the store and record them as the current version."""
    folder = store.versions_root / version_id
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for path in files:
        target = folder / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.exists():
            shutil.copy2(path, target)
        manifest_files.append({"name": path.name, "dest": str(path), "is_dir": path.is_dir()})
    store.add_version(version_id, {"files": manifest_files}, note=note)
    return folder


def rollback(store: PatternStore, version_id: str) -> list[str]:
    """Restore an earlier snapshot and retire versions adopted after it."""
    versions = store.versions()
    chosen = next((row for row in versions if row["id"] == version_id), None)
    if chosen is None:
        raise KeyError(version_id)
    restored = []
    folder = store.versions_root / version_id
    for item in chosen["manifest"].get("files") or []:
        source = folder / item["name"]
        dest = Path(item["dest"])
        if not source.exists():
            continue
        if item.get("is_dir"):
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        restored.append(str(dest))
    store.retire_after(version_id)
    return restored


def _discard_rate(session_ids: list[str], decisions_path: Path | None) -> float | None:
    if decisions_path is None or not decisions_path.exists():
        return None
    labels = load_review_decisions(decisions_path)["labels"]
    relevant = [labels.get(sid) for sid in session_ids if sid in labels]
    if not relevant:
        return None
    discards = sum(1 for label in relevant if label == "discard")
    return round(discards / len(relevant), 3)


def report(store: PatternStore, *, decisions_path: Path | None = None, error_type: str | None = None) -> list[dict]:
    """Compare sessions dated before vs after each still-active version."""
    sessions = store.sessions()
    annotations = {row["global_id"]: row["annotation"] for row in store.annotated_rows()}
    versions = [row for row in store.versions() if not row.get("retired_at")]
    reports = []
    for index, version in enumerate(versions):
        start = version["adopted_at"]
        end = versions[index + 1]["adopted_at"] if index + 1 < len(versions) else None
        before = [row for row in sessions if row.get("created_at") and row["created_at"] < start]
        after = [row for row in sessions if _in_window(row.get("created_at") or "", start, end)]

        def metrics(group: list[dict]) -> dict:
            scores = [float(row["score"]) for row in group if row.get("score") is not None]
            ids = [row["global_id"] for row in group]
            anns = [annotations[gid] for gid in ids if gid in annotations]
            corrections = sum(1 for ann in anns if (ann.get("user_correction") or "").strip())
            recurrence = 0
            if error_type:
                recurrence = sum(1 for ann in anns if ann.get("error_type") == error_type)
            return {
                "sessions": len(group),
                "mean_score": _mean(scores),
                "discard_rate": _discard_rate(ids, decisions_path),
                "user_correction_rate": round(corrections / len(anns), 3) if anns else None,
                "error_recurrence": recurrence,
            }

        reports.append(
            {
                "version": version["id"],
                "adopted_at": start,
                "before": metrics(before),
                "after": metrics(after),
            }
        )
    return reports
