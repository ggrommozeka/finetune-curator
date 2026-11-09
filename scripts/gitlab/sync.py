"""Phase 1: pull GitLab into the local store. Phase 2 never calls this."""

from __future__ import annotations

import json
import urllib.parse
from datetime import datetime

from scripts.gitlab.client import quote_id
from scripts.gitlab.anonymize import Anonymizer
from scripts.gitlab.export import list_merge_requests
from scripts.gitlab.link import DEFAULT_FIX_PATTERN
from scripts.gitlab.signals import SignalConfig
from scripts.gitlab.store import GitlabStore


def _parent_sha(commit: dict) -> str | None:
    parents = commit.get("parent_ids") or []
    if parents:
        return str(parents[0])
    return None


def _diff_paths(diff: dict) -> list[str]:
    paths = []
    for key in ("new_path", "old_path"):
        path = diff.get(key)
        if path and path not in paths:
            paths.append(path)
    return paths


def fetch_bundle(client, mr: dict, *, project_path: str | None = None) -> dict:
    """Download discussions, every commit diff, and file text at the commit and its parent."""
    project_id = int(mr["project_id"])
    iid = int(mr["iid"])
    base = f"/api/v4/projects/{project_id}/merge_requests/{iid}"
    discussions = client.get_all(base + "/discussions")
    commits = client.get_all(base + "/commits")
    diffs_by_sha: dict[str, list] = {}
    files: list[dict] = []
    seen_files: set[tuple[str, str, str]] = set()
    for commit in commits:
        sha = str(commit.get("id") or "")
        if not sha:
            continue
        diffs = client.get_all_optional(f"/api/v4/projects/{project_id}/repository/commits/{sha}/diff")
        diffs_by_sha[sha] = diffs
        parent = _parent_sha(commit)
        for diff in diffs:
            for path in _diff_paths(diff):
                encoded = quote_id(path)
                sides = [("after", sha)]
                if parent and not diff.get("new_file"):
                    sides.append(("before", parent))
                if diff.get("deleted_file"):
                    sides = [("before", parent)] if parent else []
                for side, ref in sides:
                    key = (sha, path, side)
                    if key in seen_files or not ref:
                        continue
                    seen_files.add(key)
                    content = client.get_text(
                        f"/api/v4/projects/{project_id}/repository/files/{encoded}/raw",
                        {"ref": ref},
                    )
                    if content is None:
                        continue
                    files.append({"sha": sha, "path": path, "side": side, "content": content})
    return {
        "mr": mr,
        "discussions": discussions,
        "commits": commits,
        "diffs_by_sha": diffs_by_sha,
        "files": files,
        "project_path": project_path or "",
    }


def _fetch_mr(client, project: str | int, iid: int) -> dict:
    path = f"/api/v4/projects/{quote_id(project)}/merge_requests/{iid}"
    _status, _headers, payload = client.request_json("GET", path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"MR {project}!{iid} was not a JSON object")
    return payload


def sync_store(
    client,
    store: GitlabStore,
    *,
    projects: list[str] | None = None,
    groups: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    refresh: list[str] | None = None,
) -> dict:
    """Fetch new MRs. Stored MRs are skipped unless listed in ``refresh`` (``project!iid``)."""
    projects = projects or []
    groups = groups or []
    refresh_refs = list(refresh or [])
    listed = list_merge_requests(client, projects, groups, since, until) if (projects or groups) else []
    stats = {"fetched": 0, "skipped": 0, "refreshed": 0}
    seen: set[tuple[int, int]] = set()

    for mr in listed:
        project_id = int(mr["project_id"])
        iid = int(mr["iid"])
        path = projects[0] if len(projects) == 1 and not str(projects[0]).isdigit() else _path_from_web(mr)
        forced = _is_refresh(refresh_refs, project_id, iid, path, store)
        if store.has_mr(project_id, iid) and not forced:
            seen.add((project_id, iid))
            stats["skipped"] += 1
            continue
        bundle = fetch_bundle(client, mr, project_path=path)
        store.save_bundle(
            bundle["mr"],
            bundle["discussions"],
            bundle["commits"],
            bundle["diffs_by_sha"],
            bundle["files"],
            project_path=path,
        )
        seen.add((project_id, iid))
        if forced:
            stats["refreshed"] += 1
        else:
            stats["fetched"] += 1

    for ref in refresh_refs:
        try:
            project, iid = _split_refresh(ref)
        except ValueError:
            continue
        if project.isdigit():
            project_id = int(project)
            if (project_id, iid) in seen:
                continue
            mr = _fetch_mr(client, project_id, iid)
            path = _path_from_web(mr)
        else:
            mr = _fetch_mr(client, project, iid)
            project_id = int(mr["project_id"])
            path = project
            if (project_id, iid) in seen:
                continue
        bundle = fetch_bundle(client, mr, project_path=path)
        store.save_bundle(
            bundle["mr"],
            bundle["discussions"],
            bundle["commits"],
            bundle["diffs_by_sha"],
            bundle["files"],
            project_path=path,
        )
        stats["refreshed"] += 1
    return stats


def _path_from_web(mr: dict) -> str:
    web = str(mr.get("web_url") or "")
    marker = "/-/merge_requests/"
    if marker not in web:
        return ""
    left = web.split(marker, 1)[0].rstrip("/")
    parts = urllib.parse.unquote(left).split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return ""


def _split_refresh(ref: str) -> tuple[str, int]:
    from scripts.gitlab.store import parse_mr_ref

    return parse_mr_ref(ref)


def _is_refresh(refresh_refs: list[str], project_id: int, iid: int, path: str, store: GitlabStore) -> bool:
    for ref in refresh_refs:
        try:
            project, ref_iid = _split_refresh(ref)
        except ValueError:
            continue
        if ref_iid != iid:
            continue
        if project.isdigit() and int(project) == project_id:
            return True
        if path and project == path:
            return True
        if not project.isdigit() and store.has_mr(project_id, iid):
            try:
                resolved = store.resolve_ref(ref)
            except KeyError:
                resolved = None
            if resolved == (project_id, iid):
                return True
    return False


def _anonymize_traces(traces: list[dict], *, strip_urls: bool) -> list[dict]:
    anonymizer = Anonymizer(strip_urls=strip_urls)
    for trace in traces:
        anonymizer.register_author(
            {"username": trace.get("author_username"), "name": trace.get("author_name")}
        )
        anonymizer.register_author(trace.get("commit_author") or {})
        for reply in trace.get("replies") or []:
            anonymizer.register_author(
                {"username": reply.get("author_username"), "name": reply.get("author_name")}
            )
        resolved_by = trace.get("resolved_by") or ""
        if resolved_by:
            anonymizer.register_author({"username": resolved_by, "name": resolved_by})
    cleaned = []
    for trace in traces:
        item = json.loads(json.dumps(trace))
        viewed = anonymizer.author_view(
            {"username": trace.get("author_username"), "name": trace.get("author_name")}
        )
        item["author_username"] = viewed.get("username") or ""
        item["reviewer"] = viewed.get("username") or ""
        item["author_name"] = viewed.get("name") or ""
        if item.get("resolved_by"):
            item["resolved_by"] = anonymizer.author_view({"username": trace.get("resolved_by")}).get("username") or ""
        item["body"] = anonymizer.text(item.get("body") or "")
        item["before"] = anonymizer.text(item.get("before") or "")
        item["after"] = anonymizer.text(item.get("after") or "")
        item["diff"] = anonymizer.text(item.get("diff") or "")
        anchor = dict(item.get("anchor") or {})
        anchor["snippet"] = anonymizer.text(anchor.get("snippet") or "")
        item["anchor"] = anchor
        replies = []
        for reply in item.get("replies") or []:
            reply = dict(reply)
            reply_view = anonymizer.author_view(
                {"username": reply.get("author_username"), "name": reply.get("author_name")}
            )
            reply["author_username"] = reply_view.get("username") or ""
            reply["author_name"] = reply_view.get("name") or ""
            reply["body"] = anonymizer.text(reply.get("body") or "")
            replies.append(reply)
        item["replies"] = replies
        author = item.get("commit_author") or {}
        if author:
            item["commit_author"] = anonymizer.author_view(author)
        cleaned.append(item)
    return cleaned


def process_store(
    store: GitlabStore,
    refs: list[str],
    *,
    reviewer: str | None = None,
    fix_pattern: str = DEFAULT_FIX_PATTERN,
    line_pad: int = 2,
    max_chars: int = 8000,
    anonymize: bool = False,
    anonymize_urls: bool = False,
    signal_config: SignalConfig | None = None,
    complete_fn=None,
) -> dict:
    """Phase 2. Reads the store only and replaces example rows for each MR."""
    totals = {"merge_requests": 0, "examples": 0, "comments": 0, "file_line": 0, "file": 0, "mr": 0, "unlinked": 0}
    for ref in refs:
        project_id, iid = store.resolve_ref(ref)
        bundle = store.load_bundle(project_id, iid)
        if bundle is None:
            raise KeyError(ref)
        from scripts.gitlab.export import records_from_traces
        from scripts.gitlab.trace import build_traces

        overrides = store.overrides_for(project_id, iid)
        traces = build_traces(
            bundle["discussions"],
            bundle["commits"],
            bundle["diffs_by_sha"],
            bundle["mr"],
            reviewer=reviewer,
            fix_pattern=fix_pattern,
            line_pad=line_pad,
            files=bundle.get("files") or [],
            overrides=overrides,
        )
        if anonymize or anonymize_urls:
            traces = _anonymize_traces(traces, strip_urls=anonymize_urls)
        store.replace_traces(project_id, iid, traces)
        from scripts.gitlab.classify import classify_commits
        from scripts.gitlab.polish import build_polish_records
        from scripts.gitlab.signals import SignalConfig, load_signal_config

        config = signal_config or load_signal_config(mrr_pattern=fix_pattern)
        roles = classify_commits(bundle["commits"], bundle["mr"], bundle["discussions"], config)
        store.replace_roles(project_id, iid, roles)
        records, stats = records_from_traces(bundle["mr"], traces, max_chars)
        polish = build_polish_records(
            bundle["mr"],
            bundle["commits"],
            bundle["diffs_by_sha"],
            bundle["discussions"],
            config,
            max_chars=max_chars,
            complete_fn=complete_fn,
        )
        if anonymize or anonymize_urls:
            from scripts.gitlab.examples import apply_anonymization

            polish = apply_anonymization(polish, strip_urls=anonymize_urls)
        records.extend(polish)
        stats["polish"] = len(polish)
        if anonymize_urls:
            for record in records:
                gitlab = record.get("gitlab") or {}
                gitlab.pop("web_url", None)
                record["gitlab"] = gitlab
        for record in records:
            record.pop("_authors", None)
        store.replace_examples(project_id, iid, records)
        totals["merge_requests"] += 1
        totals["examples"] += len(records)
        for key in ("comments", "file_line", "file", "mr", "unlinked"):
            totals[key] += stats.get(key, 0)
    return totals
