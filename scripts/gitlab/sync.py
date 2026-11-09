"""Phase 1: pull GitLab into the local store. Phase 2 never calls this."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from datetime import datetime

from scripts.gitlab.client import GitLabError, quote_id
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


_LOG = logging.getLogger(__name__)
# Cross-project ``group/project#123`` before a bare ``#123`` or ``issues/123``.
_ISSUE_REF = re.compile(
    r"(?P<path>[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)#(?P<iid>\d+)|(?:#|issues/)(?P<local>\d+)"
)


def linked_issue_refs(*texts: str) -> list[tuple[str, str]]:
    """Issue references from MR title and description.

    Each item is ``(project_path, iid)``. An empty project path means the MR's own project.
    """
    found: list[tuple[str, str]] = []
    for text in texts:
        for match in _ISSUE_REF.finditer(text or ""):
            iid = match.group("iid") or match.group("local")
            path = match.group("path") or ""
            item = (path, iid)
            if item not in found:
                found.append(item)
    return found


def linked_issue_iids(text: str) -> list[str]:
    """Same-project issue iids. Cross-project refs are not included."""
    return [iid for path, iid in linked_issue_refs(text) if not path]


def _base_sha(mr: dict, commits: list[dict]) -> str:
    refs = mr.get("diff_refs") if isinstance(mr.get("diff_refs"), dict) else {}
    if refs.get("base_sha"):
        return str(refs["base_sha"])
    if commits:
        parent = _parent_sha(commits[0])
        if parent:
            return parent
    return ""


def fetch_bundle(client, mr: dict, *, project_path: str | None = None) -> dict:
    """Download discussions, diffs, base-commit file snapshots, and linked issues."""
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
    base = _base_sha(mr, commits)
    changed = []
    for diffs in diffs_by_sha.values():
        for diff in diffs or []:
            for path in _diff_paths(diff):
                if path not in changed:
                    changed.append(path)
    if base:
        for path in changed:
            content = client.get_text(
                f"/api/v4/projects/{project_id}/repository/files/{quote_id(path)}/raw",
                {"ref": base},
            )
            if content is None:
                continue
            files.append({"sha": base, "path": path, "side": "base", "content": content})
    issues = []
    for issue_path, issue_iid in linked_issue_refs(str(mr.get("title") or ""), str(mr.get("description") or "")):
        project_ref: str | int = quote_id(issue_path) if issue_path else project_id
        try:
            _status, _headers, payload = client.request_json(
                "GET", f"/api/v4/projects/{project_ref}/issues/{issue_iid}"
            )
        except GitLabError as exc:
            _LOG.warning("linked issue %s#%s was not stored: %s", issue_path or project_id, issue_iid, exc)
            continue
        if isinstance(payload, dict):
            issues.append(
                {
                    "iid": payload.get("iid") or issue_iid,
                    "project_path": issue_path,
                    "title": payload.get("title") or "",
                    "description": payload.get("description") or "",
                }
            )
    return {
        "mr": mr,
        "discussions": discussions,
        "commits": commits,
        "diffs_by_sha": diffs_by_sha,
        "files": files,
        "issues": issues,
        "base_sha": base,
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
            issues=bundle.get("issues") or [],
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
            issues=bundle.get("issues") or [],
        )
        stats["refreshed"] += 1
    return stats


def _path_from_web(mr: dict) -> str:
    """Project path between the host and ``/-/merge_requests/``, including nested groups."""
    web = str(mr.get("web_url") or "")
    marker = "/-/merge_requests/"
    if marker not in web:
        return ""
    left = web.split(marker, 1)[0]
    path = urllib.parse.unquote(urllib.parse.urlparse(left).path).strip("/")
    return path


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
    fix_pattern: str = DEFAULT_FIX_PATTERN,
    line_pad: int = 2,
    max_chars: int = 8000,
    anonymize: bool = False,
    anonymize_urls: bool = False,
    signal_config: SignalConfig | None = None,
    complete_fn=None,
) -> dict:
    """Phase 2. Reads the store only and replaces example rows for each MR.

    Every reviewer's comments are stored. A reviewer filter belongs on list and
    the viewer, not on this write path.
    """
    from scripts.gitlab.classify import classify_commits
    from scripts.gitlab.examples import apply_anonymization
    from scripts.gitlab.export import records_from_traces
    from scripts.gitlab.polish import build_polish_records
    from scripts.gitlab.signals import load_signal_config
    from scripts.gitlab.trace import build_traces

    totals = {"merge_requests": 0, "examples": 0, "comments": 0, "file_line": 0, "file": 0, "mr": 0, "unlinked": 0}
    for ref in refs:
        project_id, iid = store.resolve_ref(ref)
        bundle = store.load_bundle(project_id, iid)
        if bundle is None:
            raise KeyError(ref)
        overrides = store.overrides_for(project_id, iid)
        traces = build_traces(
            bundle["discussions"],
            bundle["commits"],
            bundle["diffs_by_sha"],
            bundle["mr"],
            line_pad=line_pad,
            files=bundle.get("files") or [],
            overrides=overrides,
        )
        if anonymize or anonymize_urls:
            traces = _anonymize_traces(traces, strip_urls=anonymize_urls)
        store.replace_traces(project_id, iid, traces)
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
