"""GitLab REST client: pagination, rate limits, on-disk cache, token redaction.

The personal access token is sent as a header and is never written to the cache,
logs, or exception text.
"""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class GitLabError(RuntimeError):
    """A GitLab request failed. The message is safe to print."""


_PRIVATE_TOKEN = re.compile(r"(private[_-]?token=)[^&\s]+", re.IGNORECASE)
_HEADER_TOKEN = re.compile(r"(PRIVATE-TOKEN:\s*)\S+", re.IGNORECASE)


def redact(text: str, token: str | None = None) -> str:
    """Remove tokens and private_token query values from a string."""
    if not text:
        return ""
    cleaned = text
    if token:
        cleaned = cleaned.replace(token, "***")
    cleaned = _PRIVATE_TOKEN.sub(r"\1***", cleaned)
    cleaned = _HEADER_TOKEN.sub(r"\1***", cleaned)
    return cleaned


def quote_id(value: str | int) -> str:
    return urllib.parse.quote(str(value), safe="")


def merge_request_path(kind: str, ident: str | int) -> str:
    """API path for merged-MR listing. ``kind`` is ``project`` or ``group``."""
    quoted = quote_id(ident)
    if kind == "group":
        return f"/api/v4/groups/{quoted}/merge_requests"
    if kind == "project":
        return f"/api/v4/projects/{quoted}/merge_requests"
    raise ValueError(f"unknown GitLab scope {kind!r}")


def build_ssl_context(verify: bool, ca_file: str | None = None) -> ssl.SSLContext:
    if not verify:
        return ssl._create_unverified_context()
    if ca_file:
        path = Path(ca_file)
        if not path.is_file():
            raise GitLabError(f"CA bundle not found: {path}")
        return ssl.create_default_context(cafile=str(path))
    return ssl.create_default_context()


def _norm_headers(headers: dict | None) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in (headers or {}).items()}


def _retry_delay(headers: dict[str, str], attempt: int, now: float) -> float:
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    reset = headers.get("ratelimit-reset")
    if reset:
        try:
            return max(0.0, float(reset) - now)
        except ValueError:
            pass
    return min(30.0, 0.5 * (2**attempt))


class FixtureTransport:
    """Replay recorded responses. ``pages`` are 1-based GitLab result pages."""

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.calls: list[tuple[str, str, int]] = []

    @classmethod
    def from_file(cls, path: Path) -> FixtureTransport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        responses = data.get("responses") if isinstance(data, dict) else data
        if not isinstance(responses, list):
            raise GitLabError(f"Fixture {path} must contain a responses list")
        return cls(responses)

    def __call__(self, method: str, url: str, headers: dict) -> tuple[int, dict, bytes]:
        parsed = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)
        page = int((query.get("page") or ["1"])[0])
        self.calls.append((method, path, page))
        match = None
        for item in self.responses:
            item_path = urllib.parse.unquote(str(item.get("path") or ""))
            if path == item_path or path.endswith(item_path):
                match = item
                break
        if match is None:
            body = json.dumps({"message": "not in fixture", "path": path}).encode()
            return 404, {}, body
        pages = match.get("pages") or []
        if page - 1 >= len(pages):
            return 200, {"x-next-page": ""}, b"[]"
        page_spec = pages[page - 1]
        raw_body = page_spec.get("body", [])
        raw = raw_body if isinstance(raw_body, bytes) else json.dumps(raw_body).encode()
        return int(page_spec.get("status", 200)), _norm_headers(page_spec.get("headers")), raw


class GitLabClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        verify: bool = True,
        ca_file: str | None = None,
        cache_dir: Path | None = None,
        transport=None,
        sleep=time.sleep,
        clock=time.time,
        timeout: float = 60,
        max_retries: int = 5,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or ""
        self.verify = verify
        self.ca_file = ca_file
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transport = transport
        self.sleep = sleep
        self.clock = clock
        self.timeout = timeout
        self.max_retries = max_retries
        self._ssl = None if transport else build_ssl_context(verify, ca_file)

    def __repr__(self) -> str:
        return f"GitLabClient(base_url={self.base_url!r}, token=***)"

    def get_all(self, path: str, params: dict | None = None) -> list:
        page = 1
        seen: set[int] = set()
        items: list = []
        base_params = dict(params or {})
        base_params.setdefault("per_page", 100)
        while page not in seen:
            seen.add(page)
            query = dict(base_params)
            query["page"] = page
            _status, headers, payload = self.request_json("GET", path, query)
            if not isinstance(payload, list):
                raise GitLabError(f"Expected a JSON list from {path}")
            items.extend(payload)
            nxt = (headers.get("x-next-page") or "").strip()
            if not nxt:
                break
            try:
                nxt_page = int(nxt)
            except ValueError:
                break
            if nxt_page == page:
                break
            page = nxt_page
        return items

    def get_all_optional(self, path: str, params: dict | None = None) -> list:
        """Like ``get_all``, but a 404 is an empty list so one missing commit does not abort sync."""
        try:
            return self.get_all(path, params)
        except GitLabError as exc:
            if str(exc).startswith("GitLab 404"):
                return []
            raise

    def get_text(self, path: str, params: dict | None = None) -> str | None:
        """Fetch a raw file. ``None`` on 404. Other errors raise ``GitLabError``."""
        url = self._url(path, params)
        last_status = None
        for attempt in range(self.max_retries):
            status, headers, body = self._call("GET", url)
            headers = _norm_headers(headers)
            if status == 429 or status == 503:
                last_status = status
                self.sleep(_retry_delay(headers, attempt, self.clock()))
                continue
            if status == 404:
                return None
            if status >= 400:
                snippet = body.decode("utf-8", errors="replace")[:300]
                raise GitLabError(redact(f"GitLab {status} for {path}: {snippet}", self.token))
            return body.decode("utf-8", errors="replace")
        raise GitLabError(redact(f"GitLab rate limit persisted ({last_status}) for {path}", self.token))

    def request_json(self, method: str, path: str, params: dict | None = None):
        url = self._url(path, params)
        cached = self._read_cache(method, url)
        if cached is not None:
            return cached
        headers, payload = self._request_uncached(method, url, path)
        self._write_cache(method, url, headers, payload)
        return 200, headers, payload

    def _url(self, path: str, params: dict | None) -> str:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base_url + path
        if params:
            query = urllib.parse.urlencode(params, doseq=True)
            url = url + "?" + query
        return url

    def _request_uncached(self, method: str, url: str, path: str):
        last_status = None
        for attempt in range(self.max_retries):
            status, headers, body = self._call(method, url)
            headers = _norm_headers(headers)
            if status == 429 or status == 503:
                last_status = status
                self.sleep(_retry_delay(headers, attempt, self.clock()))
                continue
            if status >= 400:
                snippet = body.decode("utf-8", errors="replace")[:300]
                raise GitLabError(redact(f"GitLab {status} for {path}: {snippet}", self.token))
            try:
                payload = json.loads(body.decode("utf-8") or "null")
            except json.JSONDecodeError as exc:
                raise GitLabError(redact(f"GitLab returned non-JSON for {path}", self.token)) from exc
            self._pause_if_limited(headers)
            return headers, payload
        raise GitLabError(redact(f"GitLab rate limit persisted ({last_status}) for {path}", self.token))

    def _pause_if_limited(self, headers: dict[str, str]) -> None:
        remaining = headers.get("ratelimit-remaining")
        if remaining is None:
            return
        try:
            left = int(remaining)
        except ValueError:
            return
        if left > 0:
            return
        self.sleep(_retry_delay(headers, 0, self.clock()))

    def _call(self, method: str, url: str) -> tuple[int, dict, bytes]:
        request_headers = {}
        if self.token:
            request_headers["PRIVATE-TOKEN"] = self.token
        if self.transport is not None:
            try:
                return self.transport(method, url, request_headers)
            except GitLabError:
                raise
            except Exception as exc:
                raise GitLabError(redact(f"GitLab transport error: {exc}", self.token)) from exc
        req = urllib.request.Request(url, method=method)
        for key, value in request_headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, context=self._ssl, timeout=self.timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read() if exc.fp is not None else b""
            return exc.code, dict(exc.headers or {}), payload
        except urllib.error.URLError as exc:
            raise GitLabError(redact(f"GitLab connection failed: {exc.reason}", self.token)) from exc

    def _cache_file(self, method: str, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(f"{method} {url}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, method: str, url: str):
        path = self._cache_file(method, url)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        headers = _norm_headers(data.get("headers"))
        return int(data.get("status") or 200), headers, data.get("body")

    def _write_cache(self, method: str, url: str, headers: dict, payload) -> None:
        path = self._cache_file(method, url)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep only pagination headers. Never persist the request or the token.
        kept = {
            key: headers[key]
            for key in ("x-next-page", "x-page", "ratelimit-remaining")
            if key in headers
        }
        path.write_text(
            json.dumps({"status": 200, "headers": kept, "body": payload}, ensure_ascii=False),
            encoding="utf-8",
        )
