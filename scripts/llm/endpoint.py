"""Resolve an OpenAI-compatible endpoint. CLI overrides env, env overrides OpenCode."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

LLAMA_SERVER_BASE_URL = "http://localhost:8080/v1"


@dataclass
class Endpoint:
    base_url: str
    api_key: str | None
    model: str | None
    base_url_source: str
    model_source: str
    api_key_source: str


class EndpointError(ValueError):
    """The chat endpoint could not be resolved."""


def normalize_base_url(url: str) -> str:
    """Ensure an OpenAI-compatible base URL ends in /v1 when no path was given."""
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned
    path = urlparse(cleaned).path
    if path in ("", "/"):
        cleaned += "/v1"
    return cleaned


def _first(*pairs: tuple[str | None, str]) -> tuple[str | None, str | None]:
    for value, source in pairs:
        if value is not None and str(value).strip():
            return str(value).strip(), source
    return None, None


def resolve_endpoint(
    *,
    cli_base_url: str | None = None,
    cli_api_key: str | None = None,
    cli_model: str | None = None,
    env: dict | None = None,
    opencode_base_url: str | None = None,
    opencode_api_key: str | None = None,
    opencode_model: str | None = None,
    default_base_url: str = LLAMA_SERVER_BASE_URL,
) -> Endpoint:
    env = env or {}
    base_url, base_source = _first(
        (cli_base_url, "cli"),
        (env.get("FINETUNE_CURATOR_BASE_URL"), "env:FINETUNE_CURATOR_BASE_URL"),
        (env.get("OPENAI_BASE_URL"), "env:OPENAI_BASE_URL"),
        (opencode_base_url, "opencode"),
    )
    if not base_url:
        base_url, base_source = default_base_url, "default:llama-server"
    base_url = normalize_base_url(base_url)

    model, model_source = _first(
        (cli_model, "cli"),
        (env.get("FINETUNE_CURATOR_MODEL"), "env:FINETUNE_CURATOR_MODEL"),
        (env.get("OPENAI_MODEL"), "env:OPENAI_MODEL"),
        (opencode_model, "opencode"),
    )
    api_key, api_key_source = _first(
        (cli_api_key, "cli"),
        (env.get("FINETUNE_CURATOR_API_KEY"), "env:FINETUNE_CURATOR_API_KEY"),
        (env.get("OPENAI_API_KEY"), "env:OPENAI_API_KEY"),
        (opencode_api_key, "opencode"),
    )
    return Endpoint(
        base_url=base_url,
        api_key=api_key,
        model=model,
        base_url_source=base_source or "default:llama-server",
        model_source=model_source or "auto",
        api_key_source=api_key_source or "none",
    )


def format_endpoint(endpoint: Endpoint) -> str:
    """One-line description that never includes the API key value."""
    if endpoint.api_key:
        key = f"set ({endpoint.api_key_source})"
    else:
        key = "not set (optional for llama-server)"
    model = endpoint.model or "(auto)"
    return (
        f"model={model} ({endpoint.model_source})  "
        f"base_url={endpoint.base_url} ({endpoint.base_url_source})  "
        f"api_key={key}"
    )


def pick_remote_model(model_ids: list[str]) -> str:
    ids = [item for item in model_ids if item]
    if not ids:
        raise EndpointError(
            "The server returned no models. Pass --model (llama-server often accepts the GGUF name)."
        )
    if len(ids) > 1:
        listed = "\n".join(f"  - {item}" for item in ids)
        raise EndpointError(f"Multiple models on the server. Pass --model:\n{listed}")
    return ids[0]


def fetch_model_ids(base_url: str, api_key: str | None = None, timeout: float = 30) -> list[str]:
    url = normalize_base_url(base_url).rstrip("/") + "/models"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise EndpointError(f"Could not list models at {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EndpointError(f"Model list at {url} was not JSON") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise EndpointError(f"Unexpected model list payload from {url}")
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]
