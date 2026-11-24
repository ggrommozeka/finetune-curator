"""Read provider/model endpoints from OpenCode opencode.json / opencode.jsonc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.llm.parsing import parse_config_text


@dataclass
class OpenCodeModel:
    provider: str
    model: str
    api_model: str
    base_url: str | None
    api_key: str | None
    ref: str


def parse_config_file(path: Path) -> dict:
    data = parse_config_text(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def discover_config_files(
    cwd: Path,
    home: Path,
    xdg: str | None = None,
    explicit: Path | None = None,
) -> list[Path]:
    """Global config first, then project configs from farthest to closest.

    Closest project file wins when the list is merged in order. An explicit
    path replaces discovery.
    """
    if explicit is not None:
        return [explicit]
    global_dir = Path(xdg) / "opencode" if xdg else home / ".config" / "opencode"
    ordered: list[Path] = []
    for name in ("opencode.json", "opencode.jsonc"):
        ordered.append(global_dir / name)

    project: list[Path] = []
    directories = [cwd, *cwd.parents]
    for directory in directories:
        for relative in (
            Path("opencode.json"),
            Path("opencode.jsonc"),
            Path(".opencode") / "opencode.json",
            Path(".opencode") / "opencode.jsonc",
        ):
            project.append(directory / relative)
        if directory.parent == directory:
            break
    ordered.extend(reversed(project))

    seen: set[str] = set()
    unique: list[Path] = []
    for path in ordered:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_merged_config(paths: list[Path]) -> dict:
    merged: dict = {}
    found = False
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        merged = deep_merge(merged, parse_config_file(path))
        found = True
    if not found and len(paths) == 1:
        raise FileNotFoundError(paths[0])
    return merged


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _providers(config: dict) -> dict:
    current = config.get("providers")
    if isinstance(current, dict) and current:
        return current
    legacy = config.get("provider")
    if isinstance(legacy, dict):
        return legacy
    return current if isinstance(current, dict) else {}


def _first_present(dicts: list[dict], keys: tuple[str, ...]) -> str | None:
    for item in dicts:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _env_api_key(provider: dict, env: dict) -> str | None:
    names = provider.get("env") or []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        return None
    for name in names:
        value = env.get(str(name))
        if value:
            return str(value)
    return None


def _build_model(provider_id: str, model_id: str, provider: dict, model: dict, env: dict) -> OpenCodeModel:
    provider_settings = _mapping(provider.get("settings"))
    provider_options = _mapping(provider.get("options"))
    model_settings = _mapping(model.get("settings"))
    model_options = _mapping(model.get("options"))
    lookup = [model_settings, model_options, provider_settings, provider_options, model, provider]
    base_url = _first_present(lookup, ("baseURL", "baseUrl", "base_url"))
    api_key = _first_present(lookup, ("apiKey", "api_key")) or _env_api_key(provider, env)
    api_model = model.get("modelID") or model.get("modelId") or model.get("id") or model_id
    return OpenCodeModel(
        provider=provider_id,
        model=model_id,
        api_model=str(api_model),
        base_url=base_url,
        api_key=api_key,
        ref=f"{provider_id}/{model_id}",
    )


def iter_configured_models(config: dict, env: dict | None = None) -> list[OpenCodeModel]:
    env = env if env is not None else {}
    found: list[OpenCodeModel] = []
    for provider_id, provider in _providers(config).items():
        provider_map = _mapping(provider)
        models = provider_map.get("models") or {}
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            found.append(
                _build_model(str(provider_id), str(model_id), provider_map, _mapping(model), env)
            )
    return found


def resolve_model_ref(config: dict, ref: str, env: dict | None = None) -> OpenCodeModel:
    models = iter_configured_models(config, env)
    ref = (ref or "").strip()
    known = ", ".join(item.ref for item in models) or "(none)"
    if not ref:
        raise ValueError(f"OpenCode model ref is empty. Known: {known}")
    if "/" in ref:
        provider, model = ref.split("/", 1)
        for item in models:
            if item.provider == provider and item.model == model:
                return item
        raise ValueError(f"OpenCode model {ref!r} not found. Known: {known}")
    matches = [item for item in models if item.model == ref or item.api_model == ref]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"OpenCode model {ref!r} not found. Known: {known}")
    refs = ", ".join(item.ref for item in matches)
    raise ValueError(f"OpenCode model {ref!r} is ambiguous ({refs}). Pass provider/model.")
