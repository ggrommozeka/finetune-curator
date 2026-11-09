"""The one team lead, and the MRR title flag.

There is a single team lead. ``aliases`` are that person's usernames, names,
and emails, not a list of different people. A local ``gitlab.json``,
``FINETUNE_CURATOR_TEAM_LEADS``, or ``--team-lead`` supply the aliases.
Matching is case-insensitive against commit author and committer name, email,
and username.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MRR_PATTERN = r"\bmrr\b"
DEFAULT_CONFIG_NAMES = ("gitlab.json",)


@dataclass(frozen=True)
class SignalConfig:
    """Aliases belong to one team lead. ``team_leads`` is that alias tuple."""

    team_leads: tuple[str, ...]
    mrr_pattern: str

    @property
    def aliases(self) -> tuple[str, ...]:
        return self.team_leads

    def pattern(self) -> re.Pattern[str]:
        return re.compile(self.mrr_pattern, re.IGNORECASE)


def load_signal_config(
    path: Path | str | None = None,
    *,
    team_leads: list[str] | None = None,
    mrr_pattern: str | None = None,
) -> SignalConfig:
    file_leads: list[str] = []
    file_pattern = DEFAULT_MRR_PATTERN
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.extend(Path(name) for name in DEFAULT_CONFIG_NAMES)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            break
        file_leads = _aliases_from_document(data)
        if data.get("mrr_pattern"):
            file_pattern = str(data["mrr_pattern"])
        break
    env_leads = [part.strip() for part in os.environ.get("FINETUNE_CURATOR_TEAM_LEADS", "").split(",") if part.strip()]
    chosen_leads = team_leads if team_leads is not None else (env_leads or file_leads)
    chosen_pattern = mrr_pattern or os.environ.get("FINETUNE_CURATOR_MRR_PATTERN") or file_pattern
    normalized = tuple(dict.fromkeys(item.strip().lower() for item in chosen_leads if item and item.strip()))
    return SignalConfig(team_leads=normalized, mrr_pattern=chosen_pattern or DEFAULT_MRR_PATTERN)


def _aliases_from_document(data: dict) -> list[str]:
    """Read one identity. ``team_lead.aliases`` is preferred; ``team_leads`` is the same list."""
    identity = data.get("team_lead")
    raw = None
    if isinstance(identity, dict):
        raw = identity.get("aliases")
    elif isinstance(identity, str):
        raw = [identity]
    elif isinstance(identity, list):
        raw = identity
    if raw is None:
        raw = data.get("team_leads") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item) for item in raw if str(item).strip()]


def is_mrr_title(message: str, pattern: str = DEFAULT_MRR_PATTERN) -> bool:
    if not message or not pattern:
        return False
    return re.search(pattern, message, flags=re.IGNORECASE) is not None


def commit_identities(commit: dict) -> set[str]:
    values: list[str] = []
    for key in ("author_name", "author_email", "committer_name", "committer_email"):
        if commit.get(key):
            values.append(str(commit[key]))
    for role in ("author", "committer"):
        person = commit.get(role)
        if isinstance(person, dict):
            for key in ("username", "name", "email"):
                if person.get(key):
                    values.append(str(person[key]))
    return {value.strip().lower() for value in values if value and value.strip()}


def person_identities(person: dict | None) -> set[str]:
    if not isinstance(person, dict):
        return set()
    return {str(person[key]).strip().lower() for key in ("username", "name", "email") if person.get(key)}


def is_team_lead(commit: dict, config: SignalConfig) -> bool:
    if not config.team_leads:
        return False
    identities = commit_identities(commit)
    return any(lead in identities for lead in config.team_leads)


def mr_author_identities(mr: dict | None) -> set[str]:
    return person_identities((mr or {}).get("author"))
