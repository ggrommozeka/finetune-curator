"""Shared training sample."""

from __future__ import annotations

from dataclasses import dataclass, field

AGENTIC_SYSTEM = (
    "You are a coding agent in OpenCode. Complete the user's task. "
    "Use tools when you need to read or change the project, and follow tool results."
)
STYLE_SYSTEM = (
    "Rewrite the code so it follows the project's conventions. "
    "Return only the updated code."
)
REVIEW_SYSTEM = (
    "You review a diff. Point at the file and lines that have a problem and say what to change. "
    "If the change is fine, reply exactly: No issues."
)

NO_ISSUES = "No issues."


@dataclass
class Sample:
    id: str
    task: str
    source: str
    machine: str
    group: str
    messages: list[dict]
    kind: str
    dpo: dict | None = None
    gold: str = ""
    extra: dict = field(default_factory=dict)

    def _fields(self) -> dict:
        extra = self.extra or {}
        return {
            "id": self.id,
            "task": self.task,
            "source": self.source,
            "machine": self.machine or extra.get("machine") or "",
            "source_model": extra.get("source_model") or "",
            "source_provider": extra.get("source_provider") or "",
            "reviewer": extra.get("reviewer") or "",
            "group": self.group,
            "kind": self.kind,
        }

    def as_sft(self) -> dict:
        return {**self._fields(), "messages": self.messages}

    def as_dpo(self) -> dict | None:
        if not self.dpo:
            return None
        return {
            **self._fields(),
            "prompt": self.dpo["prompt"],
            "chosen": self.dpo["chosen"],
            "rejected": self.dpo["rejected"],
        }


def _with_system(system: str, messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system}, *messages]
