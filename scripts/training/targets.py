"""Default base model and chat template for each training target.

The author model is ``Qwen/Qwen3.8-27B``. It is built on the Qwen3.5 architecture,
so the accurate recipe is still bf16 LoRA. Unsloth's warning against 4-bit QLoRA
is a quantization note for that architecture, not a different checkpoint id.
Reviewer candidates are Ornith-1.0-9B and Qwen3.5-9B. Both the id and the template
can be overridden with ``--base-model`` and ``--chat-template``.
"""

from __future__ import annotations

AUTHOR = "author"
REVIEWER = "reviewer"

TARGETS = {
    AUTHOR: {
        "base_model": "Qwen/Qwen3.8-27B",
        "chat_template": "qwen3.8",
        "candidates": ["Qwen/Qwen3.8-27B"],
        "precision": "bf16-lora",
        "vram": "~56GB",
        "max_tokens": 32_000,
    },
    REVIEWER: {
        "base_model": "Qwen/Qwen3.5-9B",
        "chat_template": "qwen3",
        "candidates": [
            "deepreinforce-ai/Ornith-1.0-9B",
            "Qwen/Qwen3.5-9B",
        ],
        "precision": "bf16-lora",
        "vram": "~22GB",
        "max_tokens": 8_192,
    },
}


def target_for_tasks(tasks: set[str] | None) -> str:
    if tasks == {"review"}:
        return REVIEWER
    return AUTHOR


def resolve_target(tasks: set[str] | None, base_model: str | None, chat_template: str | None) -> dict:
    name = target_for_tasks(tasks)
    spec = dict(TARGETS[name])
    spec["name"] = name
    if base_model:
        spec["base_model"] = base_model
    if chat_template:
        spec["chat_template"] = chat_template
    return spec
