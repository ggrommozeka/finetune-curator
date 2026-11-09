"""Fit check for the at-home 27B profile.

The real command loads the checkpoint and takes a few optimizer steps. Tests use
``simulate_smoke`` so CI does not download a 27B model.
"""

from __future__ import annotations

import json
from pathlib import Path

ALLOWED_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def load_profile(path: Path | str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def validate_home_profile(profile: dict) -> None:
    training = profile.get("training") or {}
    length = int(training.get("max_seq_length") or 0)
    low = int(training.get("min_seq_length") or 2048)
    high = int(training.get("max_seq_length_limit") or 4096)
    if length < low or length > high:
        raise ValueError(f"max_seq_length {length} is outside {low}-{high}")
    rank = int(training.get("lora_r") or 0)
    if rank < int(training.get("lora_r_min") or 8) or rank > int(training.get("lora_r_max") or 16):
        raise ValueError(f"lora_r {rank} is outside 8-16")
    modules = training.get("target_modules") or []
    unknown = [name for name in modules if name not in ALLOWED_MODULES]
    if not modules or unknown:
        raise ValueError(f"target_modules must be a non-empty subset of {sorted(ALLOWED_MODULES)}")
    if training.get("load_in_4bit") is not True:
        raise ValueError("the at-home 27B profile is 4-bit QLoRA")
    if training.get("weight_cpu_offload"):
        raise ValueError("model weights stay on the GPUs; only activations may use RAM")
    if training.get("device_map") != "balanced":
        raise ValueError("shard the 4-bit model with device_map=balanced")
    if training.get("optimizer") != "paged_adamw_8bit":
        raise ValueError("optimizer must be paged_adamw_8bit")
    if training.get("gradient_checkpointing") != "unsloth":
        raise ValueError("gradient_checkpointing must be unsloth so activations can sit in pinned RAM")


def format_report(profile: dict, measurements: dict) -> dict:
    peaks = measurements.get("peak_vram_mib") or {}
    return {
        "profile": profile.get("profile"),
        "base_model": profile.get("base_model"),
        "max_seq_length": (profile.get("training") or {}).get("max_seq_length"),
        "steps": measurements.get("steps"),
        "tokens_per_sec": measurements.get("tokens_per_sec"),
        "peak_vram_mib": {str(gpu): peaks[gpu] for gpu in sorted(peaks, key=lambda item: int(item))},
        "weight_cpu_offload": False,
        "fits": measurements.get("fits", True),
    }


def simulate_smoke(profile: dict, steps: int) -> dict:
    """Planned measurements for a machine without the checkpoint. Does not load weights."""
    validate_home_profile(profile)
    training = profile["training"]
    gpus = (profile.get("hardware") or {}).get("gpu_count") or 1
    per_gpu = 14000 if gpus > 1 else 15000
    tokens = steps * int(training["per_device_train_batch_size"]) * int(training["max_seq_length"])
    return format_report(
        profile,
        {
            "steps": steps,
            "tokens_per_sec": round(tokens / max(steps, 1) / 2, 1),
            "peak_vram_mib": {index: per_gpu for index in range(gpus)},
            "fits": per_gpu < 16 * 1024,
        },
    )


def _weights_on_cpu(model) -> int:
    device_map = getattr(model, "hf_device_map", None) or {}
    return sum(1 for target in device_map.values() if str(target) in {"cpu", "disk", "meta"})


def _paged_optimizer(model, learning_rate: float):
    import torch

    optimizer_cls = torch.optim.AdamW
    try:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.PagedAdamW8bit
    except Exception:
        pass
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return optimizer_cls(parameters, lr=learning_rate)


def _dummy_steps(model, steps: int, length: int, batch_size: int, learning_rate: float) -> float:
    """Return tokens per second for a few full-length dummy batches."""
    import time

    import torch

    vocab = 128
    started = time.perf_counter()
    optimizer = _paged_optimizer(model, learning_rate)
    model.train()
    for _step in range(steps):
        device = next(model.parameters()).device
        batch = torch.randint(0, vocab, (1, length), device=device)
        outputs = model(input_ids=batch, labels=batch)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = max(time.perf_counter() - started, 1e-6)
    return (steps * batch_size * length) / elapsed


def unsloth_smoke(profile: dict, steps: int) -> dict:
    """Load the 4-bit model, take a few steps, and read peak memory from each GPU."""
    validate_home_profile(profile)
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Install a CUDA 12.8 PyTorch wheel and Unsloth before the smoke test. " + str(exc)
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("No CUDA GPU is visible. Refusing to load the 27B checkpoint on CPU.")
    training = profile["training"]
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=profile["base_model"],
        max_seq_length=int(training["max_seq_length"]),
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
        device_map="balanced",
    )
    spilled = _weights_on_cpu(model)
    if spilled:
        raise SystemExit(
            f"{spilled} modules landed on CPU or disk. bitsandbytes 4-bit loads must stay on the GPUs. "
            "Lower nothing onto system RAM; shorten the sequence or free the second GPU."
        )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(training["lora_r"]),
        lora_alpha=int(training.get("lora_alpha") or training["lora_r"]),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=training["gradient_checkpointing"],
        target_modules=list(training["target_modules"]),
    )
    length = int(training["max_seq_length"])
    tokens_per_sec = _dummy_steps(
        model,
        steps,
        length,
        int(training["per_device_train_batch_size"]),
        float(training.get("learning_rate") or 2e-4),
    )
    peaks = {
        index: round(torch.cuda.max_memory_allocated(index) / (1024 * 1024), 1)
        for index in range(torch.cuda.device_count())
    }
    return format_report(
        profile,
        {
            "steps": steps,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "peak_vram_mib": peaks,
            "fits": all(value < 16 * 1024 for value in peaks.values()),
        },
    )
