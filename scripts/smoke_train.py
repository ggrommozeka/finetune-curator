#!/usr/bin/env python3
"""Load the at-home 27B profile, take a few steps, and print peak VRAM and tokens/sec.

``--simulate`` checks the profile and prints a planned report. It does not load weights.
"""

from __future__ import annotations

import argparse
import json

from scripts.training.smoke import load_profile, simulate_smoke, unsloth_smoke


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the at-home 27B QLoRA profile.")
    parser.add_argument("--profile", default="configs/train/author-qlora-2x16gb.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--simulate", action="store_true", help="Do not load the checkpoint.")
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    report = simulate_smoke(profile, args.steps) if args.simulate else unsloth_smoke(profile, args.steps)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["fits"]:
        raise SystemExit("Peak VRAM on at least one GPU is at or above 16GB. Shorten max_seq_length.")


if __name__ == "__main__":
    main()
