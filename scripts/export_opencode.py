#!/usr/bin/env python3
"""Export OpenCode (or other LogMiner sources) into scored JSONL for review."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _which_logminer() -> list[str]:
    """Prefer `python -m logminer`, fall back to `logminer` CLI if present."""
    try:
        import logminer  # noqa: F401

        return [sys.executable, "-m", "logminer"]
    except ImportError:
        pass
    if shutil.which("logminer"):
        return ["logminer"]
    raise SystemExit(
        "LogMiner is not installed.\n"
        '  pip install "git+https://github.com/pearsonkyle/LogMiner.git"'
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run LogMiner on OpenCode history and write JSONL for the curator viewer."
    )
    p.add_argument(
        "--source",
        default="opencode",
        choices=["opencode", "claude", "cline", "codex", "qwen", "all"],
        help="LogMiner source (default: opencode)",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Optional path to logs (default: provider standard dir, e.g. ~/.local/share/opencode)",
    )
    p.add_argument(
        "--output",
        default="data/raw_scored.jsonl",
        help="Output JSONL path (default: data/raw_scored.jsonl)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Min score for filter stage. Use 0.0 to keep all for manual review (default: 0.0)",
    )
    p.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Passed to LogMiner evaluate (default: 2)",
    )
    p.add_argument(
        "--min-token-count",
        type=int,
        default=1000,
        help="Passed to LogMiner evaluate (default: 1000)",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Stop after evaluate (write *_scored.jsonl style output via run still filters; "
        "use min-score 0 to keep almost everything)",
    )
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = _which_logminer()
    # Full pipeline so we always get score/label/reasons for the viewer
    cmd += [
        "run",
        "--source",
        args.source,
        "--output",
        str(out),
        "--min-score",
        str(args.min_score),
    ]
    if args.input:
        cmd += ["--input", args.input]

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e

    # LogMiner also writes intermediates next to --output (e.g. *_scored.jsonl).
    # Prefer the scored file for the viewer if present — richer metadata.
    scored = out.with_name(out.stem + "_scored.jsonl")
    if not scored.exists():
        # Some versions use prefix pattern: training_scored.jsonl from training.jsonl
        parent = out.parent
        stem = out.stem
        candidates = list(parent.glob(f"*{stem}*scored*.jsonl")) + list(
            parent.glob("*_scored.jsonl")
        )
        if candidates:
            scored = candidates[0]

    print()
    print("Done.")
    print(f"  Filtered output : {out}")
    if scored.exists() and scored.resolve() != out.resolve():
        print(f"  Scored (review) : {scored}")
        print()
        print("For manual Keep/Discard review, point the viewer at the SCORED file:")
        print(f"  streamlit run viewer/app.py -- --input {scored}")
    else:
        print()
        print("Open viewer:")
        print(f"  streamlit run viewer/app.py -- --input {out}")


if __name__ == "__main__":
    main()
