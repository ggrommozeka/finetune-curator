#!/usr/bin/env python3
"""Export only sessions marked Keep in decisions.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_label_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Decisions file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # New format: {"labels": {id: keep|discard}, "notes": {...}}
    if isinstance(data, dict) and isinstance(data.get("labels"), dict):
        return {str(k): str(v).lower() for k, v in data["labels"].items()}
    # Legacy: {id: keep|discard}
    if isinstance(data, dict):
        return {str(k): str(v).lower() for k, v in data.items()}
    raise SystemExit("decisions.json must be an object")


def session_id(rec: dict) -> str:
    return str(rec.get("id") or rec.get("session_id") or "")


def main() -> None:
    p = argparse.ArgumentParser(description="Write JSONL with only Keep-approved sessions")
    p.add_argument("--input", required=True, help="Source JSONL (scored or filtered)")
    p.add_argument("--decisions", default="data/decisions.json")
    p.add_argument("--output", default="data/approved.jsonl")
    p.add_argument(
        "--include-undecided",
        action="store_true",
        help="Also include sessions not present in decisions.json",
    )
    args = p.parse_args()

    src = Path(args.input)
    labels = load_label_map(Path(args.decisions))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = discarded = skipped = 0
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = session_id(rec)
            decision = labels.get(sid)
            if decision == "keep":
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            elif decision == "discard":
                discarded += 1
            else:
                if args.include_undecided:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    skipped += 1

    print(f"Keep: {kept}  Discard: {discarded}  Undecided(skipped): {skipped}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
