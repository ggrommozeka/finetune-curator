# finetune-curator

Pipeline: **LogMiner export → visual review (keep/discard) → export only approved** for Unsloth / SFT.

Works with OpenCode (and other LogMiner sources: Claude Code, Cline, Codex, Qwen).

## What you get

1. **Export** — thin wrapper around [LogMiner](https://github.com/pearsonkyle/LogMiner) for OpenCode history
2. **Viewer** — Streamlit UI with **Keep / Discard**, score filters, trajectory browser
3. **Export approved** — writes only kept sessions to a clean JSONL ready for Unsloth

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# LogMiner (install from git)
pip install "git+https://github.com/pearsonkyle/LogMiner.git"
```

## Quick start

### 1. Export from OpenCode

```bash
# Keep all scores for manual review
python -m scripts.export_opencode --output data/raw_scored.jsonl --min-score 0.0
```

OpenCode data is read from `~/.local/share/opencode` by default (override with `--input`).

### 2. Review in the viewer

```bash
streamlit run viewer/app.py -- --input data/raw_scored.jsonl
```

- Filter by score / label
- Open each trajectory (user / assistant / tool turns)
- Press **Keep** or **Discard**
- Decisions are saved to `data/decisions.json`

### 3. Export only approved

```bash
python -m scripts.export_approved \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --output data/approved.jsonl
```

Load into Unsloth / HuggingFace:

```python
from datasets import load_dataset
import json

ds = load_dataset("json", data_files="data/approved.jsonl", split="train")
msgs = [json.loads(m) for m in ds[0]["messages"]]
```

## Layout

```
finetune-curator/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── export_opencode.py   # LogMiner wrapper
│   └── export_approved.py   # Keep-only export
├── viewer/
│   └── app.py               # Streamlit keep/discard UI
└── data/                    # local only (gitignored)
```

## Notes

- LogMiner quality score is a **structural filter**, not a success label. Always spot-check Keep decisions.
- After OpenCode CLI upgrades (v1 → v2 schema), re-run export; if parse fails, check `opencode --version` and DB tables.
- Do not commit real session logs if they may contain secrets.

## License

MIT
