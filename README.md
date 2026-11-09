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
source .venv/bin/activate
pip install -r requirements.txt
pip install "git+https://github.com/pearsonkyle/LogMiner.git"
```

## Quick start

### 1. Export from OpenCode

```bash
python -m scripts.export_opencode --output data/raw_scored.jsonl --min-score 0.0
```

### 2. Review in the viewer

```bash
streamlit run viewer/app.py -- --input data/raw_scored.jsonl
```

### 3. Export only approved

```bash
python -m scripts.export_approved --input data/raw_scored.jsonl --decisions data/decisions.json --output data/approved.jsonl
```

## License

MIT
