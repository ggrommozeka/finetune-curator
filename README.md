# finetune-curator

Pipeline: **LogMiner export → visual review (keep/discard) → export only approved** for Unsloth / SFT.

The same review pass can mine a second artifact: **harness rules** for the agent system prompt or `AGENTS.md` (mistakes, user corrections, loops, tool misuse).

Works with OpenCode (and other LogMiner sources: Claude Code, Cline, Codex, Qwen) and with merged GitLab merge requests.

## What you get

1. **Export** — thin wrapper around [LogMiner](https://github.com/pearsonkyle/LogMiner) for OpenCode history
2. **Viewer** — Streamlit UI with **Keep / Discard**, score filters, trajectory browser, and a **Правила** page for proposed harness rules
3. **Export approved** — writes only kept sessions to a clean JSONL ready for Unsloth
4. **Harness rules** — an OpenAI-compatible model proposes rules from discarded sessions, low scores, reviewer notes, and GitLab review comments; you approve them; export Markdown + JSON
5. **GitLab reviews** — merged MRs become SFT and DPO examples from team-lead comments and the `MRR fixes` commits that followed them

The dataset flow is unchanged. Rule files are a separate output.

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
- Optional note per session (why it failed, what the user corrected)
- Decisions are saved to `data/decisions.json`

### 3. Export only approved sessions

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

### 4. Propose harness rules

Rules are drafted from sessions you discarded, sessions under `--max-score` (default `0.6`, plus LogMiner labels `bad` / `maybe`), and any session with a reviewer note. Each rule is a short imperative instruction, with a category, rationale, source session ids, and a short evidence excerpt. Similar rules are merged.

Any OpenAI-compatible chat API works. The default endpoint is a local [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`:

```text
http://localhost:8080/v1
```

The API key is optional there. If you omit `--model`, the script uses the only id returned by `GET /v1/models`.

```bash
python -m scripts.propose_rules \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --output data/rule_proposals.json \
  --base-url http://localhost:8080/v1 \
  --context-budget 170000
```

Smoke-test the command without a server (canned reply, no network):

```bash
python -m scripts.propose_rules \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --output data/rule_proposals.json \
  --mock-response tests/fixtures/sample_rules_response.txt
```

### 5. Approve rules in the viewer

Open the viewer as in step 2 and switch the sidebar to **Правила**.

- Edit the rule text and category
- **Одобрить** / **Отклонить**
- Session buttons jump back to the source trajectory
- Decisions are saved to `data/rule_decisions.json` (proposals stay in `data/rule_proposals.json`)

```bash
streamlit run viewer/app.py -- \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --rules data/rule_proposals.json \
  --rule-decisions data/rule_decisions.json
```

### 6. Export approved rules

Markdown is grouped by category and ready to paste into a system prompt or `AGENTS.md`. JSON keeps ids, text, category, rationale, source session ids, and evidence. Only **approved** rules are written; edited text wins.

```bash
python -m scripts.export_rules \
  --proposals data/rule_proposals.json \
  --decisions data/rule_decisions.json \
  --markdown data/approved_rules.md \
  --json data/approved_rules.json
```

## Local llama.cpp

The defaults assume a server with about **200k** tokens of context. `--context-budget` (default **170000**) is the usable input budget, leaving room for the instruction prompt and the JSON answer (`--max-output-tokens`, default 4096). Typical OpenCode sessions are sent whole. A session that does not fit is split on message boundaries; short sessions are packed into one request. Lower the budget if `llama-server` was started with a smaller `-c` (for example `--context-budget 12000` on a 16k server).

Token counts are estimated as `characters / --chars-per-token` (default `3`, which over-counts English and stays inside the window). If the server reports a context overflow, lower `--chars-per-token`. Pass `--no-batch` to send every chunk in its own request.

Qwen3-family GGUF quants (for example an Unsloth Q5_K_XL around 27B) emit a `<think>` block unless thinking is disabled. By default this script appends `/no_think` and sends `chat_template_kwargs.enable_thinking=false`. Replies are still scanned and `<think>…</think>` is stripped before JSON parsing. Invalid JSON is retried (`--attempts`, default 3). Temperature defaults to `0.1`. Pass `--think` to allow thinking, or `--no-template-kwargs` if a server rejects the extra body field. The model id is never hardcoded.

```bash
llama-server \
  -m /path/to/Qwen3-27B-Q5_K_XL.gguf \
  -c 200000 \
  --jinja \
  --host 127.0.0.1 \
  --port 8080

python -m scripts.propose_rules \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --output data/rule_proposals.json \
  --base-url http://127.0.0.1:8080/v1 \
  --context-budget 170000
```

`--jinja` lets llama.cpp apply the GGUF chat template, which is what honors `/no_think` and `enable_thinking`. Check the loaded id with:

```bash
curl -s http://127.0.0.1:8080/v1/models
python -m scripts.propose_rules --list-models --base-url http://127.0.0.1:8080/v1
```

Pin it with `--model <id>` when the server exposes more than one.

## OpenCode provider config

If the model is already configured for OpenCode, pick it by `provider/model`. Project `opencode.json` / `opencode.jsonc` (including `.opencode/`) overrides `~/.config/opencode/opencode.json`. Both the current `providers.*.settings.baseURL` shape and the older `provider.*.options.baseURL` shape are read. `modelID` is what is sent to the API. Keys come from `apiKey` / `api_key` or the provider `env` list.

```bash
python -m scripts.propose_rules --list-opencode-models
python -m scripts.propose_rules --print-endpoint --from-opencode llama/qwen3-27b

python -m scripts.propose_rules \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --from-opencode llama/qwen3-27b
```

Override order, highest first: CLI flags (`--base-url`, `--api-key`, `--model`), then `FINETUNE_CURATOR_BASE_URL` / `FINETUNE_CURATOR_API_KEY` / `FINETUNE_CURATOR_MODEL`, then `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`, then the OpenCode entry, then the llama-server default base URL. A base URL with no path gets `/v1` appended.

```bash
export FINETUNE_CURATOR_BASE_URL=http://127.0.0.1:8080/v1
export FINETUNE_CURATOR_MODEL=qwen3-27b-q5
# FINETUNE_CURATOR_API_KEY is optional for llama-server
```

`--print-endpoint` shows the resolved model and base URL and never prints the key.

## GitLab merge-request reviews

Company review history can feed the same curator. This talks to a self-hosted GitLab: merged merge requests, review discussions (including inline diff notes), and follow-up commits whose message matches `MRR fixes` (case-insensitive; change it with `--fix-pattern`).

Each review comment is linked to the fix that addressed it:

1. same file and overlapping lines, comment timestamp before the commit (confidence 0.9)
2. same file, lines do not overlap (confidence 0.55)
3. otherwise the earliest later `MRR fixes` commit on the MR (confidence 0.3)

Nothing is committed. The token comes from `GITLAB_TOKEN` and is not printed. API responses can be cached under `data/gitlab_cache/` (gitignored). Pass `--anonymize` to strip author names and emails, and `--anonymize-urls` to replace `http(s)` URLs as well. `--ca-bundle` trusts a private CA; `--insecure` turns verification off.

```bash
export GITLAB_TOKEN=...   # personal access token, api scope. Do not commit it.
python -m scripts.export_gitlab_mrr \
  --base-url https://gitlab.company.internal \
  --ca-bundle /path/to/company-ca.pem \
  --group my-group \
  --since 2025-01-01 \
  --until 2026-01-01 \
  --reviewer team.lead \
  --anonymize \
  --anonymize-urls \
  --sft data/gitlab_sft.jsonl \
  --dpo data/gitlab_dpo.jsonl
```

`--project group/app` (repeatable) limits the export to specific projects. `--group` reads merged MRs from that group.

`data/gitlab_sft.jsonl` is the same messages JSONL as the OpenCode export, so the viewer keep/discard flow and `export_approved` work unchanged. Open a row to see the comment plus the before/after code. The sidebar filter **Только GitLab review** hides other sessions.

```bash
streamlit run viewer/app.py -- --input data/gitlab_sft.jsonl

python -m scripts.export_approved \
  --input data/gitlab_sft.jsonl \
  --decisions data/decisions.json \
  --output data/gitlab_approved.jsonl
```

The SFT user turn is the MR title, description, review comment, and the code before the fix. The assistant turn is the updated code. The DPO file uses Unsloth's `prompt` / `chosen` / `rejected` columns (`rejected` is the code before, `chosen` is the fix, `prompt` includes the comment). Rebuild DPO from an already reviewed SFT file without calling GitLab:

```bash
python -m scripts.export_gitlab_mrr \
  --dpo-from data/gitlab_approved.jsonl \
  --dpo data/gitlab_dpo.jsonl
```

Load the DPO file with `datasets.load_dataset("json", data_files="data/gitlab_dpo.jsonl", split="train")`.

Those review examples are also a rules source. Point `propose_rules` at the GitLab JSONL and add `gitlab_review` (it is not included in the default sources):

```bash
python -m scripts.propose_rules \
  --input data/gitlab_sft.jsonl \
  --sources gitlab_review \
  --output data/rule_proposals.json \
  --base-url http://127.0.0.1:8080/v1
```

Approve the proposed rules on the **Правила** page the same way as session-derived rules.

Smoke-test the exporter without a network or a token leak, using the recorded fixture:

```bash
GITLAB_TOKEN=example python -m scripts.export_gitlab_mrr \
  --base-url https://gitlab.example.com \
  --project group/app \
  --reviewer teamlead \
  --fixture tests/fixtures/gitlab_api.json \
  --no-cache \
  --sft /tmp/gitlab_sft.jsonl \
  --dpo /tmp/gitlab_dpo.jsonl
```

## Layout

```
finetune-curator/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── export_opencode.py    # LogMiner wrapper
│   ├── export_approved.py    # Keep-only JSONL export
│   ├── propose_rules.py      # LLM rule proposals
│   ├── export_rules.py       # Approved rules → Markdown + JSON
│   ├── export_gitlab_mrr.py  # GitLab MR review → SFT + DPO
│   ├── rules/                # selection, merge, endpoint, export
│   └── gitlab/               # GitLab client, linking, anonymize
├── viewer/
│   ├── app.py                # Streamlit keep/discard + rules review
│   ├── rules_page.py
│   └── gitlab_panel.py       # before/after review card
├── tests/
└── data/                     # local only (gitignored)
```

## Tests

```bash
python -m pytest
```

The suite covers merge, JSON parsing (thinking tags included), chunking and batching, decision files, both export formats, OpenCode config resolution, a mocked end-to-end `propose_rules` → `export_rules` run, and the GitLab exporter against a recorded API fixture. It does not call a live model or a live GitLab.

## Notes

- LogMiner quality score is a **structural filter**, not a success label. Always spot-check Keep decisions and approved rules.
- After OpenCode CLI upgrades (v1 → v2 schema), re-run export; if parse fails, check `opencode --version` and DB tables.
- Do not commit real session logs or exported rules if they may contain secrets. `data/*.jsonl`, `data/*.json`, and `data/*.md` are gitignored.
- Rule ids are a hash of the category and the normalized text, so approving the same merged wording again keeps the decision. Editing text in the viewer does not change the id.

## License

MIT
