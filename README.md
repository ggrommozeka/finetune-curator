# finetune-curator

Pipeline: **LogMiner export → visual review (keep/discard) → export only approved** for Unsloth / SFT.

The same review pass can mine a second artifact: **harness rules** for the agent system prompt or `AGENTS.md` (mistakes, user corrections, loops, tool misuse).

Works with OpenCode (and other LogMiner sources: Claude Code, Cline, Codex, Qwen) and with merged GitLab merge requests.

## What you get

1. **Export** — thin wrapper around [LogMiner](https://github.com/pearsonkyle/LogMiner) for OpenCode history
2. **One Streamlit app** — `streamlit run viewer/app.py`. Shared navigation and paths: OpenCode sessions, GitLab MRs, proposals (with clusters), export, and monitoring.
3. **Export approved** — writes only kept sessions to a clean JSONL ready for Unsloth
4. **Proposals** — one shared list (`kind`: rule or skill, text, source session ids, status). Per-session generation and GitLab review comments write that same file, and the same viewer page approves them.
5. **GitLab reviews** — merged MRs become SFT and DPO examples, and linked review comments can be proposed as rules through the shared proposal file.

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

Company review history stays on disk. Phase 1 talks to GitLab. Phase 2 never does.

**Phase 1 — sync.** Pull merged MRs (description, discussions with inline positions, commits, diffs, and file text at the commit and its parent when GitLab returns it) into `data/gitlab/` (SQLite plus raw JSON, gitignored). Re-running skips MRs already stored. `--refresh project!iid` refetches one MR. The token is `GITLAB_TOKEN` and is not printed or written into the store.

```bash
export GITLAB_TOKEN=...   # api scope. Do not commit it.
python -m scripts.export_gitlab_mrr sync \
  --base-url https://gitlab.company.internal \
  --ca-bundle /path/to/company-ca.pem \
  --group my-group \
  --since 2025-01-01 \
  --until 2026-01-01

# later, only new MRs are fetched
python -m scripts.export_gitlab_mrr sync \
  --base-url https://gitlab.company.internal \
  --group my-group

# re-fetch one MR
python -m scripts.export_gitlab_mrr sync \
  --base-url https://gitlab.company.internal \
  --refresh group/app!42
```

`--project group/app` (repeatable) limits the sync to specific projects. `--insecure` disables TLS verification.

Comments from every reviewer are stored, with the author's role on the MR (author, reviewer, assignee, or commenter). `--reviewer` is an optional exact filter, not the default.

Each comment keeps its anchor: diff SHAs (`base` / `start` / `head`), file path, old and new line range, and the snippet those lines pointed at. A comment fix is any later commit that touches those lines. The commit title is not used. Threads remember whether they were resolved, by whom, and the replies. Status is `linked` (with a confidence), `addressed_elsewhere`, `not_addressed`, or `unclear`.

There is one team lead. `gitlab.json` names that person once and lists their aliases (`team_lead.aliases`: usernames, names, emails). `--team-lead` adds another alias of the same person. Every commit by that identity becomes a before/after hunk pair. Review comments are stored with the reviewer who wrote them. Processing keeps every reviewer in the training examples; `--reviewer` and the viewer field only filter what you look at. Titles matching `mrr_pattern` (default `\bmrr\b`, so `MRR`, `MRR fixes`, and `MRR: ...`) are flagged as the primary polish, not as the comment fix. Other commits are classified per MR as `author`, `author_fix` (author commit after a review comment), `team_lead_mrr`, `team_lead_other`, or `other`.

**Phase 2 — process offline.** Pick MRs in the viewer (**GitLab** page of `streamlit run viewer/app.py`: filter by project, author, reviewer, date, MRR titles, comment count) or on the CLI. The MR page is one timeline of commits and comments. Team-lead commits are highlighted, MRR titles are marked, and each hunk has a before/after view. Comment traces still show the anchored lines and the later commit that changed them. A manual link is stored separately and overrides automatic linking. Copy `gitlab.example.json` to `gitlab.json` for that one team lead and the MRR regex. A linked comment is the earliest later commit on the same lines (confidence 0.8, or 0.9 when the thread is resolved). The same file with no line overlap is `addressed_elsewhere` (confidence 0.55).

Examples and keep/discard edits are stored in separate tables from the raw MR, so you can process again without fetching. `--anonymize` strips author names and emails from examples only; `--anonymize-urls` also replaces URLs. The raw store is left intact.

```bash
python -m scripts.export_gitlab_mrr list --fixes-only

python -m scripts.export_gitlab_mrr process \
  --mr group/app!42 --mr group/app!43 \
  --team-lead team.lead \
  --anonymize --anonymize-urls

python -m scripts.export_gitlab_mrr process --all
```

In the viewer, open an MR (description, comments, diffs), keep or discard each example, and edit the training text. Decisions survive a re-process of the same MR.

Export writes only `keep` rows unless you pass `--status`. SFT uses the Unsloth `messages` JSONL. DPO uses `prompt` / `chosen` / `rejected` (rejected is the code before the fix).

```bash
python -m scripts.export_gitlab_mrr export \
  --status keep \
  --sft data/gitlab_sft.jsonl \
  --dpo data/gitlab_dpo.jsonl
```

Optional rule proposal uses the same LLM path as session rules and does not call GitLab. `--mock-response` skips the network entirely:

```bash
python -m scripts.export_gitlab_mrr process \
  --mr group/app!42 \
  --rules \
  --mock-response tests/fixtures/sample_rules_response.txt
```

Approve those rules on the **Правила** page. `gitlab_review` is not part of the default session sources.

Smoke-test both phases with the recorded fixture:

```bash
GITLAB_TOKEN=example python -m scripts.export_gitlab_mrr sync \
  --base-url https://gitlab.example.com \
  --project group/app \
  --fixture tests/fixtures/gitlab_api.json \
  --store /tmp/gitlab-store

python -m scripts.export_gitlab_mrr process \
  --store /tmp/gitlab-store \
  --mr group/app!7 \
  --reviewer teamlead \
  --anonymize
```

## Pattern mining

This is not a replay or A/B harness. It looks across many OpenCode sessions and, when the GitLab store exists, both comment-then-fix traces and team-lead polish hunks. It recommends either a few general system-prompt rules or a task-specific skill.

Histories from several machines are imported into `data/patterns/` (gitignored). The same session in two exports is stored once and tagged with every machine. Re-import only adds sessions that were not seen for that source. OpenCode sqlite files are read as `opencode-v2` (session/message/part JSON), `opencode-v1` (role/content columns), or a session blob. Anything else is reported with the table list instead of being guessed.

```bash
python -m scripts.mine_patterns import --source laptop-a --input data/laptop-a.jsonl
python -m scripts.mine_patterns import --source laptop-b --db ~/.local/share/opencode/opencode.db
python -m scripts.mine_patterns import --source gitlab --gitlab-store data/gitlab

python -m scripts.mine_patterns annotate --base-url http://127.0.0.1:8080/v1
python -m scripts.mine_patterns cluster --min-size 3
python -m scripts.mine_patterns recommend --base-url http://127.0.0.1:8080/v1
```

Annotation is one JSON object per session (task type, project, tools, outcome, where it failed, how the user corrected it, what worked). It is cached, so a second run skips finished sessions. Clustering uses an OpenAI-compatible embeddings endpoint when `--embeddings-url` and `--embeddings-model` are set, and TF-IDF cosine otherwise. Clusters smaller than `--min-size` (default 3) are noise and are not recommended.

A pattern becomes a **rule** only when it spans at least three task types. Otherwise it becomes a **skill**: `.opencode/skills/<id>/SKILL.md` is the portable layout (YAML `name` and `description`, then steps and examples). Conflicts with existing rule text and with the same error type on successful sessions are stored on the proposal.

`recommend` writes into `data/rule_proposals.json`, the same file per-session `propose_rules` uses (`mode: per-session` drafts are replaced). Approve, edit, or reject them on the **Правила** page of `streamlit run viewer/app.py`. That page also lists clusters (frequency, failure rate, machines, example sessions) and opens the OpenCode session or the GitLab comment. The **Мониторинг** page shows before/after metrics and rollback. `--min-size 1` keeps single-session clusters for debugging.

Export of approved rules and skills is `export_rules` (Markdown, JSON, and `--skills-dir`). `mine_patterns export` calls that same writer, then records the version used by `report` and `rollback`.

```bash
python -m scripts.mine_patterns export \
  --rules-md data/patterns_rules.md \
  --rules-json data/patterns_rules.json \
  --skills-dir skills \
  --version v1

python -m scripts.mine_patterns report --decisions data/decisions.json --error-type loop
python -m scripts.mine_patterns rollback --version v1
```

Export records the file versions and the adoption time. The report compares sessions dated before and after that time: mean LogMiner score, discard rate, how often the cluster's error type comes back, and how often the user had to correct the agent. Rollback copies that version's files back and retires later versions. There is no replay loop.

`annotate` and `recommend` accept `--mock-response` for a canned JSON reply. The default live endpoint is the same llama-server client as rule proposal (`http://localhost:8080/v1`, thinking off, low temperature).

## One fine-tuning dataset

`build_dataset` reads the OpenCode JSONL and the local GitLab store and writes one Unsloth chat dataset. Three tasks share that file:

- **Agentic.** Approved sessions (`keep`) and unlabeled sessions at or above `--min-score` become multi-turn tool-calling conversations. A later user message is treated as a correction: the training trajectory drops the rejected step, and a DPO pair stores that step as `rejected` and the following assistant step as `chosen`.
- **Style.** Team-lead commits (MRR titles and the lead's other commits) and comment-then-fix pairs become "rewrite this code following project conventions" samples. The code before the edit is the DPO `rejected` completion; the code after is `chosen`.
- **Review.** A diff hunk is the input and the review comment, from every reviewer, is the output, with a suggested fix when the comment was linked to one. Hunks nobody commented on are negatives whose target is `No issues.`

Agentic and style target the Qwen3-family 27B served with llama.cpp (about 200k tokens of context at inference). Their export defaults to the Qwen3 chat template and OpenAI-style tool calls (`tool_calls` on the assistant turn, `tool` role for the result). `<think>` blocks are stripped by default so the file matches a server started with thinking off. `--thinking keep` leaves the blocks in place; use that only if inference also leaves thinking on. Training length is `--max-tokens` (default 32000). A session longer than that is cut on turn boundaries, and each piece repeats the system prompt and the first user message.

`--agentic`, `--style`, and `--review` are mixing weights for `--task all`. Held-out rows are chosen by session id or MR id, so one session or MR never appears in both `train.jsonl` and `eval.jsonl`. `--anonymize` strips emails and URLs. `manifest.json` counts rows by task, source, and machine and records `chat_template`, `thinking`, and `target_model`.

Each task is also its own dataset. `--task agentic` and `--task style` stay on the 27B defaults above. Review is the set for a separate ~9B review model: `--task review` writes only review SFT and DPO, plus the held-out eval, and defaults to 16384 tokens. Diffs keep `--review-context-lines` (default 20) on each side of the comment instead of the whole hunk. `--chat-template` can be `chatml`, `llama3`, or `gemma` for that smaller model. The messages stay in the OpenAI tool-calling schema; the template name is what Unsloth should apply.

```bash
python -m scripts.build_dataset \
  --input data/raw_scored.jsonl \
  --decisions data/decisions.json \
  --gitlab-store data/gitlab \
  --patterns-store data/patterns \
  --output-dir data/finetune \
  --agentic 0.5 --style 0.25 --review 0.25 \
  --max-tokens 32000 \
  --eval-ratio 0.1

python -m scripts.eval_dataset --eval data/finetune/eval.jsonl --mock-response "No issues."

python -m scripts.build_dataset --task agentic --output-dir data/finetune/agentic \
  --chat-template qwen3 --thinking strip --max-tokens 32000
python -m scripts.build_dataset --task style --output-dir data/finetune/style \
  --chat-template qwen3 --thinking strip --max-tokens 32000
python -m scripts.build_dataset --task review --output-dir data/finetune/review \
  --chat-template qwen3 --max-tokens 16384 --review-context-lines 20
```

The same controls are on the **Экспорт** page of `streamlit run viewer/app.py`.

### bf16 LoRA for Qwen3.5

Unsloth's [Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) says not to train Qwen3.5 with QLoRA (4-bit), dense or MoE, because the quantization gap is larger than usual. Use bf16 LoRA instead. Their published training memory is about 22GB for the 9B and about 56GB for the 27B. There is no separate Qwen3.8 training page that recommends 4-bit, so the 27B author model uses this same bf16 LoRA setup. The guide's checkpoint id is `Qwen/Qwen3.5-27B`.

The sequence length below is the training budget (32768 for the author, 16384 for the reviewer), not the long context used later in llama.cpp. Set `model_name` and `chat_template` to the pair recorded in `manifest.json` (`base_model` and `chat_template`). Reviewer candidates are `Qwen/Qwen3.5-9B` and `deepreinforce-ai/Ornith-1.0-9B` (MIT; the 9B dense model is a Qwen3.5 hybrid, and the Ornith family also includes Gemma 4 post-trains at larger sizes). Ornith's GGUF uses the Qwen3.5 chat template; pass `--chat-template gemma` only if you actually train a Gemma-based checkpoint.

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-27B",
    max_seq_length=32768,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
dataset = load_dataset("json", data_files={"train": "data/finetune/agentic/train.jsonl"})

def formatting(batch):
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in batch["messages"]
    ]
    return {"text": texts}

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"].map(formatting, batched=True),
    args=SFTConfig(
        dataset_text_field="text",
        max_length=32768,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        output_dir="outputs/qwen3.5-27b-lora",
    ),
)
trainer.train()
```

For the 9B reviewer, use the same bf16 flags with `model_name="Qwen/Qwen3.5-9B"` or `model_name="deepreinforce-ai/Ornith-1.0-9B"` and `max_length=16384`. Do not set `load_in_4bit=True`.

DPO uses `train_dpo.jsonl` (`prompt`, `chosen`, and `rejected` are message lists) with TRL's `DPOTrainer` and the same template. If the export was built with `--thinking keep`, do not disable thinking at serve time.

Two ways to serve the result with llama.cpp:

Merged GGUF, produced by Unsloth in the same quant family as the Q5_K_XL already in use:

```python
model.save_pretrained_gguf("outputs/qwen3.5-27b-gguf", tokenizer, quantization_method="q5_k_m")
```

```bash
llama-server -m outputs/qwen3.5-27b-gguf/model.gguf --jinja -c 200000 --host 127.0.0.1 --port 8080
```

Or keep a base GGUF and attach the adapter. Save the LoRA, convert it with llama.cpp's `convert_lora_to_gguf.py`, and load it beside the base model:

```python
model.save_pretrained("outputs/qwen3.5-27b-lora")
tokenizer.save_pretrained("outputs/qwen3.5-27b-lora")
```

```bash
python convert_lora_to_gguf.py outputs/qwen3.5-27b-lora --outfile adapter.gguf
llama-server -m qwen3.5-27b-q5_k_xl.gguf --lora adapter.gguf --jinja -c 200000 \
  --host 127.0.0.1 --port 8080
```

`--jinja` applies the Qwen3 chat template stored in the GGUF, including tool calls. A dataset built with `--thinking strip` should be served with thinking disabled (`chat_template_kwargs.enable_thinking` false, or the server's equivalent). A dataset built with `--thinking keep` should leave thinking on.

Score one server with `python -m scripts.eval_dataset --eval data/finetune/eval.jsonl`. To compare several base or fine-tuned checkpoints on the same held-out MR rows, pass a JSON file. Each entry is a llama-server (or any OpenAI-compatible server) or a canned reply for a dry run:

```json
[
  {"name": "qwen3.5-9b", "base_url": "http://127.0.0.1:8080/v1", "model": "Qwen3.5-9B"},
  {"name": "ornith-1.0-9b", "base_url": "http://127.0.0.1:8081/v1", "model": "Ornith-1.0-9B"},
  {"name": "review-lora", "base_url": "http://127.0.0.1:8082/v1", "model": "review-lora"}
]
```

```bash
python -m scripts.eval_dataset \
  --eval data/finetune/review/eval.jsonl \
  --models data/finetune/review/models.json
```

Model A reports the user-correction rate (completions closer to the uncorrected step than to the corrected one) and the review-comment rate on the code it wrote. Model B, for every listed model, reports comment match precision/recall and the false-positive rate on hunks that passed review with no comments. Add `--judge` to let a server decide whether a predicted review matches the real comment.

Approved rules and skills stay the harness next to Model A: export them with `export_rules` and put the rules in the system prompt or `.opencode/skills`. The review packer also inserts the approved rule text as a short conventions section when `data/rule_proposals.json` is present.

Stage 2 for Model B is a later change: agentic review with read-only tools, distilled from 27B review trajectories that match real comments and are approved in the viewer. This export is stage 1 only, with the context already assembled.

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
│   ├── export_gitlab_mrr.py  # GitLab sync / process / export
│   ├── mine_patterns.py      # import, annotate, cluster, export, rollback
│   ├── rules/                # selection, merge, endpoint, export
│   ├── gitlab/               # client, store, sync, linking, anonymize
│   └── patterns/             # multi-machine import, clusters, skills
├── viewer/
│   ├── app.py                # one Streamlit app: sessions, GitLab, proposals, export
│   ├── rules_page.py
│   ├── gitlab_page.py        # stored MRs, offline process, example review
│   ├── gitlab_panel.py       # before/after review card
│   └── patterns_page.py      # cluster review: rule vs skill
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
