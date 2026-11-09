"""Dedup, JSON recovery, and the mockable LLM retry loop."""

from scripts.rules.llm import build_chat_kwargs, request_rules, thinking_controls
from scripts.proposals.merge import make_rule_id, merge_rules, slug_category
from scripts.llm.parsing import extract_json, strip_thinking


def test_strip_thinking_variants():
    assert strip_thinking("<think>secret</think>\n{\"rules\":[]}") == '{"rules":[]}'
    assert strip_thinking("preamble</think>\n{\"ok\": true}") == '{"ok": true}'
    assert "secret" not in strip_thinking("<think>secret and {not json}")
    assert strip_thinking("no think block") == "no think block"


def test_extract_json_from_fenced_thinking_reply():
    raw = """<think>planning</think>
```json
{"rules":[{"text":"Be brief.","category":"style",}],}
```
"""
    payload = extract_json(raw)
    assert payload["rules"][0]["text"] == "Be brief."


def test_request_rules_strips_thinking_retries_and_disables_think():
    calls = {"n": 0}

    def complete(messages):
        calls["n"] += 1
        assert messages[1]["content"].endswith("/no_think")
        if calls["n"] == 1:
            return "<think>hmm</think>\nnot json"
        assert messages[-1]["content"].endswith("/no_think")
        return (
            '```json\n{"rules":[{"text":"Be brief with the user.",'
            '"category":"style","evidence":"too long"}]}\n```'
        )

    rules = request_rules(complete, "SESSION s9\nscore: 0.1\n", attempts=3)
    assert calls["n"] == 2
    assert rules[0]["category"] == "communication"
    assert rules[0]["session_ids"] == ["s9"]
    assert rules[0]["evidence"] == "too long"


def test_request_rules_can_leave_thinking_enabled():
    def complete(messages):
        assert "/no_think" not in messages[1]["content"]
        return '{"rules":[]}'

    assert request_rules(complete, "SESSION s\n", disable_thinking=False, attempts=1) == []


def test_thinking_controls_and_chat_kwargs():
    suffix, extra = thinking_controls(True)
    assert suffix.strip() == "/no_think"
    assert extra == {"chat_template_kwargs": {"enable_thinking": False}}
    assert thinking_controls(False) == ("", {})
    kwargs = build_chat_kwargs("m", [], temperature=0.1, max_tokens=32, extra_body=extra)
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "extra_body" not in build_chat_kwargs("m", [], temperature=0.1, max_tokens=32)


def test_unknown_session_ids_are_dropped():
    def complete(_messages):
        return (
            '{"rules":[{"text":"Check the error before retrying a failed command.",'
            '"category":"looping","session_ids":["nope","s1"],"evidence":"e"}]}'
        )

    rules = request_rules(complete, "SESSION s1\n[1 user]\nfix it\n", disable_thinking=False)
    assert rules[0]["session_ids"] == ["s1"]


def test_merge_unions_sessions_and_keeps_stable_id():
    rules = [
        {
            "text": "Не повторяй упавшую команду без изменений.",
            "category": "looping",
            "session_ids": ["a"],
            "rationale": "цикл",
            "evidence": "e1",
        },
        {
            "text": "Не повторяй упавшую команду без изменений!",
            "category": "циклы",
            "session_ids": ["b", "a"],
            "rationale": "снова",
            "evidence": "e2",
        },
    ]
    merged = merge_rules(rules)
    assert len(merged) == 1
    assert merged[0]["session_ids"] == ["a", "b"]
    assert "e1" in merged[0]["evidence"] and "e2" in merged[0]["evidence"]
    assert merged[0]["id"] == make_rule_id(merged[0]["category"], merged[0]["text"])
    assert merge_rules(rules)[0]["id"] == merged[0]["id"]


def test_merge_keeps_distinct_rules_and_prefers_concise_wording():
    short = "When a command fails, read the error and change approach instead of retrying it."
    long = short + " " + ("Please remember this in every repository you touch. " * 8)
    distinct = "Run the test suite after changing production code."
    merged = merge_rules(
        [
            {"text": long, "category": "looping", "session_ids": ["s1"], "rationale": "r", "evidence": "e"},
            {"text": short, "category": "loop", "session_ids": ["s2"], "rationale": "r2", "evidence": "e2"},
            {"text": distinct, "category": "verification", "session_ids": ["s3"], "rationale": "", "evidence": ""},
            {"text": "   ", "category": "other", "session_ids": ["s4"], "rationale": "", "evidence": ""},
        ]
    )
    texts = [rule["text"] for rule in merged]
    assert short in texts
    assert long not in texts
    assert distinct in texts
    looping = next(rule for rule in merged if rule["category"] == "looping")
    assert looping["session_ids"] == ["s1", "s2"]


def test_merge_cross_category_near_duplicate_prefers_specific_category():
    text = "Do not delete files the user did not mention in the request."
    merged = merge_rules(
        [
            {"text": text, "category": "other", "session_ids": ["a"], "rationale": "", "evidence": "e1"},
            {"text": text, "category": "safety", "session_ids": ["b"], "rationale": "", "evidence": "e2"},
        ]
    )
    assert len(merged) == 1
    assert merged[0]["category"] == "safety"
    assert merged[0]["session_ids"] == ["a", "b"]
    assert merged[0]["id"] == make_rule_id("safety", text)


def test_slug_category_synonyms():
    assert slug_category("Tool Use") == "tool-use"
    assert slug_category("безопасность") == "safety"
    assert slug_category("") == "other"
