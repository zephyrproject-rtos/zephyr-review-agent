# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The review loop against a scripted OpenRouter and a scripted GitHub, no network."""

from __future__ import annotations

import json

import httpx

from zephyr_reviewer.intel import SourceIntel
from zephyr_reviewer.mcp_server import build_server
from zephyr_reviewer.review import publish as pub
from zephyr_reviewer.review.agent import run_review
from zephyr_reviewer.review.config import ReviewConfig
from zephyr_reviewer.review.github_api import GitHub, PullRequest
from zephyr_reviewer.review.mcp_client import ToolBridge
from zephyr_reviewer.review.openrouter import OpenRouter
from zephyr_reviewer.review.prompts import load_profile, opening_message, system_prompt
from zephyr_reviewer.review.render import inline_comments, render_comment
from zephyr_reviewer.review.schema import parse_review

GOOD_REVIEW = {
    "summary": "Adds bus recovery to the Foo I2C driver and a Kconfig symbol for it.",
    "verdict": "request_changes",
    "confidence": "high",
    "findings": [
        {
            "file": "drivers/i2c/i2c_foo.c",
            "line": 17,
            "end_line": 19,
            "severity": "high",
            "category": "concurrency",
            "title": "Early return leaks the bus lock",
            "problem": "foo_transfer returns on error before k_sem_give, so the semaphore is never released.",
            "trigger": "Any failed transfer.",
            "impact": "Every later transfer blocks forever.",
            "recommendation": "Release the lock before returning.",
            "suggested_code": "\tif (ret < 0) {\n\t\tk_sem_give(&data->lock);\n\t\treturn ret;\n\t}",
            "confidence": "high",
            "evidence": "read_file drivers/i2c/i2c_foo.c:9-21",
        },
        {
            "file": "drivers/i2c/Kconfig",
            "line": 20,
            "severity": "medium",
            "category": "build_config",
            "title": "select of I2C_BUS_RECOVERY without a definition",
            "problem": "I2C_BUS_RECOVERY is not defined anywhere in the tree.",
            "trigger": "CONFIG_I2C_FOO_RECOVERY=y",
            "impact": "Kconfig warning; recovery never enabled.",
            "recommendation": "Define the symbol or drop the select.",
            "confidence": "medium",
        },
        {
            "file": "kernel/sem.c",
            "line": 1,
            "severity": "suggestion",
            "category": "maintainability",
            "title": "Unrelated file cited",
            "problem": "Should be demoted to summary-only.",
            "trigger": "",
            "impact": "",
            "recommendation": "",
            "confidence": "low",
        },
        {
            "file": "drivers/i2c/i2c_foo.c",
            "line": 1,
            "severity": "low",
            "category": "documentation",
            "title": "Context line cited",
            "problem": "Line 1 is not part of the diff.",
            "trigger": "",
            "impact": "",
            "recommendation": "",
            "confidence": "low",
        },
    ],
    "test_assessment": "The recovery scenario enables the option but does not exercise foo_recover().",
    "commit_message_assessment": "",
}


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _completion(message: dict, prompt_tokens: int = 1000) -> dict:
    return {
        "id": "gen-1",
        "model": "anthropic/claude-sonnet-4.5",
        "choices": [{"message": {"role": "assistant", **message}, "finish_reason": "tool_calls"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 50,
            "total_tokens": prompt_tokens + 50,
            "cost": 0.0123,
        },
    }


class ScriptedOpenRouter:
    """Replies in order; records every request payload."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        if not self.replies:
            return httpx.Response(500, json={"error": {"message": "script exhausted", "code": 500}})
        return httpx.Response(200, json=self.replies.pop(0))


def _llm(script: ScriptedOpenRouter, models=("anthropic/claude-sonnet-4.5", "openai/gpt-4.1")) -> OpenRouter:
    return OpenRouter(
        "key", list(models), client=httpx.Client(transport=httpx.MockTransport(script.handler)), max_attempts=2
    )


async def _run(mini_repo, script: ScriptedOpenRouter, **kw):
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(
        number=7,
        title="drivers: i2c: foo: add bus recovery",
        body="Adds recovery.",
        author="t",
        base_ref="main",
        base_sha=mini_repo["base"],
        head_ref="pr",
        head_sha=mini_repo["head"],
        head_repo="t/zephyr",
        draft=False,
    )
    system = system_prompt(load_profile("zephyr"), "English")
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", "zephyrproject-rtos/zephyr")
    changed = {f.path: f.new_side_lines for f in intel.files}
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_review(_llm(script), bridge, system, opening, changed, log=lambda m: None, **kw)
    return run, pr, intel


async def test_agent_investigates_then_submits(mini_repo):
    script = ScriptedOpenRouter(
        [
            _completion(
                {
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            "c1", "read_file", {"path": "drivers/i2c/i2c_foo.c", "start_line": 9, "end_line": 21}
                        ),
                        _tool_call("c2", "kconfig_symbol", {"name": "I2C_BUS_RECOVERY"}),
                    ],
                }
            ),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, pr, intel = await _run(mini_repo, script)
    assert run.review is not None and run.steps == 2 and not run.forced
    assert run.tool_call_count == 2
    assert run.usage.requests == 2 and run.usage.cost > 0.02

    first, second = script.requests
    assert first["model"] == "anthropic/claude-sonnet-4.5" and first["models"] == [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-4.1",
    ]
    assert {t["function"]["name"] for t in first["tools"]} >= {"read_file", "submit_review", "kconfig_symbol"}
    assert first["messages"][0]["role"] == "system" and "Zephyr Project" in first["messages"][0]["content"]
    assert "### Full diff" in first["messages"][1]["content"]
    # Tool results were fed back with their ids, in order.
    tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert "   17+| \tif (ret < 0) {" in tool_msgs[0]["content"]
    assert "No Kconfig definition of I2C_BUS_RECOVERY" in tool_msgs[1]["content"]

    review = run.review
    assert review.verdict == "request_changes" and review.highest_severity() == "high"
    lock, kconfig, unrelated, context = review.findings
    assert lock.inline and lock.line == 17 and lock.end_line == 19
    assert kconfig.inline and kconfig.line == 20
    assert not unrelated.inline and unrelated.line == 0 and "not changed by this pull request" in unrelated.notes[0]
    assert not context.inline and "not on the new side" in context.notes[0]

    body = render_comment(review, run, pr, repo="zephyrproject-rtos/zephyr", tag="zephyr-reviewer")
    # The posted comment carries no run statistics unless asked for.
    assert "tool call(s)" not in body and "$0.0" not in body and "anthropic/claude" not in body
    assert body.rstrip().endswith("It is advisory; verify before acting on it.</sub>")
    body = render_comment(review, run, pr, repo="zephyrproject-rtos/zephyr", tag="zephyr-reviewer", details=True)
    assert body.startswith("<!-- zephyr-reviewer -->")
    assert "🛑 **Request changes**" in body
    assert "blob/" + mini_repo["head"] + "/drivers/i2c/i2c_foo.c#L17" in body
    assert "```c\n\tif (ret < 0) {" in body
    assert "Evidence" not in body and "**Trigger:**" not in body and "**Problem:**" not in body
    assert "shown in the summary only" not in body  # internal notes stay in review.json
    assert "### Test gaps" in body and "2 tool call(s)" in body and "$0.025" in body

    inline = inline_comments(review)
    assert len(inline) == 2
    assert inline[0] == {
        "path": "drivers/i2c/i2c_foo.c",
        "line": 19,
        "side": "RIGHT",
        "start_line": 17,
        "start_side": "RIGHT",
        "body": inline[0]["body"],
    }
    assert "```suggestion" in inline[0]["body"]


async def test_agent_rejects_invalid_submission_then_accepts(mini_repo):
    bad = {"summary": "", "verdict": "maybe", "findings": "nope"}
    script = ScriptedOpenRouter(
        [
            _completion({"content": "", "tool_calls": [_tool_call("c1", "submit_review", bad)]}),
            _completion({"content": "", "tool_calls": [_tool_call("c2", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None and run.steps == 2
    rejection = [m for m in script.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "summary is required" in rejection and "verdict must be one of" in rejection
    assert "findings must be an array" in rejection


async def test_agent_forces_submission_when_budget_runs_out(mini_repo):
    script = ScriptedOpenRouter(
        [
            _completion({"content": "", "tool_calls": [_tool_call("c1", "pr_overview", {})]}),
            _completion({"content": "", "tool_calls": [_tool_call("c2", "commit_messages", {})]}),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script, max_steps=3)
    assert run.review is not None and run.forced
    last = script.requests[2]
    assert last["tool_choice"] == {"type": "function", "function": {"name": "submit_review"}}
    assert "budget is exhausted" in last["messages"][-1]["content"]


async def test_agent_nudges_prose_and_handles_unknown_tools_and_bad_json(mini_repo):
    script = ScriptedOpenRouter(
        [
            _completion({"content": "Let me think out loud instead of calling a tool."}),
            _completion(
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "x1", "type": "function", "function": {"name": "read_file", "arguments": "{not json"}},
                        _tool_call("x2", "no_such_tool", {}),
                    ],
                }
            ),
            _completion({"content": "", "tool_calls": [_tool_call("c3", "submit_review", GOOD_REVIEW)]}),
        ]
    )
    run, _, _ = await _run(mini_repo, script)
    assert run.review is not None
    msgs = script.requests[2]["messages"]
    assert any(m["role"] == "user" and "prose instead of a tool call" in m["content"] for m in msgs)
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert "not valid JSON" in tool_msgs[0]["content"] and "unknown tool" in tool_msgs[1]["content"]


async def test_agent_gives_up_after_repeated_rejections(mini_repo):
    bad = _completion({"content": "", "tool_calls": [_tool_call("c", "submit_review", {"summary": "x"})]})
    script = ScriptedOpenRouter([bad, bad, bad, bad])
    run, _, _ = await _run(mini_repo, script, max_rejections=2)
    assert run.review is None and run.errors and "verdict" in run.errors[0]


def test_openrouter_retries_then_fails_cleanly():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="busy", headers={"Retry-After": "0"})

    llm = OpenRouter("k", ["m"], client=httpx.Client(transport=httpx.MockTransport(handler)), max_attempts=2)
    import pytest

    from zephyr_reviewer.review.openrouter import OpenRouterError

    llm._sleep = lambda *a: None  # type: ignore[method-assign]
    with pytest.raises(OpenRouterError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2


def test_parse_review_keeps_only_problems_in_assessments():
    base = {"summary": "ok", "verdict": "approve", "confidence": "high", "findings": []}
    clear = {
        "test_assessment": "Tests are adequate for this change.",
        "commit_message_assessment": "No issues found.",
        "discussion_assessment": "All threads have been addressed.",
    }
    review, _ = parse_review({**base, **clear}, {})
    assert review is not None
    assert (
        review.test_assessment == "" and review.commit_message_assessment == "" and review.discussion_assessment == ""
    )
    kept = {
        "test_assessment": "The recovery scenario enables the option but does not exercise foo_recover().",
        "commit_message_assessment": "The second commit mixes the Kconfig change with the fix; split it.",
        "discussion_assessment": "@bob's request to release the lock on the error path is still unaddressed.",
    }
    review, _ = parse_review({**base, **kept}, {})
    assert review is not None
    assert review.test_assessment == kept["test_assessment"]
    assert review.commit_message_assessment == kept["commit_message_assessment"]
    assert review.discussion_assessment == kept["discussion_assessment"]
    # Short but naming a problem: kept. Long prose: kept even if it opens positively.
    review, _ = parse_review({**base, "test_assessment": "Adequate, but the error path is untested."}, {})
    assert review is not None and review.test_assessment
    review, _ = parse_review({**base, "test_assessment": "Looks good. " + "x" * 300}, {})
    assert review is not None and review.test_assessment


def test_parse_review_normalizes_vocabulary():
    review, errors = parse_review(
        {
            "summary": "ok",
            "verdict": "request changes",
            "confidence": "HIGH",
            "findings": [
                {
                    "file": "b/a.c",
                    "line": "3",
                    "severity": "Suggestion",
                    "category": "weird",
                    "title": "t",
                    "problem": "p",
                    "confidence": "?",
                }
            ],
        },
        {"a.c": {3}},
    )
    assert not errors and review is not None
    assert review.verdict == "request_changes" and review.confidence == "high"
    f = review.findings[0]
    assert (
        f.file == "a.c" and f.inline and f.severity == "suggestion" and f.category == "bug" and f.confidence == "medium"
    )


def test_publish_upserts_comment_and_posts_inline_review(mini_repo, tmp_path, monkeypatch):
    seen: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path.endswith("/issues/7/comments"):
            return httpx.Response(200, json=[{"id": 42, "body": "<!-- zephyr-reviewer -->\nold"}])
        if request.method == "GET" and request.url.path.endswith("/pulls/7/comments"):
            return httpx.Response(200, json=[])
        if request.method == "PATCH":
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/7#issuecomment-42"})
        if request.url.path.endswith("/reviews"):
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/7#pullrequestreview-1"})
        return httpx.Response(404, json={})

    gh = GitHub("tok", "o/r", client=httpx.Client(transport=httpx.MockTransport(handler)))
    review, _ = parse_review(GOOD_REVIEW, {"drivers/i2c/i2c_foo.c": {17, 18, 19}, "drivers/i2c/Kconfig": {20}})
    from zephyr_reviewer.review.agent import RunResult
    from zephyr_reviewer.review.openrouter import Usage

    run = RunResult(review, 3, Usage(), ["m"], False)
    pr = PullRequest(7, "t", "", "a", "main", "b" * 40, "pr", "h" * 40, "o/r", False)
    cfg = ReviewConfig(
        post_inline=True, comment_tag="zephyr-reviewer", fail_on_severity="high", output_dir=str(tmp_path / "out")
    )
    out_file = tmp_path / "outputs.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    outputs = pub.publish(cfg, gh, pr, run, lambda exclude: "<!-- zephyr-reviewer -->\nnew", log=lambda m: None)
    pub.write_outputs(outputs)
    pub.save_artifacts(cfg.output_dir, run, "body", pr, cfg)

    methods = [(m, p) for m, p, _ in seen]
    assert ("PATCH", "/repos/o/r/issues/comments/42") in methods  # updated in place, not duplicated
    review_req = next(b for m, p, b in seen if p.endswith("/reviews"))
    assert review_req["commit_id"] == "h" * 40 and len(review_req["comments"]) == 2
    assert outputs["comment_url"].endswith("issuecomment-42") and outputs["highest_severity"] == "high"
    assert "verdict=request_changes" in out_file.read_text()
    assert pub.gate_fails(run, cfg) is not None
    assert (tmp_path / "out" / "review.json").exists() and (tmp_path / "out" / "transcript.jsonl").exists()
    saved = json.loads((tmp_path / "out" / "review.json").read_text())
    assert saved["review"]["findings"][0]["title"] == "Early return leaks the bus lock"
