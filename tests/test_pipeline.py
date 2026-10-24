# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The specialist pipeline: planning, accounting rules, and a full scripted run."""

from __future__ import annotations

import json

import httpx

from tests.test_agent import ScriptedOpenRouter, _completion, _llm
from zephyr_reviewer.intel import SourceIntel
from zephyr_reviewer.mcp_server import build_server
from zephyr_reviewer.review.github_api import PullRequest
from zephyr_reviewer.review.mcp_client import ToolBridge
from zephyr_reviewer.review.pipeline import (
    hunk_text,
    make_verification_parser,
    plan_specialists,
    run_pipeline,
)
from zephyr_reviewer.review.prompts import load_profile, opening_message, system_prompt
from zephyr_reviewer.review.schema import parse_review


def test_planner_picks_passes_from_the_change():
    assert plan_specialists(["doc/x.rst"], "") == ["intent", "logic"]
    assert plan_specialists(["kernel/sched.c"], "z_sched_lock();") == [
        "intent",
        "logic",
        "error_handling",
        "concurrency",
    ]
    assert plan_specialists(["arch/arm/core/fault.S"], "") == ["intent", "logic", "error_handling", "concurrency"]
    assert plan_specialists(["lib/os/clock.c"], "") == ["intent", "logic", "error_handling", "resources", "security"]
    assert plan_specialists(["drivers/i2c/i2c_foo.c", "drivers/i2c/Kconfig"], "k_sem_take(&l, K_FOREVER);") == [
        "intent",
        "logic",
        "error_handling",
        "concurrency",
        "resources",
        "hardware",
        "build_config",
    ]
    assert "security" in plan_specialists(["subsys/net/ip/tcp.c"], "memcpy(a, b, n);")
    assert "build_config" in plan_specialists(["boards/vnd/b/b.dts"], "")
    assert "hardware" not in plan_specialists(["boards/vnd/b/b.dts"], "")  # no C code
    assert plan_specialists(["dts/bindings/i2c/x.yaml"], "") == ["intent", "logic", "build_config"]
    assert "error_handling" in plan_specialists(["subsys/net/ip/tcp.c"], "")


def _finding(title="t", **kw):
    f = {
        "file": "a.c",
        "line": 1,
        "severity": "medium",
        "category": "bug",
        "title": title,
        "problem": "p",
        "trigger": "",
        "impact": "",
        "recommendation": "",
        "confidence": "medium",
    }
    f.update(kw)
    return f


def test_verification_parser_enforces_full_accounting():
    parse = make_verification_parser({"C1", "C2", "C3"}, {"D1", "D2"})
    ok, errs = parse(
        {
            "findings": [{**_finding("merged"), "source_ids": ["C1", "C2"]}],
            "dismissed": [{"source_ids": ["D1"], "reason": "guarded"}],
            "hard_cases": [{"source_ids": ["C3", "D2"], "question": "is the lock held?", "severity_hint": "high"}],
        }
    )
    assert errs == [] and ok is not None
    assert [h.id for h in ok.hard_cases] == ["H1"] and ok.hard_cases[0].source_ids == ["C3", "D2"]

    # Missing ids are rejected, naming them.
    _, errs = parse({"findings": [{**_finding(), "source_ids": ["C1"]}], "dismissed": [], "hard_cases": []})
    assert errs == ["every id must be accounted for; missing: C2, C3, D1, D2"]
    # A concern may not be dismissed directly; a dismissal may not be confirmed as a finding.
    _, errs = parse(
        {
            "findings": [{**_finding(), "source_ids": ["D1"]}],
            "dismissed": [{"source_ids": ["C1"], "reason": "nah"}],
            "hard_cases": [{"source_ids": ["C2", "C3", "D2"], "question": "", "severity_hint": "low"}],
        }
    )
    assert "findings[0]: D1 is not allowed in findings (concerns you doubt go to hard_cases)" in errs
    assert "dismissed[0]: C1 is not allowed in dismissed (concerns you doubt go to hard_cases)" in errs
    # Double placement and unknown ids are rejected.
    _, errs = parse(
        {
            "findings": [{**_finding(), "source_ids": ["C1"]}, {**_finding(), "source_ids": ["C1", "C9"]}],
            "dismissed": [{"source_ids": ["D1", "D2"], "reason": "r"}],
            "hard_cases": [{"source_ids": ["C2", "C3"], "question": "", "severity_hint": "medium"}],
        }
    )
    assert "C1 appears in both findings[0] and findings[1]" in errs and "findings[1]: unknown id 'C9'" in errs


def test_speculative_findings_are_capped_at_medium():
    review, errs = parse_review(
        {
            "summary": "s",
            "verdict": "comment",
            "confidence": "high",
            "findings": [_finding("spec", severity="critical", speculative=True), _finding("solid", severity="high")],
        },
        {"a.c": {1}},
    )
    assert not errs and review is not None
    spec, solid = review.findings
    assert spec.speculative and spec.severity == "medium" and "capped at medium (was critical)" in spec.notes[0]
    assert not solid.speculative and solid.severity == "high"


# --- full pipeline against a scripted model ------------------------------------------------------
def _tc(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class StageRouter:
    """Answers each request by the stage named in its system prompt or by the terminal tool offered."""

    def __init__(self, verification_first_attempt_bad: bool = True):
        self.requests: list[dict] = []
        self.bad_once = verification_first_attempt_bad
        self.verification_attempts = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        system = body["messages"][0]["content"]
        tools = {t["function"]["name"] for t in body["tools"]}
        if "submit_concerns" in tools:
            stage = system.split("# Stage: ", 1)[1].split("\n", 1)[0].strip()
            if stage.startswith("intent"):
                out = {
                    "concerns": [
                        _finding(
                            "Commit message does not mention the Kconfig symbol",
                            file="drivers/i2c/Kconfig",
                            line=20,
                            severity="low",
                            category="commit_message",
                            evidence="commit_messages",
                        )
                    ],
                    "dismissed": [],
                }
            elif stage.startswith("concurrency"):
                out = {
                    "concerns": [
                        _finding(
                            "Early return leaks the bus lock",
                            file="drivers/i2c/i2c_foo.c",
                            line=17,
                            end_line=19,
                            severity="high",
                            category="concurrency",
                            trigger="failed transfer",
                            impact="bus wedged",
                            recommendation="give the sem",
                            evidence="read_file drivers/i2c/i2c_foo.c:9-21",
                            confidence="high",
                        )
                    ],
                    "dismissed": [
                        {
                            "title": "k_sem_init count",
                            "file": "drivers/i2c/i2c_foo.c",
                            "line": 32,
                            "reason": "binary semaphore is intended",
                            "evidence": "read_file :28-34",
                        }
                    ],
                }
            elif stage.startswith("Kconfig"):
                out = {
                    "concerns": [
                        _finding(
                            "select of undefined I2C_BUS_RECOVERY",
                            file="drivers/i2c/Kconfig",
                            line=20,
                            severity="medium",
                            category="build_config",
                            evidence="kconfig_symbol",
                        )
                    ],
                    "dismissed": [
                        {
                            "title": "I2C_FOO default y",
                            "file": "drivers/i2c/Kconfig",
                            "line": 12,
                            "reason": "gated by DT_HAS",
                            "evidence": "kconfig_symbol I2C_FOO",
                        }
                    ],
                }
            else:
                out = {"concerns": [], "dismissed": []}
            calls = [_tc("c", "submit_concerns", out)]
        elif "submit_verification" in tools:
            self.verification_attempts += 1
            if self.bad_once and self.verification_attempts == 1:
                # Forgets D2 and tries to dismiss a concern directly: must be rejected.
                calls = [
                    _tc(
                        "v",
                        "submit_verification",
                        {
                            "findings": [
                                {
                                    **_finding(
                                        "Early return leaks the bus lock",
                                        file="drivers/i2c/i2c_foo.c",
                                        line=17,
                                        end_line=19,
                                        severity="high",
                                        category="concurrency",
                                        confidence="high",
                                    ),
                                    "source_ids": ["C2"],
                                }
                            ],
                            "dismissed": [
                                {"source_ids": ["C1"], "reason": "not a code issue"},
                                {"source_ids": ["D1"], "reason": "ok"},
                            ],
                            "hard_cases": [
                                {
                                    "source_ids": ["C3"],
                                    "question": "is I2C_BUS_RECOVERY defined?",
                                    "severity_hint": "medium",
                                }
                            ],
                        },
                    )
                ]
            else:
                calls = [
                    _tc(
                        "v",
                        "submit_verification",
                        {
                            "findings": [
                                {
                                    **_finding(
                                        "Early return leaks the bus lock",
                                        file="drivers/i2c/i2c_foo.c",
                                        line=17,
                                        end_line=19,
                                        severity="high",
                                        category="concurrency",
                                        confidence="high",
                                        evidence="read_file i2c_foo.c:9-21",
                                    ),
                                    "source_ids": ["C2"],
                                }
                            ],
                            "dismissed": [{"source_ids": ["D1", "D2"], "reason": "both cite disproving code"}],
                            "hard_cases": [
                                {
                                    "source_ids": ["C3"],
                                    "question": "is I2C_BUS_RECOVERY defined anywhere?",
                                    "severity_hint": "medium",
                                },
                                {
                                    "source_ids": ["C1"],
                                    "question": "is the message really incomplete?",
                                    "severity_hint": "low",
                                },
                            ],
                        },
                    )
                ]
        elif "submit_hard_case_verdict" in tools:
            case_text = body["messages"][1]["content"].split("## Hard case", 1)[1]
            if "I2C_BUS_RECOVERY" in case_text:
                calls = [
                    _tc(
                        "h",
                        "submit_hard_case_verdict",
                        {
                            "verdict": "confirmed",
                            "reason": "kconfig_symbol shows no definition",
                            "finding": _finding(
                                "select of undefined I2C_BUS_RECOVERY",
                                file="drivers/i2c/Kconfig",
                                line=20,
                                severity="critical",
                                category="build_config",
                                speculative=True,
                                evidence="kconfig_symbol I2C_BUS_RECOVERY: no definition",
                            ),
                        },
                    )
                ]
            else:
                calls = [
                    _tc(
                        "h",
                        "submit_hard_case_verdict",
                        {"verdict": "refuted", "reason": "the second commit message names the symbol"},
                    )
                ]
        elif "submit_report" in tools:
            assert body.get("tools") and len(body["tools"]) <= 2, "report stage must not get the source tools"
            calls = [
                _tc(
                    "r",
                    "submit_report",
                    {
                        "summary": "The error path leaks the bus lock.",
                        "verdict": "comment",
                        "confidence": "high",
                        "test_assessment": "Tests are adequate.",
                        "commit_message_assessment": "",
                        "discussion_assessment": "",
                    },
                )
            ]
        else:
            raise AssertionError(f"unexpected tools {tools}")
        return httpx.Response(200, json=_completion({"content": "", "tool_calls": calls}, prompt_tokens=200))


async def test_pipeline_end_to_end(mini_repo):
    router = StageRouter()
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(7, "t", "", "author", "main", mini_repo["base"], "pr", mini_repo["head"], "o/r", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", "o/r")
    added = hunk_text(intel.diff_text)
    llm_cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    llm_cheap._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    llm_strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    llm_strong._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            llm_cheap,
            llm_strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text=added,
            log=lambda m: None,
        )

    assert run.review is not None
    rm = run.review_map
    assert rm["specialists"] == [
        "intent",
        "logic",
        "error_handling",
        "concurrency",
        "resources",
        "hardware",
        "build_config",
    ]
    # Specialist models vs verification models.
    spec_models = {
        r["model"] for r in router.requests if "submit_concerns" in {t["function"]["name"] for t in r["tools"]}
    }
    ver_models = {
        r["model"] for r in router.requests if "submit_verification" in {t["function"]["name"] for t in r["tools"]}
    }
    assert spec_models == {"cheap/model"} and ver_models == {"strong/model"}
    # The first verification attempt was rejected for breaking the accounting rules, then accepted.
    assert router.verification_attempts == 2
    vreqs = [r for r in router.requests if "submit_verification" in {t["function"]["name"] for t in r["tools"]}]
    rejection = [m for m in vreqs[1]["messages"] if m["role"] == "tool"]
    assert any("missing: D2" in m["content"] and "C1 is not allowed in dismissed" in m["content"] for m in rejection)

    fates = {c["id"]: c for c in rm["concerns"]}
    assert fates["C2"]["fate"] == "confirmed" and fates["C2"]["by"] == "verification"
    assert fates["C3"]["fate"] == "confirmed" and fates["C3"]["by"].startswith("H")
    assert fates["C1"]["fate"] == "refuted"
    assert {d["fate"] for d in rm["dismissed_concerns"]} == {"dismissed"}
    assert all(c["fate"] != "unaccounted" for c in rm["concerns"] + rm["dismissed_concerns"])

    titles = [f.title for f in run.review.findings]
    assert titles == ["Early return leaks the bus lock", "select of undefined I2C_BUS_RECOVERY"]
    lock, kconf = run.review.findings
    assert lock.inline and lock.line == 17 and "[C2]" in lock.evidence
    assert kconf.speculative and kconf.severity == "medium" and "[C3]" in kconf.evidence  # critical capped
    # The report said "comment" with a high finding standing: the deterministic rule wins.
    assert run.review.verdict == "request_changes"
    assert run.review.summary == "The error path leaks the bus lock." and run.review.test_assessment == ""
    stage_names = [s["stage"] for s in run.stages]
    n = len(rm["specialists"])
    assert stage_names[:n] == rm["specialists"] and "verification" in stage_names and "report" in stage_names
    assert sum(1 for s in stage_names if s.startswith("H")) == 2
    assert run.usage.requests == len(router.requests) and run.steps == sum(s["steps"] for s in run.stages)
    assert {t.get("stage") for t in run.transcript} >= {"intent", "verification", "report", "H1", "H2"}


async def test_pipeline_with_no_concerns_skips_verification(mini_repo):
    class Quiet(StageRouter):
        def handler(self, request):
            body = json.loads(request.content)
            self.requests.append(body)
            tools = {t["function"]["name"] for t in body["tools"]}
            if "submit_concerns" in tools:
                calls = [_tc("c", "submit_concerns", {"concerns": [], "dismissed": []})]
            else:
                assert "submit_report" in tools
                calls = [
                    _tc(
                        "r",
                        "submit_report",
                        {
                            "summary": "No issues found.",
                            "verdict": "request_changes",
                            "confidence": "high",
                            "test_assessment": "",
                            "commit_message_assessment": "",
                            "discussion_assessment": "",
                        },
                    )
                ]
            return httpx.Response(200, json=_completion({"content": "", "tool_calls": calls}))

    router = Quiet()
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    llm = _llm(ScriptedOpenRouter([]), models=("m",))
    llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            llm,
            llm,
            bridge,
            "sys",
            opening_message(pr, "ov", None, "", ""),
            {},
            changed_files=["doc/x.rst"],
            added_text="",
            specialists=["intent"],
            log=lambda m: None,
        )
    assert run.review is not None and run.review.findings == []
    assert run.review.verdict == "approve"  # request_changes without findings is corrected
    assert [s["stage"] for s in run.stages] == ["intent", "report"]
    assert run.review_map["concerns"] == [] and run.errors == []


def test_error_handling_stage_prompt_covers_returns_and_asserts():
    from zephyr_reviewer.review.pipeline import _stage_prompt

    text = _stage_prompt("error_handling")
    assert text.startswith("# Stage: error handling and assertions")
    for must in (
        "device_is_ready",
        "k_sem_take",
        "net_buf_alloc",
        "errno",
        "release what the success path would release",
        "CONFIG_ASSERT",
        "compiled out",
        "invariant",
        "runtime check",
        "syscall argument",
    ):
        assert must in text
    # The logic stage no longer claims the same ground.
    assert "belong to the error-handling stage" in _stage_prompt("logic")


def test_refuted_candidates_section_is_opt_in():
    from tests.test_agent import GOOD_REVIEW
    from zephyr_reviewer.review.agent import RunResult
    from zephyr_reviewer.review.github_api import PullRequest
    from zephyr_reviewer.review.openrouter import Usage
    from zephyr_reviewer.review.render import refuted_candidates, render_comment

    review, _ = parse_review(GOOD_REVIEW, {"drivers/i2c/i2c_foo.c": {17, 18, 19}, "drivers/i2c/Kconfig": {20}})
    rm = {
        "specialists": ["intent"],
        "concerns": [
            {"id": "C1", "stage": "intent", "title": "Kept", "fate": "confirmed", "by": "verification"},
            {
                "id": "C2",
                "stage": "security",
                "title": "Deferred close changes lifetime semantics",
                "fate": "refuted",
                "by": "H1",
                "reason": "the walker frees the node under ilock; callers never observe a freed session.",
            },
            {"id": "C3", "stage": "logic", "title": "Unused", "fate": "dismissed"},
        ],
        "dismissed_concerns": [],
        "findings": [],
        "stages": [],
    }
    run = RunResult(review, 3, Usage(), ["m"], False, review_map=rm)
    pr = PullRequest(7, "t", "", "a", "main", "b" * 40, "pr", "h" * 40, "o/r", False)
    assert refuted_candidates(run) == [
        (
            "Deferred close changes lifetime semantics",
            "security",
            "the walker frees the node under ilock; callers never observe a freed session.",
        )
    ]
    plain = render_comment(review, run, pr, repo="o/r", tag="t")
    assert "Also considered and refuted" not in plain
    shown = render_comment(review, run, pr, repo="o/r", tag="t", refuted=True)
    assert "<details><summary>Also considered and refuted: 1 candidate(s)</summary>" in shown
    assert "- Deferred close changes lifetime semantics (security): the walker frees the node under ilock" in shown
    assert "Kept" not in shown.split("Also considered")[1]
    # No review map (single agent): nothing added even when asked.
    assert "Also considered" not in render_comment(
        review, RunResult(review, 1, Usage(), ["m"], False), pr, repo="o/r", tag="t", refuted=True
    )
