# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The specialist pipeline: planning, accounting rules, and a full scripted run."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from tests.test_agent import ScriptedOpenRouter, _completion, _llm
from zephyr_reviewer.intel import SourceIntel
from zephyr_reviewer.mcp_server import build_server
from zephyr_reviewer.review.external import Claim
from zephyr_reviewer.review.followup import Dismissed, drop_dismissed
from zephyr_reviewer.review.github_api import PullRequest
from zephyr_reviewer.review.mcp_client import ToolBridge
from zephyr_reviewer.review.pipeline import (
    Concern,
    Primer,
    agreement_line,
    hunk_text,
    make_verification_parser,
    merge_concern,
    plan_specialists,
    primed,
    run_pipeline,
    verification_message,
)
from zephyr_reviewer.review.prompts import load_profile, opening_message, system_prompt
from zephyr_reviewer.review.schema import parse_review


def test_planner_picks_passes_from_the_change():
    assert plan_specialists(["doc/x.rst"], "") == ["intent", "logic"]
    # Kernel C code always gets the security pass: syscall verifiers and object checks live there.
    assert plan_specialists(["kernel/sched.c"], "z_sched_lock();") == [
        "intent",
        "logic",
        "error_handling",
        "concurrency",
        "security",
        "coding_guidelines",
    ]
    assert plan_specialists(["arch/arm/core/fault.S"], "") == [
        "intent",
        "logic",
        "error_handling",
        "concurrency",
        "coding_guidelines",
    ]
    assert plan_specialists(["lib/os/clock.c"], "") == [
        "intent",
        "logic",
        "error_handling",
        "resources",
        "security",
        "coding_guidelines",
    ]
    assert plan_specialists(["Kconfig"], "") == ["intent", "logic", "build_config"]  # no C, no guidelines pass
    assert plan_specialists(["drivers/i2c/i2c_foo.c", "drivers/i2c/Kconfig"], "k_sem_take(&l, K_FOREVER);") == [
        "intent",
        "logic",
        "error_handling",
        "concurrency",
        "resources",
        "hardware",
        "build_config",
        "coding_guidelines",
    ]
    assert "security" in plan_specialists(["subsys/net/ip/tcp.c"], "memcpy(a, b, n);")
    # Security runs for the attacker-facing areas the sightings came from, and on wire-length idioms anywhere.
    for path, text in [
        ("drivers/ieee802154/ieee802154_rf2xx.c", "rx_len = phr & 0x7f;"),
        ("subsys/fs/ext2/ext2_ops.c", ""),
        ("subsys/llext/llext_load.c", ""),
        ("kernel/poll.c", "z_vrfy_k_poll"),
        ("subsys/usb/device_next/class/usbd_cdc_ncm.c", ""),
        ("drivers/modem/hl7800.c", ""),
        ("subsys/lorawan/services/frag_transport.c", ""),
        ("drivers/foo/anything.c", "len = sys_get_le16(buf); memcpy(dst, buf, len);"),
    ]:
        assert "security" in plan_specialists([path], text), path
    assert "security" not in plan_specialists(["drivers/gpio/gpio_foo.c"], "gpio_pin_set(dev, 1);")
    assert "security" not in plan_specialists(["doc/x.rst"], "memcpy")  # no C code
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
        tools = {t["function"]["name"] for t in body["tools"]}
        if "submit_concerns" in tools:
            stage = body["messages"][1]["content"].split("# Stage: ", 1)[1].split("\n", 1)[0].strip()
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
        "coding_guidelines",
    ]
    # Specialist models vs verification models.
    spec_models = {
        r["model"] for r in router.requests if "submit_concerns" in {t["function"]["name"] for t in r["tools"]}
    }
    ver_models = {
        r["model"] for r in router.requests if "submit_verification" in {t["function"]["name"] for t in r["tools"]}
    }
    assert spec_models == {"cheap/model"} and ver_models == {"strong/model"}
    # Every specialist opens with the same system prompt, tools and opening, and its own instructions come
    # after them: the shared part is what a provider can serve from its prompt cache.
    firsts = [
        r
        for r in router.requests
        if "submit_concerns" in {t["function"]["name"] for t in r["tools"]} and len(r["messages"]) == 2
    ]
    assert len(firsts) == len(rm["specialists"])
    assert len({r["messages"][0]["content"] for r in firsts}) == 1
    assert len({json.dumps(r["tools"], sort_keys=True) for r in firsts}) == 1
    assert "# Stage: " not in firsts[0]["messages"][0]["content"]
    assert "# What you produce" in firsts[0]["messages"][0]["content"]
    tails = {r["messages"][1]["content"].removeprefix(opening) for r in firsts}
    assert all(r["messages"][1]["content"].startswith(opening) for r in firsts) and len(tails) == len(firsts)
    assert all("# Your part in this review" in t and "# Stage: " in t for t in tails)
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


def test_security_stage_prompt_encodes_the_sightings_taxonomy():
    from zephyr_reviewer.review.pipeline import _stage_prompt

    text = _stage_prompt("security")
    for must in (
        "Trust boundaries",
        "net_buf_pull",
        "buf->len >= sizeof",
        "ARRAY_SIZE",
        "uint8_t n = payload_len",
        "1U << n",
        "ROUND_UP",
        "strncpy(dst, src, sizeof(dst))",
        "rec_len",
        "CONTAINER_OF",
        "constant-time",
        "sys_csrand_get",
        "K_SYSCALL_MEMORY_READ/WRITE",
        "K_SYSCALL_OBJ_NEVER_INIT",
        "K_OBJ_ANY",
        "sibling sites",
        "FCS-good",
        "Not a boundary",
        "speculative",
    ):
        assert must in text, must
    guide = _stage_prompt("verification")  # the guide is appended at run time; check the guide file directly
    from zephyr_reviewer.review.prompts import load_guides

    assert "Bounds enforced elsewhere" in load_guides() and "strictly more privileged" in load_guides()
    assert guide  # verification prompt still loads


def test_coding_guidelines_stage_prompt_covers_the_misra_subset_and_project_rules():
    from zephyr_reviewer.review.pipeline import _stage_prompt
    from zephyr_reviewer.review.schema import CATEGORIES

    text = _stage_prompt("coding_guidelines")
    assert text.startswith("# Stage: coding guidelines")
    for must in (
        "MISRA C:2012",
        "checkpatch",
        "U` suffix",
        "1 << 31",
        "if (ptr != NULL)",
        "else if",
        "default",
        "__fallthrough",
        "ARG_UNUSED",
        "#if defined(CONFIG_X)",
        "include guards",
        "atoi",
        "memcmp",
        "A.1 Conditional compilation",
        "A.2 Inclusive language",
        "A.3 Macro name collisions",
        "A.4 libc in the kernel",
        "A.5 libc in the codebase",
        "strtok_r",
        "`low` or `suggestion`",
        "Group violations of the same rule",
        # 10.3 and 10.4 are judged on essential types, with the cases a review got wrong
        "essential type",
        "`u8a = u8b + u8c;` complies",
        "`0U` and `200U` are essentially `unsigned char`",
        "`u16 = (buf != NULL) ? cmd->opcode : 0U;` complies",
        "`cond ? u16 : 0`",
        "A proposed fix must not create another violation",
    ):
        assert must in text, must
    # The verification stage reads the guide, not the specialist's prompt: it must not confirm such a finding.
    guide = system_prompt(load_profile("zephyr"), "English")
    assert "MISRA conversions judged on the promoted type" in guide and "is not confirmed" in guide
    assert "coding_guidelines" in CATEGORIES


def test_concerns_with_the_same_file_and_title_merge_and_count_their_drafters():
    concerns: list[Concern] = []
    a = merge_concern(
        concerns, _finding("Early return leaks the bus lock", file="a.c", line=3, evidence="e1"), "logic@m1"
    )
    b = merge_concern(
        concerns, _finding("early return leaks the bus lock!", file="a.c", line=9, evidence="e2"), "concurrency@m2"
    )
    c = merge_concern(concerns, _finding("Early return leaks the bus lock", file="b.c", line=3), "logic@m1")
    assert a is b and c is not a and [x.id for x in concerns] == ["C1", "C2"]
    assert a.raised_by == ["logic@m1", "concurrency@m2"] and a.stage == "logic" and a.drafters == {"m1", "m2"}
    assert a.data["line"] == 3 and a.data["evidence"] == "e1 | e2"  # the first report's text, both evidences
    assert agreement_line(a, ["m1", "m2"]).startswith("Raised by 2 of 2 draft models (m1, m2)")
    assert agreement_line(c, ["m1", "m2"]).startswith("Raised by 1 of 2 draft models (m1)")
    assert agreement_line(a, ["m1"]) == ""  # no ensemble, no agreement talk
    msg = verification_message("open", concerns, [], ["logic", "concurrency"], ["m1", "m2"])
    assert "ran independently on 2 draft models (m1, m2)" in msg and "Raised by 1 of 2" in msg
    assert "draft models" not in verification_message("open", concerns, [], ["logic"], ["m1"])


class EnsembleRouter(StageRouter):
    """Specialists answer as StageRouter does; verification accounts for whatever ids it is shown: C* become
    findings, D* are dismissed, X* (external claims) go to hard cases, where StageRouter confirms the one that
    mentions I2C_BUS_RECOVERY and refutes the rest."""

    def handler(self, request):
        body = json.loads(request.content)
        tools = {t["function"]["name"] for t in body["tools"]}
        if "submit_verification" not in tools:
            return super().handler(request)
        self.requests.append(body)
        text = body["messages"][1]["content"]
        heads = re.findall(r"^### (C\d+) \[[^\]]*\] (.*?)  \(([^;:]*):", text, flags=re.M)
        dids = re.findall(r"^### (D\d+) ", text, flags=re.M)
        xids = re.findall(r"^### (X\d+) ", text, flags=re.M)
        out = {
            "findings": [
                {**_finding(title, file=file, line=1, severity="low", category="bug"), "source_ids": [cid]}
                for cid, title, file in heads
            ],
            "dismissed": [{"source_ids": dids, "reason": "all cite code"}] if dids else [],
            "hard_cases": [
                {"source_ids": [x], "question": "is the supplied claim real?", "severity_hint": "medium"} for x in xids
            ],
        }
        return httpx.Response(
            200, json=_completion({"content": "", "tool_calls": [_tc("v", "submit_verification", out)]})
        )


async def test_pipeline_ensemble_runs_each_specialist_per_model_and_merges(mini_repo):
    router = EnsembleRouter(verification_first_attempt_bad=False)
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    llms = []
    for name in ("cheap/model", "second/model", "strong/model"):
        llm = _llm(ScriptedOpenRouter([]), models=(name,))
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
        llms.append(llm)
    cheap, second, strong = llms
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            cheap,
            strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text="",
            specialists=["intent", "concurrency", "build_config"],
            ensemble=[second],
            log=lambda m: None,
        )
    assert run.review is not None and run.errors == []
    rm = run.review_map
    assert rm["drafters"] == ["cheap/model", "second/model"]
    spec = [r for r in router.requests if "submit_concerns" in {t["function"]["name"] for t in r["tools"]}]
    assert sorted(r["model"] for r in spec) == ["cheap/model"] * 3 + ["second/model"] * 3
    # Both models raised the same three concerns: the verifier saw three merged candidates, each raised by both,
    # while dismissals are not merged (a dismissal by one model is that model's evidence).
    assert [c["raised_by"] for c in rm["concerns"]] == [
        ["intent@cheap/model", "intent@second/model"],
        ["concurrency@cheap/model", "concurrency@second/model"],
        ["build_config@cheap/model", "build_config@second/model"],
    ]
    vreq = [r for r in router.requests if "submit_verification" in {t["function"]["name"] for t in r["tools"]}][0]
    text = vreq["messages"][1]["content"]
    assert "ran independently on 2 draft models (cheap/model, second/model)" in text
    assert text.count("Raised by 2 of 2 draft models (cheap/model, second/model)") == 3
    assert "## Concerns (3)" in text and "## Dismissed concerns (4)" in text
    # Stage records keep the stage name and say which drafter ran it; the verification saw all four dismissals.
    drafted = sorted((r["stage"], r["drafter"]) for r in run.stages if "drafter" in r)
    assert drafted == [
        ("build_config", "cheap/model"),
        ("build_config", "second/model"),
        ("concurrency", "cheap/model"),
        ("concurrency", "second/model"),
        ("intent", "cheap/model"),
        ("intent", "second/model"),
    ]
    assert {d["stage"] for d in rm["dismissed_concerns"]} == {
        "concurrency@cheap/model",
        "concurrency@second/model",
        "build_config@cheap/model",
        "build_config@second/model",
    }
    assert [f.title for f in run.review.findings] == [
        "Commit message does not mention the Kconfig symbol",
        "Early return leaks the bus lock",
        "select of undefined I2C_BUS_RECOVERY",
    ]
    assert all(c["fate"] == "confirmed" for c in rm["concerns"])


def test_verification_must_account_for_external_claim_ids():
    parse = make_verification_parser({"C1", "X1"}, {"D1"})
    out, errors = parse(
        {
            "findings": [{**_finding("t"), "source_ids": ["C1"]}],
            "dismissed": [{"source_ids": ["D1"], "reason": "r"}],
            "hard_cases": [],
        }
    )
    assert out is None and any("missing: X1" in e for e in errors)
    out, errors = parse(
        {
            "findings": [{**_finding("t"), "source_ids": ["C1", "X1"]}],
            "dismissed": [{"source_ids": ["D1"], "reason": "r"}],
            "hard_cases": [],
        }
    )
    assert out is not None and errors == [] and out.findings[0]["source_ids"] == ["C1", "X1"]
    out, errors = parse(
        {
            "findings": [],
            "dismissed": [{"source_ids": ["X1", "D1"], "reason": "r"}],
            "hard_cases": [{"source_ids": ["C1"], "question": "q", "severity_hint": "low"}],
        }
    )
    assert out is None and any("X1 is not allowed in dismissed" in e for e in errors)


async def test_pipeline_verifies_external_claims_and_reports_a_verdict_for_each(mini_repo):
    router = EnsembleRouter(verification_first_attempt_bad=False)
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    for llm in (cheap, strong):
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    claims = [
        Claim(
            "X1",
            "drivers/i2c/Kconfig:20: I2C_BUS_RECOVERY is selected but never defined",
            "drivers/i2c/Kconfig",
            20,
            "notes.txt",
        ),
        Claim("X2", "drivers/i2c/i2c_foo.c:30: the semaphore count is wrong", "drivers/i2c/i2c_foo.c", 30, "notes.txt"),
    ]
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            cheap,
            strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text="",
            specialists=["intent"],
            claims=claims,
            log=lambda m: None,
        )
    assert run.review is not None and run.errors == []
    vreq = [r for r in router.requests if "submit_verification" in {t["function"]["name"] for t in r["tools"]}][0]
    text = vreq["messages"][1]["content"]
    assert "## External claims (2)" in text and "### X1 [medium, speculative]" in text and "from external" in text
    assert "Being supplied is not evidence." in text
    verdicts = {c.id: c for c in run.review.external_claims}
    assert verdicts["X1"].verdict == "confirmed" and verdicts["X1"].finding_id.startswith("HF")
    assert verdicts["X2"].verdict == "refuted" and "second commit message" in verdicts["X2"].reason
    assert (
        verdicts["X1"].title == "I2C_BUS_RECOVERY is selected but never defined"
        and verdicts["X1"].source == "notes.txt"
    )
    titles = [f.title for f in run.review.findings]
    assert "select of undefined I2C_BUS_RECOVERY" in titles  # the hard case's confirmed finding
    xmap = {c["id"]: c for c in run.review_map["external_claims"]}
    assert xmap["X1"]["fate"] == "confirmed" and xmap["X2"]["fate"] == "refuted"
    rreq = [r for r in router.requests if "submit_report" in {t["function"]["name"] for t in r["tools"]}][0]
    assert "External claims checked: 1 confirmed, 1 refuted, 0 undecided" in rreq["messages"][1]["content"]


def test_a_hard_case_verdict_is_taken_as_it_was_validated():
    from zephyr_reviewer.review.pipeline import _parse_hard_case, make_hard_cases_parser

    # Any capitalisation is accepted, so what comes out is the spelling the pipeline compares with: a
    # "Confirmed" passed on as it came would be read as a refutation and the finding dropped.
    confirmed = {"verdict": "Confirmed", "reason": "the symbol has no definition", "finding": _finding("undefined")}
    out, errs = _parse_hard_case(confirmed)
    assert errs == [] and out["verdict"] == "confirmed" and out["finding"]["title"] == "undefined"
    out, errs = _parse_hard_case({"verdict": "REFUTED", "reason": "defined in Kconfig.foo"})
    assert errs == [] and out["verdict"] == "refuted"
    out, errs = make_hard_cases_parser(["H1", "H2"])(
        {"verdicts": [{**confirmed, "case": "H1"}, {"case": "H2", "verdict": "Refuted", "reason": "defined"}]}
    )
    assert errs == [] and [(v["case"], v["verdict"]) for v in out["verdicts"]] == [
        ("H1", "confirmed"),
        ("H2", "refuted"),
    ]
    assert _parse_hard_case({"verdict": "maybe", "reason": "r"}) == (None, ["verdict must be confirmed or refuted"])


def test_batched_hard_cases_need_one_verdict_each():
    from zephyr_reviewer.review.pipeline import make_hard_cases_parser

    parse = make_hard_cases_parser(["H2", "H5"])
    out, errs = parse({"verdicts": [{"case": "H2", "verdict": "refuted", "reason": "the guard exists"}]})
    assert out is None and errs == ["every case needs a verdict; missing: H5"]
    out, errs = parse(
        {
            "verdicts": [
                {"case": "H2", "verdict": "refuted", "reason": "the guard exists"},
                {"case": "H2", "verdict": "refuted", "reason": "again"},
                {"case": "H9", "verdict": "refuted", "reason": "?"},
                {"case": "H5", "verdict": "confirmed", "reason": "no finding given"},
            ]
        }
    )
    assert out is None
    assert "verdicts[1]: H2 is decided twice" in errs and any("unknown case 'H9'" in e for e in errs)
    assert any(e.startswith("H5: a confirmed verdict needs a complete finding") for e in errs)
    out, errs = parse(
        {
            "verdicts": [
                {"case": "H5", "verdict": "confirmed", "reason": "seen", "finding": _finding("t")},
                {"case": "H2", "verdict": "refuted", "reason": "the guard exists"},
            ]
        }
    )
    assert errs == [] and [v["case"] for v in out["verdicts"]] == ["H5", "H2"]
    assert parse({"verdicts": "no"}) == (None, ["arguments must be an object with a verdicts list"])


def test_verification_gives_every_external_claim_its_own_hard_case():
    parse = make_verification_parser({"C1", "X1", "X2"}, set())
    out, errs = parse(
        {
            "findings": [],
            "dismissed": [],
            "hard_cases": [{"source_ids": ["X1", "X2", "C1"], "question": "q", "severity_hint": "low"}],
        }
    )
    assert out is None and any(e.startswith("hard_cases[0] groups the external claims X1, X2") for e in errs)
    out, errs = parse(
        {
            "findings": [],
            "dismissed": [],
            "hard_cases": [
                {"source_ids": ["X1", "C1"], "question": "q", "severity_hint": "low"},
                {"source_ids": ["X2"], "question": "q", "severity_hint": "low"},
            ],
        }
    )
    assert errs == [] and [h.source_ids for h in out.hard_cases] == [["X1", "C1"], ["X2"]]


class BatchRouter(EnsembleRouter):
    """Answers a batch of hard cases with a verdict and a reason per case."""

    def handler(self, request):
        body = json.loads(request.content)
        tools = {t["function"]["name"] for t in body["tools"]}
        if "submit_hard_case_verdicts" not in tools:
            return super().handler(request)
        self.requests.append(body)
        text = body["messages"][1]["content"]
        verdicts = []
        for case_id, block in re.findall(r"^## Hard case (H\d+)\n(.*?)(?=^## Hard case |\Z)", text, flags=re.M | re.S):
            if "I2C_BUS_RECOVERY" in block:
                verdicts.append(
                    {
                        "case": case_id,
                        "verdict": "confirmed",
                        "reason": "kconfig_symbol shows no definition",
                        "finding": _finding(
                            "select of undefined I2C_BUS_RECOVERY",
                            file="drivers/i2c/Kconfig",
                            line=20,
                            severity="medium",
                            category="build_config",
                        ),
                    }
                )
            else:
                where = re.search(r"\((\S+?):\d+", block)
                verdicts.append(
                    {
                        "case": case_id,
                        "verdict": "refuted",
                        "reason": f"nothing wrong at {where.group(1) if where else '?'}",
                    }
                )
        calls = [_tc("hb", "submit_hard_case_verdicts", {"verdicts": verdicts})]
        return httpx.Response(200, json=_completion({"content": "", "tool_calls": calls}))


async def test_batched_hard_cases_keep_their_own_reasons(mini_repo):
    router = BatchRouter(verification_first_attempt_bad=False)
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    for llm in (cheap, strong):
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    claims = [
        Claim("X1", "drivers/i2c/Kconfig:20: I2C_BUS_RECOVERY is never defined", "drivers/i2c/Kconfig", 20, "e"),
        Claim("X2", "drivers/i2c/i2c_foo.c:30: the semaphore count is wrong", "drivers/i2c/i2c_foo.c", 30, "e"),
        Claim("X3", "drivers/i2c/i2c_foo.c:12: the name is misspelled", "drivers/i2c/i2c_foo.c", 12, "e"),
    ]
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            cheap,
            strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text="",
            specialists=["intent"],
            claims=claims,
            max_hard_cases=1,  # everything lands in one batch
            log=lambda m: None,
        )
    assert run.review is not None and run.errors == []
    batch = [r for r in router.requests if "submit_hard_case_verdicts" in {t["function"]["name"] for t in r["tools"]}]
    assert len(batch) == 1
    text = batch[0]["messages"][1]["content"]
    assert "# 3 hard cases, each decided on its own" in text and text.count("## Hard case H") == 3
    assert "a reason describes that one case and never refers to the others" in text
    verdicts = {c.id: c for c in run.review.external_claims}
    assert verdicts["X1"].verdict == "confirmed" and verdicts["X1"].finding_id.startswith("HF")
    assert verdicts["X2"].verdict == "refuted" and verdicts["X2"].reason == "nothing wrong at drivers/i2c/i2c_foo.c"
    assert verdicts["X3"].verdict == "refuted" and verdicts["X3"].reason == "nothing wrong at drivers/i2c/i2c_foo.c"
    stage = [s for s in run.stages if s["stage"] == "H1+H2+H3"][0]
    assert stage["verdict"] == "H1 confirmed, H2 refuted, H3 refuted"


def test_the_cases_beyond_the_single_ones_share_one_run_or_runs_of_a_few_each():
    from zephyr_reviewer.review.pipeline import HardCase, batch_hard_cases

    def spread(n: int, max_single: int = 8, batch: int = 0) -> list[int]:
        cases = [HardCase(f"H{i + 1}", [f"C{i + 1}"], "q", "medium") for i in range(n)]
        batches = batch_hard_cases(cases, max_single, batch)
        assert [h.id for b in batches for h in b] == [h.id for h in cases]  # every case once, order kept
        return [len(b) for b in batches]

    assert spread(0) == [] and spread(5) == [1] * 5 and spread(8) == [1] * 8  # up to the limit, a run each
    # Without a limit, all of the rest share one run: the fewest runs of the verification model.
    assert spread(9) == [1] * 7 + [2] and spread(29) == [1] * 7 + [22] and spread(29, batch=-3) == [1] * 7 + [22]
    # With one, the rest is spread over runs of at most that many cases.
    assert spread(29, batch=4) == [1] * 7 + [4, 4, 4, 4, 4, 2] and spread(9, batch=4) == [1] * 7 + [2]
    assert spread(29, batch=22) == [1] * 7 + [22] and spread(29, batch=1) == [1] * 29
    assert spread(3, max_single=1) == [3] and spread(5, max_single=1, batch=2) == [2, 2, 1]
    assert spread(4, max_single=0) == [4] and spread(4, max_single=-1, batch=1) == [1] * 4  # every case decided
    assert spread(0, max_single=-1) == [] and spread(0, max_single=0, batch=3) == []  # and none is none


def test_a_case_rated_below_the_floor_for_a_run_of_its_own_shares_one():
    from zephyr_reviewer.review.pipeline import (
        Concern,
        HardCase,
        batch_hard_cases,
        below_floor,
        draft_severity,
        merge_concern,
    )

    sev = {"C1": "high", "C2": "low", "C3": "medium", "C4": "suggestion", "C5": "Low", "C6": "", "C7": "low"}
    concerns = {k: Concern(k, "logic", {"title": k, "severity": v}) for k, v in sev.items()}
    concerns["X1"] = Concern("X1", "external", {"title": "an earlier comment", "severity": "low"})

    def case(n: int, *ids: str) -> HardCase:
        return HardCase(f"H{n}", list(ids), "q", "medium")

    # A case is rated as its highest concern; an unknown rating counts as medium.
    assert draft_severity(case(1, "C2", "C1"), concerns) == "high" and draft_severity(case(1, "C5"), concerns) == "low"
    assert draft_severity(case(1, "C6"), concerns) == "medium"
    # A dismissal in a case does not rate it: the case is rated by its concerns.
    assert draft_severity(case(1, "C2", "D1"), concerns) == "low" and below_floor(
        case(1, "C2", "D1"), "medium", concerns
    )
    # A case with no concern, or with a comment of an earlier review, has no draft rating: no floor applies.
    assert draft_severity(case(1, "D1"), concerns) is None and draft_severity(case(1, "C2", "X1"), concerns) is None
    # A concern that two passes raised keeps the first one's text, and counts with the higher of their ratings.
    merged: list[Concern] = []
    for origin, rating in (("logic@a", "low"), ("logic@b", "high"), ("security@a", "")):
        merge_concern(merged, {"file": "a.c", "title": "Lock leak", "problem": "p", "severity": rating}, origin)
    assert len(merged) == 1 and merged[0].data["severity"] == "low" and merged[0].ratings == ["low", "high", ""]
    by_id = {merged[0].id: merged[0]}
    assert draft_severity(case(1, merged[0].id), by_id) == "high" and not below_floor(
        case(1, merged[0].id), "medium", by_id
    )
    merged[0].ratings = ["low", "suggestion"]
    assert below_floor(case(1, merged[0].id), "medium", by_id)
    assert below_floor(case(1, "C2", "C4"), "medium", concerns) and not below_floor(
        case(1, "C2", "C3"), "medium", concerns
    )
    assert not below_floor(case(1, "C2"), "low", concerns) and below_floor(case(1, "C4"), "low", concerns)
    assert not below_floor(case(1, "C4"), "", concerns) and not below_floor(case(1, "C4"), "nonsense", concerns)
    assert not below_floor(case(1, "D1"), "critical", concerns) and not below_floor(case(1, "X1"), "critical", concerns)

    cases = [case(1, "C1"), case(2, "C3"), case(3, "C2"), case(4, "C4"), case(5, "C7")]

    def spread(max_single: int, floor: str, batch: int = 0) -> list[list[str]]:
        batches = batch_hard_cases(cases, max_single, batch, alone=lambda h: not below_floor(h, floor, concerns))
        assert sorted(h.id for b in batches for h in b) == [h.id for h in cases]  # every case once
        return [[h.id for h in b] for b in batches]

    assert spread(8, "") == [["H1"], ["H2"], ["H3"], ["H4"], ["H5"]]
    # Few enough for a run each, but three are rated below medium: those share one, in their order.
    assert spread(8, "medium") == [["H1"], ["H2"], ["H3", "H4", "H5"]]
    assert spread(8, "medium", batch=2) == [["H1"], ["H2"], ["H3", "H4"], ["H5"]]
    assert spread(8, "low") == [["H1"], ["H2"], ["H3"], ["H5"], ["H4"]]  # one such case alone still gets the run
    # More cases than runs: the one that would have been single but is rated too low gives its place up.
    assert spread(3, "") == [["H1"], ["H2"], ["H3", "H4", "H5"]]
    assert spread(4, "medium") == [["H1"], ["H2"], ["H3", "H4", "H5"]]
    assert spread(2, "high") == [["H1"], ["H2", "H3", "H4", "H5"]]


async def test_hard_cases_rated_below_the_floor_are_not_decided(mini_repo):
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(7, "t", "", "author", "main", mini_repo["base"], "pr", mini_repo["head"], "o/r", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", "o/r")

    async def run(floor: str):
        router = StageRouter(verification_first_attempt_bad=False)
        logged: list[str] = []
        cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
        strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
        for llm in (cheap, strong):
            llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
        async with ToolBridge(build_server(intel)) as bridge:
            result = await run_pipeline(
                cheap,
                strong,
                bridge,
                system_prompt(load_profile("zephyr"), "English"),
                opening,
                {f.path: f.new_side_lines for f in intel.files},
                changed_files=[f.path for f in intel.files],
                added_text=hunk_text(intel.diff_text),
                hard_case_min_severity=floor,
                log=logged.append,
            )
        hard = [r for r in router.requests if "submit_hard_case_verdict" in {t["function"]["name"] for t in r["tools"]}]
        return result, len(hard), logged

    # The scripted verification sends two concerns to hard cases: C3, rated medium, and C1, rated low.
    result, hard, logged = await run("")
    fates = {c["id"]: c for c in result.review_map["concerns"]}
    assert hard == 2 and fates["C1"]["fate"] == "refuted" and fates["C3"]["fate"] == "confirmed"
    assert not any("not decided" in m for m in logged)

    result, hard, logged = await run("medium")
    fates = {c["id"]: c for c in result.review_map["concerns"]}
    assert hard == 1 and fates["C3"]["fate"] == "confirmed" and fates["C3"]["by"] == "H1"
    assert (fates["C1"]["fate"], fates["C1"]["by"]) == ("skipped", "hard_case_min_severity")
    assert fates["C1"]["note"] == "rated low by the draft model, below medium; not verified, not reported"
    assert [s["stage"] for s in result.stages if s["stage"].startswith("H")] == ["H1"]
    assert any("1 hard case(s) not decided" in m and "H2" in m for m in logged)
    # What verification confirmed on its own is reported as before, and nothing is left unaccounted.
    assert "Early return leaks the bus lock" in [f.title for f in result.review.findings]
    assert "Commit message does not mention the Kconfig symbol" not in [f.title for f in result.review.findings]
    assert all(c["fate"] != "unaccounted" for c in result.review_map["concerns"])

    # A floor at the rating of the lowest case leaves all of them in.
    result, hard, logged = await run("low")
    assert hard == 2 and not any("not decided" in m for m in logged)


async def test_a_long_tail_of_hard_cases_is_decided_in_several_shared_runs(mini_repo):
    router = BatchRouter(verification_first_attempt_bad=False)
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    for llm in (cheap, strong):
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    claims = [Claim("X1", "drivers/i2c/Kconfig:20: I2C_BUS_RECOVERY is never defined", "drivers/i2c/Kconfig", 20, "e")]
    claims += [
        Claim(f"X{i}", f"drivers/i2c/i2c_foo.c:{10 + i}: point {i} is wrong", "drivers/i2c/i2c_foo.c", 10 + i, "e")
        for i in range(2, 6)
    ]
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            cheap,
            strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text="",
            specialists=["intent"],
            claims=claims,
            max_hard_cases=1,
            hard_case_batch=2,
            log=lambda m: None,
        )
    assert run.review is not None and run.errors == []
    # Five cases: two shared runs of two, and the last one alone, which is then an ordinary single case.
    assert [s["stage"] for s in run.stages if s["stage"].startswith("H")] == ["H1+H2", "H3+H4", "H5"]
    shared = [r for r in router.requests if "submit_hard_case_verdicts" in {t["function"]["name"] for t in r["tools"]}]
    assert sorted(r["messages"][1]["content"].count("## Hard case H") for r in shared) == [2, 2]
    assert all("# 2 hard cases, each decided on its own" in r["messages"][1]["content"] for r in shared)
    # The run's turns are shared, and the message says so: all cases' lookups in every turn.
    assert all(
        "You have 20 turns for all 2 cases together, so work on them side by side" in r["messages"][1]["content"]
        for r in shared
    )
    assert all("several tool calls in one answer" in r["messages"][1]["content"] for r in shared)
    verdicts = {c.id: c.verdict for c in run.review.external_claims}
    assert verdicts == {"X1": "confirmed", "X2": "refuted", "X3": "refuted", "X4": "refuted", "X5": "refuted"}


async def test_findings_pruned_before_the_report_are_not_in_it(mini_repo):
    router = StageRouter()
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(7, "t", "", "author", "main", mini_repo["base"], "pr", mini_repo["head"], "o/r", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", "o/r")
    llm_cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    llm_cheap._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    llm_strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    llm_strong._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    seen: list[int] = []

    def prune(review) -> int:
        # Every verified finding restates a comment a person dismissed, as the command line passes it.
        dismissed = [Dismissed(i, f"T{i}", f.file, f.line, f.title, "", "bob") for i, f in enumerate(review.findings)]
        seen.append(len(dismissed))
        return drop_dismissed(review, dismissed, log=lambda m: None)

    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            llm_cheap,
            llm_strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text=hunk_text(intel.diff_text),
            prune=prune,
            log=lambda m: None,
        )

    assert seen and seen[0] > 0 and run.review is not None and run.review.findings == []
    report = [r for r in router.requests if "submit_report" in {t["function"]["name"] for t in r["tools"]}][-1]
    asked = report["messages"][1]["content"]
    assert "## Verified findings (fixed; do not change them)\n(none)" in asked


async def test_parallel_runs_wait_for_the_first_answer_of_the_one_that_leads():
    primer = Primer()
    order: list[str] = []
    answered = asyncio.Event()

    def run(name: str):
        async def go(hook):
            order.append(f"{name} starts")
            if hook is not None:  # the leader: its first model request is answered later
                await answered.wait()
                hook()
            await asyncio.sleep(0)
            order.append(f"{name} ends")
            return name

        return go

    tasks = [asyncio.create_task(primed(primer, run(n))) for n in ("a", "b", "c")]
    await asyncio.sleep(0.01)
    assert order == ["a starts"]  # the others wait: nothing is in the provider's cache yet
    answered.set()
    assert await asyncio.gather(*tasks) == ["a", "b", "c"]
    assert order[0] == "a starts" and set(order[1:3]) <= {"b starts", "c starts", "a ends"}
    assert sorted(order) == sorted(f"{n} {w}" for n in "abc" for w in ("starts", "ends"))


async def test_a_leader_that_fails_or_stalls_does_not_hold_the_others_back():
    async def boom(hook):
        raise RuntimeError("model request failed")

    async def fine(hook):
        return "ran"

    primer = Primer()
    results = await asyncio.gather(primed(primer, boom), primed(primer, fine), return_exceptions=True)
    assert isinstance(results[0], RuntimeError) and results[1] == "ran"

    stalled = asyncio.Event()

    async def stall(hook):
        await stalled.wait()
        return "late"

    primer = Primer(timeout=0.01)
    leader = asyncio.create_task(primed(primer, stall))
    assert await primed(primer, fine) == "ran"  # released by the timeout, not by the leader
    stalled.set()
    assert await leader == "late"
    assert await primed(None, fine) == "ran"  # no primer: runs at once


class LosingRouter(EnsembleRouter):
    """A router on which every specialist pass of the models in ``lose`` fails."""

    def __init__(self, lose: tuple[str, ...]):
        super().__init__(verification_first_attempt_bad=False)
        self.lose = lose

    def handler(self, request):
        body = json.loads(request.content)
        tools = {t["function"]["name"] for t in body.get("tools") or []}
        if "submit_concerns" in tools and body.get("model") in self.lose:
            return httpx.Response(400, json={"error": {"message": "Provider returned error", "code": 400}})
        return super().handler(request)


async def _run_ensemble(router, mini_repo, logs: list[str], **kw):
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(7, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "o/r", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    llms = []
    for name in ("cheap/model", "second/model", "strong/model"):
        llm = _llm(ScriptedOpenRouter([]), models=(name,))
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
        llms.append(llm)
    cheap, second, strong = llms
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            cheap,
            strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text="",
            specialists=["intent", "concurrency", "build_config"],
            ensemble=[second],
            log=logs.append,
            quiet=True,
            **kw,
        )
    return run, pr


async def test_passes_that_return_nothing_are_counted_and_lower_the_confidence(mini_repo):
    from zephyr_reviewer.review.render import render_comment

    # Nothing lost and no speculative finding: the report's "high" stands, and the summary says nothing.
    logs: list[str] = []
    run, pr = await _run_ensemble(LosingRouter(()), mini_repo, logs)
    assert run.review_map["passes"] == {"planned": 6, "lost": 0} and run.review.confidence == "high"
    assert not any(f.speculative for f in run.review.findings) and not any("::warning::" in m for m in logs)
    assert "returned no result" not in render_comment(run.review, run, pr, repo="o/r", tag="t")

    # One model's three passes fail: the other model's stand, and the reader is told what is missing.
    logs = []
    run, pr = await _run_ensemble(LosingRouter(("second/model",)), mini_repo, logs)
    assert run.review is not None and run.review_map["passes"] == {"planned": 6, "lost": 3}
    assert run.review.confidence == "medium"  # the report said high; it was not told what was lost
    warnings = [m for m in logs if m.startswith("::warning::review pass ")]
    assert len(warnings) == 3 and all("@second/model returned no result: model request failed" in m for m in warnings)
    body = render_comment(run.review, run, pr, repo="o/r", tag="t")
    assert "3 of 6 planned review passes returned no result." in body
    assert "second/model" not in body and "intent" not in body  # a count, no stage and no model


class AllClearReportRouter(EnsembleRouter):
    """The report approves and says that nothing was found, whatever the findings."""

    def handler(self, request):
        body = json.loads(request.content)
        if "submit_report" not in {t["function"]["name"] for t in body.get("tools") or []}:
            return super().handler(request)
        out = {"summary": "No issues found.", "verdict": "approve", "confidence": "high"}
        return httpx.Response(200, json=_completion({"content": "", "tool_calls": [_tc("r", "submit_report", out)]}))


async def test_a_review_with_a_finding_does_not_say_that_there_is_none(mini_repo):
    from zephyr_reviewer.review.render import render_comment

    run, pr = await _run_ensemble(AllClearReportRouter(verification_first_attempt_bad=False), mini_repo, [])
    review = run.review
    assert review.findings and review.highest_severity() == "low"  # nothing that forces a verdict
    assert review.verdict == "comment" and review.summary == "Automated review completed; see findings."
    body = render_comment(review, run, pr, repo="o/r", tag="t")
    assert "No findings reported" not in body and "No issues found" not in body
    assert "worth a look, nothing blocking" in body and f"**Findings:** {len(review.findings)}" in body


async def test_a_speculative_finding_keeps_the_confidence_from_being_high(mini_repo):
    logs: list[str] = []
    run = await _run_with(StageRouter(verification_first_attempt_bad=False), mini_repo, logs)
    assert run.review_map["passes"]["lost"] == 0 and any(f.speculative for f in run.review.findings)
    assert run.review.confidence == "medium"  # the scripted report says high


async def test_when_every_pass_returns_nothing_there_is_no_review(mini_repo):
    logs: list[str] = []
    router = LosingRouter(("cheap/model", "second/model"))
    run, _ = await _run_ensemble(router, mini_repo, logs)
    assert run.review is None  # not an empty review, which would read "No findings reported"
    assert run.errors[-1] == "none of the 6 review pass(es) returned a result, so there is no review"
    assert run.review_map["passes"] == {"planned": 6, "lost": 6}
    tools = [{t["function"]["name"] for t in r["tools"]} for r in router.requests]
    assert not any({"submit_verification", "submit_report"} & t for t in tools)
    # With an earlier comment to re-check there is still something to verify and to answer.
    claim = Claim(
        "X1", "drivers/i2c/Kconfig:20: I2C_BUS_RECOVERY is selected but never defined", "drivers/i2c/Kconfig", 20
    )
    run, _ = await _run_ensemble(LosingRouter(("cheap/model", "second/model")), mini_repo, [], claims=[claim])
    assert run.review is not None and [c.verdict for c in run.review.external_claims] == ["confirmed"]
    assert run.review.confidence == "medium"


class DoubtedDismissalRouter(StageRouter):
    """Verification doubts a dismissal and sends it to a hard case of its own, which never answers."""

    def handler(self, request):
        body = json.loads(request.content)
        tools = {t["function"]["name"] for t in body["tools"]}
        if "submit_verification" in tools:
            self.requests.append(body)
            lock = _finding("Early return leaks the bus lock", file="drivers/i2c/i2c_foo.c", line=17, severity="high")
            out = {
                "findings": [{**lock, "source_ids": ["C1"]}],
                "dismissed": [],
                "hard_cases": [{"source_ids": ["D1"], "question": "is one the right count?", "severity_hint": "low"}],
            }
            return httpx.Response(
                200, json=_completion({"content": "", "tool_calls": [_tc("v", "submit_verification", out)]})
            )
        if "submit_hard_case_verdict" in tools:
            self.requests.append(body)
            return httpx.Response(200, json=_completion({"content": "I cannot tell from here.", "tool_calls": []}))
        return super().handler(request)


async def test_a_dismissal_whose_hard_case_did_not_decide_is_accounted_for(mini_repo):
    router = DoubtedDismissalRouter(verification_first_attempt_bad=False)
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    for llm in (cheap, strong):
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    async with ToolBridge(build_server(intel)) as bridge:
        run = await run_pipeline(
            cheap,
            strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text="",
            specialists=["concurrency"],
            hard_case_max_steps=1,
            log=lambda m: None,
        )
    assert run.review is not None and [f.title for f in run.review.findings] == ["Early return leaks the bus lock"]
    assert any(e.startswith("hard case H1 undecided") for e in run.errors)
    # The hard case had no concern to keep as a speculative finding, only the dismissal it was asked about.
    (dismissal,) = run.review_map["dismissed_concerns"]
    assert dismissal["id"] == "D1" and dismissal["title"] == "k_sem_init count"
    assert (dismissal["fate"], dismissal["by"]) == ("undecided", "H1")
    assert dismissal["note"] == "undecided; the dismissal stands"
    assert [s["verdict"] for s in run.stages if s["stage"] == "H1"] == ["undecided"]


class FailingVerification(StageRouter):
    """A router whose verification stage gets a provider error for its first ``failures`` requests."""

    def __init__(self, failures: int):
        super().__init__(verification_first_attempt_bad=False)
        self.failures = failures
        self.verification_requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        tools = {t["function"]["name"] for t in json.loads(request.content).get("tools") or []}
        if "submit_verification" in tools:
            self.verification_requests += 1
            if self.verification_requests <= self.failures:
                return httpx.Response(400, json={"error": {"message": "Provider returned error", "code": 400}})
        return super().handler(request)


async def _run_with(router: StageRouter, mini_repo, logs: list[str], **kw):
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(7, "t", "", "author", "main", mini_repo["base"], "pr", mini_repo["head"], "o/r", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", "o/r")
    llm_cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    llm_cheap._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    llm_strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    llm_strong._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    async with ToolBridge(build_server(intel)) as bridge:
        return await run_pipeline(
            llm_cheap,
            llm_strong,
            bridge,
            system_prompt(load_profile("zephyr"), "English"),
            opening,
            {f.path: f.new_side_lines for f in intel.files},
            changed_files=[f.path for f in intel.files],
            added_text=hunk_text(intel.diff_text),
            log=logs.append,
            quiet=True,
            **kw,
        )


async def test_a_verification_stage_that_fails_once_is_run_again(mini_repo):
    logs: list[str] = []
    healthy = await _run_with(StageRouter(verification_first_attempt_bad=False), mini_repo, [])
    run = await _run_with(FailingVerification(1), mini_repo, logs)
    assert any(m.startswith("verification did not submit (model request failed: HTTP 400") for m in logs)
    assert any("running it once more" in m for m in logs)
    # The second run of the stage is the review's: the same findings as when nothing failed.
    assert run.review is not None and run.review.findings
    assert [f.title for f in run.review.findings] == [f.title for f in healthy.review.findings]
    assert run.errors == healthy.errors
    stages = [(r["stage"], r.get("failed", False)) for r in run.stages if r["stage"] == "verification"]
    assert stages == [("verification", True), ("verification", False)]


async def test_without_verification_there_is_no_review(mini_repo):
    logs: list[str] = []
    router = FailingVerification(99)
    run = await _run_with(router, mini_repo, logs)
    assert router.verification_requests == 2  # the stage and its one repeat, each ended by the first failed request
    # Nothing to post: the specialists' candidates were not checked, merged or weighed.
    assert run.review is None
    assert len(run.errors) == 1 and run.errors[0].startswith(
        "verification did not submit, twice: model request failed: HTTP 400"
    )
    assert "were not verified, so there is no review" in run.errors[0]
    # No later stage ran, and the record keeps what the specialists raised.
    tools = [{t["function"]["name"] for t in r["tools"]} for r in router.requests]
    assert not any({"submit_hard_case_verdict", "submit_hard_case_verdicts", "submit_report"} & t for t in tools)
    assert run.review_map["concerns"] and {c["fate"] for c in run.review_map["concerns"]} == {"unverified"}


async def test_the_record_of_a_run_without_verification_keeps_what_an_earlier_run_refuted(mini_repo):
    from zephyr_reviewer.review.previous import Refutation

    healthy = await _run_with(StageRouter(verification_first_attempt_bad=False), mini_repo, [])
    first, *others = healthy.review_map["concerns"]
    assert others  # something is left for verification to fail on
    settled = Refutation(first["file"], first["line"], first["title"], "the lock is released", "a" * 40, settled=True)
    run = await _run_with(FailingVerification(99), mini_repo, [], refuted=[settled])
    assert run.review is None
    fates = {c["id"]: (c["fate"], c.get("reason", "")) for c in run.review_map["concerns"]}
    assert fates[first["id"]] == ("refuted", "the lock is released")
    assert {fates[c["id"]] for c in others} == {("unverified", "")}


class HintedRouter(BatchRouter):
    """Verification sends every concern and every earlier comment to a hard case of its own, hinted as told."""

    hints: dict[str, str] = {}

    def handler(self, request):
        body = json.loads(request.content)
        if "submit_verification" not in {t["function"]["name"] for t in body["tools"]}:
            return super().handler(request)
        self.requests.append(body)
        ids = re.findall(r"^### ([CX]\d+) ", body["messages"][1]["content"], flags=re.M)
        dids = re.findall(r"^### (D\d+) ", body["messages"][1]["content"], flags=re.M)
        out = {
            "findings": [],
            "dismissed": [{"source_ids": dids, "reason": "all cite code"}] if dids else [],
            "hard_cases": [
                {"source_ids": [i], "question": "is it real?", "severity_hint": self.hints.get(i, "low")} for i in ids
            ],
        }
        return httpx.Response(
            200, json=_completion({"content": "", "tool_calls": [_tc("v", "submit_verification", out)]})
        )


async def test_a_comment_of_an_earlier_review_is_settled_in_a_run_of_its_own_before_other_cases(mini_repo):
    from zephyr_reviewer.review.pipeline import Concern, HardCase, hard_case_order

    concerns = {"C1": Concern("C1", "logic", {}), "X1": Concern("X1", "external", {})}

    def key(ids: list[str], hint: str) -> tuple[int, int]:
        return hard_case_order(HardCase("H1", ids, "q", hint), concerns)

    # Critical and high first whatever they are about, then what settles an earlier comment, then the rest.
    assert key(["C1"], "critical") < key(["X1"], "high") < key(["C1", "X1"], "medium")
    assert key(["C1", "X1"], "medium") < key(["X1"], "low") < key(["X1"], "suggestion") < key(["C1"], "medium")
    assert key(["C1"], "medium") < key(["D1"], "low") and key(["X1"], "high") == key(["C1"], "high")

    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    claims = [
        Claim("X1", "drivers/i2c/i2c_foo.c:30: the semaphore count is wrong", "drivers/i2c/i2c_foo.c", 30, "e"),
        Claim("X2", "drivers/i2c/i2c_foo.c:12: the name is misspelled", "drivers/i2c/i2c_foo.c", 12, "e"),
    ]

    async def stages(hints: dict[str, str]) -> list[str]:
        router = HintedRouter(verification_first_attempt_bad=False)
        router.hints = hints
        cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
        strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
        for llm in (cheap, strong):
            llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
        async with ToolBridge(build_server(intel)) as bridge:
            run = await run_pipeline(
                cheap,
                strong,
                bridge,
                system_prompt(load_profile("zephyr"), "English"),
                opening,
                {f.path: f.new_side_lines for f in intel.files},
                changed_files=[f.path for f in intel.files],
                added_text="",
                specialists=["intent"],
                claims=claims,
                max_hard_cases=2,  # one run for a case alone, one for the rest
                log=lambda m: None,
            )
        assert run.review is not None and {c.id: c.verdict for c in run.review.external_claims} == {
            "X1": "refuted",
            "X2": "refuted",
        }
        return [s["stage"] for s in run.stages if s["stage"].startswith("H")]

    # H1 is the pass's own concern (C1), H2 and H3 the two earlier comments. Hinted higher than they are, the
    # concern does not take the one run for a case alone from an earlier comment unless it is high or critical.
    assert await stages({"C1": "medium", "X1": "low", "X2": "low"}) == ["H2", "H3+H1"]
    assert await stages({"C1": "high", "X1": "low", "X2": "low"}) == ["H1", "H2+H3"]
    assert await stages({"C1": "low", "X1": "low", "X2": "medium"}) == ["H3", "H2+H1"]


class SureOfItRouter(EnsembleRouter):
    """A router whose hard case on the Kconfig select confirms it as a plain high finding."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tools = {t["function"]["name"] for t in body.get("tools") or []}
        case = body["messages"][1]["content"].split("## Hard case", 1)[-1]
        if "submit_hard_case_verdict" in tools and "I2C_BUS_RECOVERY" in case:
            self.requests.append(body)
            finding = _finding(
                "select of undefined I2C_BUS_RECOVERY", file="drivers/i2c/Kconfig", line=20, severity="high"
            )
            verdict = {"verdict": "confirmed", "reason": "kconfig_symbol shows no definition", "finding": finding}
            call = _tc("h", "submit_hard_case_verdict", verdict)
            return httpx.Response(200, json=_completion({"content": "", "tool_calls": [call]}, prompt_tokens=200))
        return super().handler(request)


async def test_a_hard_case_out_of_budget_decides_nothing_about_an_earlier_comment(mini_repo):
    router = SureOfItRouter(verification_first_attempt_bad=False)
    intel = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])
    pr = PullRequest(0, "t", "", "", "main", mini_repo["base"], "pr", mini_repo["head"], "", False)
    opening = opening_message(pr, intel.overview(), intel.diff_text, "", repo="o/r")
    cheap = _llm(ScriptedOpenRouter([]), models=("cheap/model",))
    strong = _llm(ScriptedOpenRouter([]), models=("strong/model",))
    for llm in (cheap, strong):
        llm._client = httpx.Client(transport=httpx.MockTransport(router.handler))
    claims = [
        Claim(
            "X1", "drivers/i2c/Kconfig:20: I2C_BUS_RECOVERY is selected but never defined", "drivers/i2c/Kconfig", 20
        ),
        Claim("X2", "drivers/i2c/i2c_foo.c:30: the semaphore count is wrong", "drivers/i2c/i2c_foo.c", 30),
    ]

    async def run(steps: int):
        async with ToolBridge(build_server(intel)) as bridge:
            return await run_pipeline(
                cheap,
                strong,
                bridge,
                system_prompt(load_profile("zephyr"), "English"),
                opening,
                {f.path: f.new_side_lines for f in intel.files},
                changed_files=[f.path for f in intel.files],
                added_text="",
                specialists=["intent"],
                claims=claims,
                hard_case_max_steps=steps,
                log=lambda m: None,
            )

    def kconfig_finding(result) -> tuple[str, bool]:
        f = next(f for f in result.review.findings if f.title == "select of undefined I2C_BUS_RECOVERY")
        return f.severity, f.speculative

    # With budget to spare the scripted hard cases confirm X1 and refute X2, and that stands.
    settled = await run(12)
    assert {c.id: c.verdict for c in settled.review.external_claims} == {"X1": "confirmed", "X2": "refuted"}
    assert kconfig_finding(settled) == ("high", False)

    # The same verdicts from a first and last step are what the model had when the budget ran out.
    forced = await run(1)
    verdicts = {c.id: c for c in forced.review.external_claims}
    assert {k: v.verdict for k, v in verdicts.items()} == {"X1": "undecided", "X2": "undecided"}
    assert all(
        v.reason == "the verification ran out of budget before it could decide; left open" for v in verdicts.values()
    )
    # The finding is kept, as the open question it is (speculative, so at most medium), and still tied to
    # the comment it came from.
    assert kconfig_finding(forced) == ("medium", True) and verdicts["X1"].finding_id.startswith("HF")
    # The record says how each was decided; a later run does not take the refutation over.
    xmap = {c["id"]: c for c in forced.review_map["external_claims"]}
    assert (xmap["X1"]["fate"], xmap["X2"]["fate"]) == ("confirmed", "refuted")
    assert xmap["X1"]["forced"] is True and xmap["X2"]["forced"] is True
