# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""A triage run: draft models look for candidate causes in parallel, one model decides and files it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..review.agent import AgentResult, LocalTool, TerminalTool, run_agent
from ..review.mcp_client import ToolBridge
from ..review.openrouter import OpenRouter, Usage
from ..review.render import normalize_title
from .fixes import COMPARISON_TOOL, Comparison, LinkedPull, parse_comparison
from .issue import Hints, Issue
from .prompts import causes_section, comparison_message, opening_message, system_prompt
from .schema import Cause, Triage, causes_terminal, triage_terminal

# Tools of the source intelligence that are about a pull request's change. An issue has no change.
PR_ONLY_TOOLS = {"pr_overview", "commit_messages", "file_diff", "read_file_at_base", "test_candidates"}
NEAR_LINES = 8  # two candidates in one file this close together are the same candidate


@dataclass
class TriageRun:
    triage: Triage | None
    causes: list[Cause] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    models_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    passes: dict[str, int] = field(default_factory=dict)
    comparisons: list[Comparison] = field(default_factory=list)  # linked pull requests against the triage


def _record(stage: str, res: AgentResult) -> dict[str, Any]:
    return {
        "stage": stage,
        "models": res.models_used,
        "steps": res.steps,
        "tool_calls": res.tool_call_count,
        "forced": res.forced,
        "submitted": res.output is not None,
        "seconds": round(res.seconds, 1),
        "usage": res.usage.as_dict(),
    }


def merge_cause(causes: list[Cause], new: Cause, by: str) -> None:
    """Fold a candidate into the list: the same place in the same file, or the same title, is one."""
    for c in causes:
        same_place = c.file == new.file and c.line and new.line and abs(c.line - new.line) <= NEAR_LINES
        same_title = c.file == new.file and normalize_title(c.title) == normalize_title(new.title)
        if same_place or same_title:
            if by not in c.raised_by:
                c.raised_by.append(by)
            if new.evidence and new.evidence not in c.evidence:
                c.evidence = (c.evidence + " | " + new.evidence).strip(" |")
            if ["low", "medium", "high"].index(new.confidence) > ["low", "medium", "high"].index(c.confidence):
                c.confidence = new.confidence
            return
    new.raised_by = [by]
    causes.append(new)


async def run_triage(
    draft_llms: list[OpenRouter],
    verify_llm: OpenRouter,
    bridge: ToolBridge,
    issue: Issue,
    hints: Hints,
    commit: str,
    *,
    exists: Callable[[str], bool],
    resolve: Callable[[str], str | None],
    repo: str = "",
    labels: list[str] | None = None,
    extra_tools: list[LocalTool] | None = None,
    language: str = "English",
    with_comments: bool = True,
    pulls: list[LinkedPull] | None = None,
    locate_max_steps: int = 20,
    confirm_max_steps: int = 25,
    compare_max_steps: int = 15,
    max_context_tokens: int = 160_000,
    max_seconds: float = 0,
    log=print,
    quiet: bool = False,
) -> TriageRun:
    """``draft_llms`` may be empty: the deciding model then looks for the cause itself.

    ``pulls`` are pull requests that set out to fix the issue. Nothing of them reaches the agents that
    look for the cause and decide: they are read afterwards, by an agent of their own, and set against
    the triage as it then stands. The triage is not changed by what they say."""
    started = time.monotonic()
    bridge.tools = [t for t in bridge.tools if t["function"]["name"] not in PR_ONLY_TOOLS]
    run = TriageRun(None)
    opening = opening_message(issue, hints, commit, repo=repo, labels=labels, with_comments=with_comments)
    common = dict(
        max_context_tokens=max_context_tokens, max_seconds=max_seconds, extra_tools=extra_tools, log=log, quiet=quiet
    )

    def took(stage: str, res: AgentResult) -> None:
        run.usage.merge(res.usage)
        run.models_used += [m for m in res.models_used if m not in run.models_used]
        run.transcript += [{**t, "stage": stage} for t in res.transcript]
        run.stages.append(_record(stage, res))

    not_a_bug: list[str] = []
    missing: list[str] = []
    if draft_llms:
        log(f"triage: looking for candidate causes on {', '.join(llm.models[0] for llm in draft_llms)}")
        results = await asyncio.gather(
            *(
                run_agent(
                    llm,
                    bridge,
                    system_prompt("locate", language),
                    opening,
                    causes_terminal(exists),
                    max_steps=locate_max_steps,
                    label=f"locate@{llm.models[0]}",
                    **common,
                )
                for llm in draft_llms
            )
        )
        lost = 0
        for llm, res in zip(draft_llms, results, strict=True):
            label = f"locate@{llm.models[0]}"
            took(label, res)
            if res.output is None:
                lost += 1
                run.errors.append(f"{label} did not submit: {'; '.join(res.errors)}")
                log(f"::warning::triage pass {label} returned no result: {'; '.join(res.errors) or 'no reason given'}")
                continue
            for c in res.output["causes"]:
                merge_cause(run.causes, c, llm.models[0])
            if res.output["not_a_bug"]:
                not_a_bug.append(res.output["not_a_bug"])
            missing += [m for m in res.output["missing"] if m not in missing]
        run.passes = {"planned": len(draft_llms), "lost": lost}
        log(f"triage: {len(run.causes)} candidate cause(s) from {len(draft_llms) - lost} of {len(draft_llms)} pass(es)")

    message = opening
    if draft_llms:
        message += "\n" + causes_section(run.causes, not_a_bug, missing, len(draft_llms))
    res = await run_agent(
        verify_llm,
        bridge,
        system_prompt("confirm", language),
        message,
        triage_terminal(exists, resolve),
        max_steps=confirm_max_steps,
        label="confirm",
        **common,
    )
    took("confirm", res)
    run.seconds = time.monotonic() - started
    if res.output is None:
        run.errors.append(f"confirm did not submit: {'; '.join(res.errors)}")
        return run
    triage: Triage = res.output
    if res.forced:
        # Submitted because the budget ran out: the path was not read to its end, so nothing is confirmed.
        triage.provisional = True
        if triage.verdict == "confirmed":
            triage.verdict = "plausible"
    if triage.cause is not None:
        match = next((c for c in run.causes if c.file == triage.cause.file), None)
        triage.cause.raised_by = list(match.raised_by) if match else []
    run.triage = triage
    if pulls:
        log(f"triage: comparing {len(pulls)} linked pull request(s) with the result")
        res = await run_agent(
            verify_llm,
            bridge,
            system_prompt("compare", language),
            comparison_message(issue, triage, pulls, commit),
            TerminalTool(
                COMPARISON_TOOL,
                lambda args: parse_comparison(args, pulls),
                lambda out: f"{len(out)} pull request(s) compared",
            ),
            max_steps=compare_max_steps,
            label="compare",
            **common,
        )
        took("compare", res)
        if res.output is None:
            # The triage stands without it; the comment then says nothing about the pull requests.
            run.errors.append(f"compare did not submit: {'; '.join(res.errors)}")
            log(f"::warning::the linked pull requests were not compared: {'; '.join(res.errors) or 'no reason given'}")
        else:
            run.comparisons = res.output
        run.seconds = time.monotonic() - started
    return run
