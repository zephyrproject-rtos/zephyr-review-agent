# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The prompts of a triage run: the working rules, the two parts, and the opening message."""

from __future__ import annotations

from importlib import resources

from .fixes import LinkedPull
from .issue import Hints, Issue
from .schema import Cause, Triage


def _profile(name: str) -> str:
    return resources.files("zephyr_reviewer.triage").joinpath(f"profiles/{name}.md").read_text(encoding="utf-8")


def system_prompt(part: str, language: str = "English") -> str:
    """The working rules, then the part of this agent (``locate`` or ``confirm``)."""
    return _profile("triage").format(language=language).rstrip() + "\n\n---\n\n" + _profile(part).rstrip() + "\n"


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"\n... [clipped to {n} characters]"


def opening_message(
    issue: Issue,
    hints: Hints,
    commit: str,
    *,
    repo: str = "",
    labels: list[str] | None = None,
    body_chars: int = 16_000,
    comment_chars: int = 1_500,
    max_comments: int = 5,
    with_comments: bool = True,
) -> str:
    """The report, delimited as the untrusted text it is, and what it names that the tree knows."""
    out = [
        f"# Issue #{issue.number}" + (f" in {repo}" if repo else ""),
        "",
        f"Analysed against commit `{commit[:12]}`. The tools answer for that commit.",
        "",
        "<issue>",
        f"Title: {issue.title}",
        "",
        _clip(issue.body, body_chars) or "(no description)",
        "</issue>",
    ]
    facts = []
    if issue.claims_regression:
        facts.append("- The reporter marked it as a regression.")
    if issue.impact:
        facts.append(f"- Impact chosen by the reporter: {issue.impact}")
    found = hints.lines()
    if facts or found:
        out += ["", "## What the report says and names", "", *facts, *found]
    if with_comments and issue.comments:
        out += ["", "## Comments on the issue", "", "<issue_comments>"]
        for who, text in issue.comments[:max_comments]:
            out += [f"[{who}]", _clip(text, comment_chars), ""]
        out += ["</issue_comments>"]
    if labels:
        out += ["", "## Labels this repository has for areas and platforms", "", ", ".join(labels)]
    return "\n".join(out)


def causes_section(causes: list[Cause], not_a_bug: list[str], missing: list[str], drafters: int) -> str:
    """What the first pass returned, for the agent that decides."""
    out = ["", "## Candidate causes from the first pass", ""]
    if not causes:
        out.append("None. The first pass found no code that explains the report.")
    for i, c in enumerate(causes, 1):
        where = f"{c.file}:{c.line}" if c.line else c.file
        agreed = f" (raised by {len(c.raised_by)} of {drafters})" if drafters > 1 else ""
        out.append(f"{i}. **{c.title}** at `{where}`, confidence {c.confidence}{agreed}")
        out.append(f"   {c.explanation}")
        if c.evidence:
            out.append(f"   Read: {c.evidence}")
    if not_a_bug:
        out += ["", "A first pass thinks this is not a defect in the tree:"] + [f"- {t}" for t in not_a_bug]
    if missing:
        out += ["", "A first pass found the report lacking:"] + [f"- {t}" for t in missing]
    return "\n".join(out)


def comparison_message(
    issue: Issue, triage: Triage, pulls: list[LinkedPull], commit: str, *, body_chars: int = 6_000
) -> str:
    """The report, the independent analysis, and the pull requests to set against it."""
    out = [
        f"# Issue #{issue.number}: proposed fixes",
        "",
        f"Analysed against commit `{commit[:12]}`. The tools answer for that commit.",
        "",
        "<issue>",
        f"Title: {issue.title}",
        "",
        _clip(issue.body, body_chars) or "(no description)",
        "</issue>",
        "",
        "## The independent analysis",
        "",
        f"Verdict: {triage.verdict}, confidence {triage.confidence}"
        + (" (submitted when its budget ran out)" if triage.provisional else ""),
        "",
        triage.summary,
    ]
    c = triage.cause
    if triage.located and c is not None:
        out += ["", f"Cause: `{c.file}:{c.line}`" if c.line else f"Cause: `{c.file}`", c.explanation]
        if triage.fix:
            out += ["", "Fix it would make: " + triage.fix]
        if triage.patch:
            out += ["", "```diff", triage.patch, "```"]
    else:
        out += ["", "It located no cause."]
    out += ["", f"## Pull requests ({len(pulls)})"]
    for p in pulls:
        how = "declares that it closes this issue" if p.closes else "refers to this issue"
        out += [
            "",
            f"### Pull request #{p.number} ({p.state}; {how})",
            "",
            "<pull_request>",
            f"Title: {p.title}",
            "Files: " + ", ".join(f"{path} (+{a} -{d})" for path, a, d in p.files[:30]),
            "",
            p.diff,
            "</pull_request>",
        ]
    return "\n".join(out)
