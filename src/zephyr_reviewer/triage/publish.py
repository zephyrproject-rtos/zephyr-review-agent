# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""What a triage run may do to the issue, each action allowed separately and none by default."""

from __future__ import annotations

from collections.abc import Callable

from ..review.github_api import GitHub, GitHubError
from .decide import Proposal
from .issue import Issue
from .schema import Triage

ACTIONS = ("comment", "labels", "priority", "assign")
ASSIGN_TO = ("maintainer", "author")


def parse_actions(value: str) -> tuple[list[str], list[str]]:
    """(actions, errors) from a comma-separated list; "all" is every action, "none" or empty is no action."""
    names = [x.strip().lower() for x in (value or "").split(",") if x.strip()]
    if names in (["none"], []):
        return [], []
    if names == ["all"]:
        return list(ACTIONS), []
    bad = [n for n in names if n not in ACTIONS]
    return [n for n in ACTIONS if n in names], [
        f"unknown triage action {b!r}; one of {', '.join(ACTIONS)}" for b in bad
    ]


def apply(
    gh: GitHub,
    issue: Issue,
    triage: Triage,
    proposal: Proposal,
    actions: list[str],
    *,
    assign_to: str = "maintainer",
    log: Callable[[str], None] = print,
) -> dict[str, list[str]]:
    """Label and assign as far as ``actions`` allow and the triage supports; returns what was done.

    Labels need a located cause. A priority and an assignee need a confirmed one: a guess about where the
    defect is can be filed under its area, and should not rank the issue or hand it to someone. The
    author of the introducing commit is assigned only when that commit is named with high confidence and
    GitHub would accept them; otherwise, and by default, the area's maintainer. An issue that already has
    an assignee keeps it. A failure here is logged, not raised: the comment is still worth posting."""
    done: dict[str, list[str]] = {"labels": [], "assignees": []}
    if not triage.located:
        return done
    confirmed = triage.verdict == "confirmed" and not triage.provisional
    labels = list(proposal.labels) if "labels" in actions else []
    if "priority" in actions and confirmed and proposal.priority:
        labels.append(proposal.priority)
    if labels:
        try:
            gh.add_labels(issue.number, labels)
            done["labels"] = labels
            log(f"triage: added label(s) {', '.join(labels)}")
        except GitHubError as exc:
            log(f"::warning::could not add labels: {exc}")
    if "assign" in actions and confirmed and not issue.assignees:
        who: list[str] = []
        k = triage.culprit
        if assign_to == "author" and k is not None and k.confidence == "high" and proposal.culprit_login:
            who = [proposal.culprit_login]
        who = who or proposal.maintainers[:1]
        who = [w for w in who if gh.assignable(w)]
        if who:
            try:
                gh.add_assignees(issue.number, who)
                done["assignees"] = who
                log(f"triage: assigned {', '.join(who)}")
            except GitHubError as exc:
                log(f"::warning::could not assign: {exc}")
    return done
