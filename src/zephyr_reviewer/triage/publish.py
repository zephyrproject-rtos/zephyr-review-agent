# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""What a triage run may do to the issue, each action allowed separately and none by default."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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


@dataclass
class Plan:
    """What a run would change on the issue, before it does."""

    labels: list[str] = field(default_factory=list)
    priority: str = ""
    assignees: list[tuple[str, str]] = field(default_factory=list)  # (login, why), the first is the default
    withheld: list[str] = field(default_factory=list)  # what is not offered, and why

    @property
    def empty(self) -> bool:
        return not (self.labels or self.priority or self.assignees)


def plan(
    gh: GitHub,
    issue: Issue,
    triage: Triage,
    proposal: Proposal,
    actions: list[str],
    *,
    assign_to: str = "maintainer",
    attended: bool = False,
) -> Plan:
    """The changes ``actions`` allow and the triage supports.

    Labels need a located cause. Unattended, a priority and an assignee need a confirmed one: a guess
    about where the defect is can be filed under its area, and should not rank the issue or hand it to
    someone. The author of the introducing commit is a candidate only when GitHub would accept them, and
    the default only when asked for and the commit is named with high confidence; otherwise the area's
    maintainer is. An issue that already has an assignee keeps it.

    ``attended`` is a run where a person confirms each change: a likely cause is then enough to offer a
    priority and an assignee, and the author is offered beside the maintainer, since the person decides."""
    out = Plan()
    if not triage.located:
        return out
    confirmed = triage.verdict == "confirmed" and not triage.provisional
    if "labels" in actions:
        out.labels = list(proposal.labels)
    if "priority" in actions and proposal.priority:
        if confirmed or attended:
            out.priority = proposal.priority
        else:
            out.withheld.append("no priority is set on a cause that is not confirmed")
    if "assign" in actions:
        if issue.assignees:
            out.withheld.append(f"already assigned to {', '.join(issue.assignees)}")
        elif not (confirmed or attended):
            out.withheld.append("nobody is assigned on a cause that is not confirmed")
        else:
            k = triage.culprit
            author = proposal.culprit_login if k is not None else ""
            maintainer = proposal.maintainers[0] if proposal.maintainers else ""
            first_author = assign_to == "author" and k is not None and k.confidence == "high"
            candidates = []
            if maintainer:
                candidates.append((maintainer, f"maintainer of {proposal.area}" if proposal.area else "maintainer"))
            if author and author != maintainer and (attended or first_author):
                candidates.append((author, f"author of {k.commit[:12]}"))
            if first_author and author:
                candidates.sort(key=lambda c: c[0] != author)
            out.assignees = [c for c in candidates if gh.assignable(c[0])]
            if candidates and not out.assignees:
                out.withheld.append("GitHub accepts none of the candidates as an assignee")
            if not attended:
                out.assignees = out.assignees[:1]
    return out


def carry_out(gh: GitHub, issue: Issue, p: Plan, *, log: Callable[[str], None] = print) -> dict[str, list[str]]:
    """Apply a plan; returns what was done. A failure is logged, not raised: the comment is still worth
    posting. Of several candidate assignees only the first is assigned."""
    done: dict[str, list[str]] = {"labels": [], "assignees": []}
    labels = [*p.labels, *([p.priority] if p.priority else [])]
    if labels:
        try:
            gh.add_labels(issue.number, labels)
            done["labels"] = labels
            log(f"triage: added label(s) {', '.join(labels)}")
        except GitHubError as exc:
            log(f"::warning::could not add labels: {exc}")
    who = [login for login, _ in p.assignees[:1]]
    if who:
        try:
            gh.add_assignees(issue.number, who)
            done["assignees"] = who
            log(f"triage: assigned {', '.join(who)}")
        except GitHubError as exc:
            log(f"::warning::could not assign: {exc}")
    return done


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
    """Plan and carry out without asking anyone."""
    return carry_out(gh, issue, plan(gh, issue, triage, proposal, actions, assign_to=assign_to), log=log)
