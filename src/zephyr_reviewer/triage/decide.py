# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""What follows from a triage without a model: who the commit is by, which labels the tree assigns to the
file, who maintains it. The model proposes; what is applied to an issue is derived or checked here."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..intel import SourceIntel
from .issue import Issue
from .schema import Triage

LABEL_PREFIXES = ("area:", "platform:")
REGRESSION_LABEL = "Regression"


@dataclass
class Proposal:
    """What the triage would change on the issue."""

    labels: list[str] = field(default_factory=list)  # area and platform labels the issue does not have yet
    priority: str = ""  # the priority label, e.g. "priority: medium"
    maintainers: list[str] = field(default_factory=list)  # of the area to turn to first
    area: str = ""  # that area
    areas: list[str] = field(default_factory=list)  # every MAINTAINERS area that has the file, that one first
    owners: list[tuple[str, str]] = field(default_factory=list)  # (first maintainer, area), one per area
    culprit_login: str = ""  # the account of the commit's author, when known


def commit_resolver(intel: SourceIntel) -> Callable[[str], str | None]:
    """A commit named by the model counts only if it exists and is in the history under analysis."""

    def resolve(rev: str) -> str | None:
        sha = intel.head.resolve(rev)
        return sha if sha and intel.head.is_ancestor(sha) else None

    return resolve


def describe_culprit(triage: Triage, intel: SourceIntel) -> None:
    """Fill in the commit's author, date, subject and first releases from the repository."""
    k = triage.culprit
    if k is None:
        return
    info = intel.head.commit_info(k.commit)
    if info is None:
        triage.culprit = None
        return
    k.author, k.email, k.date, k.subject = info["author"], info["email"], info["date"], info["subject"]
    k.releases = intel.head.releases_containing(k.commit)[:2]


def _known(label: str, known: dict[str, str] | None) -> str:
    """The label in the repository's own spelling; any label when the repository's list is not known."""
    if known is None:
        return label
    return known.get(label.lower(), "")


def area_lines(intel: SourceIntel, paths: list[str], limit: int = 6) -> list[str]:
    """What MAINTAINERS says about files: the areas each is in and the labels of those areas."""
    m = intel.maintainers
    out: list[str] = []
    if m is None:
        return out
    for path in paths[:limit]:
        found = m.owners_for(path, is_dir=intel.head.is_dir(path))
        if found:
            parts = [f"{a.name} ({', '.join(a.labels)})" if a.labels else a.name for a, _ in found]
            out.append(f"- `{path}`: " + "; ".join(parts))
    return out


def propose(triage: Triage, intel: SourceIntel, issue: Issue, repo_labels: list[str] | None = None) -> Proposal:
    """How the issue is filed follows from MAINTAINERS for the file that has the defect: its areas, their
    labels, and whom to turn to, in the order get_maintainer would. A label the model adds counts only if
    MAINTAINERS defines it for some area; none is made up, and a label the repository does not have is
    dropped. Nothing is proposed for a report without a located cause: where the defect is decides how
    it is filed."""
    p = Proposal()
    if not triage.located:
        return p
    known = {lab.lower(): lab for lab in repo_labels} if repo_labels is not None else None
    have = {lab.lower() for lab in issue.labels}
    labels: list[str] = []
    m = intel.maintainers
    defined: set[str] = set()
    if m is not None:
        defined = {lab.lower() for area in m.areas for lab in area.labels}
        for area, _groups in m.owners_for(triage.cause.file):
            labels += area.labels
            p.areas.append(area.name)
            if area.maintainers:
                if not p.maintainers:
                    p.maintainers, p.area = list(area.maintainers), area.name
                if area.maintainers[0] not in [who for who, _ in p.owners]:
                    p.owners.append((area.maintainers[0], area.name))
    labels += [lab for lab in triage.labels if lab.lower() in defined]
    if triage.regression and triage.culprit is not None:
        labels.append(REGRESSION_LABEL)
    for lab in labels:
        name = _known(lab, known)
        if name and name.lower() not in have and name not in p.labels:
            p.labels.append(name)
    if triage.priority and not any(lab.startswith("priority:") for lab in have):
        p.priority = _known(f"priority: {triage.priority}", known)
    return p
