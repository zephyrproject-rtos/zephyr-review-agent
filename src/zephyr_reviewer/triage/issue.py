# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""A reported issue as the triage reads it: the sections of the bug form, and what the text points at."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..intel.repo import GitRepo
from ..review.discussion import sanitize

# Headings of Zephyr's bug report form, and the key each is kept under.
SECTIONS = {
    "describe the bug": "description",
    "regression": "regression",
    "steps to reproduce": "steps",
    "relevant log output": "logs",
    "impact": "impact",
    "environment": "environment",
    "additional context": "context",
}
NO_RESPONSE = "_no response_"
BUG_LABELS = {"bug"}
NOT_A_BUG_LABELS = {"enhancement", "feature request", "feature", "rfc", "question", "hardware support"}


@dataclass
class Issue:
    number: int
    title: str
    body: str
    author: str = ""
    labels: list[str] = field(default_factory=list)
    assignees: list[str] = field(default_factory=list)
    state: str = "open"
    kind: str = ""  # GitHub's issue type ("Bug", "Feature", ...), when the repository uses them
    url: str = ""
    is_pull_request: bool = False
    comments: list[tuple[str, str]] = field(default_factory=list)  # (author, text), people only
    comment_dates: list[str] = field(default_factory=list)  # when each of them was written

    @classmethod
    def from_api(cls, d: dict[str, Any], comments: list[dict[str, Any]] | None = None) -> Issue:
        people, dates = [], []
        for c in comments or []:
            user = c.get("user") or {}
            if user.get("type") == "Bot" or str(user.get("login") or "").endswith("[bot]"):
                continue
            people.append((str(user.get("login") or ""), sanitize(c.get("body") or "")))
            dates.append(str(c.get("created_at") or ""))
        return cls(
            number=int(d.get("number") or 0),
            title=sanitize(d.get("title") or ""),
            body=sanitize(d.get("body") or ""),
            author=str((d.get("user") or {}).get("login") or ""),
            labels=[str(lab.get("name") if isinstance(lab, dict) else lab) for lab in d.get("labels") or []],
            assignees=[str(a.get("login") or "") for a in d.get("assignees") or []],
            state=str(d.get("state") or "open"),
            kind=str((d.get("type") or {}).get("name") or "") if isinstance(d.get("type"), dict) else "",
            url=str(d.get("html_url") or ""),
            is_pull_request="pull_request" in d,
            comments=people,
            comment_dates=dates,
        )

    @property
    def sections(self) -> dict[str, str]:
        return split_sections(self.body)

    def is_bug_report(self) -> bool:
        """A report of a defect: typed or labelled as one, or written on the bug form. A feature request,
        an RFC or a question is not triaged."""
        labels = {lab.lower() for lab in self.labels}
        if self.kind:
            return self.kind.lower() == "bug"
        if labels & BUG_LABELS:
            return True
        if labels & NOT_A_BUG_LABELS:
            return False
        return "description" in self.sections

    @property
    def claims_regression(self) -> bool:
        return bool(re.search(r"\[[xX]\]\s*This is a regression", self.body))

    @property
    def impact(self) -> str:
        """The impact the reporter chose on the form, without its explanation."""
        text = self.sections.get("impact", "")
        return re.split(r"\s+[–-]\s+", text.strip(), maxsplit=1)[0].strip()


_HEADING = re.compile(r"^#{2,4}\s+(.+?)\s*$", re.M)


def split_sections(body: str) -> dict[str, str]:
    """The answers of the bug form by key; an unanswered field is left out."""
    out: dict[str, str] = {}
    marks = list(_HEADING.finditer(body or ""))
    for i, m in enumerate(marks):
        key = SECTIONS.get(m.group(1).strip().lower())
        if not key:
            continue
        text = body[m.end() : marks[i + 1].start() if i + 1 < len(marks) else len(body)].strip()
        if text and text.lower() != NO_RESPONSE:
            out[key] = text
    return out


@dataclass
class Hints:
    """What the report names that the tree knows: where to start looking, found without a model."""

    paths: list[str] = field(default_factory=list)
    configs: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)  # ids and release tags that resolve, as written
    boards: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = []
        for name, values in (
            ("Files named that exist in the tree", self.paths),
            ("Kconfig symbols named", self.configs),
            ("Functions named", self.functions),
            ("Commits and releases named", self.commits),
            ("Boards named", self.boards),
        ):
            if values:
                out.append(f"- {name}: " + ", ".join(f"`{v}`" for v in values))
        return out


_PATH = re.compile(r"(?<![\w.-])/?((?:[\w.+-]+/)+[\w.+-]+\.[A-Za-z][\w]{0,8})(?::\d+)?")
_CONFIG = re.compile(r"\bCONFIG_[A-Z0-9_]{3,}\b")
_CALL = re.compile(r"\b([a-z_][a-z0-9_]{3,})\s*\(")
_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
_VERSION = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?(?:-rc\d+)?)\b")
_BOARD = re.compile(r"(?:-b|--board|board[:=]?)\s+`?([a-z0-9][a-z0-9_]+(?:/[a-z0-9_]+)*)", re.I)
_NOT_FUNCTIONS = {"main", "printf", "printk", "sizeof", "defined", "while", "return", "switch"}


def find_hints(issue: Issue, repo: GitRepo, limit: int = 12) -> Hints:
    text = issue.title + "\n" + issue.body
    files = repo.fileset()

    def first(seq) -> list[str]:
        return list(dict.fromkeys(seq))[:limit]

    paths = []
    for m in _PATH.finditer(text):
        p = m.group(1)
        # A path as a build log prints it: absolute, or relative to a workspace above the tree.
        while p and p not in files and "/" in p:
            p = p.split("/", 1)[1]
        if p in files:
            paths.append(p)
    commits = [s for s in _SHA.findall(text) if not s.isdigit() and repo.resolve(s)]
    for v in _VERSION.findall(text):
        if repo.resolve("v" + v):
            commits.append("v" + v)
    boards = [
        b
        for b in _BOARD.findall(text)
        if repo.is_dir("boards") and any(f"/{b.split('/')[0]}/" in f for f in files if f.startswith("boards/"))
    ]
    return Hints(
        paths=first(paths),
        configs=first(_CONFIG.findall(text)),
        functions=first(f for f in _CALL.findall(text) if f not in _NOT_FUNCTIONS),
        commits=first(commits),
        boards=first(boards),
    )
