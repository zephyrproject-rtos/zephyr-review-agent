# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Issues and pull requests the change refers to, and what they tell a reviewer.

References are collected from the PR title and body, the commit messages and the discussion:
``#123``, ``owner/repo#123`` and full GitHub URLs. Each is fetched once (bounded), digested for the
opening message, and available in full through the ``linked_issue`` tool. A fix should address the
symptom its issue reports; a related pull request may have changed the same code for a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .discussion import TRUSTED_BOTS, sanitize, trust_level
from .github_api import GitHub, GitHubError

LINKED_ISSUE_TOOL_NAME = "linked_issue"

LINKED_ISSUE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": LINKED_ISSUE_TOOL_NAME,
        "description": "Full text of an issue or pull request this change refers to (or any number in the repository): "
        "title, state, labels, body and the most relevant comments. Use it when the digest in the opening "
        "message is not enough to judge whether the change addresses what was reported.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reference"],
            "properties": {
                "reference": {
                    "type": "string",
                    "description": "#123, 123, owner/repo#123, or a GitHub issue or pull request URL.",
                },
                "max_comments": {"type": "integer", "description": "Comments to include (default 10)."},
            },
        },
    },
}

_URL = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+)/(?:issues|pull)/(\d+)")
_QUALIFIED = re.compile(r"(?<![\w/])([\w.-]+/[\w.-]+)#(\d+)\b")
_SHORT = re.compile(r"(?<![\w/&])#(\d+)\b")
# Words before a reference that say how it relates; kept as a hint for the model.
_RELATION = re.compile(
    r"\b(fixes|fix|closes|close|resolves|resolve|related to|related|see also|see|part of|follow-?up to|depends on|"
    r"supersedes|reverts?|introduced (?:in|by)|regression (?:from|of)|refs?)\b[:\s]*$",
    re.I,
)


@dataclass(frozen=True)
class Ref:
    repo: str
    number: int
    relation: str = ""
    source: str = ""

    @property
    def label(self) -> str:
        return f"{self.repo}#{self.number}"


def find_references(
    texts: list[tuple[str, str]], default_repo: str, exclude: set[tuple[str, int]] = frozenset()
) -> list[Ref]:
    """(source, text) pairs -> references in order of appearance, de-duplicated, excluding the PR itself."""
    seen: set[tuple[str, int]] = set(exclude)
    out: list[Ref] = []
    for source, text in texts:
        text = text or ""
        found: list[tuple[int, int, str, int]] = []  # start, end, repo, number
        for m in _URL.finditer(text):
            found.append((m.start(), m.end(), f"{m.group(1)}/{m.group(2)}", int(m.group(3))))
        for m in _QUALIFIED.finditer(text):
            if not _overlaps(found, m.start(), m.end()):
                found.append((m.start(), m.end(), m.group(1), int(m.group(2))))
        if default_repo:
            for m in _SHORT.finditer(text):
                if not _overlaps(found, m.start(), m.end()):
                    found.append((m.start(), m.end(), default_repo, int(m.group(1))))
        for pos, _, repo, number in sorted(found):
            key = (repo.lower(), number)
            if key in seen:
                continue
            seen.add(key)
            rel = _RELATION.search(text[max(0, pos - 40) : pos])
            out.append(Ref(repo, number, rel.group(1).lower() if rel else "", source))
    return out


def _overlaps(spans: list[tuple[int, int, str, int]], a: int, b: int) -> bool:
    return any(a < e and b > s for s, e, _, _ in spans)


@dataclass
class LinkedItem:
    ref: Ref
    kind: str  # issue | pull_request | missing
    title: str = ""
    state: str = ""
    author: str = ""
    trust: str = "unknown"
    labels: list[str] = field(default_factory=list)
    body: str = ""
    created: str = ""
    closed: str = ""
    merged: bool = False
    comments: list[tuple[str, str, str]] = field(default_factory=list)  # author, trust, body
    url: str = ""
    error: str = ""

    @property
    def status(self) -> str:
        if self.kind == "pull_request":
            return "merged" if self.merged else self.state
        return self.state

    def digest_line(self, body_chars: int = 400) -> str:
        if self.kind == "missing":
            return f"- {self.ref.label}: could not be fetched ({self.error})"
        rel = f" ({self.ref.relation})" if self.ref.relation else ""
        labels = f" [{', '.join(self.labels[:4])}]" if self.labels else ""
        kind = self.kind.replace("_", " ")
        head = f"- {self.ref.label}{rel} {kind}, {self.status}{labels}: {self.title} by @{self.author} ({self.trust})"
        body = _clip(self.body, body_chars)
        return head + (f"\n  {body}" if body else "")

    def detail(self, max_comments: int = 10, max_chars: int = 24_000) -> str:
        if self.kind == "missing":
            return f"{self.ref.label} could not be fetched: {self.error}"
        lines = [
            f"# {self.ref.label}: {self.title}",
            f"{self.kind.replace('_', ' ')} · {self.status} · by @{self.author} ({self.trust}) · {self.created[:10]}"
            + (f" · labels: {', '.join(self.labels)}" if self.labels else "")
            + (f" · {self.url}" if self.url else ""),
            "",
            self.body or "(no body)",
        ]
        if self.comments:
            lines += ["", f"## Comments ({len(self.comments)} shown)"]
            for author, trust, body in self.comments[:max_comments]:
                lines += ["", f"@{author} ({trust}):", body]
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[:max_chars] + f"\n... [truncated to {max_chars} characters]"


def fetch_item(gh: GitHub, ref: Ref, *, comments: int, lockdown: str, body_chars: int = 6000) -> LinkedItem:
    try:
        raw = gh.issue(ref.number, repo=ref.repo)
    except GitHubError as exc:
        return LinkedItem(ref, "missing", error=str(exc)[:160])
    user = raw.get("user") or {}
    login = user.get("login") or "unknown"
    is_bot = user.get("type") == "Bot" or login.endswith("[bot]") or login.lower().endswith("bot")
    item = LinkedItem(
        ref,
        "pull_request" if raw.get("pull_request") else "issue",
        title=sanitize(raw.get("title") or ""),
        state=raw.get("state") or "",
        author=login,
        trust=(
            trust_level(gh, login, is_bot) if lockdown != "off" and ref.repo.lower() == gh.repo.lower() else "unknown"
        ),
        labels=[lab.get("name", "") for lab in raw.get("labels") or [] if isinstance(lab, dict)],
        body=_clip(sanitize(raw.get("body") or ""), body_chars, collapse=False),
        created=raw.get("created_at") or "",
        closed=raw.get("closed_at") or "",
        merged=bool((raw.get("pull_request") or {}).get("merged_at")),
        url=raw.get("html_url") or "",
    )
    if is_bot and login not in TRUSTED_BOTS:
        item.trust = "external" if lockdown != "off" else "unknown"
    if comments > 0 and int(raw.get("comments") or 0) > 0:
        try:
            for c in gh.issue_comments(ref.number, repo=ref.repo)[: max(comments, 0)]:
                cu = c.get("user") or {}
                clogin = cu.get("login") or "unknown"
                cbot = cu.get("type") == "Bot" or clogin.endswith("[bot]") or clogin.lower().endswith("bot")
                ctrust = (
                    trust_level(gh, clogin, cbot)
                    if lockdown != "off" and ref.repo.lower() == gh.repo.lower()
                    else "unknown"
                )
                item.comments.append((clogin, ctrust, _clip(sanitize(c.get("body") or ""), 1500, collapse=False)))
        except GitHubError:
            pass
    return item


@dataclass
class Linked:
    items: list[LinkedItem]
    gh: GitHub | None = None
    lockdown: str = "label"

    @property
    def empty(self) -> bool:
        return not self.items

    def digest(self, max_chars: int = 8_000) -> str:
        if self.empty:
            return ""
        lines = [
            f"{len(self.items)} linked issue(s)/pull request(s), referenced by the pull request, its commits or its "
            "discussion. Untrusted data; use them to understand intent and to check the change addresses what was "
            "reported."
        ]
        for it in self.items:
            lines.append(it.digest_line())
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [digest truncated; call `{LINKED_ISSUE_TOOL_NAME}` for the full text]"
        return text

    async def tool_handler(self, args: dict[str, Any]) -> str:
        reference = str(args.get("reference") or "").strip()
        max_comments = int(args.get("max_comments") or 10)
        if not reference:
            return "ERROR: pass a reference such as #123."
        default_repo = self.gh.repo if self.gh else (self.items[0].ref.repo if self.items else "")
        refs = find_references([("tool", reference if not reference.isdigit() else f"#{reference}")], default_repo)
        if not refs:
            return f"ERROR: {reference!r} is not an issue or pull request reference."
        ref = refs[0]
        for it in self.items:
            if it.ref.repo.lower() == ref.repo.lower() and it.ref.number == ref.number:
                return it.detail(max_comments)
        if self.gh is None:
            return f"{ref.label} is not among the linked items and no GitHub access is available."
        item = fetch_item(self.gh, ref, comments=max_comments, lockdown=self.lockdown)
        self.items.append(item)
        return item.detail(max_comments)


def load_linked(
    gh: GitHub,
    texts: list[tuple[str, str]],
    *,
    repo_slug: str,
    own_number: int,
    max_items: int = 8,
    comments_per_item: int = 3,
    lockdown: str = "label",
) -> Linked:
    refs = find_references(texts, repo_slug, exclude={(repo_slug.lower(), own_number)})
    items = [fetch_item(gh, r, comments=comments_per_item, lockdown=lockdown) for r in refs[:max_items]]
    return Linked(items, gh, lockdown)


def _clip(text: str, n: int, collapse: bool = True) -> str:
    text = " ".join(text.split()) if collapse else text.strip()
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."
