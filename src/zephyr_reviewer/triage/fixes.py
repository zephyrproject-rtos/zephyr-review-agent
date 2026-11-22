# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Pull requests that set out to fix the issue, compared with what the triage found on its own.

They are not input to the analysis. The run first decides from the report and the tree what is wrong and
how it would be fixed; only then are the pull requests read, by a separate agent, and each is set against
that result. A fix that agrees with an analysis which never saw it is evidence for both; one that
disagrees says that one of the two is wrong, which is worth a maintainer's look either way."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..review.discussion import sanitize
from ..review.github_api import GitHub, GitHubError
from .issue import Issue

RELATIONS = ["same_fix", "same_cause_different_fix", "different_cause", "unrelated", "cannot_tell"]
SUBMIT_COMPARISON = "submit_comparison"

QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      closedByPullRequestsReferences(first: 10, includeClosedPrs: true) {
        nodes { ...pr }
      }
      timelineItems(first: 60, itemTypes: [CROSS_REFERENCED_EVENT]) {
        nodes { ... on CrossReferencedEvent { source { __typename ...pr } } }
      }
    }
  }
}
fragment pr on PullRequest {
  number title state merged isDraft createdAt url
  author { login }
  files(first: 60) { nodes { path additions deletions } }
}"""


@dataclass
class LinkedPull:
    number: int
    title: str
    state: str  # open | merged | closed
    created: str
    closes: bool  # the pull request declares that it closes the issue; otherwise it only refers to it
    url: str = ""
    author: str = ""
    files: list[tuple[str, int, int]] = field(default_factory=list)
    diff: str = ""


@dataclass
class Comparison:
    number: int
    relation: str
    assessment: str
    concerns: str = ""
    # Copied from the pull request for the comment:
    title: str = ""
    state: str = ""
    closes: bool = False


def _pull(node: dict[str, Any], closes: bool) -> LinkedPull | None:
    if not node or not node.get("number"):
        return None
    state = "merged" if node.get("merged") else str(node.get("state") or "").lower()
    return LinkedPull(
        number=int(node["number"]),
        title=sanitize(node.get("title") or ""),
        state=state,
        created=str(node.get("createdAt") or ""),
        closes=closes,
        url=str(node.get("url") or ""),
        author=str((node.get("author") or {}).get("login") or ""),
        files=[
            (f["path"], int(f.get("additions") or 0), int(f.get("deletions") or 0))
            for f in ((node.get("files") or {}).get("nodes") or [])
            if f and f.get("path")
        ],
    )


def load_linked_pulls(
    gh: GitHub, number: int, *, limit: int = 3, diff_chars: int = 30_000, log=lambda m: None
) -> list[LinkedPull]:
    """Open and merged pull requests that close the issue, then those that only refer to it, newest
    first within each; with their diffs. One abandoned without merging proposes nothing any more."""
    owner, name = gh.repo.split("/", 1)
    try:
        data = gh.graphql(QUERY, {"owner": owner, "name": name, "number": number})
    except GitHubError as exc:
        log(f"::warning::could not read the pull requests linked to #{number}: {exc}")
        return []
    issue = ((data.get("repository") or {}).get("issue")) or {}
    found: dict[int, LinkedPull] = {}
    for node in (issue.get("closedByPullRequestsReferences") or {}).get("nodes") or []:
        p = _pull(node, True)
        if p:
            found[p.number] = p
    for item in (issue.get("timelineItems") or {}).get("nodes") or []:
        src = (item or {}).get("source") or {}
        if src.get("__typename") == "PullRequest":
            p = _pull(src, False)
            if p and p.number not in found:
                found[p.number] = p
    pulls = [p for p in found.values() if p.state in ("open", "merged")]
    pulls.sort(key=lambda p: (not p.closes, p.created), reverse=False)
    pulls = sorted(pulls, key=lambda p: p.created, reverse=True)
    pulls.sort(key=lambda p: not p.closes)
    pulls = pulls[:limit]
    for p in pulls:
        try:
            diff = gh.pull_diff(p.number)
        except GitHubError as exc:
            log(f"::warning::could not read the diff of #{p.number}: {exc}")
            diff = ""
        p.diff = (
            diff if len(diff) <= diff_chars else diff[:diff_chars] + f"\n... [diff clipped to {diff_chars} characters]"
        )
    return [p for p in pulls if p.diff]


def before_the_fixes(issue: Issue, pulls: list[LinkedPull]) -> Issue:
    """The issue as it stood before the first of the pull requests was opened. A comment written after
    that may say what the fix is, and the analysis is to find the cause without being told."""
    if not pulls or not issue.comment_dates:
        return issue
    first = min(p.created for p in pulls if p.created) if any(p.created for p in pulls) else ""
    if not first:
        return issue
    keep = [i for i, when in enumerate(issue.comment_dates) if when and when < first]
    copy = Issue(**{**issue.__dict__})
    copy.comments = [issue.comments[i] for i in keep]
    copy.comment_dates = [issue.comment_dates[i] for i in keep]
    return copy


COMPARISON_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_COMPARISON,
        "description": "Submit how each pull request relates to the independent analysis. This ends the run.",
        "parameters": {
            "type": "object",
            "properties": {
                "comparisons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer", "description": "The pull request"},
                            "relation": {
                                "type": "string",
                                "enum": RELATIONS,
                                "description": "same_fix: it changes what the analysis says is wrong, in the "
                                "way the analysis would. same_cause_different_fix: it addresses the same defect "
                                "another way. different_cause: it changes something else, which may or may not "
                                "explain the report. unrelated: it does not try to fix this issue. cannot_tell: "
                                "the diff does not show enough.",
                            },
                            "assessment": {
                                "type": "string",
                                "description": "One or two sentences: what the pull request changes, and how "
                                "that stands against the cause found",
                            },
                            "concerns": {
                                "type": "string",
                                "description": "What the pull request leaves open or gets wrong with respect to "
                                "this issue, if anything; empty otherwise",
                            },
                        },
                        "required": ["number", "relation", "assessment"],
                    },
                }
            },
            "required": ["comparisons"],
        },
    },
}


def parse_comparison(args: dict[str, Any], pulls: list[LinkedPull]) -> tuple[list[Comparison] | None, list[str]]:
    """Every pull request shown gets one comparison, and none that was not shown."""
    raw = args.get("comparisons")
    if not isinstance(raw, list):
        return None, ["comparisons must be an array with one entry per pull request"]
    by_number = {p.number: p for p in pulls}
    out: dict[int, Comparison] = {}
    errors: list[str] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or c.get("number") not in by_number:
            errors.append(f"comparisons[{i}].number must be one of {', '.join(str(n) for n in by_number)}")
            continue
        relation = str(c.get("relation") or "").strip().lower()
        text = " ".join(str(c.get("assessment") or "").split())
        if relation not in RELATIONS:
            errors.append(f"comparisons[{i}].relation must be one of {', '.join(RELATIONS)}")
        elif not text:
            errors.append(f"comparisons[{i}].assessment is required")
        else:
            p = by_number[c["number"]]
            out[p.number] = Comparison(
                p.number,
                relation,
                text[:800],
                " ".join(str(c.get("concerns") or "").split())[:600],
                title=p.title,
                state=p.state,
                closes=p.closes,
            )
    missing = [n for n in by_number if n not in out]
    if missing and not errors:
        errors.append(f"no comparison for pull request(s) {', '.join(str(n) for n in missing)}")
    if errors:
        return None, errors
    return [out[p.number] for p in pulls], []
