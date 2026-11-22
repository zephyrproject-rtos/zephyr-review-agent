# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Measure the triage against issues the project has already closed with a fix.

A closed bug with a merged pull request says where the defect was (the files the fix touched), how
maintainers filed it (labels, priority) and, by blame of the lines the fix removed, which commit
introduced it. Each such issue is triaged against the tree as it was just before the fix, from the
report alone: no comments, no labels, nothing a triager would not have had on the day it was opened."""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..benchmark.mine import culprit_by_blame
from ..intel import SourceIntel
from ..mcp_server import build_server
from ..review.github_api import GitHub
from ..review.mcp_client import ToolBridge
from ..review.openrouter import OpenRouter
from .decide import LABEL_PREFIXES, commit_resolver, describe_culprit, propose
from .issue import Issue, find_hints
from .pipeline import run_triage

QUERY = """
query($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Issue {
        number title body createdAt closedAt
        labels(first: 30) { nodes { name } }
        assignees(first: 5) { nodes { login } }
        closedByPullRequestsReferences(first: 3) {
          nodes {
            number merged
            mergeCommit { oid }
            commits { totalCount }
            files(first: 100) { nodes { path } }
          }
        }
      }
    }
  }
}"""


@dataclass
class Entry:
    number: int
    title: str
    body: str
    labels: list[str]  # area and platform labels maintainers gave it
    priority: str  # low | medium | high | ""
    assignees: list[str]
    fix_pr: int
    fix_commit: str  # the last commit of the fix on the main branch
    before: str  # the commit just before the fix: the tree the report was about
    fix_files: list[str]
    culprits: list[str] = field(default_factory=list)  # commits the fix's removed lines are blamed on


def _git(root: str, *args: str) -> str:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=False).stdout.strip()


def entry_from(node: dict[str, Any], root: str) -> Entry | None:
    """An entry when the issue has one merged fix whose commits are in the clone."""
    prs = [p for p in ((node.get("closedByPullRequestsReferences") or {}).get("nodes") or []) if p and p.get("merged")]
    if len(prs) != 1 or not (prs[0].get("mergeCommit") or {}).get("oid"):
        return None
    pr = prs[0]
    last = pr["mergeCommit"]["oid"]
    count = int((pr.get("commits") or {}).get("totalCount") or 1)
    before = _git(root, "rev-parse", "--verify", "--quiet", f"{last}~{count}^{{commit}}")
    files = [f["path"] for f in ((pr.get("files") or {}).get("nodes") or []) if f and f.get("path")]
    if not before or not files:
        return None
    labels = [n["name"] for n in ((node.get("labels") or {}).get("nodes") or [])]
    culprits: list[str] = []
    for sha in _git(root, "rev-list", f"{before}..{last}").splitlines():
        found = culprit_by_blame(root, sha)
        if found and found[1] >= 0.5 and found[0] not in culprits:
            culprits.append(found[0])
    return Entry(
        number=int(node["number"]),
        title=node.get("title") or "",
        body=node.get("body") or "",
        labels=sorted(lab for lab in labels if lab.lower().startswith(LABEL_PREFIXES)),
        priority=next(
            (lab.split(":", 1)[1].strip().lower() for lab in labels if lab.lower().startswith("priority:")), ""
        ),
        assignees=[a["login"] for a in ((node.get("assignees") or {}).get("nodes") or [])],
        fix_pr=int(pr["number"]),
        fix_commit=last,
        before=before,
        fix_files=files,
        culprits=culprits,
    )


def mine(
    gh: GitHub, root: str, *, since: str = "", which: str = "type:Bug", max_entries: int = 100, log=print
) -> list[Entry]:
    """``which`` is the search qualifier that selects bug reports: Zephyr types its issues, so "type:Bug";
    a repository that labels them would use "label:bug"."""
    q = f"repo:{gh.repo} is:issue is:closed reason:completed {which} linked:pr sort:updated-desc"
    if since:
        q += f" closed:>={since}"
    entries: list[Entry] = []
    after: str | None = None
    seen = skipped = 0
    for _ in range(20):
        conn = (gh.graphql(QUERY, {"q": q, "after": after}) or {}).get("search") or {}
        for node in conn.get("nodes") or []:
            if not node or "number" not in node:
                continue
            seen += 1
            e = entry_from(node, root)
            if e is None:
                skipped += 1
                continue
            entries.append(e)
            if len(entries) >= max_entries:
                break
        info = conn.get("pageInfo") or {}
        if len(entries) >= max_entries or not info.get("hasNextPage"):
            break
        after = info.get("endCursor")
    log(f"{len(entries)} of {seen} closed issues usable ({skipped} without one merged fix in this clone)")
    return entries


def save(entries: list[Entry], path: str) -> None:
    Path(path).write_text(json.dumps([asdict(e) for e in entries], indent=2) + "\n", encoding="utf-8")


def load(path: str) -> list[Entry]:
    return [Entry(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def score(entry: Entry, record: dict[str, Any]) -> dict[str, Any]:
    """One triage against what the project did. ``record`` is the run's triage.json content."""
    t = record.get("triage") or {}
    cause = t.get("cause") or {}
    located = t.get("verdict") in ("confirmed", "plausible") and bool(cause.get("file"))
    path = cause.get("file") or ""
    dirs = {f.rsplit("/", 1)[0] for f in entry.fix_files if "/" in f}
    proposed = set((record.get("proposal") or {}).get("labels") or [])
    truth = set(entry.labels)
    culprit = (t.get("culprit") or {}).get("commit") or ""
    return {
        "issue": entry.number,
        "completed": bool(t),
        "verdict": t.get("verdict") or "",
        "located": located,
        "file_hit": located and path in entry.fix_files,
        "dir_hit": located and ("/" in path and path.rsplit("/", 1)[0] in dirs),
        "labels_truth": len(truth),
        "labels_proposed": len(proposed),
        "labels_right": len(proposed & truth),
        "priority_known": bool(entry.priority and t.get("priority")),
        "priority_hit": bool(entry.priority) and t.get("priority") == entry.priority,
        "culprit_known": bool(entry.culprits),
        "culprit_named": bool(culprit),
        "culprit_hit": bool(culprit) and culprit in entry.culprits,
        "cost": float((record.get("usage") or {}).get("cost") or 0.0),
        "seconds": float(record.get("seconds") or 0.0),
    }


def report(scores: list[dict[str, Any]]) -> dict[str, Any]:
    done = [s for s in scores if s["completed"]]
    n = len(done)

    def share(key: str, among: list[dict[str, Any]] | None = None) -> float:
        among = done if among is None else among
        return round(100 * sum(1 for s in among if s[key]) / len(among), 1) if among else 0.0

    with_culprit = [s for s in done if s["culprit_known"]]
    named = [s for s in with_culprit if s["culprit_named"]]
    with_priority = [s for s in done if s["priority_known"]]
    proposed = sum(s["labels_proposed"] for s in done)
    truth = sum(s["labels_truth"] for s in done)
    right = sum(s["labels_right"] for s in done)
    return {
        "issues": len(scores),
        "completed": n,
        "verdicts": {v: sum(1 for s in done if s["verdict"] == v) for v in sorted({s["verdict"] for s in done})},
        "located_pct": share("located"),
        "file_hit_pct": share("file_hit"),
        "dir_hit_pct": share("dir_hit"),
        "file_hit_of_located_pct": share("file_hit", [s for s in done if s["located"]]),
        "label_precision_pct": round(100 * right / proposed, 1) if proposed else 0.0,
        "label_recall_pct": round(100 * right / truth, 1) if truth else 0.0,
        "priority_compared": len(with_priority),
        "priority_hit_pct": share("priority_hit", with_priority),
        "culprit_known": len(with_culprit),
        "culprit_named": len(named),
        "culprit_hit_of_named_pct": share("culprit_hit", named),
        "cost_total": round(sum(s["cost"] for s in done), 2),
        "cost_median": round(statistics.median([s["cost"] for s in done]), 3) if done else 0.0,
        "seconds_median": round(statistics.median([s["seconds"] for s in done]), 1) if done else 0.0,
    }


def render(r: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"issues: {r['issues']}, triaged: {r['completed']}, verdicts: {r['verdicts']}",
            f"cause located: {r['located_pct']}%; in a file the fix touched: {r['file_hit_pct']}% of all, "
            f"{r['file_hit_of_located_pct']}% of those located; in a directory the fix touched: {r['dir_hit_pct']}%",
            f"labels: {r['label_precision_pct']}% of those proposed were given, "
            f"{r['label_recall_pct']}% of those given were proposed",
            f"priority: same as given in {r['priority_hit_pct']}% of {r['priority_compared']} compared",
            f"introducing commit: named in {r['culprit_named']} of {r['culprit_known']} with a known one, "
            f"right in {r['culprit_hit_of_named_pct']}% of those named",
            f"cost: ${r['cost_total']} in all, ${r['cost_median']} median; {r['seconds_median']}s median",
        ]
    )


async def triage_entry(
    entry: Entry, root: str, drafters: list[OpenRouter], decider: OpenRouter, out: Path, log=print, **budgets: Any
) -> dict[str, Any]:
    """Triage one closed issue against the tree before its fix, and save the record next to its score."""
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "triage.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    issue = Issue(
        entry.number, entry.title, Issue.from_api({"number": entry.number, "body": entry.body}).body, labels=["bug"]
    )
    intel = SourceIntel(root, entry.before, entry.before)
    async with ToolBridge(build_server(intel, history=True)) as bridge:
        run = await run_triage(
            drafters,
            decider,
            bridge,
            issue,
            find_hints(issue, intel.head),
            intel.head.sha,
            exists=intel.head.exists,
            resolve=commit_resolver(intel),
            with_comments=False,
            log=log,
            quiet=True,
            **budgets,
        )
    proposal = None
    if run.triage is not None:
        describe_culprit(run.triage, intel)
        proposal = propose(run.triage, intel, issue, None)
    record = {
        "issue": entry.number,
        "commit": entry.before,
        "triage": run.triage.to_dict() if run.triage else None,
        "proposal": asdict(proposal) if proposal else {},
        "errors": run.errors,
        "passes": run.passes,
        "seconds": round(run.seconds, 1),
        "usage": run.usage.as_dict(),
        "stages": run.stages,
    }
    saved.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


async def run_benchmark(
    entries: list[Entry],
    root: str,
    drafters: list[OpenRouter],
    decider: OpenRouter,
    out_dir: str,
    *,
    concurrency: int = 2,
    log=print,
    **budgets: Any,
) -> dict[str, Any]:
    """Triage every entry, at most ``concurrency`` at a time; a finished entry is not run again."""
    gate = asyncio.Semaphore(max(1, concurrency))
    base = Path(out_dir)

    async def one(e: Entry) -> dict[str, Any]:
        async with gate:
            try:
                record = await triage_entry(e, root, drafters, decider, base / str(e.number), log, **budgets)
            except Exception as exc:  # noqa: BLE001 - one entry failing must not lose the others
                record = {"triage": None, "errors": [f"{type(exc).__name__}: {exc}"]}
            s = score(e, record)
            log(f"#{e.number}: {s['verdict'] or 'no result'}" + (", file hit" if s["file_hit"] else ""))
            return s

    scores = await asyncio.gather(*(one(e) for e in entries))
    result = {"summary": report(list(scores)), "entries": list(scores)}
    base.mkdir(parents=True, exist_ok=True)
    (base / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
