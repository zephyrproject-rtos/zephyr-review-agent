# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The ``triage`` command: read an issue, analyse it against the tree, and file what the run supports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import mcp_server
from ..intel import SourceIntel
from ..review import publish as pub
from ..review.agent import LocalTool
from ..review.config import ReviewConfig, _env
from ..review.discussion import sanitize
from ..review.github_api import GitHub, GitHubError, event_payload
from ..review.mcp_client import ToolBridge, stdio_params
from ..review.openrouter import OpenRouter
from .decide import LABEL_PREFIXES, Proposal, commit_resolver, describe_culprit, propose
from .fixes import LinkedPull, before_the_fixes, load_linked_pulls
from .issue import Issue, find_hints
from .pipeline import TriageRun, run_triage
from .publish import ACTIONS, ASSIGN_TO, Plan, carry_out, parse_actions, plan
from .render import TAG, marker, render_comment

SEARCH_ISSUES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_issues",
        "description": "Search this repository's issues, open and closed, for reports of the same defect. "
        "Give a few distinctive words: a function, an error message, a board. Returns number, state and title.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search words"}},
            "required": ["query"],
        },
    },
}


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", dest="repo_root", help="Path to the Zephyr checkout (default: cwd)")
    p.add_argument("--repository", help="GitHub owner/name (default: $GITHUB_REPOSITORY)")
    p.add_argument("--issue", type=int, help="Issue number (default: from the event that started the workflow)")
    p.add_argument("--issue-file", help="Read the issue from a JSON file as the GitHub API returns it")
    p.add_argument("--head", help="Commit to analyse against (default: HEAD)")
    p.add_argument("--model", help="Model that decides (default: as for a review)")
    p.add_argument("--draft-ensemble", help="Comma-separated models that look for candidate causes first")
    p.add_argument("--provider", help="OpenRouter provider preferences as JSON")
    p.add_argument("--api-key", help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    p.add_argument("--github-token", help="GitHub token (default: $GITHUB_TOKEN)")
    p.add_argument(
        "--apply",
        help="What the run may do to the issue, comma-separated: comment, labels, priority, assign; all; "
        "none (default: none, which prints the comment and changes nothing)",
    )
    p.add_argument(
        "--assign-to",
        choices=ASSIGN_TO,
        help="With assign: the area's maintainer (default), or the author of the introducing commit when "
        "that commit is named with high confidence",
    )
    p.add_argument(
        "--ask",
        action="store_true",
        help="Show each change and ask before making it: the labels, the priority, the assignee, then the "
        "comment. Asks about everything --apply allows, or about every action when --apply is not given",
    )
    p.add_argument("--any-kind", action="store_true", help="Triage the issue even if it is not a bug report")
    p.add_argument("--again", action="store_true", help="Triage an issue this tool already commented on")
    p.add_argument(
        "--no-linked-prs",
        action="store_true",
        help="Do not compare the pull requests that set out to fix the issue with the result",
    )
    p.add_argument("--no-comments", action="store_true", help="Do not show the issue's comments to the models")
    p.add_argument("--output-dir", help="Where triage.json and comment.md are written (default: triage-output)")
    p.add_argument("--in-process-server", action="store_true", help="Run the source tools in this process")
    p.add_argument("-q", "--quiet", action="store_true", help="Log the major steps only")
    p.add_argument("--show-details", action="store_true", help="Log models, tokens and cost of the run")


def _settings(args: argparse.Namespace) -> tuple[ReviewConfig, dict[str, Any]]:
    """The model, key and budget settings a review uses, and what is the triage's own."""
    cfg = ReviewConfig.from_env()
    for attr, key in (("repo_root", "repo_root"), ("repository", "repo_slug"), ("api_key", "openrouter_api_key")):
        if getattr(args, attr, None) is not None:
            setattr(cfg, key, getattr(args, attr))
    opt = lambda name: getattr(args, name, None)  # noqa: E731 - the benchmark's options are a subset
    if opt("github_token") is not None:
        cfg.github_token = args.github_token
    if opt("model"):
        cfg.models = [args.model] + [m for m in cfg.models[1:] if m != args.model]
    if opt("draft_ensemble") is not None:
        cfg.draft_ensemble = [m.strip() for m in args.draft_ensemble.split(",") if m.strip()]
    if opt("provider"):
        cfg.provider = json.loads(args.provider)
    cfg.quiet = cfg.quiet or bool(opt("quiet"))
    cfg.in_process_server = cfg.in_process_server or bool(opt("in_process_server"))
    own = {
        "issue": opt("issue") or (int(_env("issue_number")) if _env("issue_number").strip().isdigit() else None),
        "head": opt("head") or _env("triage_head", "HEAD"),
        "actions": opt("apply") if opt("apply") is not None else _env("triage_apply", "none"),
        "ask": bool(opt("ask")) or _env("triage_ask").strip().lower() in ("1", "true", "yes"),
        "assign_to": opt("assign_to") or _env("triage_assign_to", "maintainer"),
        "output_dir": opt("output_dir") or _env("triage_output_dir", "triage-output"),
        "tag": _env("triage_tag", TAG),
        "locate_max_steps": int(_env("triage_locate_max_steps", "20")),
        "confirm_max_steps": int(_env("triage_confirm_max_steps", "25")),
    }
    return cfg, own


def _load_issue(args: argparse.Namespace, gh: GitHub | None, number: int | None) -> Issue | None:
    if args.issue_file:
        data = json.loads(Path(args.issue_file).read_text(encoding="utf-8"))
        return Issue.from_api(data, data.get("comment_list") or [])
    if gh is None or number is None:
        return None
    return Issue.from_api(gh.issue(number), [] if args.no_comments else gh.issue_comments(number))


def _search_tool(gh: GitHub, issue: Issue) -> LocalTool:
    async def handler(arguments: dict[str, Any]) -> str:
        query = " ".join(str(arguments.get("query") or "").split())[:200]
        if not query:
            return "ERROR: give a query"
        try:
            items = [i for i in gh.search_issues(query, 10) if int(i.get("number") or 0) != issue.number]
        except GitHubError as exc:
            return f"ERROR: the search failed: {str(exc)[:200]}"
        if not items:
            return "No other issue matches."
        return "\n".join(f"#{i.get('number')} [{i.get('state')}] {sanitize(i.get('title') or '')[:160]}" for i in items)

    return LocalTool(SEARCH_ISSUES_TOOL, handler)


async def _run(
    cfg: ReviewConfig,
    own: dict,
    intel: SourceIntel,
    issue: Issue,
    gh: GitHub | None,
    labels,
    log,
    args,
    pulls: list[LinkedPull] | None = None,
):
    def llm(models: list[str]) -> OpenRouter:
        return OpenRouter(
            cfg.openrouter_api_key,
            models,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            provider=cfg.provider,
            timeout=cfg.model_timeout_seconds,
            on_event=log,
        )

    drafters = cfg.draft_models[:1] + [m for m in cfg.draft_ensemble if m not in cfg.draft_models[:1]]
    if cfg.in_process_server:
        target = mcp_server.build_server(intel, history=True)
    else:
        target = stdio_params(cfg.repo_root, intel.head.sha, intel.head.sha, cfg.max_tool_result_chars, history=True)
    async with ToolBridge(
        target, max_result_chars=cfg.max_tool_result_chars, call_timeout=float(cfg.tool_timeout_seconds)
    ) as bridge:
        return await run_triage(
            [llm([m]) for m in drafters],
            llm(cfg.models),
            bridge,
            issue,
            find_hints(issue, intel.head),
            intel.head.sha,
            exists=intel.head.exists,
            resolve=commit_resolver(intel),
            repo=cfg.repo_slug,
            labels=labels,
            extra_tools=[_search_tool(gh, issue)] if gh is not None else None,
            language=cfg.language,
            with_comments=not args.no_comments,
            pulls=pulls,
            locate_max_steps=own["locate_max_steps"],
            confirm_max_steps=own["confirm_max_steps"],
            max_context_tokens=cfg.max_context_tokens,
            max_seconds=cfg.max_stage_seconds,
            log=log,
            quiet=cfg.quiet,
        )


def has_terminal() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _answer(question: str) -> str:
    try:
        return input(question).strip().lower()
    except EOFError:
        return "n"


def _yes(question: str) -> bool:
    while True:
        a = _answer(f"{question} [y]es / [n]o: ")
        if a in ("y", "yes"):
            return True
        if a in ("n", "no", ""):
            return False
        print("please answer y or n", file=sys.stderr)


def ask_plan(p: Plan, issue: Issue) -> Plan:
    """Put each planned change to the person at the terminal; what they decline is dropped."""
    kept = Plan(withheld=list(p.withheld))
    for note in p.withheld:
        print(f"not offered: {note}", file=sys.stderr)
    if p.labels and _yes(f"Add {', '.join(p.labels)} to #{issue.number}?"):
        kept.labels = list(p.labels)
    if p.priority and _yes(f"Set {p.priority} on #{issue.number}?"):
        kept.priority = p.priority
    if len(p.assignees) == 1:
        login, why = p.assignees[0]
        if _yes(f"Assign #{issue.number} to {login} ({why})?"):
            kept.assignees = [p.assignees[0]]
    elif p.assignees:
        options = " / ".join(f"[{i}] {login} ({why})" for i, (login, why) in enumerate(p.assignees, 1))
        while True:
            a = _answer(f"Assign #{issue.number} to: {options} / [n]obody: ")
            if a in ("n", "no", "nobody", ""):
                break
            if a.isdigit() and 1 <= int(a) <= len(p.assignees):
                kept.assignees = [p.assignees[int(a) - 1]]
                break
            print(f"please answer 1 to {len(p.assignees)}, or n", file=sys.stderr)
    return kept


def ask_comment(body: str, path: Path, issue: Issue) -> tuple[bool, str]:
    """Show the comment and ask to post it, with the chance to edit comment.md first."""
    while True:
        print("\n" + body + "\n", file=sys.stderr)
        a = _answer(f"Post this comment to #{issue.number}? [y]es / [n]o / [e]dit comment.md: ")
        if a in ("y", "yes"):
            return True, body
        if a in ("n", "no", ""):
            return False, body
        if a in ("e", "edit"):
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
            subprocess.call([*editor.split(), str(path)])
            body = path.read_text(encoding="utf-8")
            continue
        print("please answer y, n or e", file=sys.stderr)


def save(out_dir: str, run: TriageRun, body: str, issue: Issue, commit: str, proposal: Proposal, applied) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "comment.md").write_text(body, encoding="utf-8")
    record = {
        "issue": issue.number,
        "commit": commit,
        "triage": run.triage.to_dict() if run.triage else None,
        "proposal": asdict(proposal),
        "applied": applied,
        "causes": [asdict(c) for c in run.causes],
        "linked_pulls": [asdict(c) for c in run.comparisons],
        "passes": run.passes,
        "errors": run.errors,
        "models": run.models_used,
        "seconds": round(run.seconds, 1),
        "usage": run.usage.as_dict(),
        "stages": run.stages,
    }
    (d / "triage.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with (d / "transcript.jsonl").open("w", encoding="utf-8") as fh:
        for t in run.transcript:
            fh.write(json.dumps(t) + "\n")
    return d


def add_benchmark_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="tcmd", required=True)
    m = sub.add_parser("mine", help="Build a dataset from closed bug reports that have one merged fix")
    m.add_argument("--repo", default=".", help="A full clone of the repository")
    m.add_argument("--repository", help="GitHub owner/name (default: $GITHUB_REPOSITORY)")
    m.add_argument("--github-token", help="GitHub token (default: $GITHUB_TOKEN)")
    m.add_argument("--since", help="Only issues closed on or after this date, e.g. 2026-09-01")
    m.add_argument("--which", default="type:Bug", help="Search qualifier selecting bug reports, e.g. label:bug")
    m.add_argument("--max", type=int, default=100)
    m.add_argument("--out", default="triage-benchmark.json")
    m.set_defaults(fn=cmd_benchmark_mine)
    r = sub.add_parser("run", help="Triage each issue against the tree before its fix, and score the result")
    r.add_argument("file", help="Dataset produced by `triage-benchmark mine`")
    r.add_argument("--repo", default=".")
    r.add_argument("--output-dir", default="triage-benchmark-runs")
    r.add_argument("--analyze-only", action="store_true", help="Score what the output directory holds; run no model")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--model")
    r.add_argument("--draft-ensemble")
    r.add_argument("--provider")
    r.add_argument("--api-key")
    r.set_defaults(fn=cmd_benchmark_run)


def cmd_benchmark_mine(args: argparse.Namespace) -> int:
    from ..cli import log
    from . import benchmark as tb

    token = args.github_token or os.environ.get("GITHUB_TOKEN", "")
    slug = args.repository or os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not slug:
        log("::error::mining needs a GitHub token and --repository")
        return 2
    entries = tb.mine(
        GitHub(token, slug), args.repo, since=args.since or "", which=args.which, max_entries=args.max, log=log
    )
    tb.save(entries, args.out)
    log(f"{len(entries)} entries written to {args.out}")
    return 0


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    from ..cli import log, run_async
    from . import benchmark as tb

    entries = tb.load(args.file)[args.offset :]
    entries = entries[: args.limit] if args.limit else entries
    if args.analyze_only:
        scores = []
        for e in entries:
            saved = Path(args.output_dir) / str(e.number) / "triage.json"
            if saved.exists():
                scores.append(tb.score(e, json.loads(saved.read_text(encoding="utf-8"))))
        print(tb.render(tb.report(scores)))
        return 0
    args.repo_root = args.repo
    cfg, own = _settings(args)
    if not cfg.openrouter_api_key:
        log("::error::an OpenRouter API key is required (--api-key or OPENROUTER_API_KEY)")
        return 2

    def llm(models: list[str]) -> OpenRouter:
        return OpenRouter(
            cfg.openrouter_api_key,
            models,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            provider=cfg.provider,
            timeout=cfg.model_timeout_seconds,
        )

    drafters = cfg.draft_models[:1] + [m for m in cfg.draft_ensemble if m not in cfg.draft_models[:1]]
    result = run_async(
        tb.run_benchmark(
            entries,
            args.repo,
            [llm([m]) for m in drafters],
            llm(cfg.models),
            args.output_dir,
            concurrency=args.concurrency,
            log=log,
            locate_max_steps=own["locate_max_steps"],
            confirm_max_steps=own["confirm_max_steps"],
        )
    )
    print(tb.render(result["summary"]))
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    from ..cli import describe_error, log, run_async  # the command line's own helpers; imported late, it imports us

    cfg, own = _settings(args)
    actions, errors = parse_actions(own["actions"])
    ask = own["ask"]
    if ask and not actions and not errors:
        actions = list(ACTIONS)  # asked about each, so every action may be offered
    if own["assign_to"] not in ASSIGN_TO:
        errors.append(f"assign-to must be one of {', '.join(ASSIGN_TO)}")
    if not cfg.openrouter_api_key:
        errors.append("an OpenRouter API key is required (--api-key or OPENROUTER_API_KEY)")
    if actions and (not cfg.github_token or not cfg.repo_slug):
        errors.append(("--ask" if ask else "--apply") + " needs a GitHub token and --repository")
    for e in errors:
        log(f"::error::{e}")
    if errors:
        return 2

    def skipped(reason: str) -> int:
        log(f"skipping triage: {reason}")
        pub.write_outputs({"skipped": "true", "verdict": "skipped"})
        return 0

    gh = GitHub(cfg.github_token, cfg.repo_slug, cfg.github_api_url) if cfg.github_token and cfg.repo_slug else None
    number = own["issue"] or int((event_payload().get("issue") or {}).get("number") or 0) or None
    try:
        issue = _load_issue(args, gh, number)
    except (GitHubError, OSError, ValueError) as exc:
        log(f"::error::could not read the issue: {exc}")
        return 2
    if issue is None:
        log("::error::no issue to triage: give --issue with a token and --repository, or --issue-file")
        return 2
    if issue.is_pull_request:
        return skipped(f"#{issue.number} is a pull request")
    if issue.state != "open" and not args.again:
        return skipped(f"#{issue.number} is {issue.state}")
    if not issue.is_bug_report() and not args.any_kind:
        return skipped(f"#{issue.number} is not a bug report")
    if gh is not None and not args.again and gh.find_comment(issue.number, marker(own["tag"])):
        return skipped(f"#{issue.number} was triaged before")

    intel = SourceIntel(cfg.repo_root, own["head"], own["head"], max_chars=cfg.max_tool_result_chars)
    commit = intel.head.sha
    repo_labels = None
    if gh is not None:
        try:
            repo_labels = gh.repo_labels()
        except GitHubError as exc:
            log(f"::warning::could not read the repository's labels, none will be proposed beyond MAINTAINERS: {exc}")
    shown = sorted(lab for lab in repo_labels or [] if lab.lower().startswith(LABEL_PREFIXES))
    pulls: list[LinkedPull] = []
    if gh is not None and issue.number and not args.no_linked_prs and _env("triage_linked_prs", "true") != "false":
        pulls = load_linked_pulls(gh, issue.number, log=log)
        if pulls:
            # They are compared with the result afterwards. What was written on the issue once the first
            # of them existed may say what the fix is, and is kept from the analysis for that reason.
            issue = before_the_fixes(issue, pulls)
            log(
                f"{len(pulls)} linked pull request(s) will be compared with the result: "
                + ", ".join(f"#{p.number}" for p in pulls)
            )
    log(f"triaging #{issue.number} against {commit[:12]}: {issue.title[:80]}")
    tb = ""
    try:
        run = run_async(_run(cfg, own, intel, issue, gh, shown, log, args, pulls))
    except Exception as exc:  # noqa: BLE001 - a failed run posts nothing; say why and keep the traceback
        tb = traceback.format_exc()
        run = TriageRun(None, errors=[describe_error(exc)])
    if run.triage is None:
        log("::error::" + "; ".join(run.errors or ["the triage did not complete"]))
        d = save(own["output_dir"], run, "", issue, commit, Proposal(), {})
        if tb:
            (d / "traceback.txt").write_text(tb, encoding="utf-8")
        return 1

    triage = run.triage
    describe_culprit(triage, intel)
    proposal = propose(triage, intel, issue, repo_labels)
    if gh is not None and triage.culprit is not None:
        triage.culprit.login = proposal.culprit_login = gh.commit_author_login(triage.culprit.commit)
    applied: dict[str, list[str]] = {}
    attended = ask and has_terminal()
    if ask and not attended:
        # As for a review: without someone to ask, nothing is changed and the result is kept to be read.
        log(f"::notice::--ask without a terminal: the triage was saved in {own['output_dir']}, nothing was changed")
        actions = []
    if gh is not None and actions:
        todo = plan(gh, issue, triage, proposal, actions, assign_to=own["assign_to"], attended=attended)
        if attended:
            shown_first = render_comment(
                triage, proposal, commit, repo=cfg.repo_slug, tag=own["tag"], comparisons=run.comparisons
            )
            print("\n" + shown_first, file=sys.stderr)
            todo = ask_plan(todo, issue)
        applied = carry_out(gh, issue, todo, log=log)
    body = render_comment(
        triage,
        proposal,
        commit,
        repo=cfg.repo_slug,
        api_url=cfg.github_api_url,
        tag=own["tag"],
        applied=applied,
        comparisons=run.comparisons,
    )
    d = save(own["output_dir"], run, body, issue, commit, proposal, applied)
    log(f"artifacts in {d}")
    url = ""
    post = gh is not None and "comment" in actions
    if post and attended:
        post, body = ask_comment(body, d / "comment.md", issue)
        if marker(own["tag"]) not in body:
            body = marker(own["tag"]) + "\n" + body  # an edit must not lose what marks the issue as triaged
    if post:
        url = gh.upsert_comment(issue.number, body, marker(own["tag"]))
        log(f"posted {url}")
    elif not attended:
        print(body)
    else:
        log("comment not posted")
    pub.write_summary(body)
    pub.write_outputs(
        {
            "skipped": "false",
            "verdict": triage.verdict,
            "confidence": triage.confidence,
            "culprit": triage.culprit.commit if triage.culprit else "",
            "labels": ",".join(applied.get("labels") or []),
            "comment_url": url,
        }
    )
    u = run.usage
    if args.show_details or cfg.show_run_details:
        log(
            f"models: {', '.join(run.models_used)}; {u.requests} requests, {u.prompt_tokens:,} prompt and "
            f"{u.completion_tokens:,} completion tokens, ${u.cost:.4f}, {run.seconds:.0f}s"
        )
    else:
        log(f"triage of #{issue.number}: {triage.verdict}, {run.seconds:.0f}s")
    if os.environ.get("GITHUB_ACTIONS") and run.errors:
        for e in run.errors:
            log(f"::warning::{e}")
    return 0
