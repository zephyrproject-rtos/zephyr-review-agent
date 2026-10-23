# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Markdown for the summary comment and the inline review comments."""

from __future__ import annotations

import hashlib
from typing import Any

from .agent import RunResult
from .github_api import PullRequest, file_url
from .schema import SEVERITIES, Finding, Review

SEVERITY_BADGE = {
    "critical": "🔴 critical",
    "high": "🟠 high",
    "medium": "🟡 medium",
    "low": "🔵 low",
    "suggestion": "⚪ suggestion",
}
VERDICT_LINE = {
    "approve": "✅ **Approve**: no blocking issues found.",
    "comment": "💬 **Comment**: worth a look, nothing blocking.",
    "request_changes": "🛑 **Request changes**: at least one issue should be fixed before merge.",
}
_LANG = {
    "c": "c",
    "h": "c",
    "S": "asm",
    "py": "python",
    "cmake": "cmake",
    "txt": "cmake",
    "yaml": "yaml",
    "yml": "yaml",
    "dts": "dts",
    "dtsi": "dts",
    "overlay": "dts",
    "rst": "rst",
    "conf": "kconfig",
    "sh": "bash",
}


def marker(tag: str) -> str:
    return f"<!-- {tag} -->"


def _lang(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if base.startswith("Kconfig") or base.endswith("defconfig"):
        return "kconfig"
    return _LANG.get(base.rsplit(".", 1)[-1], "") if "." in base else ""


def _location(f: Finding, repo: str, sha: str, api_url: str) -> str:
    if not f.file:
        return "_general_"
    label = f"`{f.file}`" if not f.line else f"`{f.file}:{f.line}`" + (f"-{f.end_line}" if f.end_line else "")
    if not repo:
        return label
    return f"[{label}]({file_url(repo, sha, f.file, f.line or None, api_url)})"


def _sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    first = text.split()[0].rstrip(",.;:")
    if first.isalpha() and first.islower():  # a plain word, not an identifier such as foo_transfer or k_sem_give
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def finding_text(f: Finding) -> str:
    """The finding's body as posted: the problem, with the trigger and impact folded in as clauses when
    they add something, and nothing else. Short enough to skim, specific enough to act on."""
    problem = _sentence(f.problem)
    extras = []
    low = problem.lower()
    if f.trigger and f.trigger.lower().rstrip(".") not in low:
        t = f.trigger.strip().rstrip(".")
        extras.append(
            t
            if t[:1].isupper() and t.split()[0].lower() in ("when", "if", "on", "only")
            else f"Happens when {t[:1].lower() + t[1:]}"
        )
    if f.impact and f.impact.lower().rstrip(".") not in low:
        extras.append(f.impact.strip().rstrip("."))
    return " ".join([problem, *(_sentence(e) for e in extras)]).strip()


def _finding_md(f: Finding, idx: int, repo: str, sha: str, api_url: str) -> str:
    badge = f"{SEVERITY_BADGE.get(f.severity, f.severity)}{' (speculative)' if f.speculative else ''}"
    lines = [f"#### {idx}. {f.title}", "", f"{badge} · {_location(f, repo, sha, api_url)}", "", finding_text(f), ""]
    if f.recommendation:
        lines += [f"**Fix:** {_sentence(f.recommendation)}", ""]
    if f.suggested_code.strip():
        lines += [f"```{_lang(f.file)}", f.suggested_code.rstrip(), "```", ""]
    return "\n".join(lines)


def render_comment(
    review: Review,
    run: RunResult,
    pr: PullRequest,
    *,
    repo: str,
    tag: str,
    api_url: str = "https://api.github.com",
    title: str = "AI review (advisory)",
    details: bool = False,
    exclude_inline: bool = False,
) -> str:
    """The top-level comment. With ``exclude_inline`` the anchored findings are left to inline comments
    and only counted here; unanchored findings always appear in full."""
    inline = [f for f in review.findings if f.inline] if exclude_inline else []
    listed = [f for f in review.findings if f not in inline]
    out = [
        marker(tag),
        f"## 🤖 {title}",
        "",
        VERDICT_LINE.get(review.verdict, review.verdict),
        "",
        f"**Findings:** {len(review.findings)} · **Highest severity:** {review.highest_severity()} · "
        f"**Confidence:** {review.confidence}",
        "",
        review.summary,
        "",
    ]
    if inline:
        counts = ", ".join(
            f"{n} {sev}" for sev, n in ((s, sum(1 for f in inline if f.severity == s)) for s in SEVERITIES) if n
        )
        out += [f"{len(inline)} finding(s) are posted as inline comments on the changed lines ({counts}).", ""]
    by_sev: dict[str, list[Finding]] = {s: [] for s in SEVERITIES}
    for f in listed:
        by_sev.setdefault(f.severity, []).append(f)
    idx = 0
    if inline and listed:
        out += ["### Findings not anchored to a changed line", ""]
    for sev in SEVERITIES:
        items = by_sev.get(sev) or []
        if not items:
            continue
        open_attr = " open" if sev in ("critical", "high", "medium") else ""
        out.append(f"<details{open_attr}><summary><b>{SEVERITY_BADGE[sev]}</b>: {len(items)} finding(s)</summary>\n")
        for f in items:
            idx += 1
            out.append(_finding_md(f, idx, repo, pr.head_sha, api_url))
        out.append("</details>\n")
    if review.test_assessment:
        out += ["### Test gaps", "", review.test_assessment, ""]
    if review.commit_message_assessment:
        out += ["### Commit message issues", "", review.commit_message_assessment, ""]
    if review.discussion_assessment:
        out += ["### Unresolved discussion", "", review.discussion_assessment, ""]
    if review.test_plan:
        out += render_test_plan(review)
    out += ["---", f"<sub>{run_details(run, pr) + ' ' if details else ''}{ADVISORY}</sub>"]
    return "\n".join(out)


ADVISORY_SHORT = "AI review, advisory; verify before acting on it."

ADVISORY = (
    "This review is generated by a language model with tool access to the source tree. "
    "It is advisory; verify before acting on it."
)


def run_details(run: RunResult, pr: PullRequest) -> str:
    """Models, steps, tool calls, tokens, cost and time of the run. Shown on the PR only when asked."""
    cost = f" · ${run.usage.cost:.3f}" if run.usage.cost else ""
    if len(run.stages) > 1:
        how = " then ".join(
            f"{st['stage']}: {', '.join(st['models']) or 'unknown'} ({st['steps']} step(s))" for st in run.stages
        )
        how = f"in two stages, {how},"
    else:
        how = f"with {', '.join(run.models_used) or 'unknown model'}"
    return (
        f"Reviewed `{pr.head_sha[:12]}` {how} via OpenRouter · "
        f"{run.steps} step(s), {run.tool_call_count} tool call(s), {run.usage.total_tokens:,} tokens{cost} · "
        f"{run.seconds:.0f}s{' · budget exhausted, review forced' if run.forced else ''}."
    )


def render_failure(run: RunResult, pr: PullRequest, tag: str) -> str:
    return "\n".join(
        [
            marker(tag),
            "## 🤖 AI review (advisory)",
            "",
            f"The review of `{pr.head_sha[:12]}` did not complete: " + "; ".join(run.errors or ["unknown error"]),
            "",
            f"<sub>{run.steps} step(s), {run.usage.total_tokens:,} tokens.</sub>",
        ]
    )


def finding_key(f: Finding) -> str:
    """Stable id of a finding at a location, used to avoid re-posting it on a later run."""
    return hashlib.sha1(f"{f.file}:{f.line}:{f.end_line}:{f.title.strip().lower()}".encode()).hexdigest()[:12]


def inline_marker(tag: str, key: str) -> str:
    return f"<!-- {tag}:finding:{key} -->"


def inline_comments(review: Review, tag: str = "zephyr-reviewer") -> list[dict[str, Any]]:
    """One review comment per anchored finding, self-contained and short, carrying a hidden marker for dedup."""
    comments = []
    for f in review.findings:
        if not f.inline:
            continue
        badge = f"{SEVERITY_BADGE.get(f.severity, f.severity)}{' (speculative)' if f.speculative else ''}"
        body = [inline_marker(tag, finding_key(f)), f"**{f.title}** · {badge}", "", finding_text(f)]
        if f.recommendation:
            body += ["", f"**Fix:** {_sentence(f.recommendation)}"]
        if f.suggested_code.strip():
            body += ["", "```suggestion", f.suggested_code.rstrip(), "```"]
        body += ["", f"<sub>{ADVISORY_SHORT}</sub>"]
        c: dict[str, Any] = {"path": f.file, "line": f.end_line or f.line, "side": "RIGHT", "body": "\n".join(body)}
        if f.end_line and f.end_line > f.line:
            c["start_line"] = f.line
            c["start_side"] = "RIGHT"
        comments.append(c)
    return comments


PRIORITY_BADGE = {"must": "🔴 must", "should": "🟡 should", "optional": "⚪ optional"}


def render_test_plan(review: Review) -> list[str]:
    from .testplan import commands

    out = ["### Test plan", "", "| Priority | Test | Platforms | Verifies |", "| --- | --- | --- | --- |"]
    for e in review.test_plan:
        test = f"`{e.suite}`" + (f"<br>`{e.scenario}`" if e.scenario else "")
        if e.build_only:
            test += "<br><sub>build only</sub>"
        plats = ", ".join(f"`{p}`" for p in e.platforms) or "_suite defaults_"
        reason = e.reason.replace("|", "\\|")
        if e.notes:
            reason += " <sub>⚠ " + "; ".join(e.notes).replace("|", "\\|") + "</sub>"
        out.append(f"| {PRIORITY_BADGE.get(e.priority, e.priority)} | {test} | {plats} | {reason} |")
    cmds = commands(review.test_plan)
    if cmds:
        out += ["", "<details><summary>twister commands</summary>", "", "```sh", *cmds, "```", "</details>"]
    out.append("")
    return out


def preview_inline(inline: list[dict[str, Any]]) -> str:
    """Human-readable listing of inline comments, as they would appear on the pull request."""
    if not inline:
        return ""
    out = [f"### {len(inline)} inline comment(s)"]
    for i, c in enumerate(inline, 1):
        where = f"{c['path']}:{c.get('start_line') or c['line']}" + (
            f"-{c['line']}" if c.get("start_line") and c.get("start_line") != c["line"] else ""
        )
        body = "\n".join(ln for ln in c["body"].splitlines() if not ln.startswith("<!-- "))
        out += ["", f"--- [{i}] {where} ---", body]
    return "\n".join(out)
