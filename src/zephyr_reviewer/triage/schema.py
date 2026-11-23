# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""What a triage run submits: candidate causes from the first pass, and the triage itself."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..review.agent import TerminalTool

VERDICTS = ["confirmed", "plausible", "cannot_confirm", "not_a_bug", "needs_info"]
LEVELS = ["high", "medium", "low"]
PRIORITIES = ["high", "medium", "low"]

SUBMIT_CAUSES = "submit_causes"
SUBMIT_TRIAGE = "submit_triage"

_CAUSE = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The defect in a few words"},
        "file": {"type": "string", "description": "Path of the file that has it"},
        "line": {"type": "integer", "description": "Line of the defect at the commit under analysis"},
        "explanation": {"type": "string", "description": "How this code produces what the report describes"},
        "evidence": {"type": "string", "description": "What you read that supports it: files, lines, symbols"},
        "confidence": {"type": "string", "enum": LEVELS},
    },
    "required": ["title", "file", "explanation", "confidence"],
}

CAUSES_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_CAUSES,
        "description": "Submit the candidate causes you found for the reported problem. This ends your pass.",
        "parameters": {
            "type": "object",
            "properties": {
                "causes": {"type": "array", "items": _CAUSE, "description": "Most likely first; empty if none"},
                "not_a_bug": {
                    "type": "string",
                    "description": "If the code behaves as documented and the report misuses it: why. Else empty.",
                },
                "missing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What the report would have to say for the problem to be located",
                },
            },
            "required": ["causes"],
        },
    },
}

TRIAGE_TOOL = {
    "type": "function",
    "function": {
        "name": SUBMIT_TRIAGE,
        "description": "Submit the triage of the issue. This ends the run.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": VERDICTS,
                    "description": "confirmed: the code has the defect, shown by reading it. plausible: a likely "
                    "cause, with a link you could not check. cannot_confirm: no cause found. not_a_bug: the code "
                    "behaves as documented. needs_info: the report lacks what is needed to look.",
                },
                "summary": {
                    "type": "string",
                    "description": "One or two plain sentences: the result. Posted on its own only when no cause "
                    "is named; with a cause, the cause's explanation is what is posted.",
                },
                "confidence": {"type": "string", "enum": LEVELS},
                "cause": {
                    "type": "object",
                    "description": "Where the defect is. Required for confirmed and plausible.",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "explanation": {
                            "type": "string",
                            "description": "One or two plain sentences a maintainer can act on: what the code "
                            "does wrong and when. Posted next to a link to the file and line, so no file names "
                            "or line numbers; name a function or symbol only when the sentence needs it.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "What you read that shows it: files, lines, symbols, the path from "
                            "the reported use to the failure. Kept with the run, not posted.",
                        },
                    },
                    "required": ["file", "explanation"],
                },
                "culprit": {
                    "type": "object",
                    "description": "The commit that introduced the defect, when the history shows it. Leave "
                    "out when it does not: the commit that last touched a line is not by that fact the cause.",
                    "properties": {
                        "commit": {"type": "string", "description": "Commit id"},
                        "confidence": {"type": "string", "enum": LEVELS},
                        "reason": {
                            "type": "string",
                            "description": "One sentence: what this commit changed that created the defect",
                        },
                    },
                    "required": ["commit", "confidence", "reason"],
                },
                "regression": {"type": "boolean", "description": "It worked in an earlier release"},
                "priority": {"type": "string", "enum": PRIORITIES},
                "priority_reason": {"type": "string", "description": "One short clause: what breaks, for whom"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels of further areas involved, as the maintainers tool lists them; those of the "
                    "cause's own file are added without being named",
                },
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What the reporter should add, one short question each; required for needs_info",
                },
                "fix": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "One sentence saying what to change; put code in patch, not here",
                        },
                        "patch": {"type": "string", "description": "A unified diff, only when small and certain"},
                    },
                    "required": ["description"],
                },
                "duplicates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"number": {"type": "integer"}, "reason": {"type": "string"}},
                        "required": ["number", "reason"],
                    },
                },
            },
            "required": ["verdict", "summary", "confidence"],
        },
    },
}


@dataclass
class Cause:
    title: str
    file: str
    explanation: str
    line: int = 0
    evidence: str = ""
    confidence: str = "low"
    raised_by: list[str] = field(default_factory=list)


@dataclass
class Culprit:
    commit: str
    confidence: str
    reason: str
    # Filled in from the repository, never from the model:
    author: str = ""
    email: str = ""
    date: str = ""
    subject: str = ""
    releases: list[str] = field(default_factory=list)
    login: str = ""


@dataclass
class Triage:
    verdict: str
    summary: str
    confidence: str = "low"
    cause: Cause | None = None
    culprit: Culprit | None = None
    regression: bool = False
    priority: str = ""
    priority_reason: str = ""
    labels: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    fix: str = ""
    patch: str = ""
    duplicates: list[tuple[int, str]] = field(default_factory=list)
    provisional: bool = False  # submitted when the budget ran out

    @property
    def located(self) -> bool:
        return self.verdict in ("confirmed", "plausible") and self.cause is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _level(value: Any, allowed: list[str], default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def _text(value: Any, limit: int = 4000) -> str:
    return " ".join(str(value or "").split())[:limit]


def parse_causes(args: dict[str, Any], exists: Callable[[str], bool]) -> tuple[dict[str, Any] | None, list[str]]:
    raw = args.get("causes")
    if not isinstance(raw, list):
        return None, ["causes must be an array (empty is fine)"]
    causes: list[Cause] = []
    errors: list[str] = []
    for i, c in enumerate(raw[:8]):
        if not isinstance(c, dict) or not _text(c.get("title")) or not _text(c.get("explanation")):
            errors.append(f"causes[{i}] needs title and explanation")
            continue
        path = str(c.get("file") or "").strip().lstrip("./")
        if not exists(path):
            errors.append(
                f"causes[{i}].file {path!r} is not a file at this commit; give the path as read_file takes it"
            )
            continue
        causes.append(
            Cause(
                title=_text(c.get("title"), 200),
                file=path,
                line=int(c.get("line") or 0) if str(c.get("line") or "0").lstrip("-").isdigit() else 0,
                explanation=_text(c.get("explanation")),
                evidence=_text(c.get("evidence")),
                confidence=_level(c.get("confidence"), LEVELS, "low"),
            )
        )
    if errors:
        return None, errors
    missing = (
        [_text(q, 300) for q in args.get("missing") or [] if _text(q)] if isinstance(args.get("missing"), list) else []
    )
    return {"causes": causes, "not_a_bug": _text(args.get("not_a_bug"), 1500), "missing": missing[:6]}, []


def parse_triage(
    args: dict[str, Any], exists: Callable[[str], bool], resolve: Callable[[str], str | None]
) -> tuple[Triage | None, list[str]]:
    """``exists`` says whether a path is a file at the commit; ``resolve`` gives a commit's full id when it
    is in the history under analysis. A cause in a file that is not there, or a culprit that is not a commit,
    is sent back: both are checked here because a comment naming either would be wrong on its face."""
    errors: list[str] = []
    verdict = str(args.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        errors.append(f"verdict must be one of {', '.join(VERDICTS)}")
    summary = _text(args.get("summary"), 1500)
    if not summary:
        errors.append("summary is required")
    cause = None
    c = args.get("cause")
    if isinstance(c, dict) and str(c.get("file") or "").strip():
        path = str(c.get("file")).strip().lstrip("./")
        if not exists(path):
            errors.append(f"cause.file {path!r} is not a file at this commit")
        elif not _text(c.get("explanation")):
            errors.append("cause.explanation is required")
        else:
            line = c.get("line")
            cause = Cause(
                title="",
                file=path,
                line=int(line) if isinstance(line, int) and line > 0 else 0,
                explanation=_text(c.get("explanation")),
                evidence=_text(c.get("evidence")),
            )
    if verdict in ("confirmed", "plausible") and cause is None and not errors:
        errors.append(f"a {verdict} verdict needs a cause with the file that has the defect")
    culprit = None
    k = args.get("culprit")
    if isinstance(k, dict) and str(k.get("commit") or "").strip():
        sha = resolve(str(k.get("commit")).strip())
        if sha is None:
            errors.append(
                f"culprit.commit {str(k.get('commit'))[:50]!r} is not a commit in this history; "
                "give an id that blame, file_history or search_history printed, or leave culprit out"
            )
        elif cause is None:
            errors.append("a culprit needs a cause")
        else:
            culprit = Culprit(sha, _level(k.get("confidence"), LEVELS, "low"), _text(k.get("reason"), 800))
    questions = (
        [_text(q, 300) for q in args.get("questions") or [] if _text(q)]
        if isinstance(args.get("questions"), list)
        else []
    )
    if verdict == "needs_info" and not questions:
        errors.append("needs_info needs the questions for the reporter")
    if errors:
        return None, errors
    fix = args.get("fix") if isinstance(args.get("fix"), dict) else {}
    dups = []
    for d in args.get("duplicates") or [] if isinstance(args.get("duplicates"), list) else []:
        if isinstance(d, dict) and isinstance(d.get("number"), int) and d["number"] > 0:
            dups.append((d["number"], _text(d.get("reason"), 300)))
    return (
        Triage(
            verdict=verdict,
            summary=summary,
            confidence=_level(args.get("confidence"), LEVELS, "low"),
            cause=cause,
            culprit=culprit,
            regression=bool(args.get("regression")),
            priority=_level(args.get("priority"), PRIORITIES, ""),
            priority_reason=_text(args.get("priority_reason"), 500),
            labels=[str(x).strip() for x in args.get("labels") or [] if isinstance(x, str) and x.strip()][:12],
            questions=questions[:6],
            fix=_text(fix.get("description"), 1500),
            patch=str(fix.get("patch") or "").strip()[:6000],
            duplicates=dups[:5],
        ),
        [],
    )


def causes_terminal(exists: Callable[[str], bool]) -> TerminalTool:
    return TerminalTool(
        CAUSES_TOOL,
        lambda args: parse_causes(args, exists),
        lambda out: f"{len(out['causes'])} candidate cause(s)",
    )


def triage_terminal(exists: Callable[[str], bool], resolve: Callable[[str], str | None]) -> TerminalTool:
    return TerminalTool(
        TRIAGE_TOOL,
        lambda args: parse_triage(args, exists, resolve),
        lambda out: f"triage: {out.verdict}, confidence {out.confidence}",
    )
