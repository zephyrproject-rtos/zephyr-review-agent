# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The specialist pipeline: focused parallel passes, consolidation with full accounting, and
per-finding re-verification of contested cases.

    plan (deterministic) -> specialists (parallel) -> verification -> hard cases (parallel) -> report

Every specialist concern and dismissal gets a stable id; verification must place each id in exactly
one of findings, dismissed or hard_cases, and may never drop a concern on its own. Hard cases are
decided one at a time with their own tool budget. The lineage is kept as a review map so a reader
can see why each candidate ended where it did. Modelled on sashiko's pipeline.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .agent import AgentResult, LocalTool, RunResult, TerminalTool, run_agent
from .mcp_client import ToolBridge
from .openrouter import OpenRouter, Usage
from .prompts import load_guides
from .schema import (
    CATEGORIES,
    CONFIDENCES,
    SEVERITIES,
    VERDICTS,
    Finding,
    Review,
    _issues_only,
    _norm_enum,
    parse_review,
)

SPECIALISTS = ["intent", "logic", "error_handling", "concurrency", "resources", "hardware", "build_config", "security"]
ALWAYS = ["intent", "logic"]

_CONCURRENCY_HINT = re.compile(
    r"\b(k_spin|k_mutex|k_sem|k_work|k_timer|k_queue|k_fifo|k_msgq|irq_|ISR_|atomic_|k_sched|k_thread|K_NO_WAIT"
    r"|K_FOREVER|z_impl_|z_vrfy_|__syscall|k_poll|k_condvar|k_event)"
)
_SECURITY_HINT = re.compile(
    r"\b(z_vrfy_|__syscall|K_SYSCALL|K_OBJ_ANY|memcpy|memmove|strcpy|strncpy|strlen|strchr|strtok|sscanf|atoi|strtol|"
    r"net_pkt|net_buf|sys_get_le|sys_get_be|sys_le16_to_cpu|sys_be16_to_cpu|le16_to_cpu|CONTAINER_OF|ARRAY_SIZE|"
    r"bt_|hci_|l2cap|rfcomm|sdp_|gatt|usb_|usbd_|usbh_|wLength|wTotalLength|bLength|shell_|json_|cbor|mgmt_|"
    r"psa_|mbedtls_|tc_|crypto|aead|poly1305|chacha|aes_|mac_|nonce|sys_rand_get|sys_csrand_get|"
    r"CONFIG_USERSPACE|k_usermode|elf|Elf|phr|PHR|dlc|DLC|rx_len|frame_len|payload_len|hdr_len|opt_len|tlv)"
)
_SECURITY_PATHS = re.compile(
    r"\b(subsys/(net|bluetooth|usb|fs|llext|mgmt|shell|lorawan|modbus|pmci|dap|debug|ipc|logging|canbus|sd|"
    r"modem|settings|midi|tee|rtio)/|drivers/(wifi|modem|ieee802154|ethernet|bluetooth|usb|tee|virtio|"
    r"pcie|nvme|mctp|lora|can)/|lib/|kernel/)"
)
_RESOURCE_HINT = re.compile(r"\b(k_malloc|k_free|k_heap_|k_mem_slab|net_buf|net_pkt|k_object|alloc|free)\b")


def hunk_text(diff_text: str) -> str:
    """Added and context lines of a diff: the code the change touches and the code around it."""
    return "\n".join(
        ln[1:]
        for ln in diff_text.splitlines()
        if (ln.startswith("+") or ln.startswith(" ")) and not ln.startswith("+++")
    )


def plan_specialists(files: list[str], added_text: str) -> list[str]:
    """Pick specialist passes from the files touched and the code in and around the hunks. Deterministic."""
    chosen = list(ALWAYS)
    c_code = any(f.endswith((".c", ".h", ".S")) for f in files)
    paths = " ".join(files)
    if c_code:
        chosen.append("error_handling")  # return codes, cleanup on error paths, assertions vs runtime checks
    if c_code and (_CONCURRENCY_HINT.search(added_text) or re.search(r"\b(kernel|arch)/", paths)):
        chosen.append("concurrency")
    if c_code and (_RESOURCE_HINT.search(added_text) or re.search(r"\b(subsys|drivers|lib)/", paths)):
        chosen.append("resources")
    if re.search(r"\b(drivers|soc|boards|modules/hal)/", paths) and c_code:
        chosen.append("hardware")
    if any(
        f.rsplit("/", 1)[-1].startswith("Kconfig")
        or f.endswith((".dts", ".dtsi", ".overlay", "CMakeLists.txt", ".cmake", ".conf", "defconfig"))
        or f.startswith("dts/bindings/")
        for f in files
    ):
        chosen.append("build_config")
    if c_code and (_SECURITY_HINT.search(added_text) or _SECURITY_PATHS.search(paths)):
        chosen.append("security")
    return [s for s in SPECIALISTS if s in chosen]


def _stage_prompt(name: str) -> str:
    return resources.files("zephyr_reviewer.review").joinpath(f"profiles/stages/{name}.md").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- data
@dataclass
class Concern:
    id: str
    stage: str
    data: dict[str, Any]  # finding-shaped

    @property
    def title(self) -> str:
        return str(self.data.get("title") or "")


@dataclass
class Dismissal:
    id: str
    stage: str
    title: str
    file: str
    line: int
    reason: str
    evidence: str


@dataclass
class HardCase:
    id: str
    source_ids: list[str]
    question: str
    severity_hint: str


@dataclass
class Verification:
    findings: list[dict[str, Any]]  # finding-shaped, each with source_ids
    dismissed: list[dict[str, Any]]  # {source_ids, reason}
    hard_cases: list[HardCase]


# ---------------------------------------------------------------------------- schemas
_FINDING_PROPS = {
    "file": {"type": "string"},
    "line": {"type": "integer"},
    "end_line": {"type": "integer"},
    "severity": {"type": "string", "enum": SEVERITIES},
    "category": {"type": "string", "enum": CATEGORIES},
    "title": {"type": "string"},
    "problem": {"type": "string"},
    "trigger": {"type": "string"},
    "impact": {"type": "string"},
    "recommendation": {"type": "string"},
    "suggested_code": {"type": "string"},
    "confidence": {"type": "string", "enum": CONFIDENCES},
    "evidence": {"type": "string"},
    "speculative": {"type": "boolean"},
}
_FINDING_REQUIRED = [
    "file",
    "line",
    "severity",
    "category",
    "title",
    "problem",
    "trigger",
    "impact",
    "recommendation",
    "confidence",
]

SUBMIT_CONCERNS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_concerns",
        "description": "Submit this specialist pass: real concerns, and what you checked and dismissed.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["concerns", "dismissed"],
            "properties": {
                "concerns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": _FINDING_REQUIRED,
                        "properties": _FINDING_PROPS,
                    },
                },
                "dismissed": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "file", "line", "reason", "evidence"],
                        "properties": {
                            "title": {"type": "string", "description": "The concern you considered."},
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                            "reason": {"type": "string", "description": "Why it is not a problem."},
                            "evidence": {"type": "string", "description": "The code or lookup that disproves it."},
                        },
                    },
                },
            },
        },
    },
}

SUBMIT_VERIFICATION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_verification",
        "description": "Consolidate all specialist concerns and dismissals. Every C* and D* id must appear in exactly "
        "one list.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["findings", "dismissed", "hard_cases"],
            "properties": {
                "findings": {
                    "type": "array",
                    "description": "Confirmed problems (merge duplicates). source_ids may reference only C* ids.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [*_FINDING_REQUIRED, "source_ids"],
                        "properties": {**_FINDING_PROPS, "source_ids": {"type": "array", "items": {"type": "string"}}},
                    },
                },
                "dismissed": {
                    "type": "array",
                    "description": "Well-justified dismissals. source_ids may reference only D* ids.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_ids", "reason"],
                        "properties": {
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                            "reason": {"type": "string"},
                        },
                    },
                },
                "hard_cases": {
                    "type": "array",
                    "description": "Contested or speculative candidates for individual verification; may reference C* "
                    "and D*.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_ids", "question", "severity_hint"],
                        "properties": {
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                            "question": {"type": "string", "description": "What must be checked to decide."},
                            "severity_hint": {"type": "string", "enum": SEVERITIES},
                        },
                    },
                },
            },
        },
    },
}

SUBMIT_HARD_CASE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_hard_case_verdict",
        "description": "Decide the hard case you were given.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "reason"],
            "properties": {
                "verdict": {"type": "string", "enum": ["confirmed", "refuted"]},
                "reason": {"type": "string", "description": "The evidence that decides it."},
                "finding": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": _FINDING_REQUIRED,
                    "properties": _FINDING_PROPS,
                    "description": "Required when confirmed.",
                },
            },
        },
    },
}

SUBMIT_REPORT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_report",
        "description": "Write the top-level assessment; the findings are fixed.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "summary",
                "verdict",
                "confidence",
                "test_assessment",
                "commit_message_assessment",
                "discussion_assessment",
            ],
            "properties": {
                "summary": {"type": "string"},
                "verdict": {"type": "string", "enum": VERDICTS},
                "confidence": {"type": "string", "enum": CONFIDENCES},
                "test_assessment": {"type": "string"},
                "commit_message_assessment": {"type": "string"},
                "discussion_assessment": {"type": "string"},
            },
        },
    },
}


# --------------------------------------------------------------------------- parsers
def _parse_concerns(args: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(args, dict):
        return None, ["arguments must be an object"]
    concerns = args.get("concerns")
    dismissed = args.get("dismissed")
    errors = []
    if not isinstance(concerns, list):
        errors.append("concerns must be an array (empty is fine)")
    if not isinstance(dismissed, list):
        errors.append("dismissed must be an array (empty is fine)")
    for i, c in enumerate(concerns if isinstance(concerns, list) else []):
        if not isinstance(c, dict) or not str(c.get("title") or "").strip() or not str(c.get("problem") or "").strip():
            errors.append(f"concerns[{i}] needs title and problem")
    for i, d in enumerate(dismissed if isinstance(dismissed, list) else []):
        if not isinstance(d, dict) or not str(d.get("title") or "").strip() or not str(d.get("reason") or "").strip():
            errors.append(f"dismissed[{i}] needs title and reason")
    if errors:
        return None, errors
    return {"concerns": concerns, "dismissed": dismissed}, []


def make_verification_parser(concern_ids: set[str], dismissal_ids: set[str]):
    """A parser enforcing full accounting: every id placed once, concerns never dropped directly."""

    def parse(args: dict[str, Any]) -> tuple[Verification | None, list[str]]:
        if not isinstance(args, dict):
            return None, ["arguments must be an object"]
        errors: list[str] = []
        findings = args.get("findings") if isinstance(args.get("findings"), list) else []
        dismissed = args.get("dismissed") if isinstance(args.get("dismissed"), list) else []
        hard = args.get("hard_cases") if isinstance(args.get("hard_cases"), list) else []
        seen: dict[str, str] = {}
        all_ids = concern_ids | dismissal_ids

        def take(ids: Any, where: str, allowed: set[str], label: str) -> list[str]:
            out = []
            for sid in ids if isinstance(ids, list) else []:
                sid = str(sid).strip()
                if sid not in all_ids:
                    errors.append(f"{where}: unknown id {sid!r}")
                    continue
                if sid not in allowed:
                    errors.append(f"{where}: {sid} is not allowed in {label} (concerns you doubt go to hard_cases)")
                if sid in seen:
                    errors.append(f"{sid} appears in both {seen[sid]} and {where}")
                seen[sid] = where
                out.append(sid)
            if not out:
                errors.append(f"{where}: source_ids must name at least one id")
            return out

        for i, f in enumerate(findings):
            if not isinstance(f, dict):
                errors.append(f"findings[{i}] must be an object")
                continue
            f["source_ids"] = take(f.get("source_ids"), f"findings[{i}]", concern_ids, "findings")
            if not str(f.get("title") or "").strip() or not str(f.get("problem") or "").strip():
                errors.append(f"findings[{i}] needs title and problem")
        for i, d in enumerate(dismissed):
            if not isinstance(d, dict):
                errors.append(f"dismissed[{i}] must be an object")
                continue
            d["source_ids"] = take(d.get("source_ids"), f"dismissed[{i}]", dismissal_ids, "dismissed")
        cases: list[HardCase] = []
        for i, h in enumerate(hard):
            if not isinstance(h, dict):
                errors.append(f"hard_cases[{i}] must be an object")
                continue
            ids = take(h.get("source_ids"), f"hard_cases[{i}]", all_ids, "hard_cases")
            sev = str(h.get("severity_hint") or "medium").lower()
            cases.append(
                HardCase(f"H{i + 1}", ids, str(h.get("question") or "").strip(), sev if sev in SEVERITIES else "medium")
            )
        missing = sorted(all_ids - set(seen), key=_id_key)
        if missing:
            errors.append("every id must be accounted for; missing: " + ", ".join(missing))
        if errors:
            return None, errors
        return Verification(findings, dismissed, cases), []

    return parse


def _parse_hard_case(args: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(args, dict):
        return None, ["arguments must be an object"]
    verdict = str(args.get("verdict") or "").lower()
    if verdict not in ("confirmed", "refuted"):
        return None, ["verdict must be confirmed or refuted"]
    if verdict == "confirmed":
        f = args.get("finding")
        if not isinstance(f, dict) or not str(f.get("title") or "").strip() or not str(f.get("problem") or "").strip():
            return None, ["a confirmed verdict needs a complete finding (title, problem, trigger, impact, ...)"]
    if not str(args.get("reason") or "").strip():
        return None, ["reason is required"]
    return args, []


def _parse_report(args: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(args, dict) or not str(args.get("summary") or "").strip():
        return None, ["summary is required"]
    if str(args.get("verdict") or "").lower().replace(" ", "_") not in VERDICTS:
        return None, [f"verdict must be one of {VERDICTS}"]
    return args, []


def _id_key(sid: str) -> tuple[str, int]:
    m = re.match(r"([A-Z]+)(\d+)", sid)
    return (m.group(1), int(m.group(2))) if m else (sid, 0)


# ------------------------------------------------------------------------ rendering
def _finding_block(d: dict[str, Any], sid: str, stage: str) -> str:
    loc = f"{d.get('file', '')}:{d.get('line', 0)}" + (f"-{d['end_line']}" if d.get("end_line") else "")
    lines = [
        f"### {sid} [{d.get('severity', '?')}{', speculative' if d.get('speculative') else ''}] {d.get('title', '')}  "
        f"({loc}; from {stage}; confidence {d.get('confidence', '?')})",
        f"Problem: {d.get('problem', '')}",
    ]
    for key in ("trigger", "impact", "recommendation", "evidence"):
        if d.get(key):
            lines.append(f"{key.capitalize()}: {d[key]}")
    if d.get("suggested_code"):
        lines += ["Suggested code:", "```", str(d["suggested_code"]).rstrip(), "```"]
    return "\n".join(lines)


def _dismissal_block(d: Dismissal) -> str:
    return (
        f"### {d.id} dismissed by {d.stage}: {d.title}  ({d.file}:{d.line})\nReason: {d.reason}\nEvidence: {d.evidence}"
    )


def verification_message(
    opening: str, concerns: list[Concern], dismissals: list[Dismissal], stages_run: list[str]
) -> str:
    parts = [opening, "", "## Specialist output", f"Stages run: {', '.join(stages_run)}.", ""]
    parts.append(f"## Concerns ({len(concerns)})" if concerns else "## Concerns (none)")
    for c in concerns:
        parts += ["", _finding_block(c.data, c.id, c.stage)]
    parts += ["", f"## Dismissed concerns ({len(dismissals)})" if dismissals else "## Dismissed concerns (none)"]
    for d in dismissals:
        parts += ["", _dismissal_block(d)]
    parts += ["", "Account for every id above, then call `submit_verification`."]
    return "\n".join(parts)


def hard_case_message(
    opening: str, case: HardCase, concerns: dict[str, Concern], dismissals: dict[str, Dismissal]
) -> str:
    parts = [
        opening,
        "",
        f"## Hard case {case.id}",
        f"Question: {case.question or '(decide whether the concern is real)'}",
        f"Severity hint: {case.severity_hint}",
        "",
    ]
    for sid in case.source_ids:
        if sid in concerns:
            parts += [_finding_block(concerns[sid].data, sid, concerns[sid].stage), ""]
        elif sid in dismissals:
            parts += [_dismissal_block(dismissals[sid]), ""]
    parts.append("Investigate with the tools, then call `submit_hard_case_verdict`.")
    return "\n".join(parts)


def report_message(opening: str, review_findings: list[Finding], refuted: int, specialists: list[str]) -> str:
    parts = [opening, "", "## Verified findings (fixed; do not change them)"]
    if not review_findings:
        parts.append("(none)")
    for i, f in enumerate(review_findings, 1):
        loc = f"{f.file}:{f.line}" if f.file else "general"
        parts.append(f"{i}. [{f.severity}{', speculative' if f.speculative else ''}] {f.title} ({loc}): {f.problem}")
    parts += [
        "",
        f"Specialists run: {', '.join(specialists)}. Candidates refuted in verification: {refuted}.",
        "",
        "Call `submit_report` with the top-level assessment.",
    ]
    return "\n".join(parts)


# -------------------------------------------------------------------------- pipeline
def _sum_usage(parts: list[Usage]) -> Usage:
    u = Usage()
    for p in parts:
        u.prompt_tokens += p.prompt_tokens
        u.completion_tokens += p.completion_tokens
        u.total_tokens += p.total_tokens
        u.cost += p.cost
        u.requests += p.requests
    return u


def _stage_record(name: str, res: AgentResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rec = {
        "stage": name,
        "models": list(res.models_used),
        "steps": res.steps,
        "tool_calls": res.tool_call_count,
        "forced": res.forced,
        "submitted": res.output is not None,
        "seconds": round(res.seconds, 1),
        "usage": {
            "prompt_tokens": res.usage.prompt_tokens,
            "completion_tokens": res.usage.completion_tokens,
            "total_tokens": res.usage.total_tokens,
            "cost": round(res.usage.cost, 4),
            "requests": res.usage.requests,
        },
    }
    rec.update(extra or {})
    return rec


async def run_pipeline(
    specialist_llm: OpenRouter,
    verify_llm: OpenRouter,
    bridge: ToolBridge,
    base_system: str,
    opening: str,
    changed_lines: dict[str, set[int]],
    *,
    changed_files: list[str],
    added_text: str,
    specialists: list[str] | None = None,
    specialist_max_steps: int = 20,
    verify_max_steps: int = 12,
    hard_case_max_steps: int = 12,
    report_max_steps: int = 6,
    max_hard_cases: int = 8,
    max_context_tokens: int = 160_000,
    max_seconds: float = 0,
    extra_tools: list[LocalTool] | None = None,
    log=print,
) -> RunResult:
    started = time.monotonic()
    stages = specialists or plan_specialists(changed_files, added_text)
    log(
        f"pipeline: specialists {', '.join(stages)} on {', '.join(specialist_llm.models)}; "
        f"verification on {', '.join(verify_llm.models)}"
    )
    records: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    usages: list[Usage] = []
    models: list[str] = []
    errors: list[str] = []

    # 1. Specialists, in parallel.
    contract = _stage_prompt("concerns_output")

    async def specialist(name: str) -> tuple[str, AgentResult]:
        system = base_system + "\n\n" + _stage_prompt(name) + contract
        terminal = TerminalTool(
            SUBMIT_CONCERNS_TOOL,
            _parse_concerns,
            lambda o: f"{len(o['concerns'])} concern(s), {len(o['dismissed'])} dismissed",
        )
        res = await run_agent(
            specialist_llm,
            bridge,
            system,
            opening,
            terminal,
            max_steps=specialist_max_steps,
            max_context_tokens=max_context_tokens,
            max_seconds=max_seconds,
            extra_tools=extra_tools,
            log=log,
            label=name,
        )
        return name, res

    results = await asyncio.gather(*(specialist(s) for s in stages))
    concerns: list[Concern] = []
    dismissals: list[Dismissal] = []
    for name, res in results:
        usages.append(res.usage)
        models += [m for m in res.models_used if m not in models]
        transcript += [{**t, "stage": name} for t in res.transcript]
        if res.output is None:
            errors.append(f"specialist {name} did not submit: {'; '.join(res.errors)}")
            records.append(_stage_record(name, res, {"concerns": [], "dismissed": []}))
            continue
        ids_c, ids_d = [], []
        for c in res.output["concerns"]:
            cid = f"C{len(concerns) + 1}"
            concerns.append(Concern(cid, name, c))
            ids_c.append(cid)
        for d in res.output["dismissed"]:
            did = f"D{len(dismissals) + 1}"
            dismissals.append(
                Dismissal(
                    did,
                    name,
                    str(d.get("title") or ""),
                    str(d.get("file") or ""),
                    int(d.get("line") or 0),
                    str(d.get("reason") or ""),
                    str(d.get("evidence") or ""),
                )
            )
            ids_d.append(did)
        records.append(_stage_record(name, res, {"concerns": ids_c, "dismissed": ids_d}))
    log(f"specialists produced {len(concerns)} concern(s) and {len(dismissals)} dismissal(s)")

    by_c = {c.id: c for c in concerns}
    by_d = {d.id: d for d in dismissals}
    fates: dict[str, dict[str, Any]] = {}
    final: list[dict[str, Any]] = []
    refuted = 0

    # 2. Verification with full accounting; the guides apply here, not in the specialists.
    guides = load_guides()
    if concerns or dismissals:
        system = base_system + "\n\n" + _stage_prompt("verification") + "\n\n" + guides
        terminal = TerminalTool(
            SUBMIT_VERIFICATION_TOOL,
            make_verification_parser(set(by_c), set(by_d)),
            lambda v: (
                f"verification: {len(v.findings)} finding(s), {len(v.dismissed)} dismissed, "
                f"{len(v.hard_cases)} hard case(s)"
            ),
        )
        vres = await run_agent(
            verify_llm,
            bridge,
            system,
            verification_message(opening, concerns, dismissals, stages),
            terminal,
            max_steps=verify_max_steps,
            max_context_tokens=max_context_tokens,
            max_seconds=max_seconds,
            max_rejections=3,
            extra_tools=extra_tools,
            log=log,
            label="verification",
        )
        usages.append(vres.usage)
        models += [m for m in vres.models_used if m not in models]
        transcript += [{**t, "stage": "verification"} for t in vres.transcript]
        if vres.output is None:
            # Fail safe: every concern becomes a hard case so nothing is dropped silently.
            errors.append(f"verification did not submit: {'; '.join(vres.errors)}; all concerns treated as hard cases")
            verification = Verification(
                [],
                [],
                [
                    HardCase(f"H{i + 1}", [c.id], "verification stage failed", str(c.data.get("severity", "medium")))
                    for i, c in enumerate(concerns)
                ],
            )
            records.append(_stage_record("verification", vres, {"failed": True}))
        else:
            verification = vres.output
            records.append(
                _stage_record(
                    "verification",
                    vres,
                    {
                        "findings": len(verification.findings),
                        "dismissed": len(verification.dismissed),
                        "hard_cases": len(verification.hard_cases),
                    },
                )
            )
        for i, f in enumerate(verification.findings, 1):
            fid = f"VF{i}"
            final.append({**f, "_id": fid})
            for sid in f["source_ids"]:
                fates[sid] = {"fate": "confirmed", "by": "verification", "as": fid}
        for d in verification.dismissed:
            for sid in d["source_ids"]:
                fates[sid] = {"fate": "dismissed", "by": "verification", "reason": d.get("reason", "")}

        # 3. Hard cases, in parallel, highest severity first; the tail is batched into one stage.
        cases = sorted(verification.hard_cases, key=lambda h: SEVERITIES.index(h.severity_hint))
        if len(cases) > max_hard_cases:
            batches = [[h] for h in cases[: max_hard_cases - 1]] + [cases[max_hard_cases - 1 :]]
        else:
            batches = [[h] for h in cases]
        system = base_system + "\n\n" + _stage_prompt("hard_case") + "\n\n" + guides

        async def decide(batch: list[HardCase]) -> tuple[list[HardCase], AgentResult]:
            if len(batch) == 1:
                case = batch[0]
            else:
                case = HardCase(
                    "+".join(h.id for h in batch),
                    [s for h in batch for s in h.source_ids],
                    "Several lower-severity cases: decide each; confirm the strongest as one finding or refute all.",
                    batch[0].severity_hint,
                )
            terminal = TerminalTool(
                SUBMIT_HARD_CASE_TOOL, _parse_hard_case, lambda o, c=case: f"{c.id}: {o['verdict']}"
            )
            res = await run_agent(
                verify_llm,
                bridge,
                system,
                hard_case_message(opening, case, by_c, by_d),
                terminal,
                max_steps=hard_case_max_steps,
                max_context_tokens=max_context_tokens,
                max_seconds=max_seconds,
                extra_tools=extra_tools,
                log=log,
                label=case.id,
            )
            return batch, res

        decisions = await asyncio.gather(*(decide(b) for b in batches)) if batches else []
        for batch, res in decisions:
            usages.append(res.usage)
            models += [m for m in res.models_used if m not in models]
            label = "+".join(h.id for h in batch)
            transcript += [{**t, "stage": label} for t in res.transcript]
            ids = [s for h in batch for s in h.source_ids]
            if res.output is None:
                # Undecided: keep the concern, as speculative, rather than lose it.
                errors.append(f"hard case {label} undecided: {'; '.join(res.errors)}")
                for h in batch:
                    src = next((by_c[s] for s in h.source_ids if s in by_c), None)
                    if src is not None:
                        fid = f"HF{len(final) + 1}"
                        final.append({**src.data, "speculative": True, "source_ids": h.source_ids, "_id": fid})
                        for s in h.source_ids:
                            fates[s] = {
                                "fate": "confirmed",
                                "by": label,
                                "as": fid,
                                "note": "undecided; kept as speculative",
                            }
                records.append(_stage_record(label, res, {"verdict": "undecided"}))
                continue
            verdict = res.output["verdict"]
            if verdict == "confirmed":
                fid = f"HF{len(final) + 1}"
                final.append({**res.output["finding"], "source_ids": ids, "_id": fid})
                for s in ids:
                    fates[s] = {"fate": "confirmed", "by": label, "as": fid, "reason": res.output["reason"]}
            else:
                refuted += 1
                for s in ids:
                    fates[s] = {"fate": "refuted", "by": label, "reason": res.output["reason"]}
            records.append(_stage_record(label, res, {"verdict": verdict}))

    # 4. Build the findings, then the report.
    review, perrs = parse_review(
        {
            "summary": "pending",
            "verdict": "comment",
            "confidence": "medium",
            "findings": [{k: v for k, v in f.items() if k not in ("source_ids", "_id")} for f in final],
        },
        changed_lines,
    )
    if review is None:
        errors.append("final findings failed validation: " + "; ".join(perrs))
        review = Review(summary="", verdict="comment", confidence="low", findings=[])
    for f, src in zip(review.findings, final, strict=False):
        f.evidence = (f.evidence + (" " if f.evidence else "") + f"[{', '.join(src.get('source_ids', []))}]").strip()

    system = base_system + "\n\n" + _stage_prompt("report")
    terminal = TerminalTool(SUBMIT_REPORT_TOOL, _parse_report, lambda o: f"report: {o['verdict']}")
    rres = await run_agent(
        verify_llm,
        None,
        system,
        report_message(opening, review.findings, refuted, stages),
        terminal,
        max_steps=report_max_steps,
        max_context_tokens=max_context_tokens,
        max_seconds=max_seconds,
        extra_tools=extra_tools,
        log=log,
        label="report",
    )
    usages.append(rres.usage)
    models += [m for m in rres.models_used if m not in models]
    transcript += [{**t, "stage": "report"} for t in rres.transcript]
    records.append(_stage_record("report", rres))
    if rres.output is not None:
        o = rres.output
        review.summary = str(o.get("summary") or "").strip()
        review.verdict = _norm_enum(o.get("verdict"), VERDICTS, "comment")
        review.confidence = _norm_enum(o.get("confidence"), CONFIDENCES, "medium")
        review.test_assessment = _issues_only(o.get("test_assessment"))
        review.commit_message_assessment = _issues_only(o.get("commit_message_assessment"))
        review.discussion_assessment = _issues_only(o.get("discussion_assessment"))
    else:
        errors.append(f"report stage did not submit: {'; '.join(rres.errors)}")
        review.summary = "Automated review completed; see findings." if review.findings else "No issues found."
    # The verdict must agree with the findings whatever the report said.
    worst = review.highest_severity()
    if worst in ("critical", "high"):
        review.verdict = "request_changes"
    elif not review.findings and review.verdict == "request_changes":
        review.verdict = "approve"

    review_map = {
        "specialists": stages,
        "concerns": [
            {
                "id": c.id,
                "stage": c.stage,
                "title": c.title,
                "file": c.data.get("file"),
                "line": c.data.get("line"),
                "severity": c.data.get("severity"),
                **fates.get(c.id, {"fate": "unaccounted"}),
            }
            for c in concerns
        ],
        "dismissed_concerns": [
            {
                "id": d.id,
                "stage": d.stage,
                "title": d.title,
                "file": d.file,
                "line": d.line,
                **fates.get(d.id, {"fate": "unaccounted"}),
            }
            for d in dismissals
        ],
        "findings": [{"id": f["_id"], "title": f.get("title"), "source_ids": f.get("source_ids", [])} for f in final],
        "stages": records,
    }
    return RunResult(
        review=review,
        steps=sum(r["steps"] for r in records),
        usage=_sum_usage(usages),
        models_used=models,
        forced=any(r["forced"] for r in records),
        errors=errors,
        transcript=transcript,
        seconds=time.monotonic() - started,
        stages=records,
        review_map=review_map,
    )
