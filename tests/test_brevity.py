# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Posted comments are short: one paragraph, one fix line, code when useful, no evidence or internals."""

from __future__ import annotations

from zephyr_reviewer.review.render import finding_text, inline_comments
from zephyr_reviewer.review.schema import Finding, Review


def _f(**kw) -> Finding:
    base = dict(
        file="a.c",
        line=3,
        severity="medium",
        category="bug",
        title="T",
        problem="The lock is leaked on error",
        inline=True,
    )
    base.update(kw)
    return Finding(**base)


def test_finding_text_folds_trigger_and_impact_without_repeating():
    assert finding_text(_f()) == "The lock is leaked on error."
    assert finding_text(_f(trigger="a transfer fails", impact="the bus wedges")) == (
        "The lock is leaked on error. Happens when a transfer fails. The bus wedges."
    )
    assert finding_text(_f(trigger="When CONFIG_NET_VLAN is enabled")) == (
        "The lock is leaked on error. When CONFIG_NET_VLAN is enabled."
    )
    # Clauses the problem already contains are not repeated.
    assert finding_text(
        _f(
            problem="The lock is leaked when a transfer fails, so the bus wedges",
            trigger="a transfer fails",
            impact="the bus wedges",
        )
    ) == ("The lock is leaked when a transfer fails, so the bus wedges.")


def test_inline_comment_shape():
    f = _f(
        recommendation="Give the semaphore before returning",
        suggested_code="\tk_sem_give(&l);",
        evidence="read_file a.c",
        notes=["internal"],
    )
    body = inline_comments(Review("s", "comment", "high", [f]), "t")[0]["body"].splitlines()
    assert body[0].startswith("<!-- t:finding:") and body[1] == "**T** · 🟡 medium"
    assert body[3] == "The lock is leaked on error."
    assert body[5] == "**Fix:** Give the semaphore before returning."
    assert body[7:10] == ["```suggestion", "\tk_sem_give(&l);", "```"]
    assert body[-1] == "<sub>AI review, advisory; verify before acting on it.</sub>"
    joined = "\n".join(body)
    assert "read_file" not in joined and "internal" not in joined and "Problem" not in joined
