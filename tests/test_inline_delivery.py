# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Findings anchored to changed lines go inline; the top-level comment carries the rest."""

from __future__ import annotations

import json

import httpx

from tests.test_agent import GOOD_REVIEW
from zephyr_reviewer.review import publish as pub
from zephyr_reviewer.review.agent import RunResult
from zephyr_reviewer.review.config import ReviewConfig
from zephyr_reviewer.review.github_api import GitHub, PullRequest
from zephyr_reviewer.review.openrouter import Usage
from zephyr_reviewer.review.render import finding_key, inline_comments, inline_marker, render_comment
from zephyr_reviewer.review.schema import parse_review

CHANGED = {"drivers/i2c/i2c_foo.c": {17, 18, 19}, "drivers/i2c/Kconfig": {20}}


def _review():
    review, _ = parse_review(GOOD_REVIEW, CHANGED)
    return review


def _run():
    review = _review()
    pr = PullRequest(7, "t", "", "a", "main", "b" * 40, "pr", "h" * 40, "o/r", False)
    return RunResult(review, 2, Usage(), ["m"], False), pr


def _gh(seen: list, existing_inline: list | None = None, reject_review: bool = False) -> GitHub:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path.endswith("/pulls/7/comments"):
            return httpx.Response(200, json=existing_inline or [])
        if request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/reviews"):
            if reject_review:
                return httpx.Response(422, json={"message": "Unprocessable: line is not part of the diff"})
            return httpx.Response(200, json={"html_url": "https://github.com/o/r/pull/7#pullrequestreview-1"})
        return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/7#issuecomment-9"})

    return GitHub("tok", "o/r", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_top_level_comment_excludes_inline_findings_but_keeps_unanchored_ones():
    run, pr = _run()
    full = render_comment(run.review, run, pr, repo="o/r", tag="t")
    split = render_comment(run.review, run, pr, repo="o/r", tag="t", exclude_inline=True)
    assert "Early return leaks the bus lock" in full and "Early return leaks the bus lock" not in split
    assert "select of I2C_BUS_RECOVERY" not in split
    assert "2 finding(s) are posted as inline comments on the changed lines (1 high, 1 medium)." in split
    assert "### Findings not anchored to a changed line" in split
    assert "Unrelated file cited" in split and "Context line cited" in split  # line 0 / off-diff stay here
    assert "**Findings:** 4" in split and "### Test gaps" in split


def test_inline_comments_are_self_contained_and_marked():
    review = _review()
    comments = inline_comments(review, "zephyr-reviewer")
    assert [c["path"] for c in comments] == ["drivers/i2c/i2c_foo.c", "drivers/i2c/Kconfig"]
    lock = comments[0]
    key = finding_key(review.findings[0])
    assert lock["body"].startswith(inline_marker("zephyr-reviewer", key))
    assert lock["body"].splitlines()[1] == "**Early return leaks the bus lock** · 🟠 high"
    assert "Evidence" not in lock["body"] and "**Trigger:**" not in lock["body"] and "**Impact:**" not in lock["body"]
    expected = (
        "foo_transfer returns on error before k_sem_give, so the semaphore is never released. "
        "Happens when any failed transfer. Every later transfer blocks forever."
    )
    assert expected in lock["body"]
    assert "**Fix:** Release the lock before returning." in lock["body"] and "```suggestion" in lock["body"]
    assert lock["body"].rstrip().endswith("verify before acting on it.</sub>")
    assert lock["start_line"] == 17 and lock["line"] == 19
    # The key depends on file and title only: reworded prose, a different anchor range or a moved line
    # after a push all still match, and punctuation or case differences in the title do not matter.
    other = _review()
    other.findings[0].problem = "different wording"
    other.findings[0].line, other.findings[0].end_line = 1617, 1668
    assert finding_key(other.findings[0]) == key
    other.findings[0].title = "Early return leaks the bus lock!"
    assert finding_key(other.findings[0]) == key
    other.findings[0].file = "other.c"
    assert finding_key(other.findings[0]) != key


def test_publish_posts_summary_then_inline_review_and_skips_already_posted():
    run, pr = _run()
    key0 = finding_key(run.review.findings[0])
    existing = [
        {
            "id": 1,
            "path": "drivers/i2c/i2c_foo.c",
            "body": inline_marker("zephyr-reviewer", key0) + "\nposted last time",
        }
    ]
    seen: list = []
    logs: list[str] = []
    cfg = ReviewConfig(comment_tag="zephyr-reviewer")  # post_inline defaults to True

    def render(exclude_inline: bool) -> str:
        return "summary-without-inline" if exclude_inline else "summary-with-all"

    out = pub.publish(cfg, _gh(seen, existing), pr, run, render, log=logs.append)
    posted_summary = [b for m, p, b in seen if m == "POST" and p.endswith("/issues/7/comments")][0]
    assert posted_summary["body"] == "summary-without-inline"
    review_req = [b for m, p, b in seen if p.endswith("/reviews")][0]
    assert [c["path"] for c in review_req["comments"]] == ["drivers/i2c/Kconfig"]  # the lock finding was skipped
    assert out["inline_count"] == 1 and out["posted"] == "true"
    assert any("1 inline comment(s) already posted" in m for m in logs)


def test_rejected_inline_review_folds_findings_back_into_the_summary():
    run, pr = _run()
    seen: list = []
    logs: list[str] = []

    def render(exclude_inline: bool) -> str:
        return "summary-without-inline" if exclude_inline else "summary-with-all"

    out = pub.publish(ReviewConfig(), _gh(seen, reject_review=True), pr, run, render, log=logs.append)
    summaries = [b["body"] for m, p, b in seen if m in ("POST", "PATCH") and "/issues/" in p]
    assert summaries == ["summary-without-inline", "summary-with-all"]
    assert out["inline_count"] == 0 and out["review_url"] == "" and out["posted"] == "true"
    assert any("folding the findings into the summary comment" in m for m in logs)


def test_no_inline_mode_puts_everything_in_the_summary():
    run, pr = _run()
    seen: list = []
    calls: list[bool] = []

    def render(exclude_inline: bool) -> str:
        calls.append(exclude_inline)
        return "all"

    pub.publish(ReviewConfig(post_inline=False), _gh(seen), pr, run, render, log=lambda m: None)
    assert calls == [False] and not any(p.endswith("/reviews") for _, p, _ in seen)


def test_saved_inline_comments_are_posted_after_editing(tmp_path):
    run, pr = _run()
    cfg = ReviewConfig(repo_slug="o/r", output_dir=str(tmp_path / "out"))
    inline = inline_comments(run.review, cfg.comment_tag)
    pub.save_artifacts(cfg.output_dir, run, "top", pr, cfg, inline)
    saved = json.loads((tmp_path / "out" / "inline_comments.json").read_text())
    assert len(saved) == 2
    (tmp_path / "out" / "inline_comments.json").write_text(json.dumps(saved[:1]))  # a human drops one
    seen: list = []
    out = pub.post_saved(cfg.output_dir, _gh(seen), inline=True, tag=cfg.comment_tag, log=lambda m: None)
    assert out["inline_count"] == 1
    review_req = [b for m, p, b in seen if p.endswith("/reviews")][0]
    assert [c["path"] for c in review_req["comments"]] == ["drivers/i2c/i2c_foo.c"]


def test_already_posted_is_recognised_by_title_when_the_old_marker_differs():
    """Comments posted by an older version keyed the marker on the line range; the title still identifies them."""
    run, pr = _run()
    seen: list = []
    old_body = (
        "<!-- zephyr-reviewer:finding:5679fa7dd6d9 -->\n**Early return leaks the bus lock**\n\n"
        "🟠 high · concurrency\n\nold prose"
    )
    existing = [{"id": 1, "path": "drivers/i2c/i2c_foo.c", "line": 1668, "body": old_body}]
    out = pub.publish(ReviewConfig(), _gh(seen, existing), pr, run, lambda e: "s", log=lambda m: None)
    review_req = [b for m, p, b in seen if p.endswith("/reviews")][0]
    assert [c["path"] for c in review_req["comments"]] == ["drivers/i2c/Kconfig"] and out["inline_count"] == 1
    # A different file with the same title is a different finding.
    existing = [{"id": 1, "path": "drivers/other.c", "line": 5, "body": old_body}]
    seen.clear()
    pub.publish(ReviewConfig(), _gh(seen, existing), pr, run, lambda e: "s", log=lambda m: None)
    assert len([b for m, p, b in seen if p.endswith("/reviews")][0]["comments"]) == 2
