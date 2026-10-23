# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Linked issues and pull requests: reference parsing, fetching, digest, tool and prompt placement."""

from __future__ import annotations

import json

import httpx

from zephyr_reviewer.review.github_api import GitHub, PullRequest
from zephyr_reviewer.review.linked import LINKED_ISSUE_TOOL, Linked, Ref, fetch_item, find_references, load_linked
from zephyr_reviewer.review.prompts import opening_message


def test_find_references_handles_every_form_and_keeps_order():
    texts = [
        (
            "description",
            "Fixes #122778 and relates to zephyrproject-rtos/zephyr#121776.\nSee https://github.com/other/repo/pull/5.",
        ),
        ("commit abc", 'Fixes: 91ac587124fd ("x")\n\nCloses #122778\nRelated: #99'),
        ("comment", "Looks like a regression from #121776; also &#123; is html, not a ref, and v1.0#4 is not either"),
    ]
    refs = find_references(texts, "zephyrproject-rtos/zephyr", exclude={("zephyrproject-rtos/zephyr", 99)})
    assert [(r.repo, r.number, r.relation, r.source) for r in refs] == [
        ("zephyrproject-rtos/zephyr", 122778, "fixes", "description"),
        ("zephyrproject-rtos/zephyr", 121776, "", "description"),
        ("other/repo", 5, "see", "description"),
    ]
    assert find_references([("x", "#12")], "") == []  # no default repository: bare numbers cannot resolve


ISSUE = {
    "number": 122778,
    "title": "VLAN frames overrun the TX buffer <!-- hidden -->",
    "state": "closed",
    "html_url": "https://github.com/o/r/issues/122778",
    "user": {"login": "reporter", "type": "User"},
    "labels": [{"name": "bug"}, {"name": "area: Ethernet"}],
    "comments": 2,
    "created_at": "2026-09-01T00:00:00Z",
    "closed_at": "2026-09-05T00:00:00Z",
    "body": "With CONFIG_NET_VLAN a tagged frame is four bytes longer than the buffer.\n\n"
    "Steps: enable VLAN, send a max-size frame.",
}
PR = {
    "number": 121776,
    "title": "Bluetooth: Controller: fix anonymous advertising",
    "state": "closed",
    "html_url": "https://github.com/o/r/pull/121776",
    "user": {"login": "jhedberg", "type": "User"},
    "labels": [],
    "comments": 0,
    "created_at": "2026-10-01T00:00:00Z",
    "closed_at": "2026-10-12T00:00:00Z",
    "body": "Split into three.",
    "pull_request": {"merged_at": "2026-10-12T00:00:00Z"},
}
COMMENTS = [
    {"user": {"login": "maint", "type": "User"}, "body": "Reproduced on nrf52840dk with VLAN enabled."},
    {"user": {"login": "reporter", "type": "User"}, "body": "Also happens with DSA tags."},
]
PERMS = {"reporter": "read", "maint": "admin", "jhedberg": "write"}


def _gh() -> GitHub:
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/repos/o/r/issues/122778":
            return httpx.Response(200, json=ISSUE)
        if p == "/repos/o/r/issues/121776":
            return httpx.Response(200, json=PR)
        if p == "/repos/o/r/issues/122778/comments":
            return httpx.Response(200, json=COMMENTS)
        if p == "/repos/other/repo/issues/5":
            return httpx.Response(
                200, json={**PR, "number": 5, "title": "other repo pr", "user": {"login": "x", "type": "User"}}
            )
        if "/collaborators/" in p:
            login = p.split("/collaborators/")[1].split("/")[0]
            return httpx.Response(200, json={"permission": PERMS.get(login, "none")})
        return httpx.Response(404, json={"message": "Not Found"})

    return GitHub("tok", "o/r", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_fetch_item_classifies_and_sanitizes():
    gh = _gh()
    issue = fetch_item(gh, Ref("o/r", 122778, "fixes"), comments=3, lockdown="label")
    assert issue.kind == "issue" and issue.status == "closed" and issue.labels == ["bug", "area: Ethernet"]
    assert issue.title == "VLAN frames overrun the TX buffer" and "hidden" not in issue.title
    assert issue.trust == "external" and [(a, t) for a, t, _ in issue.comments] == [
        ("maint", "maintainer"),
        ("reporter", "external"),
    ]
    pr = fetch_item(gh, Ref("o/r", 121776), comments=3, lockdown="label")
    assert (
        pr.kind == "pull_request"
        and pr.merged
        and pr.status == "merged"
        and pr.trust == "collaborator"
        and pr.comments == []
    )
    other = fetch_item(gh, Ref("other/repo", 5), comments=0, lockdown="label")
    assert other.kind == "pull_request" and other.trust == "unknown"  # permissions are only known for our repository
    missing = fetch_item(gh, Ref("o/r", 1), comments=0, lockdown="label")
    assert missing.kind == "missing" and "404" in missing.error


async def test_load_digest_and_tool():
    gh = _gh()
    linked = load_linked(
        gh,
        [("description", "Fixes #122778, follow-up to #121776, see other/repo#5, and #7 is me"), ("comment", "see #1")],
        repo_slug="o/r",
        own_number=7,
        max_items=3,
        lockdown="label",
    )
    assert [it.ref.number for it in linked.items] == [122778, 121776, 5]  # own PR excluded, cap respected (#1 dropped)
    digest = linked.digest()
    assert digest.startswith("3 linked issue(s)/pull request(s)")
    assert (
        "- o/r#122778 (fixes) issue, closed [bug, area: Ethernet]: VLAN frames overrun the TX buffer "
        "by @reporter (external)" in digest
    )
    assert "  With CONFIG_NET_VLAN a tagged frame is four bytes longer" in digest
    assert (
        "- o/r#121776 (follow-up to) pull request, merged: Bluetooth: Controller: fix anonymous advertising "
        "by @jhedberg (collaborator)" in digest
    )
    assert len(linked.digest(max_chars=200)) < 320 and "call `linked_issue`" in linked.digest(max_chars=200)

    full = await linked.tool_handler({"reference": "#122778"})
    assert full.startswith("# o/r#122778: VLAN frames overrun the TX buffer") and "Steps: enable VLAN" in full
    assert "@maint (maintainer):\nReproduced on nrf52840dk" in full
    assert (await linked.tool_handler({"reference": "122778"})).startswith("# o/r#122778")
    assert "not an issue or pull request reference" in await linked.tool_handler({"reference": "nonsense"})
    # An unlisted number is fetched on demand and remembered.
    missing = await linked.tool_handler({"reference": "#1"})
    assert missing.startswith("o/r#1 could not be fetched") and len(linked.items) == 4
    assert LINKED_ISSUE_TOOL["function"]["name"] == "linked_issue"


def test_opening_message_places_linked_items_before_the_discussion():
    pr = PullRequest(7, "t", "", "a", "main", "b" * 40, "pr", "h" * 40, "o/r", False)
    text = opening_message(pr, "ov", None, "", "o/r", "digest", linked="- o/r#1 issue, open: boom by @x (external)")
    assert text.index("### Linked issues and pull requests") < text.index("### Existing discussion")
    assert "<linked_items>\n- o/r#1 issue" in text
    assert "### Linked issues" not in opening_message(pr, "ov", None, "", "o/r", None, linked=None)


def test_cli_flag_and_config(monkeypatch):
    from zephyr_reviewer import cli
    from zephyr_reviewer.review.config import ReviewConfig

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert cli._config(cli.build_parser().parse_args(["review", "--base", "x", "--no-linked"])).include_linked is False
    monkeypatch.setenv("INPUT_INCLUDE_LINKED_ISSUES", "false")
    monkeypatch.setenv("INPUT_MAX_LINKED_ISSUES", "3")
    cfg = ReviewConfig.from_env()
    assert cfg.include_linked is False and cfg.max_linked_items == 3
    # Nothing is fetched without a PR, without a client, or when disabled.
    pr = PullRequest(7, "", "", "", "", "", "", "", "", False)
    assert cli._load_linked(cfg, _gh(), pr, None, None) is None
    _ = Linked, json  # referenced for readers
