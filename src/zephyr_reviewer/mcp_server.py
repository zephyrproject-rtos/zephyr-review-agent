# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""MCP server exposing Zephyr source intelligence for one pull request.

Runs over stdio (spawned per review) or streamable HTTP (a shared service).
The tool surface is the stable contract; see ``intel.service.SourceIntel``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from .intel import SourceIntel


# Typed outputs: these tools return human-readable text for the model *and* structured content for
# programmatic clients. The return annotation publishes the output schema; the function returns a
# CallToolResult carrying both.
class CommitOut(TypedDict):
    sha: str
    author: str
    subject: str


class ChangedFileOut(TypedDict):
    path: str
    status: str
    old_path: str | None
    additions: int
    deletions: int
    binary: bool
    areas: list[str]


class OverviewOut(TypedDict):
    base_sha: str
    head_sha: str
    shallow: bool
    commits: list[CommitOut]
    files: list[ChangedFileOut]


class FileGroupOut(TypedDict):
    name: str
    collaborators: list[str]


class AreaOut(TypedDict):
    name: str
    status: str
    maintainers: list[str]
    collaborators: list[str]
    labels: list[str]
    tests: list[str]
    file_groups: list[FileGroupOut]


class MaintainersOut(TypedDict):
    path: str
    areas: list[AreaOut]


class PlatformOut(TypedDict):
    identifier: str
    board: str
    full_name: str
    directory: str


class PlatformsOut(TypedDict):
    total_identifiers: int
    total_boards: int
    fragment: str
    platforms: list[PlatformOut]


class DefinitionOut(TypedDict):
    kind: str
    path: str
    line: int
    text: str


class DefinitionsOut(TypedDict):
    name: str
    definitions: list[DefinitionOut]


def _both(text: str, data: dict[str, Any]) -> Any:
    return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


INSTRUCTIONS = """\
Source intelligence for a Zephyr RTOS pull request. Every tool answers against the
pull request's head commit (and base commit where noted); nothing is executed.
Start with pr_overview, read full files with read_file before judging a hunk, and use
find_definition / find_references / kconfig_symbol / dt_binding to verify assumptions
instead of guessing. Results are plain text, bounded in size; narrow a query when told so.\
"""


HISTORY_INSTRUCTIONS = """\
Source intelligence for the Zephyr RTOS tree at one commit; nothing is executed. Read whole functions
with read_file, and use find_definition / find_references / kconfig_symbol / dt_binding to verify
assumptions instead of guessing. blame, file_history, search_history and show_commit say how code came
to be as it is. Results are plain text, bounded in size; narrow a query when told so.\
"""


def build_server(intel: SourceIntel, name: str = "zephyr-source-intel", *, history: bool = False) -> MCPServer:
    """``history`` adds the tools that trace a line to the commit that made it (issue triage). A review
    does not get them: its tool list is part of every request it pays for."""
    srv = MCPServer(name, instructions=HISTORY_INSTRUCTIONS if history else INSTRUCTIONS, version="0.1.0")

    def _guarded(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface the cause to the model, do not crash the server
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    async def guard(fn, *args, **kwargs):
        """Run a lookup in a worker thread: the intel is synchronous git plumbing, and several review
        stages call tools concurrently; blocking the event loop would serialize them behind the slowest."""
        return await asyncio.to_thread(_guarded, fn, *args, **kwargs)

    @srv.tool()
    async def pr_overview() -> OverviewOut:
        """Commits, changed files with +/- counts, and the MAINTAINERS areas each file falls in. Call this first."""
        return await guard(lambda: _both(intel.overview(), intel.overview_data()))

    @srv.tool(structured_output=False)
    async def commit_messages() -> str:
        """Full commit messages of the pull request, oldest first (Zephyr reviews the message, not only the code)."""
        return await guard(intel.commit_messages)

    @srv.tool(structured_output=False)
    async def file_diff(path: str, context_lines: int = 3) -> str:
        """Unified diff of one changed file between base and head, with the given context."""
        return await guard(intel.diff_for, path, context_lines)

    @srv.tool(structured_output=False)
    async def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file at the head commit with line numbers; lines added by the PR are marked '+'.
        At most 400 lines per call; omit end_line to read 400 lines from start_line."""
        return await guard(intel.read, path, start_line, end_line)

    @srv.tool(structured_output=False)
    async def read_file_at_base(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file as it was at the base commit, to compare old and new behavior."""
        return await guard(intel.read_base, path, start_line, end_line)

    @srv.tool(structured_output=False)
    async def search(
        pattern: str,
        path: str | None = None,
        regex: bool = True,
        whole_word: bool = False,
        ignore_case: bool = False,
        max_results: int = 100,
    ) -> str:
        """grep the head tree. path may be a directory (drivers/i2c) or a glob (drivers/**/*.c).
        Regex is POSIX extended as in git grep: use [[:space:]] not \\s, and whole_word instead of \\b."""
        return await guard(
            intel.search,
            pattern,
            path,
            regex=regex,
            whole_word=whole_word,
            ignore_case=ignore_case,
            max_results=max_results,
        )

    @srv.tool(structured_output=False)
    async def list_files(pattern: str) -> str:
        """List files matching a glob at the head commit, e.g. dts/bindings/i2c/*.yaml or tests/drivers/i2c/**."""
        return await guard(intel.list_files, pattern)

    @srv.tool()
    async def find_definition(name: str, path: str | None = None) -> DefinitionsOut:
        """Where a C function, macro, struct, typedef, Kconfig symbol (CONFIG_FOO), devicetree compatible
        (vendor,device) or CMake function is defined. Optionally restrict to a path."""
        return await guard(lambda: _both(intel.definition(name, path), intel.definition_data(name, path)))

    @srv.tool(structured_output=False)
    async def find_references(name: str, path: str | None = None, max_results: int = 60) -> str:
        """Who uses a symbol: counts per area, then individual sites, changed files first."""
        return await guard(intel.references, name, path, max_results)

    @srv.tool(structured_output=False)
    async def kconfig_symbol(name: str) -> str:
        """Every definition block of a Kconfig symbol (depends on, select, default, help) and its users
        in Kconfig, .conf/defconfig, CMake and source."""
        return await guard(intel.kconfig, name)

    @srv.tool(structured_output=False)
    async def dt_binding(compatible_or_name: str) -> str:
        """The devicetree binding YAML for a compatible, plus the drivers (DT_DRV_COMPAT) and nodes using it."""
        return await guard(intel.dt_binding, compatible_or_name)

    @srv.tool()
    async def maintainers(path: str) -> MaintainersOut:
        """MAINTAINERS.yml areas covering a path: status, maintainers, collaborators, labels, test identifiers."""
        return await guard(lambda: _both(intel.maintainers_for(path), intel.maintainers_data(path)))

    @srv.tool(structured_output=False)
    async def related_tests(path_or_symbol: str) -> str:
        """Twister suites (tests.yaml / sample.yaml) covering a path or mentioning a symbol, with tags and platforms."""
        return await guard(intel.related_tests, path_or_symbol)

    @srv.tool(structured_output=False)
    async def test_candidates() -> str:
        """Everything a test plan for this change should consider: suites the PR edits, MAINTAINERS test
        identifiers, implicated twister tags, suites covering the changed source (with scenarios, tags,
        integration_platforms, platform_allow), and the platforms implicated by changed boards, SoCs,
        architectures and driver bindings."""
        return await guard(intel.test_candidates)

    @srv.tool()
    async def list_platforms(fragment: str) -> PlatformsOut:
        """Valid twister platform identifiers (board/soc[/qualifiers]) whose name, board or vendor contains
        the fragment, e.g. 'nrf5340', 'qemu', 'native', 'stm32h7'. Use it to validate a test plan."""
        return await guard(lambda: _both(intel.platforms(fragment), intel.platforms_data(fragment)))

    @srv.tool(structured_output=False)
    async def file_history(path: str, count: int = 10) -> str:
        """Recent commits touching a path at the head commit."""
        return await guard(intel.history, path, count)

    @srv.tool(structured_output=False)
    async def blame(path: str, start_line: int, end_line: int) -> str:
        """git blame for a line range at the head commit (at most 200 lines)."""
        return await guard(intel.blame, path, start_line, end_line)

    @srv.tool(structured_output=False)
    async def project_guidelines(topic: str = "agents") -> str:
        """Zephyr contribution rules. Topics: agents, guidelines, expectations, reviewer, style-c, style-kconfig,
        style-cmake, style-devicetree, coding-guidelines, bindings, kconfig-tips, or a doc/ path."""
        return await guard(intel.guidelines, topic)

    if history:

        @srv.tool(structured_output=False)
        async def show_commit(commit: str) -> str:
            """One commit by id or release tag: its message, the files it touched, its patch, and the first
            releases that contain it. Use it on a commit that blame or search_history points at."""
            return await guard(intel.commit, commit)

        @srv.tool(structured_output=False)
        async def search_history(text: str, path: str | None = None, regex: bool = False, count: int = 15) -> str:
            """Commits that added or removed a piece of text (a call, a symbol, a condition), newest first;
            optionally under one path. With regex, commits whose patch has a changed line matching it. The
            oldest one listed is usually the commit that introduced the text."""
            return await guard(intel.search_history, text, path, regex, count)

    return srv


def add_server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", default=".", help="Path to the Zephyr checkout (default: cwd)")
    p.add_argument("--head", default="HEAD", help="Commit under review (default: HEAD)")
    p.add_argument("--base", required=True, help="Base commit or ref, e.g. upstream/main or a merge-base SHA")
    p.add_argument("--max-chars", type=int, default=24_000, help="Per-result size cap")
    p.add_argument("--history", action="store_true", help="Also offer the tools that trace code to a commit")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)


def _only_broken_pipes(exc: BaseException) -> bool:
    """True when an exception (group) consists solely of BrokenPipeError: the client went away."""
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(_only_broken_pipes(e) for e in exc.exceptions)
    return False


def serve(args: argparse.Namespace) -> int:
    intel = SourceIntel(args.repo, args.head, args.base, max_chars=args.max_chars)
    srv = build_server(intel, history=getattr(args, "history", False))
    try:
        if args.transport == "stdio":
            srv.run(transport="stdio")
        else:
            srv.run(transport="streamable-http", host=args.host, port=args.port)
    except BaseException as exc:  # noqa: BLE001 - classify, then re-raise anything real
        if _only_broken_pipes(exc):
            # The orchestrator closed the pipe (it finished or failed). Nothing to report here, and
            # printing a traceback would only hide the orchestrator's own message.
            try:
                sys.stdout.close()
            except Exception:  # noqa: BLE001
                pass
            return 0
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m zephyr_reviewer.mcp_server")
    add_server_args(p)
    return serve(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
