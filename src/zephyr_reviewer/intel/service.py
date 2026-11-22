# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""The source-intelligence facade.

This is the contract the MCP tools are written against. Today it is backed by
git plumbing over a local checkout; a shared service with an index, review
history and project knowledge replaces this class without touching the tools.
Every method returns text ready to be handed to a model, bounded in size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from . import devicetree, kconfig, prefetch, symbols, tests_index
from .diff import FileDiff, parse_unified_diff
from .maintainers import Maintainers
from .platforms import PlatformIndex
from .repo import Commit, GitRepo

_GENERIC_STEMS = {
    "common",
    "config",
    "device",
    "driver",
    "errno",
    "handler",
    "helpers",
    "kernel",
    "shared",
    "system",
    "thread",
    "timeout",
    "timers",
    "utils",
    "version",
}

_GUIDELINE_FILES = {
    "agents": "AGENTS.md",
    "guidelines": "doc/contribute/guidelines.rst",
    "expectations": "doc/contribute/contributor_expectations.rst",
    "reviewer": "doc/contribute/reviewer_expectations.rst",
    "style-c": "doc/contribute/style/code.rst",
    "style-kconfig": "doc/contribute/style/kconfig.rst",
    "style-cmake": "doc/contribute/style/cmake.rst",
    "style-devicetree": "doc/contribute/style/devicetree.rst",
    "coding-guidelines": "doc/contribute/coding_guidelines/index.rst",
    "bindings": "doc/build/dts/bindings-syntax.rst",
    "kconfig-tips": "doc/build/kconfig/tips.rst",
}


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    total: int


class SourceIntel:
    def __init__(self, root: str, head: str, base: str, *, max_chars: int = 24_000):
        self.head = GitRepo(root, head)
        self.base = GitRepo(root, base)
        self.base_sha = self.base.sha
        self.max_chars = max_chars

    # -------------------------------------------------------------- the change
    @cached_property
    def diff_text(self) -> str:
        return self.head.diff(self.base_sha)

    @cached_property
    def files(self) -> list[FileDiff]:
        return parse_unified_diff(self.diff_text)

    @cached_property
    def commits(self) -> list[Commit]:
        return self.head.commits_between(self.base_sha)

    def file_diff(self, path: str) -> FileDiff | None:
        for f in self.files:
            if f.path == path or f.old_path == path:
                return f
        return None

    @cached_property
    def maintainers(self) -> Maintainers | None:
        return Maintainers.from_repo(self.base)

    @cached_property
    def tags(self) -> tests_index.Tags | None:
        return tests_index.Tags.from_repo(self.base)

    @cached_property
    def platform_index(self) -> PlatformIndex:
        return PlatformIndex.build(self.head)

    # --------------------------------------------------------- structured views
    def overview_data(self) -> dict:
        return {
            "base_sha": self.base_sha,
            "head_sha": self.head.sha,
            "shallow": self.head.is_shallow(),
            "commits": [{"sha": c.sha, "author": c.author, "subject": c.subject} for c in self.commits],
            "files": [
                {
                    "path": f.path,
                    "status": f.status,
                    "old_path": f.old_path if f.status == "R" else None,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "binary": f.binary,
                    "areas": self._area_names(f.path),
                }
                for f in self.files
            ],
        }

    def maintainers_data(self, path: str) -> dict:
        m = self.maintainers
        areas = []
        if m is not None:
            for area, groups in m.areas_for(path, is_dir=self.head.is_dir(path)):
                areas.append(
                    {
                        "name": area.name,
                        "status": area.status,
                        "maintainers": list(area.maintainers),
                        "collaborators": list(area.collaborators),
                        "labels": list(area.labels),
                        "tests": list(area.tests),
                        "file_groups": [{"name": g.name, "collaborators": list(g.collaborators)} for g in groups],
                    }
                )
        return {"path": path, "areas": areas}

    def platforms_data(self, fragment: str) -> dict:
        idx = self.platform_index
        hits = idx.search(fragment) if fragment.strip() else []
        return {
            "total_identifiers": len(idx.by_identifier),
            "total_boards": len(idx.boards),
            "fragment": fragment,
            "platforms": [
                {
                    "identifier": i,
                    "board": idx.by_identifier[i].name,
                    "full_name": idx.by_identifier[i].full_name,
                    "directory": idx.by_identifier[i].directory,
                }
                for i in hits
            ],
        }

    def definition_data(self, name: str, path: str | None = None) -> dict:
        defs = symbols.find_definitions(self.head, name, paths=path)
        return {
            "name": name,
            "definitions": [{"kind": d.kind, "path": d.path, "line": d.line, "text": d.text[:200]} for d in defs],
        }

    def prefetch(self, budget_chars: int = 24_000, *, with_callers: bool = True) -> str:
        """Enclosing functions of every hunk and their callers, within a budget, for the opening message."""
        return prefetch.prefetch(self.head, self.files, budget_chars, with_callers=with_callers)

    # ------------------------------------------------------------------- tools
    def overview(self) -> str:
        out = [
            f"Base: {self.base_sha[:12]}  Head: {self.head.sha[:12]}",
            f"Commits: {len(self.commits)}   Files changed: {len(self.files)}",
            "",
            "## Commits",
        ]
        for c in self.commits:
            out.append(f"- {c.short} {c.subject}  ({c.author})")
        out += ["", "## Changed files (status, +added/-removed, areas)"]
        for f in self.files:
            areas = self._area_names(f.path)
            tag = f" [{', '.join(areas)}]" if areas else ""
            rename = f" (from {f.old_path})" if f.status == "R" else ""
            out.append(f"- {f.status} {f.path}{rename} +{f.additions}/-{f.deletions}{tag}")
        if self.head.is_shallow():
            out += ["", "Note: shallow clone; history and blame reach only a limited depth."]
        return self._clip("\n".join(out))

    def commit_messages(self) -> str:
        out = []
        for c in self.commits:
            out.append(f"commit {c.sha}\nAuthor: {c.author}\n\n    {c.subject}\n")
            if c.body:
                out.append("\n".join("    " + line for line in c.body.splitlines()))
            out.append("")
        return self._clip("\n".join(out) or "No commits between base and head.")

    def diff_for(self, path: str, context: int = 3) -> str:
        fd = self.file_diff(path)
        if fd is None:
            return f"{path} is not changed by this pull request. Changed files:\n" + "\n".join(
                f"  {f.path}" for f in self.files
            )
        if fd.binary:
            return f"{path}: binary file ({fd.status})"
        text = self.head.diff(
            self.base_sha, [path] + ([fd.old_path] if fd.old_path and fd.old_path != path else []), context=context
        )
        return self._clip(text)

    def read(self, path: str, start: int = 1, end: int | None = None, max_lines: int = 400) -> str:
        text = self.head.show(path)
        if text is None:
            if self.base.show(path) is not None:
                return f"{path} does not exist at the head commit (it was deleted by this pull request)."
            cands = [p for p in self.head.files() if p.endswith("/" + path.rsplit("/", 1)[-1])][:10]
            hint = ("\nSimilar paths:\n  " + "\n  ".join(cands)) if cands else ""
            return f"{path} not found at the head commit.{hint}"
        lines = text.splitlines()
        total = len(lines)
        start = max(1, start)
        if end is None or end < start:
            end = min(total, start + max_lines - 1)
        end = min(end, total, start + max_lines - 1)
        fd = self.file_diff(path)
        marks = fd.added_lines if fd else set()
        body = []
        for n in range(start, end + 1):
            mark = "+" if n in marks else " "
            body.append(f"{n:5d}{mark}| {lines[n - 1]}")
        header = f"# {path}  lines {start}-{end} of {total}" + ("  ('+' marks lines added by this PR)" if marks else "")
        return self._clip(header + "\n" + "\n".join(body))

    def read_base(self, path: str, start: int = 1, end: int | None = None, max_lines: int = 400) -> str:
        text = self.base.show(path)
        if text is None:
            return f"{path} does not exist at the base commit."
        lines = text.splitlines()
        total = len(lines)
        start = max(1, start)
        end = min(end or (start + max_lines - 1), total, start + max_lines - 1)
        body = [f"{n:5d} | {lines[n - 1]}" for n in range(start, end + 1)]
        return self._clip(f"# {path} @ base {self.base_sha[:12]}  lines {start}-{end} of {total}\n" + "\n".join(body))

    def search(
        self,
        pattern: str,
        path: str | None = None,
        *,
        regex: bool = True,
        whole_word: bool = False,
        ignore_case: bool = False,
        max_results: int = 100,
    ) -> str:
        hits, truncated = self.head.grep(
            pattern, paths=path, regex=regex, word=whole_word, ignore_case=ignore_case, max_results=max_results
        )
        if not hits:
            return f"No matches for {pattern!r}" + (f" under {path}" if path else "") + "."
        out = [f"{h.path}:{h.line}: {h.text.strip()[:200]}" for h in hits]
        if truncated:
            out.append(f"... truncated at {max_results} results; narrow the path or pattern.")
        return self._clip("\n".join(out))

    def list_files(self, pattern: str, limit: int = 200) -> str:
        files = self.head.glob(pattern, limit=limit + 1)
        if not files:
            return f"No files match {pattern!r}."
        more = "\n... more results, narrow the pattern." if len(files) > limit else ""
        return self._clip("\n".join(files[:limit]) + more)

    def definition(self, name: str, path: str | None = None) -> str:
        defs = symbols.find_definitions(self.head, name, paths=path)
        if not defs:
            return (
                f"No definition of {name!r} found"
                + (f" under {path}" if path else "")
                + ". Try find_references, or search with a regex."
            )
        out = []
        for d in defs[:40]:
            out.append(f"[{d.kind}] {d.path}:{d.line}: {d.text[:200]}")
        if len(defs) > 40:
            out.append(f"... {len(defs) - 40} more")
        return self._clip("\n".join(out))

    def references(self, name: str, path: str | None = None, max_results: int = 60) -> str:
        refs = symbols.find_references(self.head, name, paths=path)
        if not refs.hits:
            return f"No references to {name!r} found" + (f" under {path}" if path else "") + "."
        out = [f"{len(refs.hits)}{'+' if refs.truncated else ''} references to {name}. By area:"]
        for d, n in list(refs.by_dir.items())[:15]:
            out.append(f"  {n:5d}  {d}")
        out.append("")
        changed = {f.path for f in self.files}
        first = sorted(refs.hits, key=lambda h: (h.path not in changed, h.path, h.line))[:max_results]
        for h in first:
            flag = " *" if h.path in changed else ""
            out.append(f"{h.path}:{h.line}:{flag} {h.text.strip()[:160]}")
        if len(refs.hits) > max_results:
            out.append(f"... showing {max_results}; pass a path to narrow. '*' marks files changed by this PR.")
        return self._clip("\n".join(out))

    def kconfig(self, name: str) -> str:
        defs = kconfig.definitions(self.head, name)
        sym = name[len("CONFIG_") :] if name.startswith("CONFIG_") else name
        out = []
        if not defs:
            out.append(f"No Kconfig definition of {sym} at the head commit.")
            if kconfig.definitions(self.base, name):
                out.append("It exists at the base commit, so this pull request removes or renames it.")
        for d in defs:
            out.append(f"## {d.path}:{d.line}\n{d.block}\n")
        refs = kconfig.references(self.head, name)
        for key, label in (
            ("kconfig", "Kconfig references"),
            ("config", ".conf/defconfig"),
            ("cmake", "CMake"),
            ("source", "Source"),
        ):
            items = refs[key]
            if items:
                out.append(f"## {label} ({len(items)})")
                out += [f"  {p}:{ln}: {t[:140]}" for p, ln, t in items[:40]]
                if len(items) > 40:
                    out.append(f"  ... {len(items) - 40} more")
        return self._clip("\n".join(out))

    def dt_binding(self, compatible_or_name: str) -> str:
        bindings = devicetree.bindings_for_compatible(self.head, compatible_or_name)
        out = []
        if bindings:
            for b in bindings:
                out.append(f"## {b.path}\n```yaml\n{b.text}\n```")
        else:
            names = devicetree.bindings_by_name(self.head, compatible_or_name)
            if not names:
                return f"No binding declares compatible {compatible_or_name!r} and no binding file name contains it."
            out.append("No binding declares that exact compatible. Binding files with a matching name:")
            out += [f"  {n}" for n in names]
            return "\n".join(out)
        users = devicetree.compatible_users(self.head, compatible_or_name)
        for key, label in (
            ("drivers", "Drivers binding to it (DT_DRV_COMPAT)"),
            ("devicetree", "Devicetree nodes"),
            ("other", "Other references"),
        ):
            items = users[key]
            if items:
                out.append(f"## {label} ({len(items)})")
                out += [f"  {p}:{ln}: {t[:140]}" for p, ln, t in items[:30]]
        return self._clip("\n".join(out))

    def maintainers_for(self, path: str) -> str:
        m = self.maintainers
        if m is None:
            return "No MAINTAINERS.yml at the base commit."
        matches = m.areas_for(path, is_dir=self.head.is_dir(path))
        if not matches:
            return f"No MAINTAINERS.yml area covers {path}."
        out = []
        for area, groups in matches:
            out.append(f"## {area.name} ({area.status})")
            if area.maintainers:
                out.append("  maintainers: " + ", ".join("@" + x for x in area.maintainers))
            if area.collaborators:
                out.append("  collaborators: " + ", ".join("@" + x for x in area.collaborators))
            if area.labels:
                out.append("  labels: " + ", ".join(area.labels))
            if area.tests:
                out.append("  tests: " + ", ".join(area.tests))
            for g in groups:
                extra = (": " + ", ".join("@" + x for x in g.collaborators)) if g.collaborators else ""
                out.append(f"  file-group {g.name}{extra}")
        return "\n".join(out)

    def history(self, path: str, n: int = 10) -> str:
        commits = self.head.log(path, n)
        if not commits:
            return f"No history for {path} (new file, or outside the clone depth)."
        out = [f"{c.short} {c.subject}  ({c.author})" for c in commits]
        if self.head.is_shallow():
            out.append("(shallow clone: older history not available)")
        return "\n".join(out)

    def blame(self, path: str, start: int, end: int) -> str:
        if end < start:
            start, end = end, start
        end = min(end, start + 200)
        text = self.head.blame(path, start, end)
        return self._clip(text or f"No blame output for {path}:{start}-{end}.")

    # ----------------------------------------------------------------- history
    def commit(self, rev: str) -> str:
        """A commit by id or tag: message, statistics, patch, and the first releases that have it."""
        sha = self.head.resolve(rev)
        if sha is None:
            return f"No commit {rev!r} in this clone."
        text = self.head.show_commit(sha)
        if not self.head.is_ancestor(sha):
            text = "(not in the history of the commit under analysis)\n" + text
        releases = self.head.releases_containing(sha)
        if releases:
            text = f"First releases with this commit: {', '.join(releases[:3])}\n" + text
        return self._clip(text)

    def search_history(self, text: str, path: str | None = None, regex: bool = False, n: int = 15) -> str:
        """Commits that added or removed ``text``: how a line, a call or a symbol came to be as it is."""
        text = (text or "").strip()
        if len(text) < 3:
            return "Give at least three characters to search the history for."
        commits = self.head.pickaxe(text, path, max(1, min(int(n), 40)), regex)
        if not commits:
            where = f" under {path}" if path else ""
            note = " (shallow clone: older history not available)" if self.head.is_shallow() else ""
            return f"No commit{where} added or removed {text!r}{note}."
        out = [f"{c.short} {c.subject}  ({c.author})" for c in commits]
        if self.head.is_shallow():
            out.append("(shallow clone: older history not available)")
        return "\n".join(out)

    def related_tests(self, path_or_symbol: str) -> str:
        out: list[str] = []
        subject = path_or_symbol.strip()
        suites: list[tests_index.TestSuite] = []
        if ("/" in subject) or ("." in subject and self.head.exists(subject)):
            if subject.startswith(("tests/", "samples/")):
                s = tests_index.suite_containing(self.head, subject)
                if s:
                    suites.append(s)
            m = self.maintainers
            if m:
                for area, _ in m.areas_for(subject, is_dir=self.head.is_dir(subject)):
                    if area.tests:
                        out.append(f"MAINTAINERS area '{area.name}' lists test identifiers: {', '.join(area.tests)}")
            stem = subject.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if len(stem) >= 4:
                suites += tests_index.suites_referencing(self.head, stem, limit=10)
            top = "/".join(subject.split("/")[:-1][:2])
            if top.startswith(("drivers", "subsys", "lib", "kernel", "arch")):
                suites += tests_index.suites_under(self.head, "tests/" + top, limit=15)
        else:
            suites = tests_index.suites_referencing(self.head, subject, limit=25)
        seen: set[str] = set()
        for s in suites:
            if s.meta_path in seen:
                continue
            seen.add(s.meta_path)
            out.append(f"## {s.meta_path}  ({s.why})")
            if s.scenarios:
                out.append("  scenarios: " + ", ".join(s.scenarios[:20]) + (" ..." if len(s.scenarios) > 20 else ""))
            if s.tags:
                out.append("  tags: " + ", ".join(s.tags))
            if s.platforms:
                out.append("  platforms: " + ", ".join(s.platforms[:12]))
        if not out:
            return f"No twister suites found that reference {subject!r}."
        return self._clip("\n".join(out))

    def platforms(self, fragment: str = "") -> str:
        idx = self.platform_index
        if not fragment.strip():
            n_ids, n_boards = len(idx.by_identifier), len(idx.boards)
            return f"{n_ids} platform identifiers across {n_boards} boards. Pass a fragment to filter."
        hits = idx.search(fragment)
        if not hits:
            return f"No platform identifier, board name or vendor matches {fragment!r}."
        out = []
        for ident in hits:
            b = idx.by_identifier[ident]
            out.append(f"{ident}  ({b.full_name or b.name}; {b.directory})")
        if len(hits) >= 60:
            out.append("... narrow the fragment for more.")
        return self._clip("\n".join(out))

    def test_candidates(self) -> str:
        """Everything a test plan for this change should consider: suites, identifiers, tags, platforms."""
        changed = [f.path for f in self.files if f.status != "D"]
        changed_all = [f.path for f in self.files]
        out: list[str] = []
        seen: set[str] = set()

        def add_suite(s: tests_index.TestSuite | None) -> None:
            if s and s.meta_path not in seen:
                seen.add(s.meta_path)
                out.extend(s.describe())

        # 1. Suites the change edits directly.
        owned = tests_index.changed_suite_dirs(self.head, changed)
        out.append("## Suites changed by this pull request (run these first)")
        if owned:
            for d, files in sorted(owned.items()):
                add_suite(tests_index.suite_for_dir(self.head, d, f"edits {', '.join(files[:4])}"))
        else:
            out.append("(none)")

        # 2. MAINTAINERS test identifiers and tags for the touched areas.
        ids: dict[str, list[str]] = {}
        m = self.maintainers
        if m:
            for path in changed_all:
                for area, _ in m.areas_for(path):
                    if area.tests:
                        ids.setdefault(area.name, area.tests)
        out.append("")
        out.append("## Test identifiers from MAINTAINERS.yml (scenario-id prefixes for the touched areas)")
        if ids:
            out += [f"- {name}: {', '.join(tests)}" for name, tests in sorted(ids.items())]
        else:
            out.append("(none)")
        tag_names = self.tags.for_paths(changed_all) if self.tags else []
        out.append("")
        out.append("## Twister tags implicated (scripts/ci/tags.yaml): " + (", ".join(tag_names) or "(none)"))

        # 3. Suites covering the changed source: by area directory, by tag, by symbol reference.
        out.append("")
        out.append("## Suites covering the changed source")
        before = len(seen)
        tops: set[str] = set()
        stems: set[str] = set()
        for path in changed:
            if path.startswith(("tests/", "samples/", "doc/", "boards/", ".github/")):
                continue
            parts = path.split("/")
            dirs = parts[:-1]
            stem = parts[-1].rsplit(".", 1)[0]
            if dirs and dirs[0] == "kernel":
                tops.add("tests/kernel")
            elif dirs and dirs[0] in ("drivers", "subsys", "lib", "arch"):
                tops.add("tests/" + "/".join(dirs[:2]))
            elif len(dirs) >= 3 and dirs[0] == "include" and dirs[1] == "zephyr":
                # include/zephyr/<area>/<name>.h -> tests/<area>/<name>, else tests/<area>/<sub>
                area = dirs[2:4] if len(dirs) > 3 else [dirs[2], stem]
                cand = "tests/" + "/".join(area)
                tops.add(cand if self.head.is_dir(cand) else "tests/" + dirs[2])
            if path.endswith((".c", ".h")) and len(stem) >= 6 and stem.lower() not in _GENERIC_STEMS:
                stems.add(stem)
        for top in sorted(tops):
            for s in tests_index.suites_under(self.head, top, limit=12):
                add_suite(s)
        for s in tests_index.suites_with_tags(self.head, tag_names, limit=20):
            add_suite(s)
        for stem in sorted(stems)[:8]:
            for s in tests_index.suites_referencing(self.head, stem, limit=5):
                add_suite(s)
        if len(seen) == before:
            out.append("(none found; use related_tests and list_files tests/** to look further)")

        # 4. Platforms implicated by the change.
        out.append("")
        out.append("## Platforms implicated by the change")
        idx = self.platform_index
        board_idents: dict[str, list[str]] = {}
        for path in changed_all:
            if path.startswith("boards/"):
                idents = idx.for_path(path)
                if idents:
                    board_idents.setdefault(path.rsplit("/", 1)[0], idents)
        if board_idents:
            for d, idents in sorted(board_idents.items()):
                out.append(f"- board changed, {d}: " + ", ".join(idents))
        compat_boards: dict[str, set[str]] = {}
        for path in changed:
            if path.startswith("drivers/") and path.endswith(".c"):
                text = self.head.show(path, max_bytes=200_000) or ""
                for mm in re.finditer(r"^\s*#\s*define\s+DT_DRV_COMPAT\s+(\w+)", text, re.M):
                    drv = mm.group(1)
                    compats = self._compatibles_for_drv(drv)
                    for compat in compats:
                        users = devicetree.compatible_users(self.head, compat, max_results=400)
                        for p, _ln, _t in users["devicetree"]:
                            if p.startswith("boards/"):
                                compat_boards.setdefault(compat, set()).update(idx.for_path(p))
                            else:
                                via = self._boards_including(p)
                                compat_boards.setdefault(compat, set()).update(via)
                                compat_boards.setdefault(compat, set()).add(f"(dtsi: {p}: {len(via)} boards)")
        for compat, idents in sorted(compat_boards.items()):
            boards = sorted(i for i in idents if not i.startswith("(dtsi"))
            dtsis = sorted(i for i in idents if i.startswith("(dtsi"))
            line = f"- devicetree nodes with compatible {compat}: " + (", ".join(boards[:25]) or "no board DTS")
            if len(boards) > 25:
                line += f" (+{len(boards) - 25} more)"
            if dtsis:
                line += "; via SoC/shield dtsi: " + ", ".join(d[7:-1] for d in dtsis[:6])
            out.append(line)
        archs = sorted({p.split("/")[1] for p in changed_all if p.startswith("arch/") and p.count("/") >= 2})
        if archs:
            out.append("- architectures touched: " + ", ".join(archs) + " (pick emulation platforms for each)")
        socs = sorted({"/".join(p.split("/")[1:3]) for p in changed_all if p.startswith("soc/") and p.count("/") >= 2})
        if socs:
            out.append("- SoC families touched: " + ", ".join(socs) + " (use list_platforms to find their boards)")
        if not (board_idents or compat_boards or archs or socs):
            out.append("(no board, SoC, arch or driver-binding change; prefer each suite's integration_platforms)")
        out.append("")
        out.append(
            "Scenario ids are '<path>/<scenario>' for twister -s; validate platform identifiers with list_platforms."
        )
        return self._clip("\n".join(out))

    def _boards_including(self, dtsi_path: str, depth: int = 3) -> set[str]:
        """Platform identifiers of boards whose DTS includes ``dtsi_path``, following dtsi-to-dtsi includes."""
        idx = self.platform_index
        found: set[str] = set()
        seen: set[str] = set()
        frontier = [dtsi_path]
        for _ in range(depth):
            next_frontier: list[str] = []
            for path in frontier:
                if path in seen:
                    continue
                seen.add(path)
                base = path.rsplit("/", 1)[-1]
                hits, _ = self.head.grep(
                    rf"#include[[:space:]]+[<\"].*{re.escape(base)}[>\"]", regex=True, max_results=600
                )
                for h in hits:
                    if h.path.startswith("boards/") and h.path.endswith((".dts", ".dtsi")):
                        found.update(idx.for_path(h.path))
                    elif h.path.endswith(".dtsi") and h.path not in seen:
                        next_frontier.append(h.path)
            frontier = next_frontier
            if not frontier:
                break
        return found

    def _compatibles_for_drv(self, drv_compat: str) -> list[str]:
        """vnd_foo_i2c -> ['vnd,foo-i2c'] by matching binding compatibles."""
        pattern = re.escape(drv_compat).replace("_", "[,_-]")
        hits, _ = self.head.grep(
            rf'^[[:space:]]*compatible:[[:space:]]*"{pattern}"',
            paths="dts/bindings/**/*.y*ml",
            regex=True,
            max_results=20,
        )
        out: list[str] = []
        for h in hits:
            m = re.search(r'compatible:\s*"([^"]+)"', h.text)
            if m and m.group(1).replace(",", "_").replace("-", "_") == drv_compat and m.group(1) not in out:
                out.append(m.group(1))
        return out

    def guidelines(self, topic: str = "agents") -> str:
        key = topic.strip().lower()
        path = _GUIDELINE_FILES.get(key, key if "/" in key else None)
        if path is None:
            return "Unknown topic. Available: " + ", ".join(sorted(_GUIDELINE_FILES)) + ", or a doc/ path."
        text = self.head.show(path) or self.base.show(path)
        if text is None:
            return f"{path} not found in this tree."
        return self._clip(f"# {path}\n{text}")

    # ----------------------------------------------------------------- helpers
    def _area_names(self, path: str) -> list[str]:
        m = self.maintainers
        if m is None:
            return []
        return [a.name for a, _ in m.areas_for(path)]

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + f"\n... [truncated to {self.max_chars} characters; ask for a narrower range]"
