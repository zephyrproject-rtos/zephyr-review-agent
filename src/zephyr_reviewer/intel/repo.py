# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""Read-only git access pinned to one commit.

All reads go through ``git show``, ``git grep``, ``git ls-tree`` and friends
with an explicit tree-ish, so the working tree is never consulted and no file
from the commit under inspection is ever executed. A shallow clone is fine:
only history-walking queries (log, blame) degrade, and they say so.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_GREP_LINE = re.compile(r"^(?P<path>.*?):(?P<line>\d+):(?P<text>.*)$")


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class GrepHit:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class Commit:
    sha: str
    author: str
    subject: str
    body: str

    @property
    def short(self) -> str:
        return self.sha[:12]


def _pathspec(pattern: str | None) -> list[str]:
    """Turn a user-facing path or glob into a git pathspec.

    ``drivers/i2c`` and ``drivers/i2c/`` select a subtree, ``drivers/**/*.c``
    is a glob matched across directory separators.
    """
    if not pattern:
        return []
    pattern = pattern.strip()
    if not pattern or pattern in (".", "/"):
        return []
    if any(ch in pattern for ch in "*?["):
        return [f":(glob){pattern}"]
    return [pattern.rstrip("/")]


_REVISION = re.compile(r"^(?!.*\.\.)[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")  # one commit: no option, no range


class GitRepo:
    def __init__(self, root: str | Path, ref: str):
        self.root = Path(root).resolve()
        if not (self.root / ".git").exists() and not (self.root / "HEAD").exists():
            raise GitError(f"{self.root} is not a git repository")
        self.ref = ref
        self.sha = self.rev_parse(ref)
        self._files: list[str] | None = None
        self._fileset: set[str] | None = None
        self._dirs: set[str] | None = None

    # ------------------------------------------------------------------ plumbing
    def _run(self, *args: str, check: bool = True, max_bytes: int | None = None) -> str:
        cmd = ["git", "-C", str(self.root), *args]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if check and proc.returncode != 0:
            raise GitError(f"{' '.join(args[:3])} failed: {proc.stderr.decode(errors='replace').strip()}")
        out = proc.stdout
        if max_bytes is not None and len(out) > max_bytes:
            out = out[:max_bytes]
        return out.decode("utf-8", errors="replace")

    def _run_lines(self, *args: str, max_lines: int) -> tuple[list[str], bool]:
        """Stream stdout and stop after ``max_lines``; returns (lines, truncated)."""
        cmd = ["git", "-C", str(self.root), *args]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        lines: list[str] = []
        truncated = False
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                if len(lines) >= max_lines:
                    truncated = True
                    break
                lines.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        return lines, truncated

    def rev_parse(self, ref: str) -> str:
        return self._run("rev-parse", "--verify", f"{ref}^{{commit}}").strip()

    def has_commit(self, sha: str) -> bool:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, check=False
        )
        return proc.returncode == 0

    def is_shallow(self) -> bool:
        return self._run("rev-parse", "--is-shallow-repository").strip() == "true"

    # --------------------------------------------------------------------- files
    def files(self) -> list[str]:
        if self._files is None:
            out = self._run("ls-tree", "-r", "--name-only", "-z", self.sha)
            self._files = [p for p in out.split("\0") if p]
        return self._files

    def fileset(self) -> set[str]:
        if self._fileset is None:
            self._fileset = set(self.files())
        return self._fileset

    def exists(self, path: str) -> bool:
        return path in self.fileset()

    def is_dir(self, path: str) -> bool:
        if self._dirs is None:
            dirs: set[str] = set()
            for p in self.files():
                while "/" in p:
                    p = p.rsplit("/", 1)[0]
                    if p in dirs:
                        break
                    dirs.add(p)
            self._dirs = dirs
        return path.rstrip("/") in self._dirs

    def glob(self, pattern: str, limit: int = 500) -> list[str]:
        pattern = pattern.strip()
        if pattern.endswith("/"):
            pattern += "**"
        regex = _glob_to_regex(pattern)
        out = [p for p in self.files() if regex.match(p)]
        return out[:limit]

    def show(self, path: str, max_bytes: int = 2_000_000) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.sha}:{path}"], capture_output=True, check=False
        )
        if proc.returncode != 0:
            return None
        data = proc.stdout[:max_bytes]
        if b"\0" in data[:8000]:
            return None  # binary
        return data.decode("utf-8", errors="replace")

    def show_many(self, paths: list[str], max_bytes_each: int = 400_000) -> dict[str, str]:
        """Contents of many files in one ``git cat-file --batch`` round trip (binary and missing files skipped)."""
        if not paths:
            return {}
        proc = subprocess.Popen(
            ["git", "-C", str(self.root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert proc.stdin is not None and proc.stdout is not None
        request = "".join(f"{self.sha}:{p}\n" for p in paths).encode()
        out, _ = proc.communicate(request)
        result: dict[str, str] = {}
        pos = 0
        for path in paths:
            nl = out.find(b"\n", pos)
            if nl == -1:
                break
            header = out[pos:nl].decode(errors="replace").split()
            pos = nl + 1
            if len(header) < 3:  # "<sha>:<path> missing"
                continue
            size = int(header[2])
            data = out[pos : pos + size]
            pos += size + 1  # trailing newline after the object
            if b"\0" in data[:8000] or len(data) > max_bytes_each:
                continue
            result[path] = data.decode("utf-8", errors="replace")
        return result

    # ---------------------------------------------------------------------- grep
    def grep(
        self,
        pattern: str,
        *,
        paths: str | None = None,
        regex: bool = True,
        word: bool = False,
        ignore_case: bool = False,
        max_results: int = 200,
    ) -> tuple[list[GrepHit], bool]:
        args = ["grep", "-n", "-I", "--no-color"]
        args.append("-E" if regex else "-F")
        if word:
            args.append("-w")
        if ignore_case:
            args.append("-i")
        args += ["-e", pattern, self.sha, "--", *_pathspec(paths)]
        lines, truncated = self._run_lines(*args, max_lines=max_results)
        prefix = self.sha + ":"
        hits: list[GrepHit] = []
        for line in lines:
            if line.startswith(prefix):
                line = line[len(prefix) :]
            m = _GREP_LINE.match(line)
            if m:
                hits.append(GrepHit(m["path"], int(m["line"]), m["text"]))
        return hits, truncated

    # ------------------------------------------------------------------- history
    def log(self, path: str | None, n: int = 10) -> list[Commit]:
        fmt = "%H%x00%an%x00%s%x00%b%x1e"
        args = ["log", f"-n{n}", f"--format={fmt}", self.sha]
        if path:
            args += ["--", path]
        out = self._run(*args, check=False)
        return _parse_commits(out)

    def commits_between(self, base_sha: str, max_count: int = 100) -> list[Commit]:
        fmt = "%H%x00%an%x00%s%x00%b%x1e"
        out = self._run("log", f"-n{max_count}", "--reverse", f"--format={fmt}", f"{base_sha}..{self.sha}", check=False)
        return _parse_commits(out)

    def blame(self, path: str, start: int, end: int) -> str:
        return self._run(
            "blame", "--date=short", "--abbrev=12", "-L", f"{start},{end}", self.sha, "--", path, check=False
        )

    def resolve(self, rev: str) -> str | None:
        """The full id of a commit named by an id or a tag, or None. Anything that is not a plain revision
        name (an option, a range, a path) is refused before it reaches git."""
        rev = (rev or "").strip()
        if not _REVISION.match(rev):
            return None
        out = self._run("rev-parse", "--verify", "--quiet", "--end-of-options", f"{rev}^{{commit}}", check=False)
        sha = out.strip()
        return sha if re.fullmatch(r"[0-9a-f]{40,64}", sha) else None

    def is_ancestor(self, sha: str, of: str | None = None) -> bool:
        proc = subprocess.run(
            ["git", "-C", str(self.root), "merge-base", "--is-ancestor", sha, of or self.sha],
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0

    def commit_info(self, sha: str) -> dict[str, str] | None:
        """One commit's id, author, date, subject and body."""
        fmt = "%H%x00%an%x00%ae%x00%ad%x00%s%x00%b"
        out = self._run("show", "-s", "--date=short", f"--format={fmt}", sha, check=False)
        parts = out.split("\0")
        if len(parts) < 6:
            return None
        keys = ("sha", "author", "email", "date", "subject", "body")
        return {k: v.strip() for k, v in zip(keys, parts, strict=False)}

    def show_commit(self, sha: str, max_bytes: int = 200_000) -> str:
        """A commit's message, its file statistics and its patch."""
        return self._run(
            "show",
            "--no-color",
            "--no-ext-diff",
            "--stat=120",
            "--patch",
            "--date=short",
            "--format=fuller",
            sha,
            check=False,
            max_bytes=max_bytes,
        )

    def pickaxe(self, text: str, path: str | None = None, n: int = 15, regex: bool = False) -> list[Commit]:
        """Commits reachable from this one that changed the number of occurrences of ``text`` (or, with
        ``regex``, whose patch has an added or removed line matching it), newest first."""
        fmt = "%H%x00%an%x00%s%x00%b%x1e"
        args = ["log", f"-n{n}", f"--format={fmt}", ("-G" if regex else "-S") + text, self.sha]
        if path:
            args += ["--", *_pathspec(path)]
        return _parse_commits(self._run(*args, check=False))

    def releases_containing(self, sha: str, pattern: str = "v[0-9]*", limit: int = 6) -> list[str]:
        """Release tags whose history has the commit, oldest first."""
        out = self._run(
            "tag", "--list", pattern, "--contains", sha, "--sort=creatordate", "--merged", self.sha, check=False
        )
        return [t for t in out.splitlines() if t.strip()][:limit]

    # ---------------------------------------------------------------------- diff
    def merge_base(self, other: str) -> str:
        return self._run("merge-base", self.sha, other).strip()

    def diff(self, base_sha: str, paths: list[str] | None = None, context: int = 3, max_bytes: int = 4_000_000) -> str:
        args = ["diff", "--no-color", "--no-ext-diff", f"-U{context}", "--find-renames", base_sha, self.sha]
        if paths:
            args += ["--", *paths]
        return self._run(*args, max_bytes=max_bytes)

    def diff_stat(self, base_sha: str) -> str:
        return self._run("diff", "--no-color", "--stat=120", base_sha, self.sha)

    def name_status(self, base_sha: str) -> list[tuple[str, str, str | None]]:
        """[(status, path, old_path)] between base and this commit."""
        out = self._run("diff", "--name-status", "--find-renames", "-z", base_sha, self.sha)
        parts = [p for p in out.split("\0")]
        result: list[tuple[str, str, str | None]] = []
        i = 0
        while i < len(parts):
            status = parts[i]
            if not status:
                i += 1
                continue
            if status[0] in "RC":
                result.append((status[0], parts[i + 2], parts[i + 1]))
                i += 3
            else:
                result.append((status[0], parts[i + 1], None))
                i += 2
        return result


def _parse_commits(raw: str) -> list[Commit]:
    commits = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        fields = rec.split("\0")
        if len(fields) < 4:
            continue
        sha, author, subject, body = fields[0].strip(), fields[1], fields[2], fields[3]
        commits.append(Commit(sha=sha, author=author, subject=subject, body=body.strip()))
    return commits


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """fnmatch with ``**`` crossing directories and ``*`` stopping at ``/``."""
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i : i + 2] == "**":
                out.append(".*")
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    out[-1] = "(?:.*/)?"
                    i += 1
                continue
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = pattern.find("]", i)
            if j == -1:
                out.append(re.escape(ch))
            else:
                out.append(pattern[i : j + 1])
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def fnmatch_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)
