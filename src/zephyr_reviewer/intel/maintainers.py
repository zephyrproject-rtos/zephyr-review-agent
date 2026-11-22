# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""MAINTAINERS.yml lookup with the same matching rules as scripts/get_maintainer.py.

Reimplemented rather than imported so that nothing from the tree under review
is executed, and so the file can be read from the trusted base commit.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import yaml

from .repo import GitRepo


@dataclass
class Area:
    name: str
    status: str = ""
    maintainers: list[str] = field(default_factory=list)
    collaborators: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    description: str = ""
    _match: Callable[[str], object] | None = None
    _exclude: Callable[[str], object] | None = None
    _groups: list[FileGroup] = field(default_factory=list)

    def matches(self, path: str) -> tuple[bool, list[FileGroup]]:
        """(the area has the path, the file groups of the area that have it).

        A file group divides its area further: a path is in a group only if it is in the area first, as
        in get_maintainer.py. A group that lists ``dts/bindings/`` under a vendor's area has that
        vendor's bindings, not everyone's."""
        if not (self._match and self._match(path)) or (self._exclude and self._exclude(path)):
            return False, []
        return True, [g for g in self._groups if g.matches(path)]

    @staticmethod
    def deferring(groups: list[FileGroup]) -> bool:
        """The path is in the area through file groups that all defer to other areas."""
        return bool(groups) and all(g.defer for g in groups)


@dataclass
class FileGroup:
    name: str
    collaborators: list[str] = field(default_factory=list)
    defer: bool = False
    _match: Callable[[str], object] | None = None
    _exclude: Callable[[str], object] | None = None

    def matches(self, path: str) -> bool:
        if self._exclude and self._exclude(path):
            return False
        return bool(self._match and self._match(path))


def _match_fn(globs, regexes) -> Callable[[str], object] | None:
    if not (globs or regexes):
        return None
    regex = ""
    if globs:
        parts = []
        for g in globs:
            r = g.replace(".", "\\.").replace("*", "[^/]*").replace("?", "[^/]")
            if not g.endswith("/"):
                r += "$"
            parts.append(r)
        regex += "^(?:{})".format("|".join(parts))
    if regexes:
        if regex:
            regex += "|"
        regex += "|".join(regexes)
    return re.compile(regex).search


class Maintainers:
    def __init__(self, text: str):
        data = yaml.safe_load(text) or {}
        self.areas: list[Area] = []
        for name, d in data.items():
            if not isinstance(d, dict):
                continue
            area = Area(
                name=str(name),
                status=str(d.get("status", "")),
                maintainers=list(d.get("maintainers") or []),
                collaborators=list(d.get("collaborators") or []),
                labels=list(d.get("labels") or []),
                tests=list(d.get("tests") or []),
                description=str(d.get("description") or ""),
            )
            area._match = _match_fn(d.get("files"), d.get("files-regex"))
            area._exclude = _match_fn(d.get("files-exclude"), d.get("files-regex-exclude"))
            for g in d.get("file-groups") or []:
                grp = FileGroup(
                    name=str(g.get("name", "Unnamed Group")),
                    collaborators=list(g.get("collaborators") or []),
                    defer=bool(g.get("defer-to-other-areas", False)),
                )
                grp._match = _match_fn(g.get("files"), g.get("files-regex"))
                grp._exclude = _match_fn(g.get("files-exclude"), g.get("files-regex-exclude"))
                area._groups.append(grp)
            self.areas.append(area)

    @classmethod
    def from_repo(cls, repo: GitRepo, path: str = "MAINTAINERS.yml") -> Maintainers | None:
        text = repo.show(path, max_bytes=5_000_000)
        return cls(text) if text else None

    def areas_for(self, path: str, *, is_dir: bool = False) -> list[tuple[Area, list[FileGroup]]]:
        if is_dir and not path.endswith("/"):
            path += "/"
        out: list[tuple[Area, list[FileGroup]]] = []
        for area in self.areas:
            inside, groups = area.matches(path)
            if inside:
                out.append((area, groups))
        return out

    def owners_for(self, path: str, *, is_dir: bool = False) -> list[tuple[Area, list[FileGroup]]]:
        """The areas that have the path, the one to turn to first. As in get_maintainer: an area that
        has the path through file groups which defer to other areas (a subsystem's vendor drivers) comes
        after one that has it without deferring (the vendor's platform); it is still an area of the path,
        and its labels still apply."""
        found = self.areas_for(path, is_dir=is_dir)
        return sorted(found, key=lambda ag: Area.deferring(ag[1]))

    def by_name(self, name: str) -> Area | None:
        for a in self.areas:
            if a.name == name:
                return a
        return None
