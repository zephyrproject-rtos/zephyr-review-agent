# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from zephyr_reviewer.intel import SourceIntel
from zephyr_reviewer.intel.diff import parse_unified_diff
from zephyr_reviewer.intel.maintainers import Maintainers
from zephyr_reviewer.intel.repo import GitRepo, _glob_to_regex


@pytest.fixture(scope="module")
def intel(mini_repo) -> SourceIntel:
    return SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"])


def test_repo_reads_are_pinned_to_the_ref(mini_repo):
    base = GitRepo(mini_repo["root"], mini_repo["base"])
    head = GitRepo(mini_repo["root"], mini_repo["head"])
    assert "foo_recover" not in (base.show("drivers/i2c/i2c_foo.c") or "")
    assert "foo_recover" in (head.show("drivers/i2c/i2c_foo.c") or "")
    assert head.show("does/not/exist.c") is None
    assert head.is_dir("drivers/i2c") and not head.is_dir("drivers/i2c/i2c_foo.c")


def test_grep_parses_hits_and_pathspecs(mini_repo):
    head = GitRepo(mini_repo["root"], mini_repo["head"])
    hits, truncated = head.grep("k_sem_take", paths="drivers/**/*.c", regex=False, word=True)
    assert [(h.path, h.line) for h in hits] == [("drivers/i2c/i2c_foo.c", 15)]
    assert not truncated
    hits, _ = head.grep("k_sem_take", paths="kernel", regex=False)
    assert hits and hits[0].path == "kernel/sem.c"
    hits, truncated = head.grep(".", max_results=3)
    assert len(hits) == 3 and truncated


def test_glob_regex_semantics():
    assert _glob_to_regex("drivers/**/*.c").match("drivers/i2c/i2c_foo.c")
    assert _glob_to_regex("drivers/*.c").match("drivers/i2c/i2c_foo.c") is None
    assert _glob_to_regex("**/Kconfig*").match("drivers/i2c/Kconfig")
    assert _glob_to_regex("**/Kconfig*").match("Kconfig")


def test_diff_parsing_marks_new_side_lines(intel: SourceIntel):
    files = parse_unified_diff(intel.diff_text)
    paths = {f.path: f for f in files}
    assert set(paths) == {"drivers/i2c/i2c_foo.c", "drivers/i2c/Kconfig", "tests/drivers/i2c/foo/tests.yaml"}
    foo = paths["drivers/i2c/i2c_foo.c"]
    assert foo.status == "M"
    # The early return is an added line, so it is inline-commentable; line 1 is untouched.
    assert 17 in foo.added_lines and 18 in foo.added_lines
    assert 1 not in foo.new_side_lines
    assert foo.additions == 8 and foo.deletions == 0


def test_overview_lists_commits_files_and_areas(intel: SourceIntel):
    text = intel.overview()
    assert "Commits: 2" in text and "Files changed: 3" in text
    assert "drivers/i2c/i2c_foo.c +8/-0 [Drivers: I2C]" in text
    assert "add bus recovery" in text


def test_read_file_marks_added_lines_and_bounds(intel: SourceIntel):
    text = intel.read("drivers/i2c/i2c_foo.c", 14, 20)
    assert "lines 14-20 of" in text
    assert "   17+| \tif (ret < 0) {" in text
    assert "   15 | \tk_sem_take" in text
    assert "deleted" not in intel.read("drivers/i2c/i2c_foo.c")
    assert "not found" in intel.read("drivers/i2c/nope.c")
    assert "Similar paths" in intel.read("i2c_foo.c")


def test_find_definition_classifies_kinds(intel: SourceIntel):
    assert "[function] drivers/i2c/i2c_foo.c:10: static int foo_transfer(" in intel.definition("foo_transfer")
    assert "[prototype] include/zephyr/drivers/i2c.h:12" in intel.definition("i2c_transfer")
    assert "[struct] include/zephyr/drivers/i2c.h:5" in intel.definition("i2c_msg")
    assert "[macro] include/zephyr/drivers/i2c.h:14" in intel.definition("I2C_SPEED_STANDARD")
    assert "[typedef] include/zephyr/drivers/i2c.h:10" in intel.definition("i2c_api_transfer")
    assert "[kconfig] drivers/i2c/Kconfig:10" in intel.definition("CONFIG_I2C_FOO")
    assert "[dt-compatible] dts/bindings/i2c/vnd,foo-i2c.yaml:3" in intel.definition("vnd,foo-i2c")
    assert "No definition" in intel.definition("nonexistent_symbol_xyz")


def test_find_definition_ignores_calls_comments_and_resolves_syscalls(intel: SourceIntel, mini_repo):
    from zephyr_reviewer.intel.repo import GrepHit
    from zephyr_reviewer.intel.symbols import _kind_of

    assert _kind_of("k_sem_take", GrepHit("a.c", 1, "\treturn k_sem_take(&data->sem, K_FOREVER);")) is None
    assert _kind_of("k_sem_take", GrepHit("a.c", 1, "\tret = k_sem_take(&data->sem,")) is None
    assert _kind_of("k_sem_take", GrepHit("a.c", 1, " * the loop through k_sem_take(). But k_timer_start()")) is None
    assert _kind_of("k_sem_take", GrepHit("a.c", 1, "\tk_sem_take((p_sem),                      \\")) is None
    assert _kind_of("k_sem_take", GrepHit("a.c", 1, "\tk_sem_take(&data->sem,")) is None
    assert (
        _kind_of("k_sem_take", GrepHit("k.h", 1, "__syscall int k_sem_take(struct k_sem *sem, k_timeout_t timeout);"))
        == "prototype"
    )
    assert (
        _kind_of("k_sem_take", GrepHit("s.c", 1, "int z_impl_k_sem_take(struct k_sem *sem, k_timeout_t timeout)"))
        is None
    )
    assert _kind_of("foo", GrepHit("s.c", 1, "static inline const struct device *foo(void)")) == "function"
    assert _kind_of("foo", GrepHit("s.c", 1, "static int foo(const struct device *dev,")) == "function"
    # The syscall implementation is found through its z_impl_ name.
    text = intel.definition("k_sem_take")
    assert "[function (syscall impl)] kernel/sem.c:1: int z_impl_k_sem_take(" in text
    assert "drivers/i2c/i2c_foo.c" not in text


def test_find_references_excludes_definitions_and_flags_changed_files(intel: SourceIntel):
    text = intel.references("i2c_transfer")
    assert "tests/drivers/i2c/foo/src/main.c:7:" in text
    assert "include/zephyr/drivers/i2c.h:12" not in text  # the prototype is a definition
    text = intel.references("k_sem_take")
    assert "drivers/i2c/i2c_foo.c:15: *" in text  # changed file marked and listed first


def test_kconfig_symbol_block_and_users(intel: SourceIntel):
    text = intel.kconfig("CONFIG_I2C_FOO")
    assert "## drivers/i2c/Kconfig:10" in text
    assert "depends on DT_HAS_VND_FOO_I2C_ENABLED" in text
    assert "config I2C_FOO_RECOVERY" not in text.split("## Kconfig references")[0]  # block ends at next symbol
    assert "CMake" in text and "drivers/i2c/CMakeLists.txt:1" in text
    assert "drivers/i2c/Kconfig:19: depends on I2C_FOO" in text
    assert "tests/drivers/i2c/foo/prj.conf:2" in text
    new = intel.kconfig("I2C_FOO_RECOVERY")
    assert "select I2C_BUS_RECOVERY" in new
    assert "No Kconfig definition" in intel.kconfig("CONFIG_NOT_THERE")


def test_dt_binding_and_users(intel: SourceIntel):
    text = intel.dt_binding("vnd,foo-i2c")
    assert "## dts/bindings/i2c/vnd,foo-i2c.yaml" in text and "clock-frequency" in text
    assert "drivers/i2c/i2c_foo.c:2: #define DT_DRV_COMPAT vnd_foo_i2c" in text
    assert "boards/vnd/vndboard/vndboard_vndsoc_cpuapp.dts:5" in text
    assert "Binding files with a matching name" in intel.dt_binding("foo-i2c")
    assert "No binding" in intel.dt_binding("acme,none")


def test_maintainers_matching_follows_get_maintainer_rules(mini_repo):
    m = Maintainers.from_repo(GitRepo(mini_repo["root"], mini_repo["base"]))
    assert m is not None
    names = [a.name for a, _ in m.areas_for("drivers/i2c/i2c_foo.c")]
    assert names == ["Drivers: I2C"]
    assert [a.name for a, _ in m.areas_for("kernel/sem.c")] == ["Kernel"]
    assert m.areas_for("kernel/Kconfig.vm") == []  # files-exclude
    assert [a.name for a, _ in m.areas_for("doc/hardware/peripherals/i2c.rst")] == ["Documentation"]
    hits = m.areas_for("AGENTS.md")
    assert hits[0][0].name == "Documentation" and hits[0][1][0].name == "Agents"
    # A file group divides its area: what it lists outside the area's own files is not in the area.
    assert [a.name for a, _ in m.areas_for("kernel/sem.c")] == ["Kernel"]
    # A directory path matches a trailing-slash glob only when flagged as a directory.
    assert m.areas_for("drivers/i2c", is_dir=True)[0][0].name == "Drivers: I2C"


def test_maintainers_tool_text(intel: SourceIntel):
    text = intel.maintainers_for("drivers/i2c/i2c_foo.c")
    assert "## Drivers: I2C (maintained)" in text and "@alice" in text and "@bob" in text
    assert "tests: drivers.i2c" in text
    assert "No MAINTAINERS.yml area" in intel.maintainers_for("LICENSE")


def test_related_tests_for_path_and_symbol(intel: SourceIntel, mini_repo):
    text = intel.related_tests("drivers/i2c/i2c_foo.c")
    assert "lists test identifiers: drivers.i2c" in text
    assert "## tests/drivers/i2c/foo/tests.yaml" in text
    assert "scenarios: drivers.i2c.foo, drivers.i2c.foo.recovery" in text
    assert "native_sim" in text
    by_symbol = intel.related_tests("i2c_transfer")
    assert "1 reference(s) to i2c_transfer" in by_symbol
    # A file directly under a top-level directory maps to tests/<dir>, not tests/<dir>/<file>.
    assert "under tests/kernel" in SourceIntel(
        str(mini_repo["root"]), mini_repo["head"], mini_repo["base"]
    ).related_tests("kernel/sem.c") or "No twister suites" in intel.related_tests("kernel/sem.c")


def test_commit_messages_history_blame_and_guidelines(intel: SourceIntel):
    msgs = intel.commit_messages()
    assert msgs.index("add bus recovery") < msgs.index("add I2C_FOO_RECOVERY")
    assert "Signed-off-by" in msgs
    assert "add bus recovery" in intel.history("drivers/i2c/i2c_foo.c")
    assert "No history" in intel.history("drivers/i2c/nope.c")
    assert "foo_recover" in intel.blame("drivers/i2c/i2c_foo.c", 30, 33)
    assert "Be terse." in intel.guidelines("agents")
    assert "Unknown topic" in intel.guidelines("nope")


def test_file_diff_and_search(intel: SourceIntel):
    text = intel.diff_for("drivers/i2c/Kconfig")
    assert "+config I2C_FOO_RECOVERY" in text
    assert "is not changed" in intel.diff_for("kernel/sem.c")
    assert "drivers/i2c/i2c_foo.c:2:" in intel.search("DT_DRV_COMPAT", "drivers")
    assert "No matches" in intel.search("zzzz_not_here")
    assert "dts/bindings/i2c/vnd,foo-i2c.yaml" in intel.list_files("dts/bindings/**/*.yaml")


def test_results_are_clipped(mini_repo):
    small = SourceIntel(str(mini_repo["root"]), mini_repo["head"], mini_repo["base"], max_chars=200)
    text = small.read("drivers/i2c/i2c_foo.c")
    assert len(text) < 400 and "truncated to 200 characters" in text


def test_definition_classifier_is_linear_on_pathological_lines():
    """A long identifier prefix on a line that is not a definition used to backtrack exponentially."""
    import time

    from zephyr_reviewer.intel.prefetch import _is_function_header
    from zephyr_reviewer.intel.repo import GrepHit
    from zephyr_reviewer.intel.symbols import _kind_of

    long_ident = "x" * 60
    lines = [
        f"\t{long_ident} = Z_EXC_HANDLE(foo);",  # assignment: prefix fails the specifier test
        f"{long_ident}{long_ident} Z_EXC_HANDLE(a, b) {{",  # single huge token
        "\t" + " ".join(["aaa"] * 40) + " = Z_EXC_HANDLE(x);",
        "#if defined(Z_EXC_HANDLE) && " + "(" * 50 + "y" + ")" * 50,
    ]
    t = time.monotonic()
    for line in lines:
        _kind_of("Z_EXC_HANDLE", GrepHit("a.c", 1, line))
        _is_function_header(line)
    assert time.monotonic() - t < 1.0
    # Real definitions still classify.
    assert _kind_of("foo", GrepHit("a.c", 1, "static const struct device *foo(void)")) == "function"
    assert _kind_of("foo", GrepHit("a.c", 1, "unsigned long long **foo(int a,")) == "function"
    assert _is_function_header("static inline int foo(void)") == "foo"
    assert _is_function_header("const struct k_thread *const foo(void)") == "foo"
