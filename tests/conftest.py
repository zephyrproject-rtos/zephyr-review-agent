# Copyright (c) 2026 Anas Nashif
# SPDX-License-Identifier: Apache-2.0

"""A miniature Zephyr-shaped repository with a base branch and a 'pull request' on top."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

MAINTAINERS = """\
"Drivers: I2C":
  status: maintained
  maintainers:
    - alice
  collaborators:
    - bob
  labels:
    - "area: I2C"
  files:
    - drivers/i2c/
    - include/zephyr/drivers/i2c.h
    - dts/bindings/i2c/
    - tests/drivers/i2c/
  tests:
    - drivers.i2c
"Kernel":
  status: maintained
  maintainers:
    - carol
  files:
    - kernel/
    - include/zephyr/kernel.h
  files-exclude:
    - kernel/Kconfig.vm
"Documentation":
  status: maintained
  maintainers:
    - dave
  files:
    - AGENTS.md
  files-regex:
    - \\.rst$
  file-groups:
    - name: Agents
      collaborators:
        - erin
      files:
        - AGENTS.md
        - kernel/
"""

I2C_H = """\
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ZEPHYR_INCLUDE_DRIVERS_I2C_H_
#define ZEPHYR_INCLUDE_DRIVERS_I2C_H_

struct i2c_msg {
\tuint8_t *buf;
\tuint32_t len;
};

typedef int (*i2c_api_transfer)(const struct device *dev, struct i2c_msg *msgs, uint8_t num);

int i2c_transfer(const struct device *dev, struct i2c_msg *msgs, uint8_t num_msgs);

#define I2C_SPEED_STANDARD 0x1U

#endif
"""

I2C_FOO_BASE = """\
/* SPDX-License-Identifier: Apache-2.0 */
#define DT_DRV_COMPAT vnd_foo_i2c

#include <zephyr/drivers/i2c.h>

struct foo_data {
\tstruct k_sem lock;
};

static int foo_transfer(const struct device *dev, struct i2c_msg *msgs, uint8_t num)
{
\tstruct foo_data *data = dev->data;
\tint ret;

\tk_sem_take(&data->lock, K_FOREVER);
\tret = foo_do_transfer(dev, msgs, num);
\tk_sem_give(&data->lock);
\treturn ret;
}

static int foo_init(const struct device *dev)
{
\tstruct foo_data *data = dev->data;

\tk_sem_init(&data->lock, 1, 1);
\treturn 0;
}
"""

I2C_FOO_PR = (
    I2C_FOO_BASE.replace(
        "\tk_sem_take(&data->lock, K_FOREVER);\n\tret = foo_do_transfer(dev, msgs, num);\n\tk_sem_give(&data->lock);\n",
        "\tk_sem_take(&data->lock, K_FOREVER);\n\tret = foo_do_transfer(dev, msgs, num);\n"
        "\tif (ret < 0) {\n\t\treturn ret;\n\t}\n\tk_sem_give(&data->lock);\n",
    )
    + """
int foo_recover(const struct device *dev)
{
\treturn foo_bus_recover(dev);
}
"""
)

KCONFIG_I2C_BASE = """\
# SPDX-License-Identifier: Apache-2.0

menuconfig I2C
\tbool "I2C drivers"
\thelp
\t  Enable I2C drivers.

if I2C

config I2C_FOO
\tbool "Foo I2C driver"
\tdefault y
\tdepends on DT_HAS_VND_FOO_I2C_ENABLED
\thelp
\t  Driver for the Foo I2C controller.

endif # I2C
"""

KCONFIG_I2C_PR = KCONFIG_I2C_BASE.replace(
    "endif # I2C",
    'config I2C_FOO_RECOVERY\n\tbool "Bus recovery"\n\tdepends on I2C_FOO\n\tselect I2C_BUS_RECOVERY\n\nendif # I2C',
)

BINDING = """\
# SPDX-License-Identifier: Apache-2.0
description: Foo I2C controller
compatible: "vnd,foo-i2c"
include: [i2c-controller.yaml, pinctrl-device.yaml]
properties:
  clock-frequency:
    type: int
    required: true
"""

TESTS_YAML = """\
tests:
  drivers.i2c.foo:
    tags:
      - drivers
      - i2c
    integration_platforms:
      - native_sim
  drivers.i2c.foo.recovery:
    extra_configs:
      - CONFIG_I2C_FOO_RECOVERY=y
"""

TEST_MAIN = """\
#include <zephyr/ztest.h>
#include <zephyr/drivers/i2c.h>

ZTEST(i2c_foo, test_transfer)
{
\tstruct i2c_msg msg = {0};
\tzassert_equal(i2c_transfer(dev, &msg, 1), 0);
}
"""

BOARD_DTS = """\
#include <arm/vnd/vndsoc.dtsi>

/ {
\ti2c0: i2c@40000000 {
\t\tcompatible = "vnd,foo-i2c";
\t\tclock-frequency = <100000>;
\t};
};
"""

BOARD_YML = """\
board:
  name: vndboard
  full_name: Vendor Board
  vendor: vnd
  socs:
  - name: vndsoc
    variants:
    - name: ns
      cpucluster: cpuapp
"""

NATIVE_BOARD_YML = """\
boards:
- name: native_sim
  full_name: Native simulator
  vendor: zephyr
  socs:
  - name: native
    variants:
    - name: "64"
"""

TAGS_YAML = """\
kernel:
  files:
    - kernel/
    - arch/
i2c:
  files:
    - drivers/i2c/
    - include/zephyr/drivers/i2c.h
"""

KERNEL_TESTS_YAML = """\
common:
  tags:
    - kernel
tests:
  kernel.semaphore:
    platform_allow:
      - native_sim
      - vndboard/vndsoc
  kernel.semaphore.build:
    build_only: true
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture(scope="session")
def mini_repo(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("zephyr")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test")
    _write(root, "MAINTAINERS.yml", MAINTAINERS)
    _write(root, "AGENTS.md", "# Zephyr RTOS: instructions for AI coding agents\n\nBe terse.\n")
    _write(root, "include/zephyr/drivers/i2c.h", I2C_H)
    _write(root, "drivers/i2c/i2c_foo.c", I2C_FOO_BASE)
    _write(root, "drivers/i2c/Kconfig", KCONFIG_I2C_BASE)
    _write(root, "drivers/i2c/CMakeLists.txt", "zephyr_library_sources_ifdef(CONFIG_I2C_FOO i2c_foo.c)\n")
    _write(root, "dts/bindings/i2c/vnd,foo-i2c.yaml", BINDING)
    _write(root, "boards/vnd/vndboard/vndboard_vndsoc_cpuapp.dts", BOARD_DTS)
    _write(root, "boards/vnd/vndboard/vndboard_vndsoc_cpuapp_ns.dts", BOARD_DTS)
    _write(root, "boards/vnd/vndboard/board.yml", BOARD_YML)
    _write(root, "boards/native/native_sim/native_sim_native.dts", "/ { };\n")
    _write(root, "boards/native/native_sim/native_sim_native_64.dts", "/ { };\n")
    _write(root, "boards/native/native_sim/board.yml", NATIVE_BOARD_YML)
    _write(root, "scripts/ci/tags.yaml", TAGS_YAML)
    _write(root, "dts/arm/vnd/vndsoc.dtsi", "#include <arm/vnd/vndsoc_common.dtsi>\n/ { soc { }; };\n")
    _write(root, "dts/arm/vnd/vndsoc_common.dtsi", "/ { chosen { }; };\n")
    _write(root, "tests/kernel/semaphore/tests.yaml", KERNEL_TESTS_YAML)
    _write(root, "tests/kernel/semaphore/src/main.c", "#include <zephyr/ztest.h>\n")
    _write(root, "tests/drivers/i2c/common/helpers.h", "/* shared */\n")
    _write(root, "tests/drivers/i2c/target/tests.yaml", "tests:\n  drivers.i2c.target:\n    tags:\n      - i2c\n")
    _write(root, "tests/drivers/i2c/foo/tests.yaml", TESTS_YAML)
    _write(root, "tests/drivers/i2c/foo/src/main.c", TEST_MAIN)
    _write(root, "tests/drivers/i2c/foo/prj.conf", "CONFIG_I2C=y\nCONFIG_I2C_FOO=y\n")
    _write(root, "doc/hardware/peripherals/i2c.rst", "I2C\n===\n")
    _write(root, "kernel/Kconfig.vm", "config KERNEL_VM_BASE\n\thex\n")
    _write(root, "kernel/sem.c", "int z_impl_k_sem_take(struct k_sem *sem, k_timeout_t timeout)\n{\n\treturn 0;\n}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "Initial tree")
    base = _git(root, "rev-parse", "HEAD")

    _git(root, "checkout", "-q", "-b", "pr")
    _write(root, "drivers/i2c/i2c_foo.c", I2C_FOO_PR)
    _git(
        root,
        "commit",
        "-q",
        "-am",
        "drivers: i2c: foo: add bus recovery\n\nAdd foo_recover().\n\nSigned-off-by: T <t@example.com>",
    )
    _write(root, "drivers/i2c/Kconfig", KCONFIG_I2C_PR)
    _write(root, "tests/drivers/i2c/foo/tests.yaml", TESTS_YAML + "    tags:\n      - recovery\n")
    _git(root, "add", "-A")
    _git(
        root,
        "commit",
        "-q",
        "-m",
        "drivers: i2c: foo: add I2C_FOO_RECOVERY Kconfig\n\nSigned-off-by: T <t@example.com>",
    )
    head = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", "main")
    return {"root": root, "base": base, "head": head}
