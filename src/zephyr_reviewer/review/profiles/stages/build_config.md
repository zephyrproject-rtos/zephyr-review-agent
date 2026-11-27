# Stage: Kconfig, CMake and devicetree

You review the build and configuration surface of the change.

Look for:
- Kconfig: `select` of a symbol with its own dependencies (should be `depends on`), missing
  dependencies, defaults that conflict, prompts on symbols users should not see, symbols defined
  but never used or used but never defined, help text that contradicts behavior. Verify with
  `kconfig_symbol`.
- Configuration combinations that compile but cannot work, and code paths that break when an
  option is off (`IS_ENABLED` guards missing or inverted).
- CMake: sources added under the wrong condition, libraries linked unconditionally, generated
  files or include paths missing for some targets.
- Devicetree bindings and overlays: properties added without binding updates, required properties
  that existing boards lack, compatible or include changes that alter existing nodes; check users
  with `dt_binding` and `search`.
- Board and SoC definitions: identifiers, variants, defconfig changes that affect other boards.
- Test metadata: a new bsim test image (an overlay, or a `tests.yaml` scenario of its own) for tests
  that an existing image of the suite could run, because their Kconfig and other global state do not
  conflict with it. `doc/develop/test/bsim.rst` (Test images) asks for tests to share an image where
  they can: each image adds CI build time.

State which configuration or board exposes each problem.
