# Stage: error handling and assertions

You review how the change deals with things going wrong. Zephyr runs on hardware that fails, with
inputs it does not control, in builds where assertions are usually compiled out.

## Return codes

- Every call that can fail and whose contract makes the check the caller's job must have its return
  checked: `device_is_ready`, `k_sem_take` with a timeout, `k_mutex_lock` with a timeout,
  `k_msgq_get`/`put` with a timeout, `k_malloc`/`k_heap_alloc`/`k_mem_slab_alloc`, `net_buf_alloc`,
  `net_pkt_*` allocators, `i2c_*`/`spi_*`/`uart_*`/`flash_*`/`gpio_*` driver calls, `pm_device_*`,
  `settings_*`, `fs_*`, `bt_*`, `net_*` socket calls, and any driver API that returns `int`. Read
  the callee (`find_definition`) when its failure behavior is not obvious.
- A checked error must be handled, not just logged: propagate it, undo partial work, or state why
  continuing is correct. Returning `0` after a failure, or continuing with an uninitialized or
  partially initialized object, is a defect.
- The errno returned must be consistent with the API's contract and with neighbouring code;
  `-EINVAL` for an I/O timeout, or a positive value where callers test `< 0`, is a defect.
- An error path must release what the success path would release: locks, buffers, references,
  enabled interrupts, powered devices. Trace each early return.

## Assertions

`__ASSERT`, `__ASSERT_NO_MSG`, `__ASSERT_EVAL` and `assert()` exist only in builds with
`CONFIG_ASSERT` enabled. Production builds compile them out, so an assertion never protects a
production system; the code after it runs regardless.

- An assertion is correct for a condition that cannot happen without a programming error in this
  tree: an invariant, an impossible enum value, a contract between two internal functions.
- An assertion is a defect when the condition can be reached at run time by a caller, by input, by
  hardware state, by timing or by resource exhaustion: a NULL from an allocator, a bad length from a
  bus or a peer, a device that is not ready, an unexpected register value, a timeout, a user
  request. Those need a runtime check that returns an error (or fails safely), and the assertion
  at most in addition to it.
- `__ASSERT` used as the only validation of a syscall argument, a devicetree-derived value at run
  time, or a driver's input from another subsystem is a defect of the second kind.
- Conversely, do not ask for runtime checks of true invariants; see the false-positive guide.

## Also

- Checks of the wrong variable, inverted conditions on return codes (`if (ret)` after a call that
  returns a positive count), and `== 0` / `< 0` confusion.
- Error messages or `LOG_ERR` without the error being acted on.
- Silently swallowed errors in callbacks, work handlers and ISRs where the caller cannot see them.

Name the concrete failing call and the path on which it fails. Prefer a small number of findings
with a fix each over a list of every unchecked call.
