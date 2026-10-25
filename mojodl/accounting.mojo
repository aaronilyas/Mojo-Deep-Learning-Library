"""Accounting layer over the platform allocator (ADR-0004).

Every tensor allocation and free passes through accounted_alloc / accounted_free
so current and peak byte counts stay accurate for later memory_stats().
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.format import Writable, Writer
from std.python import PythonObject, Python


@fieldwise_init
struct Accounting(Movable, Writable):
    """Process-wide tensor memory counters.

    Owned by the `_engine` module as a singleton Mojo value. Public Python
    `memory_stats()` / `reset_peak_memory_stats()` read and update these fields.
    """

    var current: Int
    var peak: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Accounting(current=", self.current, ", peak=", self.peak, ")"
        )


def _accounting_ptr() raises -> Pointer[Accounting, MutUnsafeAnyOrigin]:
    var eng = Python.import_module("mojodl._engine")
    return eng._accounting.downcast_value_ptr[Accounting]()


def memory_stats() raises -> PythonObject:
    """Return {"current_bytes": int, "peak_bytes": int} from the Mojo counters."""
    var acct = _accounting_ptr()
    var d = Python.dict()
    d["current_bytes"] = PythonObject(acct[].current)
    d["peak_bytes"] = PythonObject(acct[].peak)
    return d


def reset_peak_memory_stats() raises:
    """Set peak_bytes equal to the current live byte count."""
    var acct = _accounting_ptr()
    acct[].peak = acct[].current


def accounted_alloc(nbytes: Int) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    """Allocate `nbytes` via the platform allocator and record them.

    Raises if `nbytes` is not positive. Updates current and peak counters.
    Counter update failures propagate (not swallowed).
    """
    if nbytes < 1:
        raise Error("accounted_alloc: nbytes must be positive")
    var ptr = unsafe_alloc[UInt8](nbytes)
    var acct = _accounting_ptr()
    acct[].current += nbytes
    if acct[].current > acct[].peak:
        acct[].peak = acct[].current
    return ptr


def accounted_free(
    ptr: Pointer[UInt8, MutUntrackedOrigin], nbytes: Int
):
    """Free storage previously returned by accounted_alloc and debit the count.

    Non-raising so Tensor.__deinit__ can call it. Counter update failures are
    swallowed only when the accounting singleton is already gone (interpreter
    shutdown); storage is still freed in that case.
    """
    if nbytes < 1:
        return
    ptr.unsafe_free()
    try:
        var acct = _accounting_ptr()
        acct[].current -= nbytes
    except e:
        # Interpreter teardown: _engine / _accounting may already be gone.
        pass
