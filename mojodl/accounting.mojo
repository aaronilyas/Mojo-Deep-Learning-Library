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

    Owned by the `_engine` module as a singleton Mojo value. Not part of the
    public Python API yet; later `memory_stats()` will read these fields.
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


def accounted_alloc(nbytes: Int) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    """Allocate `nbytes` via the platform allocator and record them.

    Raises if `nbytes` is not positive. Updates current and peak counters.
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

    Non-raising so Tensor.__deinit__ can call it. If the accounting singleton
    is already gone (interpreter shutdown), storage is still freed.
    """
    if nbytes < 1:
        return
    ptr.unsafe_free()
    try:
        var acct = _accounting_ptr()
        acct[].current -= nbytes
    except e:
        pass
