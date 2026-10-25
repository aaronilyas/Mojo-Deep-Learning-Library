"""Contiguous fp32 tensor storage in the Mojo engine.

Storage is row-major Float32. Allocation goes through the accounting layer.
Python holds only the address and metadata; element data never lives in a
Python/NumPy object owned by the Tensor.
"""

from std.memory import Pointer, unsafe_memcpy
from accounting import accounted_alloc, accounted_free


def alloc_storage(nbytes: Int) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    """Allocate `nbytes` of tensor element storage through accounting."""
    return accounted_alloc(nbytes)


def free_storage(
    ptr: Pointer[UInt8, MutUntrackedOrigin], nbytes: Int
) raises:
    """Free tensor element storage through accounting."""
    accounted_free(ptr, nbytes)


def copy_from_buffer(
    dest: Pointer[UInt8, MutUntrackedOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int,
):
    """Copy `nbytes` from an external buffer into engine storage."""
    if nbytes < 1:
        return
    unsafe_memcpy(dest=dest, src=src, count=nbytes)


def copy_to_buffer(
    dest: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutUntrackedOrigin],
    nbytes: Int,
):
    """Copy `nbytes` from engine storage into an external buffer."""
    if nbytes < 1:
        return
    unsafe_memcpy(dest=dest, src=src, count=nbytes)
