"""Python extension module for the Mojo engine.

Exposes answer() (smoke test) and the tensor storage primitives used by the
pure-Python Tensor veneer. Element storage is allocated and freed only here,
through the accounting layer.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer
from accounting import Accounting
from tensor_core import (
    alloc_storage,
    free_storage,
    copy_from_buffer,
    copy_to_buffer,
)


@export
def PyInit__engine() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_engine")
        _ = m.add_type[Accounting]("Accounting")
        m.def_function[answer]("answer", docstring="Return the answer to everything.")
        m.def_function[tensor_alloc](
            "tensor_alloc",
            docstring="Allocate nbytes of fp32 tensor storage; returns address as int.",
        )
        m.def_function[tensor_free](
            "tensor_free",
            docstring="Free tensor storage at address for nbytes.",
        )
        m.def_function[tensor_copy_from](
            "tensor_copy_from",
            docstring="Copy nbytes from a contiguous NumPy buffer into engine storage.",
        )
        m.def_function[tensor_copy_to](
            "tensor_copy_to",
            docstring="Copy nbytes from engine storage into a contiguous NumPy buffer.",
        )
        var mod = m.finalize()
        var acct = Accounting(0, 0)
        Python.add_object(mod, "_accounting", PythonObject(alloc=acct^))
        return mod
    except e:
        abort(String("error creating Python Mojo module: ", e))


def answer() -> PythonObject:
    return PythonObject(42)


def tensor_alloc(nbytes_obj: PythonObject) raises -> PythonObject:
    var nbytes = Int(py=nbytes_obj)
    if nbytes < 1:
        return PythonObject(0)
    var ptr = alloc_storage(nbytes)
    return PythonObject(Int(ptr))


def tensor_free(addr_obj: PythonObject, nbytes_obj: PythonObject) raises:
    var addr = Int(py=addr_obj)
    var nbytes = Int(py=nbytes_obj)
    if addr == 0 or nbytes < 1:
        return
    var ptr = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    free_storage(ptr, nbytes)


def tensor_copy_from(
    addr_obj: PythonObject, arr: PythonObject, nbytes_obj: PythonObject
) raises:
    var addr = Int(py=addr_obj)
    var nbytes = Int(py=nbytes_obj)
    if addr == 0 or nbytes < 1:
        return
    var dest = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    var src = arr.ctypes.data.unsafe_get_as_pointer[DType.uint8]()
    copy_from_buffer(dest, src, nbytes)


def tensor_copy_to(
    addr_obj: PythonObject, arr: PythonObject, nbytes_obj: PythonObject
) raises:
    var addr = Int(py=addr_obj)
    var nbytes = Int(py=nbytes_obj)
    if addr == 0 or nbytes < 1:
        return
    var src = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)
    var dest = arr.ctypes.data.unsafe_get_as_pointer[DType.uint8]()
    copy_to_buffer(dest, src, nbytes)
