"""Python extension module for the Mojo engine.

Exposes the Tensor type that owns fp32 storage, shape, strides, and device.
Element storage is allocated and freed only inside Tensor, through the
accounting layer. Meta tensors hold no storage.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from accounting import (
    Accounting,
    memory_stats,
    reset_peak_memory_stats,
)
from tensor_core import Tensor, empty
from ops import add, mul


@export
def PyInit__engine() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_engine")
        _ = m.add_type[Accounting]("Accounting")
        _ = (
            m.add_type[Tensor]("Tensor")
            .def_py_init[Tensor.py_init]()
            .def_method[Tensor.shape_tuple]("shape")
            .def_method[Tensor.device_str]("device")
            .def_method[Tensor.copy_to]("copy_to")
        )
        m.def_function[memory_stats](
            "memory_stats",
            docstring="Return current and peak tensor memory in bytes.",
        )
        m.def_function[reset_peak_memory_stats](
            "reset_peak_memory_stats",
            docstring="Set peak tensor memory equal to the current live count.",
        )
        m.def_function[empty](
            "empty",
            docstring="Create an uninitialized cpu Tensor or a Meta tensor.",
        )
        m.def_function[add](
            "add",
            docstring="Same-shape elementwise add Reference kernel / Meta version.",
        )
        m.def_function[mul](
            "mul",
            docstring="Same-shape elementwise multiply Reference kernel / Meta version.",
        )

        var mod = m.finalize()
        var acct = Accounting(0, 0)
        Python.add_object(mod, "_accounting", PythonObject(alloc=acct^))
        return mod
    except e:
        abort(String("error creating Python Mojo module: ", e))
