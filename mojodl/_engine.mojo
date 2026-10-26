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
from ops import add, mul, sum, ones, expand_scalar, add_inplace, clone
from ops_reduce import sum_dim, sum_backward, mean_dim, mean_backward
from ops_shape import reshape


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
        m.def_function[sum](
            "sum",
            docstring="Whole-tensor sum Reference kernel / Meta version.",
        )
        m.def_function[ones](
            "ones",
            docstring="Create a ones-filled cpu Tensor or a Meta tensor.",
        )
        m.def_function[expand_scalar](
            "expand_scalar",
            docstring="Expand a scalar gradient to a target shape (sum backward).",
        )
        m.def_function[add_inplace](
            "add_inplace",
            docstring="In-place same-shape add for gradient accumulation.",
        )
        m.def_function[clone](
            "clone",
            docstring="Deep-copy a tensor for owned leaf .grad buffers.",
        )
        m.def_function[sum_dim](
            "sum_dim",
            docstring="Sum over all elements or one axis; Reference / Meta.",
        )
        m.def_function[sum_backward](
            "sum_backward",
            docstring="Broadcast sum upstream gradient to the input shape.",
        )
        m.def_function[mean_dim](
            "mean_dim",
            docstring="Mean over all elements or one axis; Reference / Meta.",
        )
        m.def_function[mean_backward](
            "mean_backward",
            docstring="Broadcast mean upstream gradient times 1/count to input shape.",
        )
        m.def_function[reshape](
            "reshape",
            docstring="Zero-copy reshape view; Reference and Meta share validation.",
        )

        var mod = m.finalize()
        var acct = Accounting(0, 0)
        Python.add_object(mod, "_accounting", PythonObject(alloc=acct^))
        return mod
    except e:
        abort(String("error creating Python Mojo module: ", e))
