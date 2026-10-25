"""Python extension module for the Mojo engine.

Exposes answer() (smoke test) and the Tensor type that owns fp32 storage,
shape, and strides. Element storage is allocated and freed only inside
Tensor, through the accounting layer.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from accounting import Accounting
from tensor_core import Tensor


@export
def PyInit__engine() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_engine")
        _ = m.add_type[Accounting]("Accounting")
        _ = (
            m.add_type[Tensor]("Tensor")
            .def_py_init[Tensor.py_init]()
            .def_method[Tensor.shape_tuple]("shape")
            .def_method[Tensor.copy_to]("copy_to")
        )
        m.def_function[answer]("answer", docstring="Return the answer to everything.")
        var mod = m.finalize()
        var acct = Accounting(0, 0)
        Python.add_object(mod, "_accounting", PythonObject(alloc=acct^))
        return mod
    except e:
        abort(String("error creating Python Mojo module: ", e))


def answer() -> PythonObject:
    return PythonObject(42)
