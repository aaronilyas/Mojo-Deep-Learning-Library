from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


@export
def PyInit__engine() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_engine")
        m.def_function[answer]("answer", docstring="Return the answer to everything.")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module: ", e))


def answer() -> PythonObject:
    return PythonObject(42)
