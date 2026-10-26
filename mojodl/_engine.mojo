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
from ops_shape import reshape, transpose
from ops_linalg import matmul, matmul_nt, matmul_tn
from ops_activation import (
    relu,
    relu_backward,
    exp,
    log,
    log_backward,
    softmax,
    softmax_backward,
    log_softmax,
    log_softmax_backward,
)
from ops_elementwise import (
    broadcast_add,
    broadcast_sub,
    broadcast_mul,
    broadcast_div,
    neg,
    reduce_to_shape,
)
from ops_constructors import zeros, rand, randn
from ops_cross_entropy import cross_entropy, cross_entropy_backward
from ops_optim import sgd_step, sgd_momentum_step


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
        m.def_function[transpose](
            "transpose",
            docstring="Swap two axes into a contiguous tensor; Reference / Meta.",
        )
        m.def_function[matmul](
            "matmul",
            docstring="2-D matmul Reference kernel / Meta version.",
        )
        m.def_function[matmul_nt](
            "matmul_nt",
            docstring="a @ b.T without materializing transpose (matmul backward).",
        )
        m.def_function[matmul_tn](
            "matmul_tn",
            docstring="a.T @ b without materializing transpose (matmul backward).",
        )
        m.def_function[relu](
            "relu",
            docstring="Elementwise ReLU Reference kernel / Meta version.",
        )
        m.def_function[relu_backward](
            "relu_backward",
            docstring="ReLU backward: grad where x > 0 else 0 (Reference / Meta).",
        )
        m.def_function[exp](
            "exp",
            docstring="Elementwise exp Reference kernel / Meta version.",
        )
        m.def_function[log](
            "log",
            docstring="Elementwise natural log Reference kernel / Meta version.",
        )
        m.def_function[log_backward](
            "log_backward",
            docstring="Log backward: grad / x (Reference / Meta).",
        )
        m.def_function[softmax](
            "softmax",
            docstring="Softmax along dim Reference kernel / Meta version.",
        )
        m.def_function[softmax_backward](
            "softmax_backward",
            docstring="Softmax backward: y * (grad - sum(grad*y)) (Reference / Meta).",
        )
        m.def_function[log_softmax](
            "log_softmax",
            docstring="Log-softmax along dim Reference kernel / Meta version.",
        )
        m.def_function[log_softmax_backward](
            "log_softmax_backward",
            docstring="Log-softmax backward: grad - exp(y)*sum(grad) (Reference / Meta).",
        )
        m.def_function[broadcast_add](
            "broadcast_add",
            docstring="Broadcasting elementwise add Reference kernel / Meta version.",
        )
        m.def_function[broadcast_sub](
            "broadcast_sub",
            docstring="Broadcasting elementwise subtract Reference kernel / Meta version.",
        )
        m.def_function[broadcast_mul](
            "broadcast_mul",
            docstring="Broadcasting elementwise multiply Reference kernel / Meta version.",
        )
        m.def_function[broadcast_div](
            "broadcast_div",
            docstring="Broadcasting elementwise divide Reference kernel / Meta version.",
        )
        m.def_function[neg](
            "neg",
            docstring="Elementwise negate Reference kernel / Meta version.",
        )
        m.def_function[reduce_to_shape](
            "reduce_to_shape",
            docstring="Reduce a gradient to a target shape (broadcasting backward).",
        )
        m.def_function[zeros](
            "zeros",
            docstring="Create a zeros-filled cpu Tensor or a Meta tensor.",
        )
        m.def_function[rand](
            "rand",
            docstring="Create a uniform-[0,1) cpu Tensor or a Meta tensor.",
        )
        m.def_function[randn](
            "randn",
            docstring="Create a standard-normal cpu Tensor or a Meta tensor.",
        )
        m.def_function[cross_entropy](
            "cross_entropy",
            docstring="Mean cross-entropy Reference kernel / Meta version.",
        )
        m.def_function[cross_entropy_backward](
            "cross_entropy_backward",
            docstring="Cross-entropy logits gradient Reference kernel / Meta version.",
        )
        m.def_function[sgd_step](
            "sgd_step",
            docstring="In-place SGD update param -= lr * grad; Reference / Meta.",
        )
        m.def_function[sgd_momentum_step](
            "sgd_momentum_step",
            docstring=(
                "In-place SGD with momentum: buf = m*buf + grad; "
                "param -= lr*buf; Reference / Meta."
            ),
        )

        var mod = m.finalize()
        var acct = Accounting(0, 0)
        Python.add_object(mod, "_accounting", PythonObject(alloc=acct^))
        return mod
    except e:
        abort(String("error creating Python Mojo module: ", e))
