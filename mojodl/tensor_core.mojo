"""Contiguous fp32 tensor storage owned by the Mojo engine.

`Tensor` owns the storage pointer, byte size, shape, row-major strides, and
dtype. Allocation and free go through the accounting layer. Python holds only
a reference to this engine object; addresses and byte counts never cross the
boundary.
"""

from std.memory import Pointer, unsafe_memcpy
from std.format import Writable, Writer
from std.python import PythonObject, Python
from accounting import accounted_alloc, accounted_free


def _row_major_strides(shape: List[Int]) -> List[Int]:
    """Element strides for a contiguous row-major layout."""
    var n = len(shape)
    var strides = List[Int](capacity=n)
    for _ in range(n):
        strides.append(0)
    if n == 0:
        return strides^
    strides[n - 1] = 1
    var i = n - 2
    while i >= 0:
        strides[i] = strides[i + 1] * shape[i + 1]
        i -= 1
    return strides^


@fieldwise_init
struct Tensor(Movable, Writable):
    """Eager fp32 tensor whose element storage lives in the Mojo engine."""

    var data: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var nbytes: Int
    var shape: List[Int]
    var strides: List[Int]
    # 0 = float32; only dtype supported in this milestone.
    var dtype_code: Int

    def __deinit__(deinit self):
        # Free storage exactly once when the Python wrapper is collected.
        # Destructors must not raise; accounted_free is non-raising.
        if self.nbytes < 1:
            return
        accounted_free(self.data.value(), self.nbytes)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Tensor(nbytes=", self.nbytes, ", ndim=", len(self.shape), ")")

    @staticmethod
    def py_init(
        out self: Tensor, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Construct from a contiguous float32 NumPy array (args[0])."""
        if len(args) != 1:
            raise Error("Tensor() takes exactly 1 argument (a NumPy array)")
        var arr = args[0]
        var shape_obj = arr.shape
        var ndim = len(shape_obj)
        var shape = List[Int](capacity=Int(ndim))
        var nelem = 1
        for i in range(ndim):
            var d = Int(py=shape_obj[i])
            shape.append(d)
            nelem *= d
        var nbytes = nelem * 4  # float32
        var strides = _row_major_strides(shape)

        var data: Optional[Pointer[UInt8, MutUntrackedOrigin]]
        if nbytes < 1:
            data = None
        else:
            data = Optional(accounted_alloc(nbytes))
            var src = arr.ctypes.data.unsafe_get_as_pointer[DType.uint8]()
            unsafe_memcpy(dest=data.value(), src=src, count=nbytes)

        self = Tensor(
            data^,
            nbytes,
            shape^,
            strides^,
            0,
        )

    @staticmethod
    def shape_tuple(
        self_ptr: Pointer[Self, MutAnyOrigin],
    ) raises -> PythonObject:
        """Return the shape as a Python tuple of ints."""
        var items = Python.list()
        for i in range(len(self_ptr[].shape)):
            items.append(PythonObject(self_ptr[].shape[i]))
        var builtins = Python.import_module("builtins")
        return builtins.tuple(items)

    @staticmethod
    def copy_to(
        self_ptr: Pointer[Self, MutAnyOrigin],
        arr: PythonObject,
    ) raises:
        """Copy element storage into a contiguous float32 NumPy array."""
        var nbytes = self_ptr[].nbytes
        if nbytes < 1:
            return
        var dest = arr.ctypes.data.unsafe_get_as_pointer[DType.uint8]()
        unsafe_memcpy(dest=dest, src=self_ptr[].data.value(), count=nbytes)
