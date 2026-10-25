"""Contiguous fp32 tensor storage owned by the Mojo engine.

`Tensor` owns the storage pointer, byte size, shape, row-major strides,
dtype, and device. Allocation and free go through the accounting layer.
Meta tensors carry shape and dtype only (no storage). Python holds only a
reference to this engine object; addresses and byte counts never cross the
boundary.
"""

from std.memory import Pointer, unsafe_memcpy
from std.format import Writable, Writer
from std.python import PythonObject, Python
from accounting import accounted_alloc, accounted_free

# device_code values: 0 = cpu, 1 = meta


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


def _format_shape(shape: List[Int]) -> String:
    """Format a shape as a Python-tuple-like string, e.g. "(2, 3)" or "(7,)"."""
    var s = String("(")
    for i in range(len(shape)):
        if i > 0:
            s += ", "
        s += String(shape[i])
    if len(shape) == 1:
        s += ","
    s += ")"
    return s


def _nbytes_for_shape(shape: List[Int]) raises -> Int:
    """fp32 byte count for a dense shape, with overflow detection.

    0-d shapes are 1 element (4 bytes). Raises a tagged OverflowError when
    the element count or byte size would exceed Int.MAX, before any
    allocation. Message names the offending shape.
    """
    var nelem = 1
    for i in range(len(shape)):
        var d = shape[i]
        if d != 0 and nelem > Int.MAX // d:
            raise Error(
                "OverflowError: shape "
                + _format_shape(shape)
                + " exceeds the maximum tensor size"
            )
        nelem *= d
    if nelem != 0 and nelem > Int.MAX // 4:
        raise Error(
            "OverflowError: shape "
            + _format_shape(shape)
            + " exceeds the maximum tensor size"
        )
    return nelem * 4


def _require_numpy_f32_buffer(
    arr: PythonObject, expected_nbytes: Int, *, writeable: Bool
) raises:
    """Raise unless `arr` is a safe float32 NumPy buffer of `expected_nbytes`.

    Requires a NumPy ndarray with dtype float32, C-contiguous layout, and
    `arr.nbytes == expected_nbytes`. When `writeable` is true, also requires
    a writeable buffer (for copy destinations).
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    if not builtins.isinstance(arr, np.ndarray):
        raise Error("TypeError: expected a NumPy ndarray")
    if arr.dtype != np.float32:
        raise Error("TypeError: expected dtype float32")
    if not arr.flags.c_contiguous:
        raise Error("ValueError: array must be C-contiguous")
    if writeable and not arr.flags.writeable:
        raise Error("ValueError: array must be writeable")
    if Int(py=arr.nbytes) != expected_nbytes:
        raise Error("ValueError: array nbytes does not match tensor byte size")


@fieldwise_init
struct Tensor(Movable, Writable):
    """Eager fp32 tensor; Meta tensors hold shape/dtype only."""

    var data: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var nbytes: Int
    var shape: List[Int]
    var strides: List[Int]
    # 0 = float32; only dtype supported in this milestone.
    var dtype_code: Int
    # 0 = cpu (has storage when nbytes > 0); 1 = meta (never has storage).
    var device_code: Int

    def __deinit__(deinit self):
        # Free storage exactly once when the Python wrapper is collected.
        # Meta tensors never allocate; destructors must not raise.
        if self.device_code == 1:  # meta: never allocated
            return
        if self.nbytes < 1:
            return
        accounted_free(self.data.value(), self.nbytes)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Tensor(nbytes=",
            self.nbytes,
            ", ndim=",
            len(self.shape),
            ", device=",
            self.device_code,
            ")",
        )

    @staticmethod
    def py_init(
        out self: Tensor, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Construct from a contiguous float32 NumPy array (args[0])."""
        if len(args) != 1:
            raise Error("Tensor() takes exactly 1 argument (a NumPy array)")
        var arr = args[0]
        # Reject non-ndarrays before touching .shape / allocating.
        var np = Python.import_module("numpy")
        var builtins = Python.import_module("builtins")
        if not builtins.isinstance(arr, np.ndarray):
            raise Error("TypeError: expected a NumPy ndarray")
        var shape_obj = arr.shape
        var ndim = len(shape_obj)
        var shape = List[Int](capacity=Int(ndim))
        for i in range(ndim):
            shape.append(Int(py=shape_obj[i]))
        var nbytes = _nbytes_for_shape(shape)
        # Validate dtype, contiguity, and size before any allocation or memcpy.
        _require_numpy_f32_buffer(arr, nbytes, writeable=False)
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
            0,  # cpu
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
    def device_str(
        self_ptr: Pointer[Self, MutAnyOrigin],
    ) raises -> PythonObject:
        """Return the device name as a Python str ('cpu' or 'meta')."""
        if self_ptr[].device_code == 1:
            return PythonObject("meta")
        return PythonObject("cpu")

    @staticmethod
    def copy_to(
        self_ptr: Pointer[Self, MutAnyOrigin],
        arr: PythonObject,
    ) raises:
        """Copy element storage into a contiguous float32 NumPy array."""
        if self_ptr[].device_code == 1:
            raise Error(
                "mojodl.DeviceError: cannot copy data from a Meta tensor"
            )
        var nbytes = self_ptr[].nbytes
        if nbytes < 1:
            return
        _require_numpy_f32_buffer(arr, nbytes, writeable=True)
        var dest = arr.ctypes.data.unsafe_get_as_pointer[DType.uint8]()
        unsafe_memcpy(dest=dest, src=self_ptr[].data.value(), count=nbytes)


def empty(shape_obj: PythonObject, device_code_obj: PythonObject) raises -> PythonObject:
    """Create an uninitialized cpu Tensor or a Meta tensor (no storage).

    `shape_obj` is a sequence of non-negative ints. `device_code_obj` is
    0 (cpu) or 1 (meta). Cpu allocates through accounting; Meta never
    calls accounted_alloc.
    """
    var device_code = Int(py=device_code_obj)
    var ndim = len(shape_obj)
    var shape = List[Int](capacity=Int(ndim))
    for i in range(ndim):
        shape.append(Int(py=shape_obj[i]))
    var strides = _row_major_strides(shape)

    # Validate size for both cpu and meta before any allocation.
    var nbytes_checked = _nbytes_for_shape(shape)

    var data: Optional[Pointer[UInt8, MutUntrackedOrigin]]
    var nbytes: Int
    if device_code == 1:  # meta
        data = None
        nbytes = 0
    else:
        nbytes = nbytes_checked
        if nbytes < 1:
            data = None
        else:
            data = Optional(accounted_alloc(nbytes))

    var result = Tensor(
        data^,
        nbytes,
        shape^,
        strides^,
        0,
        device_code,
    )
    return PythonObject(alloc=result^)
