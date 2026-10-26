"""Tests for Tensor.reshape(*shape) as a zero-copy view on cpu and Meta.

PyTorch (CPU) is the numerical oracle. Covers forward/backward, shared-storage
lifetime and memory accounting, Meta dry-run paths, and ShapeError / TypeError
on bad shapes. Reshape always returns a view (every tensor is contiguous).
"""

import gc

import numpy as np
import pytest
import torch

import mojodl

_FORWARD_CASES = [
    ((6,), (2, 3)),
    ((2, 3), (3, 2)),
    ((2, 3, 4), (4, -1)),
    ((2, 3, 4), (-1,)),
    ((2, 3, 4), (2, -1, 2)),
    ((), (1,)),
    ((1,), ()),
    ((1, 1), ()),
    ((0, 3), (3, 0)),
    ((0, 3), (-1, 3)),
]

# tuple form for every case; varargs only when target is non-empty
# (empty target as varargs is reshape(), which is TypeError — covered elsewhere)
_FORWARD_PARAMS = [
    (in_shape, target, form)
    for in_shape, target in _FORWARD_CASES
    for form in ("varargs", "tuple")
    if form == "tuple" or target != ()
]


def _make_data(shape):
    if any(d == 0 for d in shape):
        return np.zeros(shape, dtype=np.float32)
    if shape == ():
        return np.array(1.5, dtype=np.float32)
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize(
    "in_shape,target,form",
    _FORWARD_PARAMS,
    ids=[f"{s}->{t}-{form}" for s, t, form in _FORWARD_PARAMS],
)
def test_reshape_forward_matches_torch(in_shape, target, form):
    a_np = _make_data(in_shape)
    a = mojodl.Tensor(a_np)

    if form == "varargs":
        result = a.reshape(*target)
    else:
        result = a.reshape(target)

    expected = torch.from_numpy(a_np).reshape(target)

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == tuple(expected.shape)
    assert result.dtype == mojodl.float32
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_reshape_no_args_raises_type_error():
    a = mojodl.Tensor(np.zeros((1,), dtype=np.float32))
    with pytest.raises(TypeError):
        a.reshape()


def test_reshape_no_args_on_meta_raises_type_error():
    a = mojodl.empty((1,), mojodl.float32, device="meta")
    with pytest.raises(TypeError):
        a.reshape()


def test_reshape_forward_list_form_matches_torch():
    a_np = _make_data((6,))
    a = mojodl.Tensor(a_np)

    result = a.reshape([2, 3])
    expected = torch.from_numpy(a_np).reshape([2, 3])

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == tuple(expected.shape)
    assert result.dtype == mojodl.float32
    np.testing.assert_array_equal(result.numpy(), expected.numpy())


def test_reshape_is_zero_copy_on_cpu():
    t = mojodl.Tensor(np.zeros((64, 64), dtype=np.float32))
    mojodl.reset_peak_memory_stats()
    before = mojodl.memory_stats()

    v = t.reshape(-1)

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    assert v.shape == (4096,)
    del t, v
    gc.collect()


def test_reshape_view_keeps_storage_alive_after_base_deleted():
    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)

    t = mojodl.Tensor(data)
    assert mojodl.memory_stats()["current_bytes"] - baseline == 16384

    v = t.reshape(-1)
    del t
    gc.collect()

    assert mojodl.memory_stats()["current_bytes"] - baseline == 16384
    np.testing.assert_array_equal(v.numpy(), data.reshape(-1))

    del v
    gc.collect()
    assert mojodl.memory_stats()["current_bytes"] == baseline


def test_reshape_view_freed_before_base_returns_to_baseline():
    gc.collect()
    baseline = mojodl.memory_stats()["current_bytes"]
    data = np.zeros((64, 64), dtype=np.float32)

    t = mojodl.Tensor(data)
    assert mojodl.memory_stats()["current_bytes"] - baseline == 16384

    v = t.reshape(-1)
    del v
    gc.collect()

    assert mojodl.memory_stats()["current_bytes"] - baseline == 16384

    del t
    gc.collect()
    assert mojodl.memory_stats()["current_bytes"] == baseline


def test_reshape_view_of_view_keeps_data_after_intermediates_deleted():
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    t = mojodl.Tensor(data)
    mid = t.reshape(4096)
    v = mid.reshape(16, 256)

    del t, mid
    gc.collect()

    np.testing.assert_array_equal(v.numpy(), data.reshape(16, 256))
    del v
    gc.collect()


def test_reshape_backward_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    w_np = rng.standard_normal((4, 6)).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.reshape(4, -1) * w).sum().backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch_x.reshape(4, -1) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (2, 3, 4)
    assert x.grad.dtype == mojodl.float32
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_reshape_chain_backward_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    w_np = rng.standard_normal((24,)).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    w = mojodl.Tensor(w_np)
    (x.reshape(6, 4).reshape(24) * w).sum().backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (torch_x.reshape(6, 4).reshape(24) * torch_w).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (2, 3, 4)
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_reshape_backward_when_x_used_twice_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    w1_np = rng.standard_normal((6, 4)).astype(np.float32)
    w2_np = rng.standard_normal((2, 3, 4)).astype(np.float32)

    x = mojodl.Tensor(a_np, requires_grad=True)
    w1 = mojodl.Tensor(w1_np)
    w2 = mojodl.Tensor(w2_np)
    ((x.reshape(6, 4) * w1).sum() + (x * w2).sum()).backward()

    torch_x = torch.tensor(a_np, requires_grad=True)
    torch_w1 = torch.tensor(w1_np)
    torch_w2 = torch.tensor(w2_np)
    ((torch_x.reshape(6, 4) * torch_w1).sum() + (torch_x * torch_w2).sum()).backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.shape == (2, 3, 4)
    np.testing.assert_allclose(
        x.grad.numpy(), torch_x.grad.numpy(), rtol=1e-6, atol=1e-6
    )


def test_reshape_meta_returns_meta_without_allocating():
    t = mojodl.empty((2, 3, 4), mojodl.float32, device="meta")
    before = mojodl.memory_stats()

    result = t.reshape(4, -1)

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == (4, 6)
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]
    del t, result
    gc.collect()


def test_reshape_meta_backward_fills_meta_grad_without_allocating():
    before = mojodl.memory_stats()

    x = mojodl.empty((2, 3, 4), device="meta", requires_grad=True)
    x.reshape(-1).sum().backward()

    assert isinstance(x.grad, mojodl.Tensor)
    assert x.grad.device == "meta"
    assert x.grad.shape == (2, 3, 4)
    assert x.grad.dtype == mojodl.float32

    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


_SHAPE_ERROR_CASES = [
    ((2, 3), (4, 2), "numel_or_shapes"),
    ((2, 3), (-1, -1), "two_inferred"),
    ((2, 3), (-2, 3), "negative_dim"),
    ((2, 3), (4, -1), "not_divisible"),
    ((0, 3), (-1, 0), "zero_ambiguous"),
]


@pytest.mark.parametrize(
    "in_shape,target,kind",
    _SHAPE_ERROR_CASES,
    ids=[k for _, _, k in _SHAPE_ERROR_CASES],
)
def test_reshape_bad_shape_raises_shape_error(in_shape, target, kind):
    a = mojodl.Tensor(np.zeros(in_shape, dtype=np.float32))

    with pytest.raises(mojodl.ShapeError) as exc_info:
        a.reshape(*target)

    if kind == "numel_or_shapes":
        msg = str(exc_info.value)
        shapes_ok = "(2, 3)" in msg and "(4, 2)" in msg
        numel_ok = "6" in msg
        assert shapes_ok or numel_ok


@pytest.mark.parametrize(
    "in_shape,target,kind",
    _SHAPE_ERROR_CASES,
    ids=[k for _, _, k in _SHAPE_ERROR_CASES],
)
def test_reshape_bad_shape_on_meta_raises_shape_error(in_shape, target, kind):
    a = mojodl.empty(in_shape, mojodl.float32, device="meta")

    with pytest.raises(mojodl.ShapeError) as exc_info:
        a.reshape(*target)

    if kind == "numel_or_shapes":
        msg = str(exc_info.value)
        shapes_ok = "(2, 3)" in msg and "(4, 2)" in msg
        numel_ok = "6" in msg
        assert shapes_ok or numel_ok


@pytest.mark.parametrize(
    "args",
    [(2.0, 3), (True, 6)],
    ids=["float", "bool"],
)
def test_reshape_non_int_dim_raises_type_error(args):
    a = mojodl.Tensor(np.zeros(6, dtype=np.float32))

    with pytest.raises(TypeError):
        a.reshape(*args)
