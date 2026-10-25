"""NumPy-style broadcasting for elementwise Tensor ops (currently + only).

PyTorch (CPU) is the numerical oracle. Tests drive only the public Python API.
OPS and BROADCAST_CASES are module-level tables so later ops (-, *, /) can
share the same parametrized coverage.
"""

import operator

import numpy as np
import pytest
import torch

import mojodl

OPS = {
    "add": operator.add,
    "mul": operator.mul,
    "sub": operator.sub,
}

BROADCAST_CASES = [
    # (a_shape, b_shape, out_shape, id)
    ((3, 4), (3, 4), (3, 4), "same"),
    ((3, 4), (), (3, 4), "scalar_right"),
    ((), (3, 4), (3, 4), "scalar_left"),
    ((3, 4), (1,), (3, 4), "ones_right"),
    ((3, 4), (4,), (3, 4), "row"),
    ((3, 4), (1, 4), (3, 4), "row_explicit"),
    ((3, 4), (3, 1), (3, 4), "column"),
    ((2, 3, 4), (3, 1), (2, 3, 4), "rank_mismatch"),
    ((2, 1, 4), (3, 1), (2, 3, 4), "both_sides"),
]

_OP_NAMES = list(OPS.keys())
_CASE_IDS = [c[3] for c in BROADCAST_CASES]


def _make_pair(a_shape, b_shape, seed=0):
    """Random float32 arrays; |b| in [0.5, 2] so division can reuse this later."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(a_shape).astype(np.float32)
    # Map U(0,1) -> [0.5, 2], then random sign.
    b = (0.5 + 1.5 * rng.random(b_shape)).astype(np.float32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=b_shape)
    b = b * signs
    # 0-d arithmetic can yield a numpy scalar; keep real ndarrays for from_numpy.
    return np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)


@pytest.mark.parametrize("op_name", _OP_NAMES)
@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [(c[0], c[1], c[2]) for c in BROADCAST_CASES],
    ids=_CASE_IDS,
)
def test_elementwise_forward_matches_torch(op_name, a_shape, b_shape, out_shape):
    op = OPS[op_name]
    a_np, b_np = _make_pair(a_shape, b_shape)

    result = op(mojodl.Tensor(a_np), mojodl.Tensor(b_np))
    expected = op(torch.from_numpy(a_np), torch.from_numpy(b_np)).numpy()

    assert isinstance(result, mojodl.Tensor)
    assert result.shape == out_shape
    assert result.dtype == mojodl.float32
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("op_name", _OP_NAMES)
@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [(c[0], c[1], c[2]) for c in BROADCAST_CASES],
    ids=_CASE_IDS,
)
def test_elementwise_backward_matches_torch(op_name, a_shape, b_shape, out_shape):
    op = OPS[op_name]
    a_np, b_np = _make_pair(a_shape, b_shape, seed=1)
    rng = np.random.default_rng(2)
    w_np = rng.standard_normal(out_shape).astype(np.float32)

    a = mojodl.Tensor(a_np, requires_grad=True)
    b = mojodl.Tensor(b_np, requires_grad=True)
    w = mojodl.Tensor(w_np)  # non-grad weight
    (op(a, b) * w).sum().backward()

    torch_a = torch.tensor(a_np, requires_grad=True)
    torch_b = torch.tensor(b_np, requires_grad=True)
    torch_w = torch.tensor(w_np)
    (op(torch_a, torch_b) * torch_w).sum().backward()

    assert isinstance(a.grad, mojodl.Tensor)
    assert isinstance(b.grad, mojodl.Tensor)
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    np.testing.assert_allclose(
        a.grad.numpy(), torch_a.grad.numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        b.grad.numpy(), torch_b.grad.numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("op_name", _OP_NAMES)
@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [(c[0], c[1], c[2]) for c in BROADCAST_CASES],
    ids=_CASE_IDS,
)
def test_elementwise_meta_forward(op_name, a_shape, b_shape, out_shape):
    op = OPS[op_name]
    a = mojodl.empty(a_shape, device="meta")
    b = mojodl.empty(b_shape, device="meta")
    before = mojodl.memory_stats()

    result = op(a, b)

    assert isinstance(result, mojodl.Tensor)
    assert result.device == "meta"
    assert result.shape == out_shape
    assert result.dtype == mojodl.float32
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


@pytest.mark.parametrize("op_name", _OP_NAMES)
@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [(c[0], c[1], c[2]) for c in BROADCAST_CASES],
    ids=_CASE_IDS,
)
def test_elementwise_meta_backward(op_name, a_shape, b_shape, out_shape):
    op = OPS[op_name]
    a = mojodl.empty(a_shape, device="meta", requires_grad=True)
    b = mojodl.empty(b_shape, device="meta", requires_grad=True)
    w = mojodl.empty(out_shape, device="meta")
    before = mojodl.memory_stats()

    (op(a, b) * w).sum().backward()

    assert isinstance(a.grad, mojodl.Tensor)
    assert isinstance(b.grad, mojodl.Tensor)
    assert a.grad.device == "meta"
    assert b.grad.device == "meta"
    assert a.grad.shape == a_shape
    assert b.grad.shape == b_shape
    after = mojodl.memory_stats()
    assert after["current_bytes"] == before["current_bytes"]
    assert after["peak_bytes"] == before["peak_bytes"]


@pytest.mark.parametrize("op_name", _OP_NAMES)
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_elementwise_incompatible_shapes_raise_shape_error(op_name, device):
    op = OPS[op_name]
    if device == "meta":
        a = mojodl.empty((2, 3), device="meta")
        b = mojodl.empty((4,), device="meta")
    else:
        a = mojodl.Tensor(np.zeros((2, 3), dtype=np.float32))
        b = mojodl.Tensor(np.zeros((4,), dtype=np.float32))

    assert issubclass(mojodl.ShapeError, ValueError)
    with pytest.raises(mojodl.ShapeError) as exc_info:
        op(a, b)

    msg = str(exc_info.value)
    assert "(2, 3)" in msg
    assert "(4,)" in msg
