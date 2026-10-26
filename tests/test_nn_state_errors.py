"""Tests for strict load_state_dict rejection via mojodl.StateDictError.

Loading is strict: the state's keys must match named_parameters() exactly.
Wrong names, shapes, or non-float32-convertible values raise StateDictError
naming the offending parameter(s); rejected loads leave the Module unchanged.
A non-mapping state raises TypeError. StateDictError is not implemented yet,
so these tests fail with AttributeError or DID NOT RAISE until that slice lands.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import mojodl


def _mlp(seed: int):
    """Sequential(Linear(4, 8), ReLU(), Linear(8, 3)) with distinct Linear seeds."""
    return mojodl.nn.Sequential(
        mojodl.nn.Linear(4, 8, seed=seed),
        mojodl.nn.ReLU(),
        mojodl.nn.Linear(8, 3, seed=seed + 1),
    )


def _model_and_good():
    """Model under test plus a twin's state_dict (different values)."""
    model = _mlp(seed=0)
    twin = _mlp(seed=100)
    good = twin.state_dict()
    # Twin values must differ so a non-atomic partial load would be visible.
    for name, param in model.named_parameters():
        assert not np.array_equal(param.numpy(), good[name])
    return model, good


def test_state_dict_error_is_value_error_subclass():
    assert issubclass(mojodl.StateDictError, ValueError)


def test_load_state_dict_unexpected_name_raises_state_dict_error():
    model, good = _model_and_good()
    bad = dict(good)
    bad["0.wieght"] = good["0.weight"].copy()

    with pytest.raises(mojodl.StateDictError) as exc_info:
        model.load_state_dict(bad)

    assert "0.wieght" in str(exc_info.value)


def test_load_state_dict_renamed_key_raises_state_dict_error():
    model, good = _model_and_good()
    bad = {("0.wieght" if k == "0.weight" else k): v.copy() for k, v in good.items()}

    with pytest.raises(mojodl.StateDictError) as exc_info:
        model.load_state_dict(bad)

    msg = str(exc_info.value)
    assert "0.wieght" in msg


def test_load_state_dict_missing_name_raises_state_dict_error():
    model, good = _model_and_good()
    bad = {k: v.copy() for k, v in good.items() if k != "2.bias"}

    with pytest.raises(mojodl.StateDictError) as exc_info:
        model.load_state_dict(bad)

    assert "2.bias" in str(exc_info.value)


def test_load_state_dict_wrong_shape_raises_state_dict_error():
    model, good = _model_and_good()
    bad = {k: v.copy() for k, v in good.items()}
    # 0.weight is (8, 4); feed the transposed shape (4, 8).
    assert bad["0.weight"].shape == (8, 4)
    bad["0.weight"] = np.zeros((4, 8), dtype=np.float32)

    with pytest.raises(mojodl.StateDictError) as exc_info:
        model.load_state_dict(bad)

    msg = str(exc_info.value)
    assert "0.weight" in msg
    assert "(8, 4)" in msg
    assert "(4, 8)" in msg


def test_load_state_dict_multiple_problems_raises_one_state_dict_error():
    model, good = _model_and_good()
    # Unexpected "0.wieght", missing "2.bias", wrong shape on "0.weight".
    bad = {k: v.copy() for k, v in good.items() if k != "2.bias"}
    bad["0.wieght"] = good["0.weight"].copy()
    bad["0.weight"] = np.zeros((4, 8), dtype=np.float32)

    with pytest.raises(mojodl.StateDictError) as exc_info:
        model.load_state_dict(bad)

    msg = str(exc_info.value)
    assert "0.wieght" in msg
    assert "2.bias" in msg
    assert "0.weight" in msg


@pytest.mark.parametrize(
    "bad_value",
    ["abc", [[1.0, 2.0], [3.0]]],
    ids=["non_numeric_string", "ragged_nested_list"],
)
def test_load_state_dict_non_float32_convertible_raises_state_dict_error(bad_value):
    model, good = _model_and_good()
    bad = {k: v.copy() for k, v in good.items()}
    bad["0.bias"] = bad_value

    with pytest.raises(mojodl.StateDictError) as exc_info:
        model.load_state_dict(bad)

    assert "0.bias" in str(exc_info.value)


def test_load_state_dict_non_mapping_raises_type_error():
    model, good = _model_and_good()
    pairs = list(good.items())

    with pytest.raises(TypeError):
        model.load_state_dict(pairs)


def _unexpected_added(good):
    bad = {k: v.copy() for k, v in good.items()}
    bad["0.wieght"] = good["0.weight"].copy()
    return bad


def _renamed_key(good):
    return {
        ("0.wieght" if k == "0.weight" else k): v.copy() for k, v in good.items()
    }


def _missing_bias(good):
    return {k: v.copy() for k, v in good.items() if k != "2.bias"}


def _wrong_shape(good):
    bad = {k: v.copy() for k, v in good.items()}
    bad["0.weight"] = np.zeros((4, 8), dtype=np.float32)
    return bad


def _multiple_problems(good):
    bad = {k: v.copy() for k, v in good.items() if k != "2.bias"}
    bad["0.wieght"] = good["0.weight"].copy()
    bad["0.weight"] = np.zeros((4, 8), dtype=np.float32)
    return bad


def _bad_string(good):
    bad = {k: v.copy() for k, v in good.items()}
    bad["0.bias"] = "abc"
    return bad


def _bad_ragged(good):
    bad = {k: v.copy() for k, v in good.items()}
    bad["0.bias"] = [[1.0, 2.0], [3.0]]
    return bad


@pytest.mark.parametrize(
    "make_bad",
    [
        _unexpected_added,
        _renamed_key,
        _missing_bias,
        _wrong_shape,
        _multiple_problems,
        _bad_string,
        _bad_ragged,
    ],
    ids=[
        "unexpected_added",
        "renamed_key",
        "missing_bias",
        "wrong_shape",
        "multiple_problems",
        "bad_string",
        "bad_ragged",
    ],
)
def test_rejected_load_state_dict_is_atomic(make_bad):
    """Failed load leaves parameters and current_bytes exactly unchanged."""
    model, good = _model_and_good()
    bad = make_bad(good)

    gc.collect()
    before_bytes = mojodl.memory_stats()["current_bytes"]
    before_state = {k: v.copy() for k, v in model.state_dict().items()}

    with pytest.raises(mojodl.StateDictError):
        model.load_state_dict(bad)

    gc.collect()
    after_bytes = mojodl.memory_stats()["current_bytes"]
    after_state = model.state_dict()

    assert after_bytes == before_bytes
    assert list(after_state.keys()) == list(before_state.keys())
    for key in before_state:
        np.testing.assert_array_equal(after_state[key], before_state[key])
