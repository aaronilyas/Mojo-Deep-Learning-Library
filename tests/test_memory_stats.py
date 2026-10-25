"""Tests for mojodl.memory_stats() and mojodl.reset_peak_memory_stats().

Peak memory is the high-water mark of process tensor storage (GLOSSARY.md).
Counters are process-wide, so every test reads a baseline and asserts deltas.
"""

import gc

import numpy as np

import mojodl


def test_memory_stats_returns_dict_with_int_byte_keys():
    stats = mojodl.memory_stats()
    assert isinstance(stats, dict)
    assert set(stats.keys()) == {"current_bytes", "peak_bytes"}
    assert isinstance(stats["current_bytes"], int)
    assert isinstance(stats["peak_bytes"], int)


def test_creating_fp32_tensor_raises_current_bytes_by_exact_nbytes():
    baseline = mojodl.memory_stats()["current_bytes"]
    shape = (256, 256)
    nbytes = 256 * 256 * 4
    t = mojodl.Tensor(np.zeros(shape, dtype=np.float32))
    assert mojodl.memory_stats()["current_bytes"] - baseline == nbytes
    del t
    gc.collect()


def test_del_and_gc_collect_returns_current_bytes_to_baseline():
    baseline = mojodl.memory_stats()["current_bytes"]
    t = mojodl.Tensor(np.zeros((256, 256), dtype=np.float32))
    del t
    gc.collect()
    assert mojodl.memory_stats()["current_bytes"] == baseline


def test_peak_bytes_tracks_high_water_mark_after_free():
    baseline = mojodl.memory_stats()
    one_mib = 1024 * 1024
    # 1 MiB fp32 = 262144 elements
    big = mojodl.Tensor(np.zeros(one_mib // 4, dtype=np.float32))
    del big
    gc.collect()
    small = mojodl.Tensor(np.zeros(64, dtype=np.float32))
    after = mojodl.memory_stats()
    # Peak is process-wide; compare to baseline current so prior peaks do not
    # mask the 1 MiB high-water this test just established.
    assert after["peak_bytes"] >= baseline["current_bytes"] + one_mib
    assert after["current_bytes"] < baseline["current_bytes"] + one_mib
    del small
    gc.collect()


def test_reset_peak_memory_stats_sets_peak_equal_to_current():
    # Drive peak above current: allocate, free, leave a smaller live tensor.
    big = mojodl.Tensor(np.zeros(1024 * 256, dtype=np.float32))
    del big
    gc.collect()
    live = mojodl.Tensor(np.zeros(64, dtype=np.float32))
    before = mojodl.memory_stats()
    assert before["peak_bytes"] > before["current_bytes"]

    mojodl.reset_peak_memory_stats()
    after = mojodl.memory_stats()
    assert after["peak_bytes"] == after["current_bytes"]
    del live
    gc.collect()


def test_peak_bytes_always_ge_current_bytes():
    baseline = mojodl.memory_stats()
    assert baseline["peak_bytes"] >= baseline["current_bytes"]

    t = mojodl.Tensor(np.zeros((128, 128), dtype=np.float32))
    mid = mojodl.memory_stats()
    assert mid["peak_bytes"] >= mid["current_bytes"]

    del t
    gc.collect()
    after = mojodl.memory_stats()
    assert after["peak_bytes"] >= after["current_bytes"]
