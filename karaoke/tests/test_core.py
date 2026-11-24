import math
import threading

import numpy as np
import pytest

import karaoke
from karaoke.core import (
    estimate_delay_gain,
    window_sum_error,
    MonoInputError,
    NonFiniteInputError,
    _stft,
    _istft,
    _shift_right,
)


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _k(s=1.0):
    return karaoke.Karaoke(s)


CASES = [
    ("length", lambda: len(_k().process([0.5, 0.5], [0.5, 0.5])[0]) == 2),
    ("mono_removed", lambda: abs(_k().process([0.5], [0.5])[0][0]) < 0.05),
    ("side_kept", lambda: abs(_k().process([0.5], [-0.5])[0][0]) > 0.4),
    ("identity", lambda: abs(_k(0.0).process([0.3], [0.2])[0][0] - 0.3) < 1e-12),
    ("center_db", lambda: _k().center_db([0.5], [0.5]) <= -12.0),
    ("range", lambda: all(abs(v) <= 1.0 for v in _k().process([0.9], [0.9])[0])),
    ("bad_strength", lambda: _raises(lambda: karaoke.Karaoke(2.0))),
    ("bad_len", lambda: _raises(lambda: _k().process([0.1], [0.1, 0.2]))),
    ("empty", lambda: _k().process([], []) == ([], [])),
    ("type", lambda: isinstance(_k().process([0.1], [0.2]), tuple)),
    ("det", lambda: _k().process([0.1], [0.2]) == _k().process([0.1], [0.2])),
    ("nan", lambda: _raises(lambda: _k().process([float('nan')], [0.0]))),
    ("stereo_pair", lambda: len(_k().process([0.1], [0.2])) == 2),
    ("monotonic", lambda: _k(1.0).center_db([0.4], [0.4]) <= _k(0.5).center_db([0.4], [0.4])),
    ("bad_strength_neg", lambda: _raises(lambda: karaoke.Karaoke(-0.1))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True


# --------------------------------------------------------------------------
# 升级题面的硬约束
# --------------------------------------------------------------------------
rng = np.random.default_rng(20240101)
N = 20000
CENTER = rng.standard_normal(N) * 0.45
SIDE = rng.standard_normal(N) * 0.45


def _shift(x, d):
    """线性延迟 y[n] = x[n-d]，零填充（无环绕）。"""
    y = np.zeros_like(x)
    if d >= 0:
        y[d:] = x[: len(x) - d]
    else:
        y[: len(x) + d] = x[-d:]
    return y


def _center_atten_db(c, lo, ro):
    mid_out = 0.5 * (np.asarray(lo) + np.asarray(ro))
    return 10.0 * np.log10(max(np.dot(mid_out, mid_out), 1e-20) / np.dot(c, c))


def _side_loss_db(s, lo, ro):
    side_out = 0.5 * (np.asarray(lo) - np.asarray(ro))
    return -10.0 * np.log10(max(np.dot(side_out, side_out), 1e-20) / np.dot(s, s))


def test_window_cola_sum_is_one():
    """WOLA 分析/合成窗积移位和恒为 1（<=1e-9）。"""
    assert window_sum_error() <= 1e-9


@pytest.mark.parametrize("n", [1, 2, 7, 100, 2047, 2048, 2049, 10000])
def test_stft_perfect_reconstruction(n):
    x = rng.standard_normal(n)
    spec, pre, length, _ = _stft(x)
    xr = _istft(spec, pre, length)
    assert xr.shape == (n,)
    assert np.max(np.abs(xr - x)) < 1e-9


def test_delay_estimation_within_one_sample():
    """±3 采样通道延迟下，互相关估计误差 <= 1 采样（长信号/真实音频域）。"""
    n = 20000
    c = rng.standard_normal(n) * 0.45
    s = rng.standard_normal(n) * 0.45
    for d in (-3, -2, -1, 0, 1, 2, 3):
        left = c + s
        right = _shift(0.7 * c - s, d)
        lag, _ = estimate_delay_gain(left, right)
        assert abs(lag - (-d)) <= 1, f"d={d} got lag={lag}"


def test_gain_estimation_accuracy():
    """纯增益失配下 RMS 增益估计误差 < 2%。"""
    c = rng.standard_normal(20000) * 0.5
    for g in (0.5, 0.9, 1.0, 1.3, 2.0):
        _, gain = estimate_delay_gain(c, g * c)
        assert abs(gain - 1.0 / g) / (1.0 / g) < 0.02


def test_pure_side_not_rescaled():
    s = rng.standard_normal(20000) * 0.5
    lag, gain = estimate_delay_gain(s, -s)
    assert lag == 0 and abs(gain - 1.0) < 1e-12


def test_pure_center_attenuation_15db():
    c = rng.standard_normal(N) * 0.45
    lo, ro = _k().process(c, c)
    assert _center_atten_db(c, lo, ro) <= -15.0


def test_pure_side_loss_under_03db():
    s = rng.standard_normal(N) * 0.45
    lo, ro = _k().process(s, -s)
    assert _side_loss_db(s, lo, ro) <= 0.3


def test_mixed_center_and_side_metrics():
    c, s = CENTER, SIDE
    lo, ro = _k().process(c + s, c - s)
    assert _center_atten_db(c, lo, ro) <= -15.0
    assert _side_loss_db(s, lo, ro) <= 0.3


@pytest.mark.parametrize("d", [-3, -2, -1, 0, 1, 2, 3])
def test_delay_mismatch_robustness(d):
    """±3 采样延迟 + 1.15 增益失配下仍满足衰减/侧损指标。"""
    c, s = CENTER, SIDE
    left = c + s
    right = _shift(1.15 * c - s, d)
    lo, ro = _k().process(left, right)
    assert _center_atten_db(c, lo, ro) <= -15.0
    assert _side_loss_db(s, lo, ro) <= 0.3


def test_mono_input_distinct_error():
    with pytest.raises(MonoInputError):
        _k().process([0.1], None)
    with pytest.raises(MonoInputError):
        _k().process(np.zeros((1, 4)), np.zeros((1, 4)))


def test_nonfinite_distinct_error():
    with pytest.raises(NonFiniteInputError):
        _k().process([float("nan"), 0.1], [0.1, 0.1])
    with pytest.raises(NonFiniteInputError):
        _k().process([0.1, 0.1], [float("inf"), 0.1])


def test_error_types_are_distinct_value_errors():
    assert issubclass(MonoInputError, ValueError)
    assert issubclass(NonFiniteInputError, ValueError)
    assert MonoInputError is not NonFiniteInputError


def test_strength_monotonic_and_identity():
    assert _k(0.0).process([0.3], [0.2]) == ([0.3], [0.2])
    dbs = [_k(s).center_db(CENTER[:4096], CENTER[:4096])
           for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(dbs[i] >= dbs[i + 1] for i in range(4))


def test_performance_100k_under_400ms():
    n = 100_000
    a = rng.standard_normal(n)
    b = _shift(a, 2) * 0.9 + 0.2 * rng.standard_normal(n)
    k = _k()
    k.process(a[:4096], b[:4096])  # 预热
    import time
    t0 = time.perf_counter()
    k.process(a, b)
    assert (time.perf_counter() - t0) * 1000.0 < 400.0


def test_thread_safety():
    k = _k()
    errors = []

    def worker(seed):
        try:
            local = np.random.default_rng(seed).standard_normal(8192)
            for _ in range(20):
                out = k.process(local, _shift(local, 1))
                if len(out[0]) != 8192:
                    raise RuntimeError("bad length")
        except Exception:  # noqa: BLE001
            import traceback
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors


def test_output_bounded_and_length_preserved():
    for n in (1, 5, 1000):
        a = rng.standard_normal(n) * 0.9
        b = rng.standard_normal(n) * 0.9
        lo, ro = _k().process(a, b)
        assert len(lo) == n and len(ro) == n
        assert all(abs(v) <= 1.0 + 1e-12 for v in lo + ro)
        assert all(math.isfinite(v) for v in lo + ro)
