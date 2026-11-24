import math

import pytest

import pathsteer

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

A = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0)]

CASES = [
    ("weights_sum", lambda: abs(sum(pathsteer.Steering(1.0, 2.0).weights()) - 1.0) < 1e-12),
    ("count", lambda: len(pathsteer.Steering(1.0, 2.0).step(A, 0.1)) == 2),
    ("separation", lambda: pathsteer.Steering(1.0, 2.0).step(A, 0.1)[0][2] < 0.0),
    ("speed_limit", lambda: all(math.hypot(p[2], p[3]) <= 2.0 + 1e-9 for p in pathsteer.Steering(1.0, 2.0).step(A, 0.1))),
    ("alone", lambda: pathsteer.Steering(1.0, 2.0).step([(0.0, 0.0, 0.0, 0.0)], 0.1) == [(0.0, 0.0, 0.0, 0.0)]),
    ("bad_radius", lambda: _raises(lambda: pathsteer.Steering(0.0, 2.0))),
    ("bad_vmax", lambda: _raises(lambda: pathsteer.Steering(1.0, 0.0))),
    ("empty", lambda: pathsteer.Steering(1.0, 2.0).step([], 0.1) == []),
    ("type", lambda: isinstance(pathsteer.Steering(1.0, 2.0).step(A, 0.1), list)),
    ("det", lambda: pathsteer.Steering(1.0, 2.0).step(A, 0.1) == pathsteer.Steering(1.0, 2.0).step(A, 0.1)),
    ("bad_dt", lambda: _raises(lambda: pathsteer.Steering(1.0, 2.0).step(A, 0.0))),
    ("len", lambda: len(pathsteer.Steering(1.0, 2.0).weights()) == 3),
    ("finite", lambda: all(math.isfinite(v) for p in pathsteer.Steering(1.0, 2.0).step(A, 0.1) for v in p)),
    ("weights_range", lambda: all(0.0 <= w <= 1.0 for w in pathsteer.Steering(1.0, 2.0).weights())),
    ("single", lambda: isinstance(pathsteer.Steering(1.0, 2.0).step([(0.0, 0.0, 0.0, 0.0)], 0.1), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
