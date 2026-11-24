import math

import pytest

import texgrad

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m(w=8, h=8):
    return texgrad.MipChain(w, h)

CASES = [
    ("sizes_count", lambda: len(_m().sizes()) >= 3),
    ("first", lambda: _m().sizes()[0] == (8, 8)),
    ("halve", lambda: _m().sizes()[1] == (4, 4)),
    ("last", lambda: _m().sizes()[-1] == (1, 1)),
    ("odd", lambda: _m(5, 3).sizes()[1] == (2, 1)),
    ("constant", lambda: all(abs(v - 0.5) < 1e-12 for row in _m().downsample(1, [[0.5] * 8 for _ in range(8)]) for v in row)),
    ("bad_w", lambda: _raises(lambda: texgrad.MipChain(0, 8))),
    ("bad_level", lambda: _raises(lambda: _m().downsample(99, [[0.5]]))),
    ("empty_img", lambda: _raises(lambda: _m().downsample(0, []))),
    ("type", lambda: isinstance(_m().sizes(), list)),
    ("det", lambda: _m().sizes() == _m().sizes()),
    ("shape", lambda: len(_m(4, 4).downsample(1, [[1.0] * 4 for _ in range(4)])) == 2),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in _m().downsample(1, [[1.0] * 8 for _ in range(8)]) for v in row)),
    ("single", lambda: len(_m(1, 1).sizes()) == 1),
    ("bad_h", lambda: _raises(lambda: texgrad.MipChain(8, 0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
