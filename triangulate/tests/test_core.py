import math

import pytest

import triangulate

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

SQ = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
CONCAVE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 1.0), (0.0, 2.0)]

CASES = [
    ("count", lambda: len(triangulate.EarClip().triangulate(SQ)) == 2),
    ("concave", lambda: len(triangulate.EarClip().triangulate(CONCAVE)) == 3),
    ("area_sum", lambda: abs(sum(triangulate.EarClip().area([CONCAVE[i] for i in t]) for t in triangulate.EarClip().triangulate(CONCAVE)) - triangulate.EarClip().area(CONCAVE)) < 1e-9),
    ("square_area", lambda: abs(triangulate.EarClip().area(SQ) - 4.0) < 1e-9),
    ("valid_index", lambda: all(0 <= i < 4 for t in triangulate.EarClip().triangulate(SQ) for i in t)),
    ("bad_self", lambda: _raises(lambda: triangulate.EarClip().triangulate([(0, 0), (2, 2), (2, 0), (0, 2)]))),
    ("bad_short", lambda: _raises(lambda: triangulate.EarClip().triangulate([(0, 0), (1, 1)]))),
    ("empty", lambda: _raises(lambda: triangulate.EarClip().triangulate([]))),
    ("type", lambda: isinstance(triangulate.EarClip().triangulate(SQ), list)),
    ("det", lambda: triangulate.EarClip().triangulate(SQ) == triangulate.EarClip().triangulate(SQ)),
    ("tri_len", lambda: all(len(t) == 3 for t in triangulate.EarClip().triangulate(SQ))),
    ("positive", lambda: all(triangulate.EarClip().area([SQ[i] for i in t]) > 0 for t in triangulate.EarClip().triangulate(SQ))),
    ("float_area", lambda: isinstance(triangulate.EarClip().area(SQ), float)),
    ("concave_area", lambda: triangulate.EarClip().area(CONCAVE) > 0.0),
    ("no_dup_index", lambda: len(set(triangulate.EarClip().triangulate(SQ)[0])) == 3),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
