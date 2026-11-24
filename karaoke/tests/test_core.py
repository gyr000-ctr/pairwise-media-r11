import math

import pytest

import karaoke

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
