import math

import pytest

import otsu

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.1, 0.2], [0.8, 0.9]]

CASES = [
    ("threshold_range", lambda: 0.0 <= otsu.Otsu().threshold(IMG) <= 1.0),
    ("between_mid", lambda: 0.2 <= otsu.Otsu().threshold(IMG) <= 0.8),
    ("binary_values", lambda: set(v for row in otsu.Otsu().binarize(IMG, 0.5) for v in row) <= {0.0, 1.0}),
    ("hist_sum", lambda: abs(sum(otsu.Otsu().histogram(IMG)) - 1.0) < 1e-12),
    ("bad_constant", lambda: _raises(lambda: otsu.Otsu().threshold([[0.5, 0.5], [0.5, 0.5]]))),
    ("bad_empty", lambda: _raises(lambda: otsu.Otsu().threshold([]))),
    ("type", lambda: isinstance(otsu.Otsu().threshold(IMG), float)),
    ("det", lambda: otsu.Otsu().threshold(IMG) == otsu.Otsu().threshold(IMG)),
    ("shape", lambda: len(otsu.Otsu().binarize(IMG, 0.5)) == 2),
    ("hist_len", lambda: len(otsu.Otsu().histogram(IMG)) == 256),
    ("bad_ragged", lambda: _raises(lambda: otsu.Otsu().binarize([[0.1, 0.2], [0.3]], 0.5))),
    ("binarize_type", lambda: isinstance(otsu.Otsu().binarize(IMG, 0.5), list)),
    ("bad_t", lambda: _raises(lambda: otsu.Otsu().binarize(IMG, 1.5))),
    ("monotone_order", lambda: otsu.Otsu().binarize(IMG, 0.5)[0][0] <= otsu.Otsu().binarize(IMG, 0.5)[1][1]),
    ("hist_range", lambda: all(v >= 0.0 for v in otsu.Otsu().histogram(IMG))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
