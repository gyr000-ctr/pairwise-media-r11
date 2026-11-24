"""多阈值 Otsu（三类分割）验收测试。"""

import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import otsu
from otsu import Otsu, OtsuMode, MultiOtsuResult, LEVELS, VARIANCE_TOL


# --------------------------------------------------------------------------
# 辅助构造
# --------------------------------------------------------------------------
def image_from_counts(counts):
    """按 256 bin 计数构造嵌套 list 图像（每 bin 取中点值）。"""
    row = []
    for i, c in enumerate(counts):
        value = (i + 0.5) / LEVELS
        row.extend([value] * c)
    # 单行二维序列，保证所有行等长（避免触发 ragged 校验）。
    return [row]


def random_counts(seed, nonzero_bins=None, total=4000):
    rng = random.Random(seed)
    counts = [0] * LEVELS
    if nonzero_bins is None:
        nonzero_bins = rng.randint(3, 40)
    bins = rng.sample(range(LEVELS), nonzero_bins)
    weights = [rng.random() + 0.05 for _ in bins]
    scale = total / sum(weights)
    for b, w in zip(bins, weights):
        counts[b] = max(1, int(w * scale))
    return counts


# --------------------------------------------------------------------------
# 阈值合法性
# --------------------------------------------------------------------------
def test_three_cluster_image():
    image = [[0.05] * 8 + [0.5] * 8 + [0.95] * 8 for _ in range(6)]
    result = Otsu().multi_threshold(image)
    assert isinstance(result, MultiOtsuResult)
    assert result.n_classes == 3
    assert result.mode is OtsuMode.THREE_CLASS
    assert len(result.thresholds) == 2
    t1, t2 = result.thresholds
    assert 0.0 < t1 < 0.5 < t2 < 1.0
    assert isinstance(t1, float) and isinstance(t2, float)


def test_thresholds_strictly_increasing_and_in_range():
    for seed in range(30):
        counts = random_counts(seed)
        result = Otsu().multi_threshold(image_from_counts(counts))
        assert result.n_classes == 3
        t1, t2 = result.thresholds
        assert 0.0 <= t1 < t2 <= 1.0
        # 严格递增
        assert t2 - t1 > 0.0


def test_deterministic():
    counts = random_counts(42)
    image = image_from_counts(counts)
    r1 = Otsu().multi_threshold(image)
    r2 = Otsu().multi_threshold(image)
    assert r1 == r2


# --------------------------------------------------------------------------
# 分解法 vs O(L^2) 暴力枚举
# --------------------------------------------------------------------------
def test_decomposition_matches_bruteforce():
    for seed in range(60):
        counts = random_counts(seed)
        image = image_from_counts(counts)
        fast = Otsu().multi_threshold(image)
        brute = Otsu().bruteforce_multi(image)
        assert fast.thresholds == brute.thresholds
        assert fast.mode == brute.mode
        assert abs(fast.between_variance - brute.between_variance) <= VARIANCE_TOL


def test_variance_matches_definition_formula():
    """独立重算：用定义式 sigma_b^2 = sum_k W_k (mu_k - mu_T)^2 验证。"""
    for seed in range(30):
        counts = random_counts(seed)
        total = sum(counts)
        result = Otsu().multi_threshold(image_from_counts(counts))
        t1, t2 = result.thresholds
        # 阈值取 bin 上边界 t=(b+1)/256，故切分 bin 序号为 int(t*256)-1。
        b1 = int(t1 * LEVELS) - 1
        b2 = int(t2 * LEVELS) - 1

        groups = [[], [], []]
        for i, c in enumerate(counts):
            groups[0 if i <= b1 else (1 if i <= b2 else 2)].extend([i] * c)
        probs = [c / total for c in counts]
        mu_t = sum(i * p for i, p in enumerate(probs))
        sigma = 0.0
        for g in groups:
            w = len(g) / total
            mu = sum(g) / len(g)
            sigma += w * (mu - mu_t) ** 2
        assert abs(sigma - result.between_variance) <= 1e-9


def test_independent_numpy_reference():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    for trial in range(20):
        # 随机 3~5 个高斯簇拼成图像
        k = rng.integers(3, 6)
        centers = rng.random(k)
        parts = []
        for c in centers:
            parts.append(np.clip(rng.normal(c, 0.03, size=400), 0.0, 1.0))
        x = np.concatenate(parts)
        image = x.reshape(40, -1).tolist()

        counts = np.bincount(np.minimum((x * LEVELS).astype(int), LEVELS - 1),
                             minlength=LEVELS).astype(float)
        p = counts / counts.sum()
        w = np.cumsum(p)
        m = np.cumsum(np.arange(LEVELS) * p)
        m_t = m[-1]
        best, best_pair = -np.inf, None
        for a in range(LEVELS - 2):
            if w[a] <= 0:
                continue
            for b in range(a + 1, LEVELS - 1):
                w1 = w[b] - w[a]
                w2 = 1.0 - w[b]
                if w1 <= 0 or w2 <= 0:
                    continue
                mu0 = m[a] / w[a]
                mu1 = (m[b] - m[a]) / w1
                mu2 = (m_t - m[b]) / w2
                val = (w[a] * (mu0 - m_t) ** 2 + w1 * (mu1 - m_t) ** 2
                       + w2 * (mu2 - m_t) ** 2)
                if val > best:
                    best, best_pair = val, (a, b)

        result = Otsu().multi_threshold(image)
        got = tuple(int(t * LEVELS) - 1 for t in result.thresholds)
        assert got == best_pair
        assert abs(result.between_variance - float(best)) <= 1e-9


# --------------------------------------------------------------------------
# 单峰 / 退化直方图
# --------------------------------------------------------------------------
def test_single_peak_returns_single_threshold_with_marker():
    # 只有两个非空 bin：无法支撑三个非空类
    image = [[0.1] * 10 + [0.9] * 10 for _ in range(4)]
    result = Otsu().multi_threshold(image)
    assert result.n_classes == 2
    assert result.degenerate is True
    assert result.mode is OtsuMode.SINGLE_PEAK
    assert result.marker is OtsuMode.SINGLE_PEAK
    assert len(result.thresholds) == 1
    (t,) = result.thresholds
    assert 0.1 < t < 0.9
    # 标记可与正常三类结果区分
    normal = Otsu().multi_threshold(
        [[0.05] * 6 + [0.5] * 6 + [0.95] * 6 for _ in range(4)]
    )
    assert normal.marker is not result.marker


def test_single_peak_equals_binary_otsu():
    image = [[0.2] * 12 + [0.8] * 12 for _ in range(3)]
    result = Otsu().multi_threshold(image)
    assert result.thresholds[0] == Otsu().threshold(image)
    brute = Otsu().bruteforce_multi(image)
    assert brute.mode is OtsuMode.SINGLE_PEAK
    assert brute.thresholds == result.thresholds


def test_exactly_three_nonempty_bins():
    counts = [0] * LEVELS
    counts[10], counts[128], counts[240] = 7, 11, 9
    result = Otsu().multi_threshold(image_from_counts(counts))
    assert result.n_classes == 3
    t1, t2 = result.thresholds
    assert t1 < t2


def test_constant_image_raises():
    with pytest.raises(ValueError):
        Otsu().multi_threshold([[0.5] * 4 for _ in range(4)])
    with pytest.raises(ValueError):
        Otsu().threshold([[0.5] * 4 for _ in range(4)])


def test_empty_image_raises():
    with pytest.raises(ValueError):
        Otsu().multi_threshold([])


def test_nonfinite_and_out_of_range_raise():
    with pytest.raises(ValueError):
        Otsu().multi_threshold([[0.1, float("nan")]])
    with pytest.raises(ValueError):
        Otsu().multi_threshold([[0.1, float("inf")]])
    with pytest.raises(ValueError):
        Otsu().multi_threshold([[0.1, 1.5]])
    with pytest.raises(ValueError):
        Otsu().multi_threshold([[0.1, -0.2]])


def test_ragged_image_raises():
    with pytest.raises(ValueError):
        Otsu().multi_threshold([[0.1, 0.2], [0.3]])


def test_non_numeric_pixels_raise_typeerror():
    with pytest.raises(TypeError):
        Otsu().multi_threshold([[0.1, "0.2"]])
    with pytest.raises(TypeError):
        Otsu().multi_threshold([[0.1, None]])
    # bool 不得被当作 0/1 静默接受
    with pytest.raises(TypeError):
        Otsu().multi_threshold([[0.1, True]])


def test_binarize_validates_threshold_and_image():
    with pytest.raises(ValueError):
        Otsu().binarize([[0.1, 0.2]], 1.5)
    with pytest.raises(ValueError):
        Otsu().binarize([[0.1, 0.2]], float("nan"))
    with pytest.raises(TypeError):
        Otsu().binarize([[0.1, 0.2]], object())
    out = Otsu().binarize([[0.1, 0.9]], 0.5)
    assert out == [[0.0, 1.0]]


# --------------------------------------------------------------------------
# 直方图归一化
# --------------------------------------------------------------------------
def test_histogram_normalized():
    for seed in range(10):
        counts = random_counts(seed, total=3000)
        image = image_from_counts(counts)
        h = Otsu().histogram(image)
        assert len(h) == LEVELS
        assert all(v >= 0.0 for v in h)
        assert abs(sum(h) - 1.0) < 1e-12


# --------------------------------------------------------------------------
# 性能：100 万像素 <= 600 ms
# --------------------------------------------------------------------------
def test_performance_megapixel():
    rng = random.Random(7)
    n = 1000
    image = [
        [rng.choice((0.08, 0.5, 0.92)) + rng.random() * 0.02 for _ in range(n)]
        for _ in range(n)
    ]
    # 预热（导入/缓存效应不计入计时）
    Otsu().multi_threshold(image)

    start = time.perf_counter()
    result = Otsu().multi_threshold(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert result.n_classes == 3
    assert elapsed_ms <= 600.0, f"too slow: {elapsed_ms:.1f} ms"


# --------------------------------------------------------------------------
# 并发安全
# --------------------------------------------------------------------------
def test_thread_safety():
    counts_a = random_counts(1)
    counts_b = random_counts(2)
    image_a = image_from_counts(counts_a)
    image_b = image_from_counts(counts_b)
    shared = Otsu()
    expected_a = shared.multi_threshold(image_a)
    expected_b = shared.multi_threshold(image_b)

    errors = []

    def worker(idx):
        try:
            for _ in range(25):
                image = image_a if idx % 2 == 0 else image_b
                expected = expected_a if idx % 2 == 0 else expected_b
                got = shared.multi_threshold(image)
                assert got == expected
                # 基础 API 同样并发可用
                h = shared.histogram(image)
                assert abs(sum(h) - 1.0) < 1e-12
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(16)))
    assert not errors


def test_no_shared_mutable_state():
    """实例跨调用复用时结果互不影响（与独立实例一致）。"""
    a = Otsu()
    standalone = Otsu()
    for seed in range(10):
        image = image_from_counts(random_counts(seed))
        assert a.multi_threshold(image) == standalone.multi_threshold(image)
