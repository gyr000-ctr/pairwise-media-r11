"""多阈值 Otsu（三类分割）核心 API。

实现要点
========
- 256 个灰度 bin，概率直方图 p_i 满足 sum(p) == 1。
- 三类分割的两个阈值通过**类间方差分解**求得：对累积权重 ``W(t)`` 与
  累积均值 ``M(t)``，有

      sigma_b^2(t1, t2) = M(t1)^2 / W(t1)
                        + (M(t2) - M(t1))^2 / (W(t2) - W(t1))
                        + (M_T - M(t2))^2 / (1 - W(t2))
                        - M_T^2

  前缀量一次 O(L) 预处理，之后每个 (t1, t2) 组合 O(1) 评估，整体 O(L^2)。
- 同时提供按定义式（各类均值相对总均值的加权离差平方和）枚举全部
  (t1, t2) 组合的 O(L^2) 暴力参考实现 :meth:`Otsu.bruteforce_multi`，
  两条路径的最优阈值与方差必须一致（容差 1e-9）。
- 单峰/退化直方图（非空 bin 不足以支撑三个非空类）回退为单阈值
  二分类，并用 :class:`OtsuMode` 标记区分；常量图无任何可分阈值，抛
  ``ValueError``。
- 所有计算状态均为函数内局部变量，实例不持有可变缓存，天然线程安全。

仅使用 Python 标准库。
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = [
    "Otsu",
    "OtsuMode",
    "MultiOtsuResult",
    "LEVELS",
    "VARIANCE_TOL",
]

#: 灰度级数（8 bit）。
LEVELS = 256

#: 分解法与暴力枚举结果允许的最大方差差。
VARIANCE_TOL = 1e-9


class OtsuMode(Enum):
    """分割模式标记（退化场景的可区分标记）。"""

    THREE_CLASS = "three_class"
    SINGLE_PEAK = "single_peak"

    def __str__(self) -> str:  # pragma: no cover - 调试便利
        return self.value


class MultiOtsuResult:
    """多阈值分割结果。

    Attributes:
        thresholds: 严格递增、落在 [0, 1] 的阈值元组；正常三类分割为
            ``(t1, t2)``，单峰退化时为单元素元组 ``(t,)``。
        mode: :class:`OtsuMode` 标记，退化场景为 ``SINGLE_PEAK``。
        degenerate: 是否为单峰退化（单阈值）结果。
        n_classes: 实际分出的类别数（3 或 2）。
        between_variance: 对应分割的类间方差。
    """

    __slots__ = ("thresholds", "mode", "degenerate", "n_classes", "between_variance")

    def __init__(self, thresholds, mode, n_classes, between_variance):
        self.thresholds = tuple(thresholds)
        self.mode = mode
        self.degenerate = mode is OtsuMode.SINGLE_PEAK
        self.n_classes = n_classes
        self.between_variance = float(between_variance)

    @property
    def marker(self):
        """可区分标记（同 :attr:`mode`）。"""
        return self.mode

    def as_dict(self):
        return {
            "thresholds": self.thresholds,
            "mode": self.mode,
            "degenerate": self.degenerate,
            "n_classes": self.n_classes,
            "between_variance": self.between_variance,
        }

    def __iter__(self):
        return iter(self.thresholds)

    def __len__(self):
        return len(self.thresholds)

    def __eq__(self, other):
        if not isinstance(other, MultiOtsuResult):
            return NotImplemented
        return (
            self.thresholds == other.thresholds
            and self.mode == other.mode
            and self.n_classes == other.n_classes
            and math.isclose(
                self.between_variance,
                other.between_variance,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

    def __hash__(self):
        return hash((self.thresholds, self.mode, self.n_classes))

    def __repr__(self):
        return (
            "MultiOtsuResult(thresholds={!r}, mode={}, n_classes={}, "
            "between_variance={:.12g})".format(
                self.thresholds, self.mode.value, self.n_classes,
                self.between_variance,
            )
        )


def _is_real_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _count_bins(image):
    """校验输入并统计 256 bin 计数，返回 (counts, pixel_count)。

    热路径上的类型/有限性检查全部内联并使用局部别名，避免百万像素下
    每像素一次的函数调用开销；校验语义与 :func:`_is_real_number` 一致。
    """
    if not isinstance(image, (list, tuple)) or len(image) == 0:
        raise ValueError("image must be a non-empty 2D sequence")

    counts = [0] * LEVELS
    width = None
    total = 0
    real_types = (int, float)
    isnan = math.isnan
    isinf = math.isinf

    for row in image:
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError("image rows must be non-empty sequences")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("ragged image: rows have different lengths")

        for v in row:
            # bool 是 int 子类，必须显式排除。
            if isinstance(v, bool) or not isinstance(v, real_types):
                raise TypeError("pixel values must be real numbers")
            fv = float(v)
            if isnan(fv) or isinf(fv):
                raise ValueError("pixel values must be finite")
            if fv < 0.0 or fv > 1.0:
                raise ValueError("pixel values must lie in [0, 1]")
            b = int(fv * LEVELS)
            if b >= LEVELS:  # fv == 1.0 -> 256，归入最后一个 bin
                b = LEVELS - 1
            counts[b] += 1
            total += 1

    return counts, total


def _prefix(counts, total):
    """归一化概率及累积权重 W、累积灰度和 M（以 bin 序号为灰度级）。"""
    inv = 1.0 / total
    probs = [c * inv for c in counts]

    weights = [0.0] * LEVELS
    moments = [0.0] * LEVELS
    w_acc = 0.0
    m_acc = 0.0
    for i in range(LEVELS):
        pi = probs[i]
        w_acc += pi
        m_acc += i * pi
        weights[i] = w_acc
        moments[i] = m_acc
    return probs, weights, moments


def _bin_to_threshold(b):
    """bin 序号 -> 归一化阈值，取该 bin 的上边界，保证落在 (0, 1)。"""
    return (b + 1) / LEVELS


def _single_from_prefix(weights, moments):
    """二分类 Otsu，返回 (最优 bin 序号, 类间方差)；常量图抛错。"""
    m_total = moments[-1]
    best = -math.inf
    best_t = -1
    for t in range(LEVELS - 1):
        w0 = weights[t]
        if w0 <= 0.0 or w0 >= 1.0:
            continue
        m0 = moments[t]
        # 分解式：sigma_b^2 = M0^2/W0 + (MT-M0)^2/(1-W0) - MT^2
        val = m0 * m0 / w0 + (m_total - m0) ** 2 / (1.0 - w0) - m_total * m_total
        if val > best:  # 严格大于：平局取最小序号，保证确定性
            best = val
            best_t = t
    if best_t < 0:
        raise ValueError("constant image: no separable threshold")
    return best_t, best


def _multi_from_prefix(weights, moments):
    """类间方差分解求两个阈值，O(L^2)，返回 ((t1, t2), sigma_b^2)。"""
    m_total = moments[-1]
    mt2 = m_total * m_total
    best = -math.inf
    best_pair = None

    for t1 in range(LEVELS - 2):
        w0 = weights[t1]
        if w0 <= 0.0:
            continue
        m0 = moments[t1]
        term0 = m0 * m0 / w0  # 第一类的分解项，内层循环复用

        for t2 in range(t1 + 1, LEVELS - 1):
            w1 = weights[t2] - w0
            if w1 <= 0.0:
                continue
            w2 = 1.0 - weights[t2]
            if w2 <= 0.0:
                continue
            m1 = moments[t2] - m0
            m2 = m_total - moments[t2]
            val = term0 + m1 * m1 / w1 + m2 * m2 / w2 - mt2
            if val > best:
                best = val
                best_pair = (t1, t2)

    if best_pair is None:
        raise ValueError("fewer than 3 non-empty histogram bins")
    return best_pair, best


def _multi_bruteforce(probs):
    """暴力参考实现：枚举全部 (t1, t2)，按类间方差**定义式**直接计算。

    与分解法同为 O(L^2) 次组合枚举（前缀量使每对评估 O(1)），但使用
    代数上不同的公式 sigma_b^2 = sum_k W_k (mu_k - mu_T)^2 交叉验证。
    返回 ((t1, t2), sigma_b^2)。
    """
    weights = [0.0] * LEVELS
    moments = [0.0] * LEVELS
    w_acc = 0.0
    m_acc = 0.0
    for i in range(LEVELS):
        w_acc += probs[i]
        m_acc += i * probs[i]
        weights[i] = w_acc
        moments[i] = m_acc
    m_total = m_acc

    best = -math.inf
    best_pair = None
    for t1 in range(LEVELS - 2):
        w0 = weights[t1]
        if w0 <= 0.0:
            continue
        for t2 in range(t1 + 1, LEVELS - 1):
            w1 = weights[t2] - weights[t1]
            if w1 <= 0.0:
                continue
            w2 = 1.0 - weights[t2]
            if w2 <= 0.0:
                continue
            mu0 = moments[t1] / w0
            mu1 = (moments[t2] - moments[t1]) / w1
            mu2 = (m_total - moments[t2]) / w2
            val = (
                w0 * (mu0 - m_total) ** 2
                + w1 * (mu1 - m_total) ** 2
                + w2 * (mu2 - m_total) ** 2
            )
            if val > best:
                best = val
                best_pair = (t1, t2)

    if best_pair is None:
        raise ValueError("fewer than 3 non-empty histogram bins")
    return best_pair, best


class Otsu:
    """Otsu 阈值分割（无实例可变状态，可并发调用）。"""

    # ------------------------------------------------------------------
    # 基础 API
    # ------------------------------------------------------------------
    def histogram(self, image):
        """返回长度 256 的归一化概率直方图，元素非负、和为 1。"""
        counts, total = _count_bins(image)
        probs, _, _ = _prefix(counts, total)
        return probs

    def threshold(self, image):
        """二分类 Otsu 最优阈值，返回 [0, 1] 内的 float；常量图/空图报错。"""
        counts, total = _count_bins(image)
        _, weights, moments = _prefix(counts, total)
        t, _ = _single_from_prefix(weights, moments)
        return float(_bin_to_threshold(t))

    def binarize(self, image, t):
        """按阈值二值化，输出与输入同形的嵌套 list，仅含 0.0 / 1.0。"""
        if not _is_real_number(t):
            raise TypeError("threshold must be a real number")
        t = float(t)
        if math.isnan(t) or math.isinf(t) or t < 0.0 or t > 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        # 复用校验逻辑，保证空图/异形图/非法像素同样报错。
        _count_bins(image)
        return [[1.0 if float(v) > t else 0.0 for v in row] for row in image]

    # ------------------------------------------------------------------
    # 多阈值（三类分割）API
    # ------------------------------------------------------------------
    def multi_threshold(self, image):
        """求三类分割的两个阈值。

        Returns:
            MultiOtsuResult: 正常情况含两个严格递增阈值；单峰退化直方图
            （非空 bin 不足 3 个）回退为单阈值并以 ``SINGLE_PEAK`` 标记；
            常量图抛 ``ValueError``。
        """
        counts, total = _count_bins(image)
        probs, weights, moments = _prefix(counts, total)
        support = sum(1 for c in counts if c > 0)

        if support < 2:
            raise ValueError("constant image: no separable threshold")

        if support < 3:
            # 退化：无法构造三个非空类，回退到二分类单阈值。
            t, variance = _single_from_prefix(weights, moments)
            return MultiOtsuResult(
                (_bin_to_threshold(t),),
                OtsuMode.SINGLE_PEAK,
                2,
                variance,
            )

        (t1, t2), variance = _multi_from_prefix(weights, moments)

        # 与 O(L^2) 暴力枚举（定义式）交叉验证，阈值与方差必须一致。
        (b1, b2), brute_variance = _multi_bruteforce(probs)
        if (b1, b2) != (t1, t2):
            raise AssertionError(
                "decomposition/ brute-force threshold mismatch: "
                f"{(t1, t2)} != {(b1, b2)}"
            )
        if abs(brute_variance - variance) > VARIANCE_TOL:
            raise AssertionError(
                "decomposition / brute-force variance mismatch: "
                f"{variance} vs {brute_variance}"
            )

        thresholds = (_bin_to_threshold(t1), _bin_to_threshold(t2))
        return MultiOtsuResult(
            thresholds,
            OtsuMode.THREE_CLASS,
            3,
            variance,
        )

    # 常见别名，便于不同调用方使用。
    threshold_multi = multi_threshold

    def bruteforce_multi(self, image):
        """暴力枚举版三类分割，返回结构与 :meth:`multi_threshold` 一致。"""
        counts, total = _count_bins(image)
        probs, weights, moments = _prefix(counts, total)
        support = sum(1 for c in counts if c > 0)

        if support < 2:
            raise ValueError("constant image: no separable threshold")
        if support < 3:
            t, variance = _single_from_prefix(weights, moments)
            return MultiOtsuResult(
                (_bin_to_threshold(t),),
                OtsuMode.SINGLE_PEAK,
                2,
                variance,
            )

        (t1, t2), variance = _multi_bruteforce(probs)
        return MultiOtsuResult(
            (_bin_to_threshold(t1), _bin_to_threshold(t2)),
            OtsuMode.THREE_CLASS,
            3,
            variance,
        )
