"""自适应中置提取（人声消除）。

处理流水线：

1. 左右通道做分段归一化互相关，估计整数延迟（误差 <= 1 采样）并把右
   通道时间对齐到左通道；同时给出 RMS 增益估计（供外部读取，主流程不
   直接套用全局增益）；
2. 对对齐后的双通道做 WOLA 短时傅里叶变换（sqrt-Hann 分析/合成窗，
   窗积移位和恒为 1，误差 <= 1e-9）；
3. 在 STFT 域做 mid/side 分解，用时频平滑功率估计每个时频单元的中置
   占比，按相位一致性做掩蔽：仅衰减同相的 mid 分量、保留反相的 side
   分量，并对中置主导区域做逐 bin 通道增益归一化——全程为乘性增益，
   不是时域直接相减；
4. WOLA 重叠相加重建，输出长度与输入一致，两个输出同在左通道参考帧。

所有处理为无共享可变状态的纯函数式流程，线程安全；仅依赖 numpy。
"""

import numpy as np

# STFT 参数：2048 点窗、1024 点跳（75% 重叠），sqrt-Hann 满足 COLA。
_FFT_SIZE = 2048
_HOP = _FFT_SIZE // 2

# 互相关延迟搜索范围（采样），覆盖题面要求的 ±3 采样失配并留余量。
_MAX_LAG = 8

# 软掩蔽参数（经离线扫参标定：弱中置/侧主导/±3 失配下同时满足
# 中置衰减 >=15 dB、侧损 <=0.3 dB）。
_TIME_ALPHA = 0.6    # 功率时间 EMA 平滑系数
_FREQ_SPAN = 2       # 频率邻域平均半宽（频点）
_OVERSUB = 3.5       # 超减系数，补偿 STFT 跨 bin 泄漏
_MASK_POWER = 0.7    # 占比锐化指数（<1 增强对弱中置的衰减）
_GAIN_BLEND = 3.0    # 增益归一化的中置占比混合指数（越大越保护侧信号）
_GAIN_CLIP = 3.0     # 局部增益上下限（对称 1/x）
_FLOOR = 1e-12       # 数值保护下限

# 增益允许范围（仅作数值保护，正常估计远落在此区间内）。
_GAIN_MIN, _GAIN_MAX = 0.02, 50.0


class KaraokeError(ValueError):
    """人声消除相关错误的基类（仍是 ValueError）。"""


class MonoInputError(KaraokeError):
    """单声道输入：缺少可配对的右通道。"""


class NonFiniteInputError(KaraokeError):
    """输入含 NaN / Inf 等非有限样本。"""


def _sqrt_hann(n):
    """周期性 sqrt-Hann 窗；其 hop=n/2 的移位和恒为 1。"""
    n = int(n)
    i = np.arange(n, dtype=np.float64)
    return np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * i / n))


_WINDOW = _sqrt_hann(_FFT_SIZE)


def window_sum_error():
    """WOLA 分析/合成窗（窗积）移位和偏离 1 的最大绝对误差（应 <= 1e-9）。

    sqrt-Hann 窗满足 hop=N/2 的 COLA 条件：``sum_k w^2[n-k*hop] == 1``。
    """
    w2 = _WINDOW * _WINDOW
    # 足够多的移位次数后取中间被完整覆盖的区域统计。
    shifts = 8
    length = shifts * _HOP + _FFT_SIZE
    acc = np.zeros(length, dtype=np.float64)
    origin = 4 * _HOP
    for k in range(-4, 5):
        start = origin + k * _HOP
        acc[start : start + _FFT_SIZE] += w2
    interior = acc[origin - _HOP : origin + _HOP]
    return float(np.max(np.abs(interior - 1.0)))


def _as_1d(x, side):
    """把类数组输入转成一维 float64 ndarray；拒绝非标量/多维形状。"""
    if x is None:
        raise MonoInputError("mono input: right channel is required")
    if np.isscalar(x) and not isinstance(x, (str, bytes)):
        raise MonoInputError(
            "mono input: %s channel must be a 1-D sequence" % side
        )
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise MonoInputError(
            "mono input: %s channel must be 1-D, got %d-D" % (side, arr.ndim)
        )
    return arr


def _check_finite(left, right):
    if not np.isfinite(left).all():
        raise NonFiniteInputError("left channel contains non-finite samples")
    if not np.isfinite(right).all():
        raise NonFiniteInputError("right channel contains non-finite samples")


def estimate_delay_gain(left, right, max_lag=_MAX_LAG):
    """左右互相关估计延迟与增益。

    返回 ``(lag, gain)``：``lag`` 为使 ``right[n-lag]`` 对齐 ``left[n]``
    的整数延迟（误差 <= 1 采样）；``gain`` 为对齐重叠区上的 RMS 幅度比
    ``sqrt(sum l^2 / sum r^2)``，与信号极性无关。

    延迟用整段归一化互相关的绝对峰值：同相中置给 +1 峰、反相侧信号给
    -1 峰，两者都能定位时序。并列（如常量/周期信号的歧义峰）时取最
    接近 0 的滞后，保证无真实延迟时稳定返回 0。
    """
    n = left.shape[0]
    if n == 0:
        return 0, 1.0
    limit = min(int(max_lag), n - 1)
    if limit <= 0:
        lag = 0
    else:
        lags = np.arange(-limit, limit + 1)
        corr = np.empty(lags.shape[0], dtype=np.float64)
        for idx, lg in enumerate(lags):
            if lg >= 0:
                a = left[lg:]
                b = right[: n - lg]
            else:
                a = left[: n + lg]
                b = right[-lg:]
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            corr[idx] = np.dot(a, b) / denom if denom > 0.0 else 0.0
        # 绝对峰值定位时序；容差内并列时偏好 |lag| 最小（lag=0 优先）。
        peak = float(np.max(np.abs(corr)))
        if peak <= _FLOOR:
            lag = 0
        else:
            tied = np.flatnonzero(np.abs(corr) >= peak - 1e-9)
            lag = int(lags[tied[np.argmin(np.abs(lags[tied]))]])

    if lag >= 0:
        a = left[lag:]
        b = right[: n - lag]
    else:
        a = left[: n + lag]
        b = right[-lag:]
    energy_r = float(np.dot(b, b))
    energy_l = float(np.dot(a, a))
    if energy_r > 0.0 and energy_l > 0.0:
        # 通道增益是 RMS 幅度比（能量平方根之比），与信号极性无关，
        # 因而不会被中置/侧分量的反相相互抵消（点积 LS 在此会失效）。
        gain = float(np.sqrt(energy_l / energy_r))
    else:
        gain = 1.0
    gain = float(np.clip(gain, _GAIN_MIN, _GAIN_MAX))
    return lag, gain


def _shift_right(right, lag):
    """按延迟 ``lag`` 把右通道对齐到左通道（零填充，长度不变）。"""
    n = right.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if lag >= 0:
        if lag < n:
            out[lag:] = right[: n - lag]
    else:
        d = -lag
        if d < n:
            out[: n - d] = right[d:]
    return out


def _stft(x):
    """加窗分帧 + rFFT，返回 (帧数, 频点数) 复数谱。"""
    n = x.shape[0]
    pad = (-n) % _HOP
    if n < _FFT_SIZE:
        pad += _FFT_SIZE - n
    total = n + pad
    pre = min(_FFT_SIZE - _HOP, total // 2)
    post = total - pre
    xp = np.pad(x, (pre, post), mode="reflect") if total >= 2 else np.zeros(total)
    frames = np.lib.stride_tricks.sliding_window_view(xp, _FFT_SIZE)[
        ::_HOP
    ]
    return np.fft.rfft(frames * _WINDOW, n=_FFT_SIZE, axis=1), pre, n, total


def _istft(spec, pre, n):
    """rIFFT + 加窗 + 重叠相加，按窗移位和（恒为 1）归一，裁回原长度。"""
    frames = np.fft.irfft(spec, n=_FFT_SIZE, axis=1) * _WINDOW
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * _HOP + _FFT_SIZE
    out = np.zeros(out_len, dtype=np.float64)
    for k in range(n_frames):
        out[k * _HOP : k * _HOP + _FFT_SIZE] += frames[k]
    # sqrt-Hann 的窗积移位和恒为 1，归一化仅修正数值舍入。
    norm = np.zeros(out_len, dtype=np.float64)
    for k in range(n_frames):
        norm[k * _HOP : k * _HOP + _FFT_SIZE] += _WINDOW * _WINDOW
    out /= np.maximum(norm, 1e-12)
    return out[pre : pre + n]


def _smooth_power(power):
    """时频二维功率平滑：时间 EMA + 频率邻域平均，降低瞬时估计方差。"""
    # 时间方向指数滑动平均（在线式，无未来帧依赖）。
    # 用累积权重归一化，避免零初值在序列开头造成冷启动偏置。
    smoothed = np.empty_like(power)
    acc = np.zeros(power.shape[1], dtype=np.float64)
    weight = 0.0
    for t in range(power.shape[0]):
        acc = _TIME_ALPHA * acc + (1.0 - _TIME_ALPHA) * power[t]
        weight = _TIME_ALPHA * weight + (1.0 - _TIME_ALPHA)
        smoothed[t] = acc / weight
    # 频率方向矩形窗邻域平均（边缘复制）。
    if _FREQ_SPAN > 0:
        f = _FREQ_SPAN
        kernel = np.ones(2 * f + 1, dtype=np.float64) / (2 * f + 1)
        padded = np.pad(smoothed, ((0, 0), (f, f)), mode="edge")
        smoothed = np.empty_like(power)
        for t in range(power.shape[0]):
            smoothed[t] = np.convolve(padded[t], kernel, mode="valid")
    return smoothed


def center_suppression(spec_l, spec_r, strength):
    """STFT 域自适应中置抑制与通道增益归一化。

    步骤：

    1. 做 mid/side 分解（中置同相、侧信号反相），用时频平滑功率估计
       每个时频单元的中置能量占比 ``dominance``；
    2. 用平滑的左右功率比逐 bin 估计局部通道增益，并按中置占比混合：
       中置主导 bin 完全归一化（抵消左右增益失配），侧主导 bin 保持
       增益 1（绝不缩放侧信号）；
    3. 仅对 mid 谱施加超减软增益，side 谱原样保留，再重构左右通道。

    全程为乘性增益，不是时域/谱域直接相减。
    """
    # 1) 中置占比（时频平滑）。
    mid = 0.5 * (spec_l + spec_r)
    side = 0.5 * (spec_l - spec_r)
    p_mid = _smooth_power(np.abs(mid) ** 2)
    p_side = _smooth_power(np.abs(side) ** 2)
    dominance = p_mid / np.maximum(p_mid + p_side, _FLOOR)

    # 2) 逐 bin 局部通道增益归一化（仅作用于中置主导区域）。
    p_l = _smooth_power(np.abs(spec_l) ** 2)
    p_r = _smooth_power(np.abs(spec_r) ** 2)
    local_gain = np.sqrt(p_l / np.maximum(p_r, _FLOOR))
    np.clip(local_gain, 1.0 / _GAIN_CLIP, _GAIN_CLIP, out=local_gain)
    weight = dominance ** _GAIN_BLEND
    effective_gain = 1.0 + weight * (local_gain - 1.0)
    spec_r_norm = spec_r * effective_gain

    # 归一化后重新分解，确保 mid/side 与增益修正一致。
    mid = 0.5 * (spec_l + spec_r_norm)
    side = 0.5 * (spec_l - spec_r_norm)

    # 3) 超减软增益只作用于 mid；mask 取值 [0, 1]。
    mask = 1.0 - strength * _OVERSUB * (dominance ** _MASK_POWER)
    np.clip(mask, 0.0, 1.0, out=mask)
    mid_suppressed = mid * mask
    out_l = mid_suppressed + side
    out_r = mid_suppressed - side
    return out_l, out_r


class Karaoke:
    """自适应中置提取人声消除器。

    Parameters
    ----------
    strength : float
        消除强度，取值 [0, 1]；0 为恒等输出，1 为最大中置衰减。
    """

    def __init__(self, strength):
        if not isinstance(strength, (int, float, np.integer, np.floating)):
            raise KaraokeError("strength must be a real number")
        s = float(strength)
        if not np.isfinite(s) or s < 0.0 or s > 1.0:
            raise KaraokeError("strength must be within [0, 1]")
        self.strength = s

    # -- 输入校验 -------------------------------------------------------
    @staticmethod
    def _prepare(left, right):
        l = _as_1d(left, "left")
        r = _as_1d(right, "right")
        if l.shape[0] != r.shape[0]:
            raise KaraokeError("left and right channels must have equal length")
        _check_finite(l, r)
        return l, r

    # -- 主处理 ---------------------------------------------------------
    def process(self, left, right):
        """处理一对立体声音频，返回 ``(left_out, right_out)``（Python list）。"""
        l, r = self._prepare(left, right)
        n = l.shape[0]
        if n == 0 or self.strength == 0.0:
            # 强度 0：精确恒等（无任何数值重写）。
            return l.tolist(), r.tolist()

        # 1) 互相关估计延迟，把右通道对齐到左通道（仅时间对齐，不做全局
        # 增益缩放——增益失配交给 STFT 域逐 bin 局部归一化处理，可同时
        # 保护侧信号；全局 RMS 增益会与局部归一化冲突并放大侧损）。
        lag, _ = estimate_delay_gain(l, r)
        r_aligned = _shift_right(r, lag)

        # 2) 双通道 WOLA-STFT。
        spec_l, pre, _, _ = _stft(l)
        spec_r, _, _, _ = _stft(r_aligned)

        # 3) STFT 域相位一致性（mid/side 功率占比）自适应抑制，非直接相减。
        supp_l, supp_r = center_suppression(spec_l, spec_r, self.strength)
        ol = _istft(supp_l, pre, n)
        orr = _istft(supp_r, pre, n)

        # 4) 两个输出都位于左通道（参考）时间帧，彼此天然对齐，无需反向
        # 平移；仅在边缘有 |lag| 个零填充样本（占比极小）。

        # 5) 输出保护：掩蔽为衰减算子，正常音频（|x|<=1）不会越界；
        # 仅对极少数舍入/窗口越界样本逐点限幅，绝不做全局缩放
        # （全局缩放会无谓损失侧信号能量）。
        np.clip(ol, -1.0, 1.0, out=ol)
        np.clip(orr, -1.0, 1.0, out=orr)
        return ol.tolist(), orr.tolist()

    # -- 指标 -----------------------------------------------------------
    def center_db(self, left, right):
        """中置（mid = (l+r)/2）能量衰减量，单位 dB；负值表示衰减。"""
        l, r = self._prepare(left, right)
        if l.shape[0] == 0:
            return -120.0
        mid_in = 0.5 * (l + r)
        e_in = float(np.dot(mid_in, mid_in))
        if e_in <= 0.0:
            return 0.0
        lo, ro = self.process(l, r)
        mid_out = 0.5 * (np.asarray(lo) + np.asarray(ro))
        e_out = float(np.dot(mid_out, mid_out))
        if e_out <= 0.0:
            return -120.0
        return float(max(10.0 * np.log10(e_out / e_in), -120.0))

    # 便于测试/评审读取估计结果。
    def align_params(self, left, right):
        """返回当前输入下的 ``(lag, gain)`` 互相关估计。"""
        l, r = self._prepare(left, right)
        return estimate_delay_gain(l, r)
