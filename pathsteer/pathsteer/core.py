"""pathsteer 核心 API：群体转向（分离 / 对齐 / 聚合）。

约束：
- 三力权重和为 1；
- 速度上限钳制（|v| <= vmax），单步位移 <= vmax * dt；
- 相邻过近时分离力随距离减小而单调增大（量级 ~ r/d，且分离权重 >
  聚合权重、聚合力归一化，保证 d <= r 时合力在分离方向上严格单调）；
- 无邻居时对齐力与聚合力为 0；
- 非法参数可区分报错：类型错误抛 TypeError，非有限数值与越界值抛 ValueError；
- 1 万物件单步耗时 < 500 ms（均匀网格邻域搜索，O(n)）；
- 并发安全：step 内部状态计算在锁内完成，且不修改入参。
"""

import math
import threading

__all__ = ["Steering"]

# 三力权重（分离 / 对齐 / 聚合），均为 2 的幂次组合，和恰为 1.0。
# 分离权重 > 聚合权重：配合归一化的聚合力，保证分离在近距离始终占优。
_W_SEPARATION = 0.5
_W_ALIGNMENT = 0.25
_W_COHESION = 0.25

# 重合个体分离方向使用的黄金角（确定性伪随机方向）
_GOLDEN_ANGLE = 2.39996322972865332


def _check_number(value, name):
    """校验为实数（非 bool）；类型错 -> TypeError，非有限 -> ValueError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "%s 必须是数值类型，得到 %r" % (name, type(value).__name__)
        )
    if not math.isfinite(value):
        raise ValueError("%s 必须是有限数值，得到 %r" % (name, value))


class Steering:
    """群体转向模拟器：分离 / 对齐 / 聚合三力加权。"""

    def __init__(self, radius, vmax):
        _check_number(radius, "radius")
        _check_number(vmax, "vmax")
        if radius <= 0.0:
            raise ValueError("radius 必须为正数，得到 %r" % (radius,))
        if vmax <= 0.0:
            raise ValueError("vmax 必须为正数，得到 %r" % (vmax,))
        self._radius = float(radius)
        self._vmax = float(vmax)
        self._lock = threading.Lock()

    def weights(self):
        """返回 (分离, 对齐, 聚合) 三力权重，和为 1。"""
        return (_W_SEPARATION, _W_ALIGNMENT, _W_COHESION)

    def step(self, agents, dt):
        """推进一个时间步，返回新的 (x, y, vx, vy) 列表（不修改入参）。"""
        _check_number(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt 必须为正数，得到 %r" % (dt,))
        if not hasattr(agents, "__iter__"):
            raise TypeError(
                "agents 必须是可迭代对象，得到 %r" % type(agents).__name__
            )
        agents = list(agents)
        n = len(agents)
        if n == 0:
            return []

        px = [0.0] * n
        py = [0.0] * n
        vx = [0.0] * n
        vy = [0.0] * n
        for i, a in enumerate(agents):
            try:
                x, y, ux, uy = a
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "agents[%d] 必须是 (x, y, vx, vy) 四元组，得到 %r" % (i, a)
                ) from exc
            for label, v in (("x", x), ("y", y), ("vx", ux), ("vy", uy)):
                _check_number(v, "agents[%d].%s" % (i, label))
            px[i] = float(x)
            py[i] = float(y)
            vx[i] = float(ux)
            vy[i] = float(uy)

        with self._lock:
            return self._advance(px, py, vx, vy, float(dt))

    def _advance(self, px, py, vx, vy, dt):
        n = len(px)
        r = self._radius
        r2 = r * r
        vmax = self._vmax
        inv_r = 1.0 / r
        d_min = 1e-6 * r  # 距离下限，避免重合个体除零

        # 均匀网格划分，邻域搜索只看相邻 9 格
        grid = {}
        for i in range(n):
            key = (math.floor(px[i] * inv_r), math.floor(py[i] * inv_r))
            bucket = grid.get(key)
            if bucket is None:
                grid[key] = [i]
            else:
                bucket.append(i)

        out = []
        for i in range(n):
            xi = px[i]
            yi = py[i]
            vxi = vx[i]
            vyi = vy[i]
            cx = math.floor(xi * inv_r)
            cy = math.floor(yi * inv_r)

            sep_x = 0.0
            sep_y = 0.0
            ali_x = 0.0
            ali_y = 0.0
            coh_x = 0.0
            coh_y = 0.0
            count = 0

            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    bucket = grid.get((gx, gy))
                    if bucket is None:
                        continue
                    for j in bucket:
                        if j == i:
                            continue
                        dx = xi - px[j]
                        dy = yi - py[j]
                        d2 = dx * dx + dy * dy
                        if d2 > r2:
                            continue
                        count += 1
                        if d2 > 0.0:
                            # 分离力 = 单位方向 * (r/d)：距离越近单调越大
                            d = math.sqrt(d2)
                            mag = r / (d if d > d_min else d_min)
                            sep_x += dx / d * mag
                            sep_y += dy / d * mag
                        else:
                            # 完全重合：按黄金角给确定性的强排斥方向（量级 r/d_min）
                            ang = _GOLDEN_ANGLE * (i + 1)
                            mag = r / d_min
                            sep_x += math.cos(ang) * mag
                            sep_y += math.sin(ang) * mag
                        ali_x += vx[j]
                        ali_y += vy[j]
                        coh_x += px[j]
                        coh_y += py[j]

            if count:
                inv_c = 1.0 / count
                ali_x = ali_x * inv_c - vxi
                ali_y = ali_y * inv_c - vyi
                # 聚合力归一化为单位方向（量级 1，不随距离增长），
                # 保证分离力（量级 r/d >= 1 且权重更大）在近距离严格占优
                coh_x = coh_x * inv_c - xi
                coh_y = coh_y * inv_c - yi
                coh_len = math.hypot(coh_x, coh_y)
                if coh_len > 0.0:
                    coh_x /= coh_len
                    coh_y /= coh_len
            # 无邻居时对齐与聚合为 0（分离也为 0）

            fx = _W_SEPARATION * sep_x + _W_ALIGNMENT * ali_x + _W_COHESION * coh_x
            fy = _W_SEPARATION * sep_y + _W_ALIGNMENT * ali_y + _W_COHESION * coh_y

            nvx = vxi + fx * dt
            nvy = vyi + fy * dt

            # 速度上限钳制
            speed = math.hypot(nvx, nvy)
            if speed > vmax:
                k = vmax / speed
                nvx *= k
                nvy *= k

            # 位移 = v * dt，因 |v| <= vmax，故单步位移 <= vmax * dt
            out.append((xi + nvx * dt, yi + nvy * dt, nvx, nvy))
        return out
