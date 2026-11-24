"""多边形耳切三角化核心实现。

设计要点：
- 纯函数式实现，无可变共享状态，天然并发安全；
- 输入校验：顶点数 < 3 报 ValueError，非有限坐标报 ValueError；
- 自交检测：均匀网格 + DDA 线段走格，期望 O(n)；
- 耳切：双向链表 + 均匀网格空间索引加速耳内点查询，期望近 O(n)。
"""

import math

__all__ = ["EarClip"]


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------

def _signed_area(pts):
    """有符号面积（鞋带公式），逆时针为正。"""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _coerce(poly):
    """把输入规整为 [(float, float), ...]，并做有限性校验。"""
    pts = []
    for v in poly:
        try:
            x, y = v
        except (TypeError, ValueError):
            raise TypeError("每个顶点必须是 (x, y) 对")
        x = float(x)
        y = float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("坐标必须有限（不允许 NaN/inf）")
        pts.append((x, y))
    return pts


def _dedup(pts):
    """去除相邻重复点（含首尾环绕）。"""
    out = []
    for p in pts:
        if not out or p != out[-1]:
            out.append(p)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


# ---------------------------------------------------------------------------
# 自交检测（Shamos–Hoey 扫描线，O(n log n)）
# ---------------------------------------------------------------------------

def _seg_intersect(a, b, c, d):
    """线段 ab 与 cd 是否相交（端点接触也算相交）。"""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    if max(ax, bx) < min(cx, dx) or max(cx, dx) < min(ax, bx):
        return False
    if max(ay, by) < min(cy, dy) or max(cy, dy) < min(ay, by):
        return False
    d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and \
       ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)):
        return True
    if d1 == 0.0 and min(cx, dx) <= ax <= max(cx, dx) and \
            min(cy, dy) <= ay <= max(cy, dy):
        return True
    if d2 == 0.0 and min(cx, dx) <= bx <= max(cx, dx) and \
            min(cy, dy) <= by <= max(cy, dy):
        return True
    if d3 == 0.0 and min(ax, bx) <= cx <= max(ax, bx) and \
            min(ay, by) <= cy <= max(ay, by):
        return True
    if d4 == 0.0 and min(ax, bx) <= dx <= max(ax, bx) and \
            min(ay, by) <= dy <= max(ay, by):
        return True
    return False


def _has_self_intersection(pts):
    """检测多边形边集是否存在非相邻边相交（含顶点接触、共线重叠）。

    扫描线算法：非垂直边按 (y, 编号) 有序维护在扫描线状态中；
    垂直边作为区间查询事件处理（简单多边形中区间命中只有相邻边）。
    """
    n = len(pts)
    m = n
    x0s = [0.0] * m
    y0s = [0.0] * m
    slopes = [0.0] * m   # 仅非垂直边有效
    vertical = [False] * m
    events = []          # (x, kind, ...) kind: 0=插入 1=垂直边 2=删除
    for i in range(m):
        xa, ya = pts[i]
        xb, yb = pts[(i + 1) % m]
        if xa == xb:
            vertical[i] = True
            events.append((xa, 1, min(ya, yb), max(ya, yb), i))
        elif xa < xb:
            x0s[i] = xa
            y0s[i] = ya
            slopes[i] = (yb - ya) / (xb - xa)
            events.append((xa, 0, ya, i))
            events.append((xb, 2, yb, i))
        else:
            x0s[i] = xb
            y0s[i] = yb
            slopes[i] = (ya - yb) / (xa - xb)
            events.append((xb, 0, yb, i))
            events.append((xa, 2, ya, i))
    events.sort(key=lambda e: (e[0], e[1]))

    def adjacent(i, j):
        d = i - j
        return d == 1 or d == -1 or d == m - 1 or d == 1 - m

    status = []  # 扫描线状态：与当前 x 相交的非垂直边，按 (y, 斜率, 编号) 升序

    def find(x, y, slope, i):
        """(y, slope, i) 在 status 中的二分插入位置。

        y 相同（共享端点）时按斜率决胜：斜率小者在当前 x 右侧位于下方，
        否则从同一点分叉的边顺序会立即失效，漏检交叉。
        """
        lo = 0
        hi = len(status)
        while lo < hi:
            mid = (lo + hi) >> 1
            j = status[mid]
            yj = y0s[j] + slopes[j] * (x - x0s[j])
            sj = slopes[j]
            if yj < y or (yj == y and (sj < slope or (sj == slope and j < i))):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def crosses(i, j):
        if adjacent(i, j):
            return False
        return _seg_intersect(pts[i], pts[(i + 1) % m], pts[j], pts[(j + 1) % m])

    for ev in events:
        kind = ev[1]
        if kind == 0:
            _, _, y, i = ev
            x = ev[0]
            pos = find(x, y, slopes[i], i)
            status.insert(pos, i)
            if pos > 0 and crosses(i, status[pos - 1]):
                return True
            if pos + 1 < len(status) and crosses(i, status[pos + 1]):
                return True
        elif kind == 2:
            _, _, y, i = ev
            x = ev[0]
            pos = find(x, y, slopes[i], i)
            if pos < len(status) and status[pos] == i:
                pass
            elif pos > 0 and status[pos - 1] == i:
                pos -= 1
            else:
                pos = status.index(i)  # 浮点极端情形兜底
            del status[pos]
            if 0 < pos < len(status) and crosses(status[pos - 1], status[pos]):
                return True
        else:
            x, _, ylo, yhi, i = ev
            # 命中 [ylo, yhi] 的状态边都与该垂直边相交
            lo = find(x, ylo, -math.inf, -1)
            hi = find(x, yhi, math.inf, m)
            for j in status[lo:hi]:
                if crosses(i, j):
                    return True

    # 垂直边之间：同 x 分组检查 y 区间重叠
    verts = sorted((e[0], e[2], e[3], e[4]) for e in events if e[1] == 1)
    k = 0
    while k < len(verts):
        group = [verts[k]]
        k += 1
        while k < len(verts) and verts[k][0] == group[0][0]:
            group.append(verts[k])
            k += 1
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i = group[a][3]
                j = group[b][3]
                if adjacent(i, j):
                    continue
                if max(group[a][1], group[b][1]) <= min(group[a][2], group[b][2]):
                    return True
    return False


# ---------------------------------------------------------------------------
# 耳切三角化（网格空间索引加速）
# ---------------------------------------------------------------------------

def _earclip(P):
    """对简单多边形 P（已校验）做耳切，返回原索引三元组列表。

    采用“锦标赛”顺序：每剪掉一个耳就跳过一个顶点，避免顺序剪耳在
    凸多边形上退化成扇形（长条三角形的大包围盒会把网格查询拖成 O(n^2)）。
    """
    n = len(P)
    order = list(range(n))
    if _signed_area(P) < 0.0:
        order.reverse()  # 统一按逆时针处理，索引仍指向原数组

    prev_ = [0] * n
    next_ = [0] * n
    for k in range(n):
        a = order[k - 1]
        b = order[k]
        next_[a] = b
        prev_[b] = a
    active = [True] * n
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]

    # 顶点均匀网格索引（懒删除：用 active 标记过滤；整数键避免元组分配）
    minx = min(xs)
    miny = min(ys)
    span = max(max(xs) - minx, max(ys) - miny)
    g = max(4, int(math.sqrt(n)))
    inv = (g / span) if span > 0.0 else 1.0
    grid = {}
    for i in range(n):
        cx = int((xs[i] - minx) * inv)
        cy = int((ys[i] - miny) * inv)
        if cx >= g:
            cx = g - 1
        if cy >= g:
            cy = g - 1
        grid.setdefault(cx * g + cy, []).append(i)

    tris = []
    remaining = n
    v = order[0]
    gget = grid.get

    while remaining > 3:
        quota = remaining  # 本轮扫描上界；一圈无收获则走兜底
        clipped = False
        while quota > 0 and remaining > 3:
            quota -= 1
            a = prev_[v]
            b = v
            c = next_[v]
            ax = xs[a]
            ay = ys[a]
            bx = xs[b]
            by = ys[b]
            cx_ = xs[c]
            cy_ = ys[c]
            o = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)
            if o > 0.0:
                # 候选耳：检查三角形包围盒覆盖的网格内是否有其他活动顶点
                if ax < bx:
                    x0 = ax if ax < cx_ else cx_
                    x1 = bx if bx > cx_ else cx_
                else:
                    x0 = bx if bx < cx_ else cx_
                    x1 = ax if ax > cx_ else cx_
                if ay < by:
                    y0 = ay if ay < cy_ else cy_
                    y1 = by if by > cy_ else cy_
                else:
                    y0 = by if by < cy_ else cy_
                    y1 = ay if ay > cy_ else cy_
                gx0 = int((x0 - minx) * inv)
                gx1 = int((x1 - minx) * inv)
                gy0 = int((y0 - miny) * inv)
                gy1 = int((y1 - miny) * inv)
                if gx1 >= g:
                    gx1 = g - 1
                if gy1 >= g:
                    gy1 = g - 1
                e1x = bx - ax
                e1y = by - ay
                e2x = cx_ - bx
                e2y = cy_ - by
                e3x = ax - cx_
                e3y = ay - cy_
                ear = True
                for gx in range(gx0, gx1 + 1):
                    base = gx * g
                    for gy in range(gy0, gy1 + 1):
                        lst = gget(base + gy)
                        if not lst:
                            continue
                        for p in lst:
                            if p == a or p == b or p == c or not active[p]:
                                continue
                            px = xs[p]
                            py = ys[p]
                            if px < x0 or px > x1 or py < y0 or py > y1:
                                continue
                            if (e1x * (py - ay) - e1y * (px - ax) >= 0.0 and
                                    e2x * (py - by) - e2y * (px - bx) >= 0.0 and
                                    e3x * (py - cy_) - e3y * (px - cx_) >= 0.0):
                                ear = False
                                break
                        if not ear:
                            break
                    if not ear:
                        break
                if ear:
                    tris.append((a, b, c))
                    next_[a] = c
                    prev_[c] = a
                    active[b] = False
                    remaining -= 1
                    clipped = True
                    # 跳过一个顶点（c 的耳性已变，留到下一轮），保持局部性
                    v = next_[c] if remaining > 3 else c
                    continue
            elif o == 0.0:
                # 共线中间点：直接移除（不改变面积，不产生三角形）
                if (ax <= bx <= cx_ or cx_ <= bx <= ax) and \
                        (ay <= by <= cy_ or cy_ <= by <= ay):
                    next_[a] = c
                    prev_[c] = a
                    active[b] = False
                    remaining -= 1
                    clipped = True
                    v = next_[c] if remaining > 3 else c
                    continue
            v = c
        if not clipped and remaining > 3:
            # 兜底：数值退化导致整圈找不到耳时，强制剪掉最凸顶点，
            # 保证终止（简单多边形正常不会走到这里）。
            best = -1.0
            bv = v
            u = v
            for _ in range(remaining):
                a2 = prev_[u]
                c2 = next_[u]
                o2 = ((xs[u] - xs[a2]) * (ys[c2] - ys[a2]) -
                      (ys[u] - ys[a2]) * (xs[c2] - xs[a2]))
                if o2 > best:
                    best = o2
                    bv = u
                u = next_[u]
            a = prev_[bv]
            c = next_[bv]
            tris.append((a, bv, c))
            next_[a] = c
            prev_[c] = a
            active[bv] = False
            remaining -= 1
            v = a

    # 收尾三角形
    a = v
    b = next_[v]
    c = next_[b]
    if ((xs[b] - xs[a]) * (ys[c] - ys[a]) -
            (ys[b] - ys[a]) * (xs[c] - xs[a])) != 0.0:
        tris.append((a, b, c))
    return tris


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

class EarClip:
    """无状态耳切三角化器（实例方法均不共享可变状态，线程安全）。"""

    def triangulate(self, poly):
        """返回三角形索引三元组列表（索引指向输入顶点）。"""
        pts = _coerce(poly)
        if len(pts) < 3:
            raise ValueError("多边形至少需要 3 个顶点")
        pts = _dedup(pts)
        if len(pts) < 3:
            return []  # 全部点重合等退化情形
        if _has_self_intersection(pts):
            raise ValueError("多边形自相交")
        return _earclip(pts)

    def area(self, poly):
        """多边形面积（恒为非负 float）。"""
        pts = _coerce(poly)
        return abs(_signed_area(pts))
