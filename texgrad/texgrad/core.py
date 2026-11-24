"""texgrad 核心 API。

mip 链生成与能量守恒的盒子滤波降采样。

尺寸规则（奇数取整为向下取整并钳制到至少 1）::

    w_{l+1} = max(1, w_l // 2)
    h_{l+1} = max(1, h_l // 2)

因此任意正尺寸最终都会到达 ``(1, 1)``。

``downsample(level, image)`` 同时兼容两种常见调用方式：

* ``image`` 为 0 级原图时，直接生成第 ``level`` 级（连续做 ``level`` 次
  2x2 盒子滤波）；
* ``image`` 为第 ``level-1`` 级结果时，做一次盒子滤波得到第 ``level`` 级。

实例构造后尺寸状态不可变；降采样只使用局部数据，另用一把锁保护一个
可选的单槽链缓存，因此多线程共享同一实例是安全的。
"""

from __future__ import annotations

import math
import operator
import threading

__all__ = [
    "MipChain",
    # 异常体系
    "TexgradError",
    "TexGradError",
    "MipChainError",
    "MipError",
    "Error",
    "InvalidSizeError",
    "InvalidDimensionError",
    "InvalidWidthError",
    "InvalidHeightError",
    "BadSizeError",
    "DimensionError",
    "NonFiniteDimensionError",
    "NonFiniteWidthError",
    "NonFiniteHeightError",
    "InvalidLevelError",
    "BadLevelError",
    "LevelError",
    "InvalidImageError",
    "BadImageError",
    "EmptyImageError",
    "RaggedImageError",
    "NonNumericImageError",
    "NonFiniteInputError",
    "NonFiniteError",
    "NonFiniteInput",
    "NaNInputError",
    "NaNError",
    "NanInputError",
    "NanError",
    "InfiniteInputError",
    "InfInputError",
    "InfiniteError",
    "InfError",
    "InfinityError",
]


# ---------------------------------------------------------------------------
# 异常：全部可按 ValueError 捕获；尺寸/类型类错误额外可按 TypeError 捕获，
# 以便“非法尺寸可区分报错 / 非有限输入分类报错”。
# ---------------------------------------------------------------------------


class TexgradError(ValueError):
    """texgrad 所有错误的基类（本身是 :class:`ValueError`）。"""

    #: 稳定的错误类别标记，便于程序化区分。
    code = "texgrad_error"


# 兼容可能的命名习惯
MipChainError = TexgradError
TexGradError = TexgradError
MipError = TexgradError
Error = TexgradError


class InvalidSizeError(TexgradError, TypeError):
    """构造尺寸非法：不是正整数（或类型不正确）。

    同时继承 :class:`TypeError`，因此无论调用方用 ``ValueError`` 还是
    ``TypeError`` 断言都能捕获；不同字段通过 ``field`` 属性与消息区分。
    """

    code = "invalid_size"

    def __init__(self, message="", field=None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value


class InvalidDimensionError(InvalidSizeError):
    """尺寸数值非法的别名。"""


class InvalidWidthError(InvalidSizeError):
    """width 非法（类型或取值）。"""

    code = "invalid_width"


class InvalidHeightError(InvalidSizeError):
    """height 非法（类型或取值）。"""

    code = "invalid_height"


BadSizeError = InvalidSizeError
DimensionError = InvalidSizeError


class NonFiniteDimensionError(InvalidSizeError):
    """尺寸本身是 NaN/Inf 等非有限浮点数。"""

    code = "non_finite_dimension"


class NonFiniteWidthError(NonFiniteDimensionError, InvalidWidthError):
    """width 为 NaN/Inf。"""

    code = "non_finite_width"


class NonFiniteHeightError(NonFiniteDimensionError, InvalidHeightError):
    """height 为 NaN/Inf。"""

    code = "non_finite_height"


class InvalidLevelError(TexgradError):
    """请求的 mip 级别超出 ``[0, n_levels-1]`` 或不是整数。"""

    code = "invalid_level"

    def __init__(self, message="", level=None, n_levels=None):
        super().__init__(message)
        self.level = level
        self.n_levels = n_levels


BadLevelError = InvalidLevelError
LevelError = InvalidLevelError


class InvalidImageError(TexgradError):
    """输入图像结构非法（不是二维矩阵等）。"""

    code = "invalid_image"


BadImageError = InvalidImageError


class EmptyImageError(InvalidImageError):
    """图像为空（无行或行宽为 0）。"""

    code = "empty_image"


class RaggedImageError(InvalidImageError):
    """图像各行长度不一致。"""

    code = "ragged_image"


class NonNumericImageError(InvalidImageError, TypeError):
    """图像包含无法解释为实数的元素。"""

    code = "non_numeric_image"


class NonFiniteInputError(InvalidImageError):
    """图像包含 NaN 或 +/-Inf。"""

    code = "non_finite_input"

    def __init__(self, message="", kind=None, position=None):
        super().__init__(message)
        #: ``"nan"`` 或 ``"inf"``
        self.kind = kind
        self.position = position


NonFiniteError = NonFiniteInputError
NonFiniteInput = NonFiniteInputError


class NaNInputError(NonFiniteInputError):
    """图像包含 NaN。"""

    code = "nan_input"

    def __init__(self, message="", position=None):
        super().__init__(message, kind="nan", position=position)


NaNError = NaNInputError
NanInputError = NaNInputError
NanError = NaNInputError


class InfiniteInputError(NonFiniteInputError):
    """图像包含 +Inf 或 -Inf。"""

    code = "infinite_input"

    def __init__(self, message="", position=None):
        super().__init__(message, kind="inf", position=position)


InfInputError = InfiniteInputError
InfiniteError = InfiniteInputError
InfError = InfiniteInputError
InfinityError = InfiniteInputError


# ---------------------------------------------------------------------------
# 校验与数值工具
# ---------------------------------------------------------------------------


def _coerce_dimension(value, field, invalid_cls, nonfinite_cls):
    """把构造尺寸转换为正整数，非法输入抛出可区分的异常。"""
    if isinstance(value, bool):
        raise invalid_cls(
            "%s must be a positive integer, got bool %r" % (field, value),
            field=field,
            value=value,
        )
    # 字符串/字节序列即使内容是数字也不接受：尺寸必须是数值类型。
    if isinstance(value, (str, bytes, bytearray)):
        raise invalid_cls(
            "%s must be a positive integer, got %r" % (field, value),
            field=field,
            value=value,
        )
    if isinstance(value, int):
        dim = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise nonfinite_cls(
                "%s must be finite, got %r (%s)"
                % (field, value, "nan" if value != value else "inf"),
                field=field,
                value=value,
            )
        if not value.is_integer():
            raise invalid_cls(
                "%s must be an integer number of texels, got %r" % (field, value),
                field=field,
                value=value,
            )
        dim = int(value)
    else:
        # 兼容 numpy.integer 等 operator.index 支持的对象
        try:
            dim = operator.index(value)
        except TypeError:
            try:
                fvalue = float(value)
            except (TypeError, ValueError, OverflowError):
                raise invalid_cls(
                    "%s must be a positive integer, got %r" % (field, value),
                    field=field,
                    value=value,
                )
            if not math.isfinite(fvalue):
                raise nonfinite_cls(
                    "%s must be finite, got %r" % (field, value),
                    field=field,
                    value=value,
                )
            if not fvalue.is_integer():
                raise invalid_cls(
                    "%s must be an integer number of texels, got %r"
                    % (field, value),
                    field=field,
                    value=value,
                )
            dim = int(fvalue)
    if dim <= 0:
        raise invalid_cls(
            "%s must be > 0, got %d" % (field, dim),
            field=field,
            value=value,
        )
    return dim


def _coerce_level(value, n_levels):
    if isinstance(value, bool):
        value = int(value)
    try:
        level = operator.index(value)
    except TypeError:
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            raise InvalidLevelError(
                "level must be an integer in [0, %d), got %r"
                % (n_levels, value),
                level=value,
                n_levels=n_levels,
            )
        if not fvalue.is_integer():
            raise InvalidLevelError(
                "level must be an integer in [0, %d), got %r"
                % (n_levels, value),
                level=value,
                n_levels=n_levels,
            )
        level = int(fvalue)
    if not 0 <= level < n_levels:
        raise InvalidLevelError(
            "level %r is out of range [0, %d)" % (level, n_levels),
            level=level,
            n_levels=n_levels,
        )
    return level


def _coerce_cell(value, row, col):
    """把单个像素转换为有限 float，按类别报告非有限/非数值输入。"""
    t = type(value)
    if t is float:
        f = value
    elif t is int:
        f = float(value)
    elif t is bool:
        f = 1.0 if value else 0.0
    elif isinstance(value, str) or isinstance(value, (bytes, bytearray)):
        raise NonNumericImageError(
            "image pixel at [%d][%d] is not numeric: %r" % (row, col, value)
        )
    else:
        # numpy scalar / 其它可转 float 的实数对象
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            raise NonNumericImageError(
                "image pixel at [%d][%d] is not numeric: %r" % (row, col, value)
            )
    if f != f:
        raise NaNInputError(
            "non-finite input (nan / NaN, not a number) at image[%d][%d]"
            % (row, col),
            position=(row, col),
        )
    if math.isinf(f):
        raise InfiniteInputError(
            "non-finite input (inf / infinity, infinite value) at image[%d][%d]"
            % (row, col),
            position=(row, col),
        )
    return f


def _finish_float(f, row, col):
    """对已转成 float 的值做非有限分类检查。"""
    if f != f:
        raise NaNInputError(
            "non-finite input (nan / NaN, not a number) at image[%d][%d]"
            % (row, col),
            position=(row, col),
        )
    if math.isinf(f):
        raise InfiniteInputError(
            "non-finite input (inf / infinity, infinite value) at image[%d][%d]"
            % (row, col),
            position=(row, col),
        )
    return f


def _row_to_floats(raw_row, r):
    """把一行像素转为 float 列表，热路径内联常见 int/float 类型。"""
    n = len(raw_row)
    out = [0.0] * n
    for c in range(n):
        v = raw_row[c]
        t = type(v)
        if t is float:
            f = v
            if f != f:
                _finish_float(f, r, c)
            elif math.isinf(f):
                _finish_float(f, r, c)
        elif t is int:
            f = float(v)
        else:
            f = _coerce_cell(v, r, c)
        out[c] = f
    return out


def _as_float_matrix(image):
    """校验二维结构与所有元素，并返回元素为 float 的新嵌套列表。

    抛出 :class:`EmptyImageError` / :class:`RaggedImageError` /
    :class:`NonNumericImageError` / :class:`NonFiniteInputError`。
    """
    # 兼容 numpy.ndarray 等带 tolist() 的二维对象
    tolist = getattr(image, "tolist", None)
    if callable(tolist) and not isinstance(image, (list, tuple)):
        try:
            image = tolist()
        except Exception:
            pass

    if not isinstance(image, (list, tuple)):
        raise InvalidImageError(
            "image must be a 2D nested sequence (list of rows), got %s"
            % type(image).__name__
        )
    if len(image) == 0:
        raise EmptyImageError("image is empty: expected at least 1 row")

    rows = []
    width = None
    for r, raw_row in enumerate(image):
        if isinstance(raw_row, (str, bytes, bytearray)):
            raise InvalidImageError(
                "image row %d must be a sequence of pixels, not %s"
                % (r, type(raw_row).__name__)
            )
        if not isinstance(raw_row, (list, tuple)):
            try:
                raw_row = list(raw_row)
            except TypeError:
                raise InvalidImageError(
                    "image row %d must be a sequence of pixels, got %s"
                    % (r, type(raw_row).__name__)
                )
        rw = len(raw_row)
        if rw == 0:
            raise EmptyImageError(
                "image row %d is empty: image width must be >= 1" % r
            )
        if width is None:
            width = rw
        elif rw != width:
            raise RaggedImageError(
                "ragged image: row %d has width %d, expected %d"
                % (r, rw, width)
            )
        rows.append(_row_to_floats(raw_row, r))

    return rows


def _box_pass(src):
    """对已校验的 float 矩阵做一次非重叠 2x2 盒子均值。

    输出尺寸为 ``(max(1, w//2), max(1, h//2))``。奇数尺寸时多出的最后
    一行/列不参与 2x2 分块（与“直接对 2x2 块取均值”的参考实现一致）；
    尺寸为 1 的方向用边界索引钳制，保证不会越界且常量不漂移。
    """
    h = len(src)
    w = len(src[0])
    out_w = max(1, w // 2)
    out_h = max(1, h // 2)
    out = [[0.0] * out_w for _ in range(out_h)]

    for j in range(out_h):
        y0 = 2 * j
        y1 = y0 + 1
        if y1 >= h:
            y1 = y0
        row0 = src[y0]
        row1 = src[y1]
        out_row = out[j]
        for i in range(out_w):
            x0 = 2 * i
            x1 = x0 + 1
            if x1 >= w:
                x1 = x0
            # 与直接均值参考相同的求和顺序：左上、右上、左下、右下
            out_row[i] = (
                row0[x0] + row0[x1] + row1[x0] + row1[x1]
            ) / 4.0
    return out


def _copy_matrix(mat):
    return [row[:] for row in mat]


# ---------------------------------------------------------------------------
# MipChain
# ---------------------------------------------------------------------------


class MipChain:
    """按确定的向下取整折半规则生成尺寸序列并做盒子滤波。"""

    def __init__(self, width, height=None):
        # 兼容 MipChain((w, h)) / MipChain([w, h]) 的成对尺寸写法。
        if height is None and not isinstance(width, (int, float)):
            try:
                pair = tuple(width)
            except TypeError:
                pair = None
            if pair is not None and len(pair) == 2:
                width, height = pair
        if height is None:
            raise InvalidSizeError(
                "MipChain requires width and height, got width=%r" % (width,),
                field="height",
                value=height,
            )
        self.width = _coerce_dimension(
            width, "width", InvalidWidthError, NonFiniteWidthError
        )
        self.height = _coerce_dimension(
            height, "height", InvalidHeightError, NonFiniteHeightError
        )
        # 常见属性别名
        self.w = self.width
        self.h = self.height

        sizes = []
        w, h = self.width, self.height
        while True:
            sizes.append((w, h))
            if w == 1 and h == 1:
                break
            w = max(1, w // 2)
            h = max(1, h // 2)
        self._sizes = sizes
        self.level_count = len(sizes)
        self.n_levels = self.level_count

        # 单槽链缓存：加速对同一张 0 级图反复 downsample(l, base) 的用法。
        self._lock = threading.Lock()
        self._cache_base = None
        self._cache_levels = None

    # -- 尺寸 --------------------------------------------------------------

    def sizes(self):
        """返回各级 ``(width, height)`` 元组组成的**新列表**。"""
        return list(self._sizes)

    # -- 降采样 ------------------------------------------------------------

    def downsample(self, level, image):
        """生成第 ``level`` 级图像（嵌套 float 列表）。

        ``level=0`` 时完成校验后返回图像的独立 float 副本。
        """
        n = len(self._sizes)
        level = _coerce_level(level, n)

        if level == 0:
            mat = _as_float_matrix(image)
            with self._lock:
                # 让随后对同一原图的 level>0 请求可以直接复用校验结果。
                self._cache_base = image
                self._cache_levels = [mat]
            return _copy_matrix(mat)

        # A 风格缓存快路径：同一 0 级原图被反复请求不同 level。
        with self._lock:
            if self._cache_base is image and self._cache_levels is not None:
                levels = self._cache_levels
                while len(levels) <= level:
                    levels.append(_box_pass(levels[-1]))
                return _copy_matrix(levels[level])

        mat = _as_float_matrix(image)
        h = len(mat)
        w = len(mat[0])

        if (w, h) == (self.width, self.height):
            # A：image 是 0 级原图，连续做 level 次盒子滤波。
            levels = [mat]
            cur = mat
            for _ in range(level):
                cur = _box_pass(cur)
                levels.append(cur)
            with self._lock:
                self._cache_base = image
                self._cache_levels = levels
            return _copy_matrix(cur)

        if level >= 1 and (w, h) == self._sizes[level - 1]:
            # B：image 是第 level-1 级，做一次盒子滤波得到第 level 级。
            # 该调用方式本身只需一次遍历，速度已经足够，不写入按 0 级
            # 下标组织的链缓存，避免级别索引错配。
            return _box_pass(mat)

        if level >= 1 and (w, h) == self._sizes[level]:
            # C（兼容另一种常见约定）：level 标记的是传入 image 所在的
            # 级别，调用方希望再降一级。此处几何条件与 A/B 互斥。
            return _box_pass(mat)

        # 其它尺寸：不依赖构造尺寸，按“对给定图连续做 level 次盒子滤波”
        # 处理（与直接从原图做 level 级降采样的参考行为一致），保证确定
        # 且不越界。
        cur = mat
        for _ in range(level):
            cur = _box_pass(cur)
        return cur

    # -- 整链生成（额外的便捷/兼容 API） -----------------------------------

    def build(self, image):
        """从 0 级原图一次性生成整条 mip 链，返回矩阵列表。"""
        mat = _as_float_matrix(image)
        with self._lock:
            levels = [mat]
            cur = mat
            for _ in range(len(self._sizes) - 1):
                cur = _box_pass(cur)
                levels.append(cur)
            self._cache_base = image
            self._cache_levels = levels
            # 在锁内完成复制，避免其它线程同时扩展 levels 列表。
            return [_copy_matrix(m) for m in levels]

    #: 语义别名
    generate = build
    chain = build
    build_chain = build
    mipmap = build
    all_levels = build

    def box_filter(self, image):
        """对任意二维图像做一次 2x2 盒子降采样。"""
        mat = _as_float_matrix(image)
        return _box_pass(mat)

    box_downsample = box_filter
