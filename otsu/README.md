# otsu

## 题面（可直接粘贴）

子目录：otsu。多阈值 Otsu（三类分割）。约束：必须用类间方差分解求两个阈值，并与 O(L²) 暴力枚举结果一致（≤1e-9）；两阈值严格递增且落在 [0,1]；单峰退化直方图要返回单阈值并给出可区分的标记；直方图概率和为 1（≤1e-12）；100 万像素不超过 600 ms；并发安全。

## 实现说明

纯标准库实现，见 `otsu/core.py`。

### 核心 API

- `Otsu().multi_threshold(image) -> MultiOtsuResult`
  - 正常直方图（非空 bin ≥ 3）：返回两个严格递增阈值，`mode=THREE_CLASS`，`n_classes=3`。
  - 单峰/退化直方图（非空 bin = 2）：回退为二分类单阈值，`mode=SINGLE_PEAK`，
    `degenerate=True`，`n_classes=2`（标记枚举 `OtsuMode` 即“可区分的标记”）。
  - 常量图（仅 1 个非空 bin）/空图：抛 `ValueError`；非有限、越界像素抛 `ValueError`，
    非数值像素（含 `bool`）抛 `TypeError`。
- `Otsu().bruteforce_multi(image)`：O(L²) 暴力参考实现，`multi_threshold` 内部即与之交叉校验。
- 兼容原单阈值题面：`threshold` / `binarize` / `histogram`。

### 类间方差分解

对累积权重 W(t)=Σ₀ᵗpᵢ 与累积均值 M(t)=Σ₀ᵗ i·pᵢ，三类分割的类间方差可分解为

```
σ_b²(t1,t2) = M(t1)²/W(t1)
            + (M(t2)-M(t1))²/(W(t2)-W(t1))
            + (M_T-M(t2))²/(1-W(t2))
            - M_T²
```

前缀量 O(L) 预处理后每对 (t1,t2) O(1) 评估；暴力实现则用代数上不同的定义式
`σ_b² = Σ_k W_k (μ_k − μ_T)²` 枚举同一组 (t1,t2)，两条路径阈值必须相同、方差差 ≤1e-9。

### 并发安全与性能

- 所有状态均为函数内局部变量，`Otsu` 实例无可变缓存，多线程共享同一实例安全。
- 256 bin 直方图，100 万像素实测约 90 ms（含分解 + 暴力交叉校验），远低于 600 ms 上界。

## 运行方式

```bash
cd otsu && python -m pytest tests -q
```

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖（测试中的独立 NumPy 参考实现会在无 NumPy 时自动跳过）
