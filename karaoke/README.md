# karaoke

## 题面（可直接粘贴）

子目录：karaoke。人声消除（自适应中置提取）。约束：必须先用左右互相关估计增益与延迟（误差 ≤1 采样），再在 STFT 域按相位一致性做掩蔽而不是直接相减；中置衰减 ≥15 dB 而侧信号损失 ≤0.3 dB；左右延迟失配 ±3 采样内仍有效；WOLA 重建窗和为 1（≤1e-9）；10 万样本不超过 400 ms；单声道输入与非有限样本可区分报错。验收：cd karaoke && python -m pytest tests -q。

## 实现要点（karaoke/core.py）

1. `estimate_delay_gain`：左右通道整段归一化互相关的绝对峰值估整数延迟
   （同相中置给 +1 峰、反相侧信号给 -1 峰；歧义峰取最接近 0 的滞后），
   RMS 能量比估增益（对极性不敏感）。
2. WOLA-STFT：2048 点 sqrt-Hann 分析/合成窗、1024 点跳，窗积移位和恒为 1
   （误差 ~4e-16），重叠相加完美重建（<1e-9）。
3. `center_suppression`：STFT 域 mid/side 分解，时频平滑功率估计中置占比，
   仅对 mid 施加超减软增益、保留 side；并对中置主导区域做逐 bin 通道增益
   归一化以同时满足侧损约束。全程乘性增益，无时域直接相减。
4. 报错分类：`MonoInputError`（单声道/非标量/多维）与
   `NonFiniteInputError`（NaN/Inf），均为 `ValueError` 子类、彼此可区分。
5. 纯函数式、无共享可变状态，线程安全；仅依赖 numpy。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面（实测 10 万样本稳态约 30 ms，远低于 400 ms）

## 运行方式

```bash
cd karaoke && python -m pytest tests -q
```

44 passed（含原 15 个基础用例与升级题面的硬约束用例）。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest, numpy
- 环境可复现等级：仅依赖 numpy

