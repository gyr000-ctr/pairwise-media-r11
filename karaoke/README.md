# karaoke

## 题面（可直接粘贴）

子目录：karaoke。人声消除（中置相位抵消）。约束：中置信号衰减 ≥12 dB、侧信号保留（损失 ≤0.1 dB）；强度 0 恒等；单声道输入报错；10 万样本不超过 200 ms；非有限样本可区分报错。验收：cd karaoke && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd karaoke && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
