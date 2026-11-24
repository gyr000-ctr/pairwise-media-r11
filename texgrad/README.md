# texgrad

## 题面（可直接粘贴）

子目录：texgrad。mip 链生成与能量守恒。约束：每级尺寸折半（奇数取整规则确定）；最末级 1x1；常量图逐级不变（≤1e-12）；盒子滤波与直接均值一致（≤1e-9）；非法尺寸可区分报错；1024x1024 建链不超过 400 ms；并发安全。验收：cd texgrad && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd texgrad && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
