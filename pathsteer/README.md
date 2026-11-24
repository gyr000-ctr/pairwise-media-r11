# pathsteer

## 题面（可直接粘贴）

子目录：pathsteer。群体转向（分离/对齐/聚合）。约束：三力权重和为 1（≤1e-12）；速度上限钳制；相邻过近时分离力单调增大；无邻居时对齐与聚合为 0；单步位移不超过 vmax*dt（≤1e-9）；非法参数可区分报错；1 万物件不超过 500 ms。验收：cd pathsteer && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd pathsteer && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
