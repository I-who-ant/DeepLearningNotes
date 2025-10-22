# 优化器速查表 (Optimizer Cheat Sheet)

## 🎯 核心概念

### 优化器是什么?
```
输入: 梯度 (param.grad)
   ↓
优化器: 决定如何更新参数
   ↓
输出: 参数更新量
   ↓
结果: loss 下降
```

### 基础公式
```python
# 最简单的梯度下降
param_new = param_old - learning_rate × gradient

# 但实际优化器更复杂,用各种技巧加速收敛
```

---

## 📊 优化器对比表

| 优化器 | 收敛速度 | 泛化性能 | 超参数敏感度 | 适用场景 | 推荐指数 |
|--------|---------|---------|------------|---------|---------|
| **SGD** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | CV大模型 | ⭐⭐ |
| **SGD+Momentum** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | CV通用 | ⭐⭐⭐⭐⭐ |
| **Adagrad** | ⭐⭐ | ⭐⭐ | ⭐ | 稀疏数据 | ⭐⭐ |
| **RMSprop** | ⭐⭐⭐ | ⭐⭐ | ⭐ | RNN | ⭐⭐⭐ |
| **Adam** | ⭐⭐⭐ | ⭐⭐ | ⭐ | 通用原型 | ⭐⭐⭐⭐ |
| **AdamW** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | NLP,大模型 | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速选择指南

### 计算机视觉 (CNN/ResNet/VGG)
```python
# 首选: SGD + Momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,              # 较大学习率
    momentum=0.9,        # 标准动量
    weight_decay=1e-4    # L2正则化
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

### 自然语言处理 (Transformer/BERT/GPT)
```python
# 首选: AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,                    # 较小学习率
    betas=(0.9, 0.98),          # NLP常用beta
    weight_decay=0.01           # 较大正则化
)

# 配合Warmup
# (需要自己实现或用transformers库)
```

### 快速原型 / 不确定任务
```python
# 推荐: Adam 或 AdamW
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001  # 默认值
)
```

---

## 🔧 参数详解

### 1. 学习率 (lr / learning_rate)
**最重要的超参数!**

```python
# 不同优化器的常用范围
SGD系列:  0.01 ~ 0.1
Adam系列: 0.0001 ~ 0.001
```

**如何选择?**
- 从小开始 (如 1e-4)
- 观察 loss 曲线
- 逐步增大直到出现震荡
- 选择震荡前的最大值

**学习率太大:** loss 震荡或爆炸
**学习率太小:** 收敛太慢

---

### 2. 动量 (momentum)
用于 SGD

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9  # 标准值
)
```

**作用:** 保留历史梯度,加速收敛,减少震荡

**常用值:**
- `0.9`: 标准选择
- `0.95 ~ 0.99`: 大 batch size 时

---

### 3. 权重衰减 (weight_decay)
L2 正则化,防止过拟合

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    weight_decay=1e-4  # 常用值
)
```

**常用范围:**
- `0`: 无正则化
- `1e-5 ~ 1e-3`: 小数据集
- `0.01 ~ 0.1`: 大模型 (Transformer)

**注意:** Adam 配合 weight_decay 效果不好,用 AdamW!

---

### 4. Betas (β₁, β₂)
用于 Adam/AdamW

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)  # 默认值
)
```

- **β₁**: 一阶矩(动量)衰减率
- **β₂**: 二阶矩(方差)衰减率

**通常不需要改!**

特殊情况:
- NLP/大batch: `(0.9, 0.98)`
- 噪声大: `(0.5, 0.999)`

---

## 💻 完整训练代码模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 1. 创建模型
model = MyModel()

# 2. 选择优化器
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# 3. 学习率调度器 (可选)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 4. 损失函数
criterion = nn.CrossEntropyLoss()

# 5. 训练循环
for epoch in range(100):
    for images, labels in train_loader:

        # 5.1 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 5.2 反向传播
        optimizer.zero_grad()  # ⚠️ 必须先清空梯度!
        loss.backward()        # 计算梯度

        # 5.3 梯度裁剪 (可选,RNN/Transformer常用)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5.4 参数更新
        optimizer.step()  # 优化器根据梯度更新参数

    # 5.5 学习率调整
    scheduler.step()

    # 5.6 打印信息
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={current_lr:.6f}")
```

---

## 🎓 优化器原理速览

### SGD (Stochastic Gradient Descent)
```python
param = param - lr × grad
```
- 最简单
- 容易震荡

---

### SGD + Momentum
```python
velocity = momentum × velocity + grad
param = param - lr × velocity
```
- 保留历史速度
- 减少震荡
- CV任务首选

**类比:** 球从山坡滚下,有惯性

---

### Adam (Adaptive Moment Estimation)
```python
m = β₁ × m + (1-β₁) × grad           # 动量
v = β₂ × v + (1-β₂) × grad²          # 方差
m_hat = m / (1 - β₁^t)                # 偏差修正
v_hat = v / (1 - β₂^t)
param = param - lr × m_hat / (√v_hat + ε)
```
- 结合动量和自适应学习率
- 收敛快
- 通用性强

---

### AdamW (Adam with decoupled Weight decay)
```python
# 在Adam基础上修正weight_decay
# 更好的泛化性能
```
- Adam的改进版
- 现代最佳实践
- NLP任务首选

---

## 📈 学习率调度器

### StepLR (阶梯衰减)
```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,  # 每30个epoch
    gamma=0.1      # 学习率 ×= 0.1
)
```

### CosineAnnealingLR (余弦退火)
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100  # 周期
)
```

### ReduceLROnPlateau (自适应降低)
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)
# 使用: scheduler.step(val_loss)
```

---

## 🐛 常见问题

### Q: loss 不下降?
1. 检查学习率是否太小
2. 检查是否忘记 `optimizer.zero_grad()`
3. 检查梯度是否消失 (打印 `param.grad`)
4. 尝试不同的优化器

### Q: loss 震荡或爆炸?
1. 降低学习率
2. 使用梯度裁剪
3. 使用 BatchNorm
4. 检查数据是否归一化

### Q: 训练很慢?
1. 增大学习率
2. 使用自适应优化器 (Adam/AdamW)
3. 使用学习率预热 (Warmup)
4. 增大 batch size

### Q: 过拟合?
1. 增大 `weight_decay`
2. 使用 Dropout
3. 数据增强
4. 减小模型复杂度

---

## 🎯 记忆口诀

> **"CV用SGD,NLP用AdamW,原型快速Adam跑"**

> **"梯度告诉方向,优化器决定步法"**

> **"学习率最重要,太大爆炸太小慢"**

---

## 📚 推荐阅读顺序

1. **新手**: 直接用 Adam (lr=0.001)
2. **进阶**: 了解 SGD+Momentum,学会调 lr
3. **高级**: 根据任务选择,配合学习率调度器
4. **专家**: 理解原理,自己实现优化器

---

**最后提醒:**
- 优化器只是工具,数据质量和模型设计更重要!
- 先用默认参数跑通,再精细调参
- 多看 loss 曲线,少凭感觉猜测

**祝训练顺利!** 🚀
