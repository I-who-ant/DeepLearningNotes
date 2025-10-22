# SGD优化器训练演示说明

## 📋 功能说明

在 `src/experiments/torch_nn/CIFARSeeback.py` 中新增了 `demo_sgd_training()` 函数,完整演示了使用SGD优化器训练CIFAR-10模型的过程。

## 🎯 实现内容

### 1. 训练配置
- **模型**: `seeback` (CIFAR-10分类模型, 96,426个参数)
- **优化器**: SGD (lr=0.01, 无momentum)
- **损失函数**: CrossEntropyLoss
- **训练数据**: 100个样本 (模拟数据)
- **Batch size**: 16
- **Epochs**: 50
- **总批次**: 7个batch/epoch

### 2. 训练流程

```python
for epoch in range(50):
    for batch in batches:
        # 1. 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 2. 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        # 3. 打印loss
        print(f"Epoch [{epoch}/{50}] Batch [{batch}/{7}] Loss: {loss:.4f}")
```

### 3. 输出格式

#### 训练过程输出
```
======================================================================
SGD优化器训练演示
======================================================================

模型参数总数: 96,426
使用优化器: SGD (lr=0.01, momentum=0)
训练数据: 100个样本, batch_size=16, 总批次=7

开始训练 (50 epochs)...
----------------------------------------------------------------------
Epoch [ 1/50] Batch [1/7] Loss: 2.3041
Epoch [ 1/50] Batch [2/7] Loss: 2.3495
Epoch [ 1/50] Batch [3/7] Loss: 2.3062
...
Epoch [50/50] Batch [7/7] Loss: 2.1160
  → Epoch 50 平均Loss: 2.2267

----------------------------------------------------------------------
训练完成!
初始loss: 2.3297
最终loss: 2.1160
总下降: 0.2137
```

#### 模型性能测试
```
======================================================================
测试最终模型性能
======================================================================
训练数据准确率: 12/100 = 12.00%

第一个样本预测概率分布:
  真实类别: 2
  预测类别: 8
  [ ]    类别0: 0.1096 (10.96%)
  [ ]    类别1: 0.1120 (11.20%)
  [✓]    类别2: 0.0990 (9.90%)  ← 真实类别
  ...
  [ ] 👉 类别8: 0.1354 (13.54%)  ← 预测类别
  ...
```

### 4. 可视化功能

代码包含loss曲线可视化功能,需要安装matplotlib:

```bash
pip install matplotlib
```

运行后会生成:
- **文件**: `artifacts/sgd_training_loss.png`
- **内容**:
  - 蓝色曲线: 每个batch的loss
  - 红色曲线: 移动平均loss (10-batch窗口)
  - 绿点: 初始loss
  - 红点: 最终loss

## 🔍 代码关键点解析

### 1. 优化器创建
```python
optimizer = torch.optim.SGD(
    model.parameters(),  # 需要优化的参数
    lr=0.01              # 学习率
)
```

### 2. 训练步骤
```python
# ⚠️ 关键步骤,顺序不能错!

optimizer.zero_grad()  # 1. 清空之前的梯度
loss.backward()        # 2. 计算当前梯度
optimizer.step()       # 3. 更新参数
```

### 3. Loss记录
```python
loss_history = []  # 记录每个batch的loss

# 训练循环中
current_loss = loss.item()  # 获取标量值
loss_history.append(current_loss)
```

### 4. 模型评估
```python
model.eval()  # 切换到评估模式
with torch.no_grad():  # 不计算梯度
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y).sum().item() / len(y)
```

## 📊 观察与分析

### 1. Loss下降趋势
```
初始loss: ~2.33
最终loss: ~2.12
总下降: ~0.21
```

**分析:**
- Loss有所下降,说明模型在学习
- 下降幅度较小,可能原因:
  - 学习率较小 (0.01)
  - 没有使用momentum
  - 训练数据较少 (100个样本)
  - 模型初始化的随机性

### 2. 准确率分析
```
训练数据准确率: ~12%
```

**分析:**
- 10分类随机猜测准确率为10%
- 当前准确率略高于随机
- 说明模型刚开始学习,还需要更多训练

### 3. 概率分布
```
各类别概率分布较为均匀 (9%-13%)
```

**分析:**
- 模型还未充分学习
- 对各类别的判断不够自信
- 需要更长时间训练或更好的超参数

## 🚀 改进建议

### 1. 提高学习效率
```python
# 添加momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9  # 添加动量
)
```

### 2. 增加训练数据
```python
num_samples = 1000  # 增加到1000个样本
```

### 3. 使用学习率调度
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)

# 训练循环中
for epoch in range(100):
    # ... 训练代码 ...
    scheduler.step()  # 调整学习率
```

### 4. 尝试其他优化器
```python
# 使用Adam优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

## 📝 运行方法

```bash
# 进入项目目录
cd /home/seeback/PycharmProjects/DeepLearning

# 运行完整演示
python src/experiments/torch_nn/CIFARSeeback.py

# 或只运行SGD训练部分
python -c "from src.experiments.torch_nn.CIFARSeeback import demo_sgd_training; demo_sgd_training()"
```

## 🎓 学习要点

### 1. 优化器的作用
- 接收梯度 (param.grad)
- 决定参数更新策略
- 目标: 让loss下降

### 2. 训练循环的核心步骤
```
前向传播 → 计算loss → 反向传播 → 更新参数
```

### 3. 必须记住的顺序
```python
optimizer.zero_grad()  # 1. 先清空
loss.backward()        # 2. 再计算
optimizer.step()       # 3. 最后更新
```

### 4. Loss打印的意义
- 观察训练趋势
- 判断是否收敛
- 发现异常情况 (如loss爆炸)

## 🔗 相关文件

- **主文件**: `src/experiments/torch_nn/CIFARSeeback.py`
- **优化器详解**: `src/experiments/torch_nn/OptimizerExplained.py`
- **反向传播详解**: `src/experiments/torch_nn/BackpropExplained.py`
- **速查表**: `docs/optimizer_cheatsheet.md`

---

**Happy Learning!** 🎉
