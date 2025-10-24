# 模型训练示例系列

这个目录包含一系列从简到繁的PyTorch模型训练示例,每个示例都专注于一个核心概念,循序渐进地展示完整的深度学习训练流程。

## 📚 示例概览

### 01_basic_training.py - 最基础的训练循环
**核心概念**: `backward()` 和 `step()` 优化过程

**包含内容**:
- ✅ 简单的3层CNN模型
- ✅ CIFAR-10数据加载
- ✅ 4步训练循环: `forward → zero_grad → backward → step`
- ✅ 每100批次打印loss

**运行**:
```bash
python src/experiments/model_train/01_basic_training.py
```

**学到什么**:
- 神经网络训练的基本流程
- 如何使用DataLoader批量加载数据
- 前向传播、反向传播和参数更新的关系

---

### 02_with_validation.py - 添加验证集
**核心概念**: 训练/验证划分,`model.train()` vs `model.eval()`

**在示例1基础上新增**:
- ✅ 训练集/验证集划分 (80%/20%)
- ✅ `model.train()` 和 `model.eval()` 模式切换
- ✅ 验证集准确率计算
- ✅ **TensorBoard 可视化**
- ✅ 过拟合检测

**运行**:
```bash
python src/experiments/model_train/02_with_validation.py
```

**TensorBoard**:
```bash
tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/02_with_validation
```

**学到什么**:
- 为什么需要验证集
- Dropout和BatchNorm在训练/评估模式下的不同行为
- 如何使用TensorBoard监控训练过程
- 如何检测模型是否过拟合

---

### 03_save_load_model.py - 模型保存和加载
**核心概念**: Checkpoint机制,训练恢复,模型部署

**在示例2基础上新增**:
- ✅ 保存checkpoint (模型+优化器+训练状态)
- ✅ 保存最佳模型
- ✅ 每5个epoch保存一个checkpoint
- ✅ 从checkpoint恢复训练
- ✅ 加载模型进行推理
- ✅ **TensorBoard 可视化**

**运行**:
```bash
python src/experiments/model_train/03_save_load_model.py
```

**TensorBoard**:
```bash
tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/03_save_load_model
```

**生成的文件**:
```
artifacts/checkpoints/03_save_load_model/
├── latest_checkpoint.pth      # 最新的checkpoint
├── best_model.pth             # 验证准确率最高的模型
└── checkpoint_epoch_X.pth     # 每5个epoch的快照
```

**学到什么**:
- 如何保存和加载PyTorch模型
- Checkpoint应该包含哪些信息
- 如何从中断处恢复训练
- 如何使用训练好的模型进行推理

---

### 04_lr_scheduler.py - 学习率调度
**核心概念**: 学习率衰减策略

**在示例3基础上新增**:
- ✅ **StepLR** - 每隔固定epoch降低学习率
- ✅ **CosineAnnealingLR** - 余弦退火
- ✅ **ReduceLROnPlateau** - 自适应降低学习率
- ✅ **ExponentialLR** - 指数衰减
- ✅ **MultiStepLR** - 多阶段降低学习率
- ✅ 学习率可视化
- ✅ 对比不同调度器效果
- ✅ **TensorBoard 可视化学习率变化**

**运行**:
```bash
python src/experiments/model_train/04_lr_scheduler.py
```

**TensorBoard**:
```bash
tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/04_lr_scheduler
```

**学到什么**:
- 为什么需要调整学习率
- 不同学习率调度策略的特点和适用场景
- 如何选择合适的学习率调度器
- 学习率对训练的影响

**调度器对比**:
| 调度器 | 特点 | 适用场景 |
|--------|------|----------|
| StepLR | 固定步长 | 基础训练 |
| CosineAnnealingLR | 平滑衰减 | Fine-tuning |
| ReduceLROnPlateau | 自适应 | 不确定训练轮数 |
| ExponentialLR | 指数衰减 | 可控衰减速度 |
| MultiStepLR | 灵活 | 自定义降低时机 |

---

### 05_data_augmentation.py - 数据增强
**核心概念**: 数据增强技术,防止过拟合

**在示例4基础上新增**:
- ✅ **随机水平翻转** (50%概率)
- ✅ **随机裁剪** (padding=4)
- ✅ **随机旋转** (±15°)
- ✅ **颜色抖动** (亮度/对比度/饱和度/色调)
- ✅ **随机擦除** (模拟遮挡,30%概率)
- ✅ **Dropout** (FC层dropout=0.5)
- ✅ 可视化增强效果
- ✅ 对比有无增强的训练效果
- ✅ 过拟合检测和分析
- ✅ **TensorBoard 可视化**

**运行**:
```bash
python src/experiments/model_train/05_data_augmentation.py
```

**TensorBoard**:
```bash
tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/05_data_augmentation
```

**生成的可视化**:
- `artifacts/data_augmentation_demo.png` - 数据增强效果展示

**学到什么**:
- 常用的数据增强技术
- 数据增强如何提高模型泛化能力
- 如何缓解过拟合
- 何时应该使用数据增强

**数据增强技术分类**:
```
1. 几何变换:
   - RandomHorizontalFlip (水平翻转)
   - RandomRotation (旋转)
   - RandomCrop (裁剪)

2. 颜色变换:
   - ColorJitter (颜色抖动)
   - RandomGrayscale (灰度化)

3. 遮挡:
   - RandomErasing (随机擦除)

4. 正则化:
   - Dropout (丢弃神经元)
```

---

### 06_detailed_evaluation.py - 详细的模型评估与分析 ⭐新增
**核心概念**: `argmax`、混淆矩阵、分类指标、模型诊断

**全新内容**:
- ✅ **argmax详细解释** (1维、2维、Batch的使用示例)
- ✅ **torch.max vs torch.argmax** 对比
- ✅ **测试集完整评估** (准确率、loss)
- ✅ **每个类别的指标** (精确率、召回率、F1分数)
- ✅ **混淆矩阵可视化** (找出易混淆的类别对)
- ✅ **Top-K准确率** (Top-1, Top-3, Top-5)
- ✅ **预测置信度分析** (正确/错误预测的置信度分布)
- ✅ **错误样本可视化** (展示16个错误预测样本)

**运行**:
```bash
python src/experiments/model_train/06_detailed_evaluation.py
```

**生成的可视化**:
- `artifacts/confusion_matrix.png` - 混淆矩阵热力图
- `artifacts/confidence_analysis.png` - 置信度分布图
- `artifacts/error_samples.png` - 错误预测样本

**学到什么**:
- **argmax的原理和使用方法** (这是你要求的重点!)
- 如何解读神经网络的输出
- 如何全面评估分类模型
- 精确率、召回率、F1分数的含义
- 如何诊断模型的弱点
- 如何分析预测置信度
- 如何找出模型容易混淆的类别

**评估指标详解**:
```
1. 准确率 (Accuracy):
   所有预测中正确的比例
   Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. 精确率 (Precision):
   预测为该类的样本中,真正属于该类的比例
   Precision = TP / (TP + FP)

   示例: 预测为猫的100张图中,真正是猫的有90张
   精确率 = 90/100 = 90%

3. 召回率 (Recall):
   真正属于该类的样本中,被正确预测的比例
   Recall = TP / (TP + FN)

   示例: 真正是猫的100张图中,正确识别出90张
   召回率 = 90/100 = 90%

4. F1分数:
   精确率和召回率的调和平均数
   F1 = 2 * (Precision * Recall) / (Precision + Recall)

5. Top-K准确率:
   预测概率前K的类别中包含正确类别的比例
   Top-1: 最高概率就是正确类别
   Top-5: 前5个最高概率中包含正确类别
```

**argmax使用示例** (重点!):
```python
# ============================================
# 示例1: 一维数组
# ============================================
import numpy as np

scores = np.array([0.1, 0.3, 0.8, 0.2, 0.5])
max_idx = np.argmax(scores)  # 返回 2 (最大值0.8的索引)

# ============================================
# 示例2: 神经网络输出 (CIFAR-10)
# ============================================
# 假设网络输出10个类别的分数 (logits)
output = np.array([2.1, -0.5, 1.8, 0.3, -1.2, 0.8, 3.5, 1.1, 0.2, -0.8])
#                  飞机  汽车   鸟   猫   鹿   狗   青蛙  马   船   卡车

# 使用argmax找出分数最高的类别
pred_class = np.argmax(output)  # 返回 6 (青蛙的分数3.5最高)

# ============================================
# 示例3: Batch预测 (多个样本)
# ============================================
# 假设有3个样本,每个样本10个类别
batch_output = np.array([
    [2.1, -0.5, 1.8, 0.3, -1.2, 0.8, 3.5, 1.1, 0.2, -0.8],  # 样本1
    [1.2, 2.8, 0.3, 0.5, 1.1, 0.2, 0.7, 0.9, 1.5, 0.4],      # 样本2
    [0.1, 0.2, 3.2, 1.1, 0.8, 1.5, 0.9, 1.2, 0.5, 0.3],      # 样本3
])

# axis=1 表示在类别维度(第1维)上找最大值
batch_preds = np.argmax(batch_output, axis=1)
# 返回 [6, 1, 2] (样本1预测青蛙,样本2预测汽车,样本3预测鸟)

# ============================================
# PyTorch版本
# ============================================
import torch

# 方法1: 使用argmax
output_tensor = torch.tensor(output)
pred = torch.argmax(output_tensor)  # 返回 6

# 方法2: 使用max (推荐,因为同时返回最大值)
max_value, pred = torch.max(output_tensor, dim=0)
# max_value = 3.5, pred = 6

# Batch版本
batch_tensor = torch.tensor(batch_output)
batch_preds = torch.argmax(batch_tensor, dim=1)  # [6, 1, 2]

# 或者
max_values, batch_preds = torch.max(batch_tensor, dim=1)
# max_values: [3.5, 2.8, 3.2]
# batch_preds: [6, 1, 2]
```

**为什么需要argmax?**
```
神经网络的输出是每个类别的"分数" (logits),不是最终预测。
我们需要找出分数最高的类别作为预测结果。

例如:
  输入: 一张猫的图片
  网络输出: [飞机:0.2, 汽车:0.1, ..., 猫:2.8, ..., 狗:1.5]
  argmax: 找出2.8对应的索引 -> 类别3 (猫)
  最终预测: 猫 ✅
```

---

## 🎯 学习路径

建议按顺序学习这6个示例:

```
01_basic_training.py
    ↓ 理解基本训练流程
02_with_validation.py
    ↓ 掌握验证和评估
03_save_load_model.py
    ↓ 学会保存和加载模型
04_lr_scheduler.py
    ↓ 优化训练策略
05_data_augmentation.py
    ↓ 提高模型性能
06_detailed_evaluation.py ⭐新增
    ↓ 深入理解模型评估
```

---

## 📊 TensorBoard 使用

所有示例(从02开始)都集成了TensorBoard可视化。

### 启动TensorBoard:
```bash
# 查看所有实验
tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board

# 查看特定实验
tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/02_with_validation
```

### 浏览器访问:
打开浏览器访问 `http://localhost:6006`

### 可视化内容:
- 📈 训练/验证Loss曲线
- 📈 训练/验证准确率曲线
- 📈 学习率变化曲线
- 📈 过拟合指标 (训练-验证准确率差值)

---

## 🛠️ 技术栈

- **框架**: PyTorch
- **数据集**: CIFAR-10
- **可视化**: TensorBoard, Matplotlib
- **模型**: SimpleCNN (3层卷积 + 2层全连接)

---

## 📦 项目结构

```
src/experiments/model_train/
├── 01_basic_training.py          # 示例1: 最基础的训练循环
├── 02_with_validation.py         # 示例2: 添加验证集
├── 03_save_load_model.py         # 示例3: 模型保存和加载
├── 04_lr_scheduler.py            # 示例4: 学习率调度
├── 05_data_augmentation.py       # 示例5: 数据增强
├── 06_detailed_evaluation.py     # 示例6: 详细的模型评估 ⭐新增
└── README.md                     # 本文件

artifacts/
├── checkpoints/                  # 模型checkpoint
│   ├── 03_save_load_model/
│   │   ├── latest_checkpoint.pth
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_X.pth
│   └── best_model_eval.pth      # 示例6训练的模型
├── confusion_matrix.png         # 混淆矩阵 ⭐新增
├── confidence_analysis.png      # 置信度分析 ⭐新增
├── error_samples.png            # 错误样本 ⭐新增
└── data_augmentation_demo.png   # 数据增强可视化

/home/seeback/PycharmProjects/DeepLearning/tensor_board/
├── 02_with_validation/           # 示例2的日志
├── 03_save_load_model/           # 示例3的日志
├── 04_lr_scheduler/              # 示例4的日志
│   ├── StepLR/
│   ├── CosineAnnealingLR/
│   └── ReduceLROnPlateau/
└── 05_data_augmentation/         # 示例5的日志
    ├── without_augmentation/
    └── with_augmentation/
```

---

## 💡 核心概念对照表

| 示例 | 核心概念 | 关键代码 | 效果 |
|------|----------|----------|------|
| 01 | 训练循环 | `backward()`, `step()` | 理解基本流程 |
| 02 | 验证集 | `model.eval()`, `torch.no_grad()` | 评估泛化能力 |
| 03 | Checkpoint | `torch.save()`, `torch.load()` | 保存训练成果 |
| 04 | 学习率调度 | `scheduler.step()` | 提升收敛速度 |
| 05 | 数据增强 | `transforms.RandomXXX()` | 防止过拟合 |
| 06 | 模型评估 | `argmax()`, `confusion_matrix()` | 深入理解模型 |

---

## 🎓 常见问题

### Q1: 为什么验证集不使用数据增强?
**A**: 验证集用于评估模型的真实性能。数据增强会改变数据分布,导致验证结果不准确。

### Q2: 什么时候应该保存checkpoint?
**A**:
- 每个epoch结束时保存最新checkpoint
- 验证准确率提升时保存最佳模型
- 每N个epoch保存一个快照(便于回滚)

### Q3: 如何选择学习率调度器?
**A**:
- **StepLR**: 训练轮数固定且已知
- **CosineAnnealingLR**: Fine-tuning预训练模型
- **ReduceLROnPlateau**: 训练轮数不确定,根据loss自适应

### Q4: 数据增强会增加训练时间吗?
**A**: 会,但增加幅度不大(通常10-20%)。收益(准确率提升+过拟合缓解)远大于成本。

### Q5: 如何判断模型过拟合?
**A**:
- 训练准确率远高于验证准确率 (差距>10%)
- 验证loss不再下降,甚至上升
- 训练loss持续下降,验证loss停滞

---

## 📝 最佳实践总结

1. **始终使用验证集** - 不要只看训练loss
2. **定期保存checkpoint** - 防止训练中断导致的损失
3. **使用学习率调度** - 加速收敛,提高最终性能
4. **应用数据增强** - 特别是小数据集
5. **监控TensorBoard** - 及时发现训练问题
6. **记录超参数** - 在checkpoint中保存所有配置
7. **固定随机种子** - 确保实验可复现

---

## 🚀 下一步学习

完成这5个示例后,可以继续学习:

- **混合精度训练** (AMP) - 加速训练,减少显存
- **分布式训练** (DDP) - 多GPU训练
- **早停** (Early Stopping) - 自动停止训练
- **梯度裁剪** (Gradient Clipping) - 防止梯度爆炸
- **迁移学习** (Transfer Learning) - 使用预训练模型
- **自定义Dataset** - 处理自己的数据

---

## 📧 反馈与贡献

如果你有任何问题或建议,欢迎提issue或PR!

---

**作者**: Seeback
**日期**: 2025-10-23
**版本**: v1.0
