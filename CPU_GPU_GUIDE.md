# CPU/GPU 训练详解

## 📋 你当前的环境

根据检测结果:
- ✅ **PyTorch版本**: 2.5.1
- ⚠️ **CUDA可用**: False (没有GPU)
- 💻 **当前设备**: CPU
- 🔧 **CPU线程数**: 8

## ❓ 常见问题解答

### Q1: 我现在是用CPU还是GPU训练?
**A**: 你目前使用的是 **CPU** 训练。

之前的代码中有 `images.cuda()` 和 `labels.cuda()`,但因为你的环境没有GPU,这些代码会报错。我已经修复了所有示例,现在会自动检测并使用CPU。

### Q2: 没有GPU可以训练吗?
**A**: **完全可以!** CPU训练没有任何问题,只是速度会慢一些。

- CPU训练: 完全可行,适合学习和小规模实验
- GPU训练: 速度快10-100倍,适合大规模训练

### Q3: CPU和GPU的区别是什么?
**A**:
| 对比项 | CPU | GPU |
|--------|-----|-----|
| **速度** | 慢 (基准) | 快 (10-100倍) |
| **适合任务** | 小模型、小数据集 | 大模型、大数据集 |
| **成本** | 免费 (本地) | 需要硬件/云服务 |
| **学习** | ✅ 完全够用 | 锦上添花 |

### Q4: 如何让代码同时支持CPU和GPU?
**A**: 使用以下模式(已经在所有示例中修复):

```python
# ✅ 正确做法 (自动适配)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    # ... 训练代码
```

```python
# ❌ 错误做法 (固定使用GPU,没有GPU会报错)
model = model.cuda()  # RuntimeError if no GPU

for images, labels in train_loader:
    images = images.cuda()  # RuntimeError if no GPU
    # ... 训练代码
```

## 🔧 已修复的问题

### 修复前 (会报错):
```python
# 01_basic_training.py (旧版)
images = images.cuda()  # ❌ 在你的环境会报错
labels = labels.cuda()  # ❌ RuntimeError: CUDA not available
```

### 修复后 (自动适配):
```python
# 01_basic_training.py (新版)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

images = images.to(device)  # ✅ 自动使用CPU
labels = labels.to(device)  # ✅ 自动使用CPU
```

## 💡 CPU训练优化建议

由于你使用CPU训练,可以通过以下方式加速:

### 1. 减小batch size
```python
# GPU可以用大batch
train_loader = DataLoader(dataset, batch_size=128)

# CPU建议用小batch
train_loader = DataLoader(dataset, batch_size=32)  # 更快
```

### 2. 增加DataLoader的worker数量
```python
# 利用你的8个CPU线程
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4  # 使用4个进程并行加载数据
)
```

### 3. 减少训练数据量 (学习时)
```python
# 使用部分数据快速实验
from torch.utils.data import Subset

# 只用前1000个样本
train_subset = Subset(train_dataset, range(1000))
train_loader = DataLoader(train_subset, batch_size=32)
```

### 4. 减少epoch数量
```python
# GPU训练
train(model, ..., num_epochs=50)  # 可以训练很多轮

# CPU训练
train(model, ..., num_epochs=10)  # 先用少量epoch测试
```

## 📊 速度对比示例

在示例7中,你可以看到CPU的实际训练速度:

```bash
python src/experiments/model_train/07_cpu_gpu_training.py
```

典型的速度对比:
- **CPU**: 1个epoch约30-60秒 (取决于数据量和模型大小)
- **GPU**: 1个epoch约3-5秒 (快10-20倍)

## 🚀 如何获得GPU

如果将来想用GPU加速训练,有几个选择:

### 免费选项:
1. **Google Colab** (推荐)
   - 免费提供GPU (每天有限额)
   - 直接在浏览器运行
   - 网址: https://colab.research.google.com

2. **Kaggle Notebooks**
   - 免费GPU
   - 每周30小时
   - 网址: https://www.kaggle.com/code

### 付费选项:
1. **云服务器**
   - 阿里云、腾讯云、AWS、Azure
   - 按小时计费
   - 适合大规模训练

2. **本地GPU**
   - 购买NVIDIA显卡 (如RTX 3060, 4060等)
   - 一次性投资
   - 适合长期使用

## ✅ 总结

**你当前的情况:**
- ✅ 没有GPU,使用CPU训练
- ✅ 所有示例已修复,可以正常运行
- ✅ CPU训练完全够用,只是慢一些
- ✅ 已经提供了CPU优化建议

**建议:**
1. 先用CPU学习基础知识 (完全够用!)
2. 需要时使用Google Colab获得免费GPU
3. 代码已经自动适配,无需修改即可在GPU上运行

**运行示例:**
```bash
# 查看详细的CPU/GPU说明
python src/experiments/model_train/07_cpu_gpu_training.py

# 运行其他示例 (已修复,自动使用CPU)
python src/experiments/model_train/01_basic_training.py
python src/experiments/model_train/02_with_validation.py
# ... 等等
```

所有代码现在都会自动检测设备并使用CPU训练,不会报错! 🎉
