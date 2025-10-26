# 学习路径

深度学习学习代码的推荐学习顺序。

## 基础入门

### 1. PyTorch基础

**学习内容**:
- `src/experiments/torch_nn/TensorTest.py` - Tensor操作
- `src/experiments/torch_nn/DatasetFormatsGuide.py` - 数据格式
- `src/experiments/torch_nn/TorchvisionDatasetsGuide.py` - 数据集使用
- `CPU_GPU_GUIDE.md` - CPU/GPU使用
- `src/experiments/model_train/understanding_device.py` - 设备管理

### 2. 神经网络组件

**学习内容**:
- `src/experiments/torch_nn/LossTest.py` - 损失函数
- `src/experiments/torch_nn/OptimizerExplained.py` - 优化器
- `src/experiments/torch_nn/BackpropExplained.py` - 反向传播
- `docs/optimizer_cheatsheet.md` - 优化器参考

### 3. 网络层

**学习内容**:
- `src/experiments/torch_nn/LinnerTest.py` - 全连接层
- `src/experiments/torch_nn/Nonlinear.py` - 激活函数
- `src/experiments/torch_nn/Conv2dTest.py` - 卷积层
- `src/experiments/torch_nn/PoolTest.py` - 池化层

### 4. 完整训练流程

按顺序学习 `src/experiments/model_train/`:

1. `01_basic_training.py` - 基础训练循环
2. `02_with_validation.py` - 验证集使用
3. `03_save_load_model.py` - 模型保存加载
4. `04_lr_scheduler.py` - 学习率调度
5. `05_data_augmentation.py` - 数据增强
6. `06_detailed_evaluation.py` - 详细评估
7. `07_cpu_gpu_training.py` - CPU/GPU对比

---

## 进阶学习

### 预训练模型

**学习内容**:
- `src/experiments/torch_nn/TorchvisionModelsGuide.py` - 预训练模型使用
- `src/experiments/torch_nn/ModelSaveLoadGuide.py` - 模型保存

### Transformer

**学习内容**:
- `src/experiments/transformer/` - Transformer相关代码

### 高级主题

**学习内容**:
- `docs/source_code_explain/` - 源码解析
- 模型压缩、剪枝、量化等

---

## 推荐顺序

**初学者**:
1. 基础入门 (1-4) 按顺序学习
2. 进阶学习 根据需要选择

**有基础**:
- 直接从 "完整训练流程" 开始
- 查缺补漏其他内容

**查找特定内容**:
- 查看 `src/experiments/torch_nn/README.md` 了解15个基础脚本
- 查看 `docs/README.md` 了解文档索引

---

## 配套资源

| 类型 | 位置 | 说明 |
|-----|------|------|
| 基础组件 | `src/experiments/torch_nn/` | 15个脚本 |
| 训练流程 | `src/experiments/model_train/` | 7个示例 |
| 交互笔记本 | `notebooks/` | Jupyter笔记本 |
| 文档 | `docs/` | 各类文档 |

---

## 学习建议

1. 先看理论代码(`torch_nn/`)理解原理
2. 再跑完整示例(`model_train/`)看效果
3. 在Jupyter中动手实验
4. 修改代码参数观察变化
