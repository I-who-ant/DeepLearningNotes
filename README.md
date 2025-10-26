# 深度学习学习代码

个人深度学习学习代码仓库,用于理解PyTorch基础和常见的模型训练方法。

## 项目说明

这是一个自用的深度学习学习代码库,包含:
- PyTorch核心组件的学习代码
- 常见模型训练流程的实现
- 一些基础的深度学习概念验证

## 快速开始

```bash
# 安装依赖
pip install torch torchvision tensorboard matplotlib numpy

# 运行训练示例
cd src/experiments/model_train
python 01_basic_training.py

# 查看训练过程(可选)
tensorboard --logdir=tensor_board/
```

## 目录结构

```
DeepLearning/
├── src/
│   ├── core/              # 工具函数
│   ├── datasets/          # 数据集相关
│   └── experiments/
│       ├── torch_nn/      # PyTorch基础组件(15个)
│       ├── model_train/   # 训练流程示例(7个)
│       ├── transformer/   # Transformer相关
│       └── prototyping/   # 其他实验代码
│
├── notebooks/             # Jupyter笔记本
├── docs/                  # 文档说明
└── LEARNING_PATH.md       # 学习顺序建议
```

## 主要内容

### PyTorch基础 (`src/experiments/torch_nn/`)

15个脚本,涵盖:
- Tensor操作、损失函数、优化器
- 卷积层、池化层、全连接层
- 数据集处理、预训练模型等

查看 `src/experiments/torch_nn/README.md` 了解详情。

### 训练流程 (`src/experiments/model_train/`)

7个完整的训练示例:
1. 基础训练循环
2. 验证集使用
3. 模型保存/加载
4. 学习率调度
5. 数据增强
6. 详细评估
7. CPU/GPU训练

查看 `src/experiments/model_train/README.md` 了解详情。

## 学习建议

- 先看 `torch_nn/` 理解基础概念
- 再跑 `model_train/` 完整训练流程
- 查看 `LEARNING_PATH.md` 了解推荐学习顺序

## 环境说明

- Python 3.8+
- PyTorch 2.0+
- 支持CPU训练(无GPU也可运行)

## 数据集

使用MNIST和CIFAR-10数据集(代码会自动下载)。

---

**注**: 这是个人学习项目,代码和注释仅供参考。
