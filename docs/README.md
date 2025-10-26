# 深度学习文档中心 📚

欢迎来到深度学习项目文档中心!这里汇总了所有学习资源和参考文档。

## 📋 文档导航

### 🚀 快速入门

1. **项目总览** → [/README.md](../README.md)
   - 项目介绍、学习路径、快速开始
   - **适合**: 新用户第一次接触项目

2. **仓库阅读指南** → [仓库阅读指南.md](仓库阅读指南.md)
   - 仓库结构说明
   - 文件组织逻辑
   - **适合**: 想深入了解项目结构的用户

### 📖 教程文档 (Tutorials)

循序渐进的完整教程,适合系统学习:

#### 模型训练教程
- **模型构建教程** → [model_building_tutorial.md](model_building_tutorial.md)
  - 从零构建神经网络模型
  - 模型设计最佳实践
  - **学习时间**: 45分钟

- **SGD训练演示** → [sgd_training_demo_readme.md](sgd_training_demo_readme.md)
  - SGD优化器完整训练流程
  - 可视化训练过程
  - **学习时间**: 30分钟

### 📚 参考指南 (Guides)

快速查阅的参考资料:

- **优化器速查表** → [optimizer_cheatsheet.md](optimizer_cheatsheet.md)
  - SGD、Adam、AdamW等优化器对比
  - 参数选择建议
  - **适合**: 快速查阅优化器配置

### 🔍 源码解析 (Source Code Explain)

深入理解PyTorch内部实现:

- **Transforms详解** → [source_code_explain/transforms.md](source_code_explain/transforms.md)
  - torchvision.transforms源码解析
  - 数据增强原理
  - **适合**: 想深入理解数据预处理的用户

### 💡 概念详解 (Concepts)

核心概念的深度讲解:

- **CPU/GPU设备管理** → [../CPU_GPU_GUIDE.md](../CPU_GPU_GUIDE.md)
  - CPU vs GPU训练对比
  - 设备选择与数据转移
  - **适合**: 理解设备管理机制

- **设备管理深入理解** → [../src/experiments/model_train/understanding_device.py](../src/experiments/model_train/understanding_device.py)
  - torch.device()详解
  - 多GPU使用
  - **适合**: 深入理解设备概念

- **GPU加速原理** → [../src/experiments/model_train/why_gpu_faster.py](../src/experiments/model_train/why_gpu_faster.py)
  - 为什么GPU比CPU快100-1000倍
  - 并行计算架构
  - **适合**: 理解GPU加速原理

---

## 🗂️ 按学习阶段分类

### 🟢 初学者 (刚开始学习)

**必读文档**:
1. [/README.md](../README.md) - 项目总览
2. [仓库阅读指南.md](仓库阅读指南.md) - 项目结构
3. [model_building_tutorial.md](model_building_tutorial.md) - 模型构建
4. [sgd_training_demo_readme.md](sgd_training_demo_readme.md) - 训练流程

**配套代码**:
- `src/experiments/torch_nn/` - 核心组件学习
- `src/experiments/model_train/01-03*.py` - 基础训练

### 🟡 进阶学习 (有基础)

**推荐文档**:
1. [optimizer_cheatsheet.md](optimizer_cheatsheet.md) - 优化器选择
2. [source_code_explain/transforms.md](source_code_explain/transforms.md) - 数据增强
3. [CPU_GPU_GUIDE.md](../CPU_GPU_GUIDE.md) - 设备管理

**配套代码**:
- `src/experiments/model_train/04-07*.py` - 高级训练技巧
- `notebooks/` - 交互式实验

### 🔵 高级应用 (深入研究)

**深度文档**:
1. 源码解析系列 - `source_code_explain/`
2. 设备管理机制 - `understanding_device.py`
3. GPU加速原理 - `why_gpu_faster.py`

**配套代码**:
- `src/experiments/transformer/` - Transformer实现
- 自定义项目开发

---

## 🔧 按功能分类

### 数据处理
- [source_code_explain/transforms.md](source_code_explain/transforms.md) - 数据增强
- `src/experiments/torch_nn/DatasetFormatsGuide.py` - 数据格式
- `src/experiments/torch_nn/TorchvisionDatasetsGuide.py` - 内置数据集

### 模型构建
- [model_building_tutorial.md](model_building_tutorial.md) - 模型构建教程
- `src/experiments/torch_nn/Conv2dTest.py` - 卷积层
- `src/experiments/torch_nn/LinnerTest.py` - 全连接层

### 训练优化
- [optimizer_cheatsheet.md](optimizer_cheatsheet.md) - 优化器速查
- [sgd_training_demo_readme.md](sgd_training_demo_readme.md) - SGD训练
- `src/experiments/model_train/04_lr_scheduler.py` - 学习率调度

### 模型评估
- `src/experiments/model_train/06_detailed_evaluation.py` - 详细评估
- `src/experiments/model_train/03_save_load_model.py` - 模型保存

### 硬件与性能
- [CPU_GPU_GUIDE.md](../CPU_GPU_GUIDE.md) - CPU/GPU使用
- `understanding_device.py` - 设备管理
- `why_gpu_faster.py` - GPU加速原理

---

## 📊 学习路径推荐

### 路径一:完整系统学习 (4-6周)

```
第1周: 项目总览 → 仓库阅读指南 → torch_nn基础组件(1-7)
第2周: torch_nn网络层(8-11) → 模型构建教程
第3周: model_train训练示例(01-04)
第4周: model_train高级技巧(05-07) → SGD训练演示
第5周: 优化器速查表 → Transforms详解 → 数据增强实践
第6周: 设备管理 → GPU原理 → 综合项目实战
```

### 路径二:快速上手 (1-2周)

```
第1周: 项目总览 → 模型构建教程 → model_train(01-03)
第2周: 优化器速查表 → model_train(04-07) → 实战项目
```

### 路径三:查漏补缺 (自定义)

根据"按功能分类"选择需要学习的主题,每个文档都可以独立学习。

---

## 🆕 最近更新

- **2024-10**: 添加详细评估指标教程 (06_detailed_evaluation.py)
- **2024-10**: 添加CPU/GPU训练对比 (07_cpu_gpu_training.py)
- **2024-10**: 完善优化器速查表
- **2024-10**: 添加SGD训练可视化演示

---

## 📝 文档规范

### 文档组织结构

```
docs/
├── README.md                    # 本文件,文档索引
├── 仓库阅读指南.md              # 项目结构说明
├── tutorials/                   # 教程文档(循序渐进)
│   ├── model_building_tutorial.md
│   └── sgd_training_demo_readme.md
├── guides/                      # 参考指南(快速查阅)
│   └── optimizer_cheatsheet.md
├── concepts/                    # 概念详解(深度理解)
│   └── (CPU/GPU相关文档建议移至此处)
├── source_code_explain/         # 源码解析
│   └── transforms.md
└── archive/                     # 归档文档
```

### 如何使用文档

1. **首次学习**: 从"快速入门"开始
2. **系统学习**: 按"学习路径推荐"进行
3. **查找资料**: 使用"按功能分类"快速定位
4. **深入研究**: 参考"源码解析"系列

---

## 🤝 贡献文档

欢迎贡献新的文档或改进现有文档!

### 文档编写规范

- 使用中文编写
- 包含代码示例
- 标注学习时间
- 说明前置知识
- 添加实际应用场景

### 提交流程

1. 在相应目录创建文档
2. 更新本README.md的索引
3. 提交Pull Request

---

## 🔗 外部资源

### 官方文档
- [PyTorch官方文档](https://pytorch.org/docs/)
- [Torchvision文档](https://pytorch.org/vision/stable/index.html)

### 推荐教程
- PyTorch官方教程
- fast.ai课程
- Deep Learning Book

---

⭐ **提示**: 所有文档都配有对应的代码示例,建议边读文档边运行代码!

📧 有问题或建议?欢迎在Issue中提出!
