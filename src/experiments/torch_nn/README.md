# PyTorch核心组件详解 🔥

本目录包含PyTorch核心组件的详细讲解和代码示例,适合系统学习深度学习基础知识。

## 📚 学习顺序

建议按照以下顺序学习(文件名前的编号表示推荐学习顺序):

### 🟢 第一阶段:基础概念 (必学)

1. **数据集格式详解** → `DatasetFormatsGuide.py`
   - 常见数据格式:MNIST ubyte、CIFAR pickle、Arrow、HDF5等
   - 格式对比与选择指南
   - **学习时间**: 30分钟

2. **Torchvision数据集使用** → `TorchvisionDatasetsGuide.py`
   - 内置数据集下载与使用
   - MNIST、CIFAR-10、ImageNet等
   - **学习时间**: 20分钟

3. **原始数据集读取** → `RawDatasetReader.py`
   - 直接读取ubyte、pickle等原始格式
   - 理解数据集内部结构
   - **学习时间**: 30分钟

### 🟡 第二阶段:神经网络组件 (核心)

4. **损失函数详解** → `LossTest.py`
   - 交叉熵损失、MSE、L1Loss等
   - 分类与回归任务的损失选择
   - **学习时间**: 40分钟
   - **前置知识**: 基础数学(概率、导数)

5. **优化器详解** → `OptimizerExplained.py`
   - SGD、Adam、AdamW等优化器原理
   - 学习率、动量、权重衰减
   - **学习时间**: 45分钟
   - **前置知识**: 损失函数

6. **反向传播机制** → `BackpropExplained.py`
   - 自动求导原理
   - backward()和step()详解
   - **学习时间**: 50分钟
   - **前置知识**: 优化器

7. **Logits详解** → `LogitsExplained.py`
   - Logits vs Probabilities
   - Softmax的作用
   - **学习时间**: 25分钟
   - **前置知识**: 损失函数

### 🔵 第三阶段:网络层组件 (实战)

8. **全连接层** → `LinnerTest.py`
   - nn.Linear原理与使用
   - 权重初始化
   - **学习时间**: 30分钟
   - **前置知识**: 反向传播

9. **激活函数** → `Nonlinear.py`
   - ReLU、Sigmoid、Tanh等
   - 激活函数的选择
   - **学习时间**: 30分钟
   - **前置知识**: 全连接层

10. **卷积层详解** → `Conv2dTest.py`
    - 卷积核、步长、填充
    - 特征图计算公式
    - **学习时间**: 60分钟
    - **前置知识**: 全连接层、激活函数

11. **池化层详解** → `PoolTest.py`
    - MaxPool、AvgPool原理
    - 池化层的作用
    - **学习时间**: 30分钟
    - **前置知识**: 卷积层

### 🟣 第四阶段:高级主题 (进阶)

12. **距离度量** → `DistanceTest.py`
    - 欧氏距离、余弦距离等
    - 相似度计算
    - **学习时间**: 25分钟
    - **前置知识**: 基础数学

13. **预训练模型使用** → `TorchvisionModelsGuide.py`
    - ResNet、VGG等经典模型
    - 迁移学习与微调
    - **学习时间**: 40分钟
    - **前置知识**: 卷积层、池化层

14. **模型保存与加载** → `ModelSaveLoadGuide.py`
    - state_dict vs 完整模型
    - checkpoint机制
    - **学习时间**: 35分钟
    - **前置知识**: 基础网络训练

15. **CIFAR完整示例** → `CIFARSeeback.py`
    - CIFAR-10分类完整流程
    - 综合应用前面所学
    - **学习时间**: 60分钟
    - **前置知识**: 前面所有内容

---

## 🎯 快速索引

### 按主题分类

#### 数据处理
- `DatasetFormatsGuide.py` - 数据格式详解
- `TorchvisionDatasetsGuide.py` - 内置数据集
- `RawDatasetReader.py` - 原始数据读取

#### 训练核心
- `LossTest.py` - 损失函数
- `OptimizerExplained.py` - 优化器
- `BackpropExplained.py` - 反向传播
- `LogitsExplained.py` - Logits概念

#### 网络层
- `LinnerTest.py` - 全连接层
- `Nonlinear.py` - 激活函数
- `Conv2dTest.py` - 卷积层
- `PoolTest.py` - 池化层

#### 高级应用
- `DistanceTest.py` - 距离度量
- `TorchvisionModelsGuide.py` - 预训练模型
- `ModelSaveLoadGuide.py` - 模型保存
- `CIFARSeeback.py` - 完整示例

---

## 📖 学习建议

### 初学者路径 (4-6周)
1. 先学习数据处理(1-3)
2. 再学习训练核心(4-7)
3. 然后学习网络层(8-11)
4. 最后学习高级主题(12-15)

### 有基础路径 (2-3周)
1. 快速浏览数据处理(1-3)
2. 重点学习训练核心和网络层(4-11)
3. 深入高级主题(12-15)

### 查漏补缺路径
- 根据快速索引选择需要补充的主题
- 每个文件都可以独立学习

---

## 💡 使用技巧

### 如何运行示例

```bash
# 进入目录
cd src/experiments/torch_nn

# 运行任意脚本
python LossTest.py
python Conv2dTest.py
# ...
```

### 如何调试

1. **使用print调试**:所有脚本都有详细的print输出
2. **修改参数观察变化**:尝试修改代码中的参数
3. **结合文档学习**:参考PyTorch官方文档

### 常见问题

**Q: 文件太多,不知道从哪里开始?**
- 按照"学习顺序"从1开始学习

**Q: 某个概念不理解怎么办?**
- 先学习前置知识
- 运行代码观察输出
- 查看代码注释

**Q: 如何验证自己掌握了?**
- 能够独立运行并理解输出
- 尝试修改代码实现变种
- 在实际项目中应用

---

## 🔗 相关资源

### 完整训练示例
学完本目录后,建议学习:
- `src/experiments/model_train/` - 7个完整训练流程示例
- 从基础训练循环到高级评估技巧

### 交互式笔记本
- `notebooks/` - Jupyter交互式学习
- 可以边学边实验

### 项目文档
- `/README.md` - 项目总览
- `CPU_GPU_GUIDE.md` - CPU/GPU使用指南

---

## 📊 学习进度跟踪

建议使用以下方式跟踪学习进度:

- [ ] 第一阶段:基础概念 (3个文件)
- [ ] 第二阶段:神经网络组件 (4个文件)
- [ ] 第三阶段:网络层组件 (4个文件)
- [ ] 第四阶段:高级主题 (4个文件)

---

⭐ **提示**: 每个Python文件都包含详细的中文注释和示例代码,可以直接运行学习!
