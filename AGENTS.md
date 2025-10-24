# AGENTS 深度学习方法学习规范

## 1. 仓库定位
- 面向自学或团队研修深度学习核心方法的工程化知识库。
- 聚焦 PyTorch 生态（尤其是 `torch.nn`、`torch.optim`、`torch.utils` 等），通过最小可运行示例理解每个组件的行为、约束与适用场景。
- 目标是建立一套可复用的实验脚手架与学习流程，帮助使用者快速从概念到实现、从 API 阅读到实战调优。

## 2. 核心指导原则
- **KISS**：示例与脚手架保持最小化，明确输入输出、删去无关代码。
- **YAGNI**：仅在验证当前方法时引入依赖或扩展，不为未来可能用途预留复杂接口。
- **DRY**：公共工具（日志、计时、可视化）集中在 `utils/` 下，避免在不同示例中重复实现。
- **SOLID**：
  - **S**：每份示例脚本专注展示一种方法或概念。
  - **O**：通过配置或策略模式扩展新实验，而非直接修改旧示例。
  - **L**：自定义模块应可替换原生 `nn.Module`，保持 forward 接口一致。
  - **I**：针对训练、评估、可视化分别提供精简接口，避免“胖控制器”。
  - **D**：高层实验脚本依赖抽象（如数据加载器接口、模型工厂），便于切换实现。

## 3. 学习路径设计
1. **概念梳理**：阅读官方文档与源码注释，记录关键参数、输入输出形状、默认行为。
2. **最小示例**：构建仅包含该方法的 Notebook/脚本，验证 forward/backward 是否符合期望。
3. **对比实验**：与替代方案或不同超参组合比较，分析性能、稳定性、资源消耗。
4. **边界探测**：尝试极端参数、异常输入，观察警告或错误并记录总结。
5. **沉淀笔记**：撰写 Markdown 记录使用心得、常见坑位、最佳实践。

## 4. 仓库推荐结构
```
DeepLearning/
├── artifacts/         # 训练产生的日志、权重、可视化等衍生物
├── docs/                # Markdown 指南、原理解析、术语表
├── notes/               # 个人或团队学习笔记
├── notebooks/           # 交互式实验，命名规则 <主题>_<方法>.ipynb
├── src/
│   ├── datasets/        # 数据集封装、数据增强
│   ├── models/          # 标准与自定义 nn.Module
│   ├── experiments/     # 实验脚本，按主题分目录
│   ├── optimizers/      # 优化器变体与学习率调度
│   └── utils/           # 计时、日志、可视化、配置解析
├── configs/             # YAML/JSON 配置，描述实验参数
├── tests/               # 单元测试与端到端冒烟测试
└── AGENTS.md            # 本学习规范
```

## 5. API 深度拆解方案
- **阅读层面**：先看函数/类签名，再读官方示例，最后查看源码实现细节。
- **类型约束**：记录输入张量形状、数据类型、广播规则、梯度需求。
- **参数敏感度**：通过网格或随机搜索快速评估关键参数影响，例如 `nn.Conv2d` 的 `kernel_size`、`padding` 等。
- **可视化辅助**：利用 TensorBoard、Matplotlib 绘制损失曲线、梯度直方图、特征图。
- **兼容性检查**：确认该组件与其它模块（如混合精度、分布式训练）的适配情况。

## 6. `torch.nn` 方法族学习要点
### 6.1 基础模块
- `nn.Module` 生命周期：构造 -> 参数初始化 -> 前向计算 -> 反向传播。
- 参数管理：使用 `named_parameters()`、`requires_grad` 判定可训练性。
- 子模块组合：通过 `nn.Sequential`、自定义 `forward` 实现复杂拓扑，同时保持层级清晰。

### 6.2 常用层类型
- **线性层**：`nn.Linear`、`nn.Bilinear`，关注形状约束与权重初始化。
- **卷积层**：比较 `nn.Conv1d/2d/3d`、`nn.ConvTranspose2d` 的步幅与填充逻辑。
- **归一化**：`nn.BatchNorm`、`nn.LayerNorm`、`nn.GroupNorm` 等差异及适用场景。
- **注意力/Transformer**：`nn.MultiheadAttention` 参数含义、掩码处理、缓存机制。
- **序列模型**：`nn.RNN`、`nn.LSTM`、`nn.GRU` 的状态管理、双向/多层配置。

### 6.3 损失函数
- 分类：`nn.CrossEntropyLoss`、`nn.NLLLoss`、`nn.BCEWithLogitsLoss` 等输入要求。
- 回归：`nn.MSELoss`、`nn.SmoothL1Loss` 的稳定性与鲁棒性对比。
- 自定义损失：继承 `nn.Module`，确保 `forward` 返回标量，必要时对输入做数值稳定处理。

### 6.4 正则与约束
- `nn.Dropout`、`nn.Dropout2d` 在训练/推理模式下行为差异。
- `nn.Parameter` 与 `torch.no_grad()` 的配合使用。
- 梯度裁剪与权重衰减的实现方式与副作用。

## 7. 优化器与调度器
- `torch.optim` 常用优化器比较（SGD、Adam、AdamW、RMSprop），记录适用数据分布与调参建议。
- 自定义优化器：继承 `Optimizer`，实现 `step` 与状态持久化。
- 学习率调度：`torch.optim.lr_scheduler` 中分段、多阶、余弦等策略，结合 warmup 的配置方式。
- 监控：每次迭代记录学习率、动量、梯度范数，辅助诊断训练不收敛问题。

## 8. 数据与管道
- 数据集模块：使用 `torch.utils.data.Dataset`、`DataLoader` 定义懒加载、缓存、并行读取。
- 数据增强：区分离线增强（预处理）与在线增强（`transforms`）。
- 批量构造：掌握 `collate_fn` 的定制方法、对变长序列的填充策略。
- 数据可视化：采样若干批次展示标签、分布、统计量。

## 9. 基础训练循环模板
```python
import torch
from torch import nn, optim

class DemoNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

model = DemoNet(in_dim=784, hidden=256, out_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        logits = model(inputs)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
- 通过 Hook 机制 (`register_forward_hook`, `register_backward_hook`) 探查特征与梯度。
- 使用 `torch.compile` 或混合精度(`torch.cuda.amp`) 时记录额外注意事项。

## 10. 实验管理策略
- 配置驱动：所有可调参数写入 YAML/JSON，实验脚本通过解析器加载。
- 随机性控制：固定 `torch`, `numpy`, `random` 的种子，记录 CUDA `deterministic` 设置。
- 结果记录：使用 `TensorBoard`、`Weights & Biases` 或自建日志，保存指标与超参。
- 对比实验模板：统一输出表格（方法、参数、指标、时长）便于复盘。

## 11. 评估与可解释性
- 评估脚本拆分：数据准备 -> 推理 -> 指标计算 -> 可视化。
- 指标体系：准确率、召回、F1、ROC/AUC、Top-K、延迟、显存占用等。
- 可解释工具：Grad-CAM、Integrated Gradients、特征重要性分析；记录适配流程与限制。
- 误差分析：聚合错误样本，分类常见错误原因，提出重试计划（数据增强、架构调整）。

## 12. 文档与知识沉淀
- `docs/` 下为每类方法建立专属指南，例如 `docs/nn/conv_layers.md`、`docs/optim/adam_variants.md`。
- 每篇指南包含：概念摘要、API 参数说明、最小示例、注意事项、延伸阅读。
- 维护术语表与常见问答，统一命名、防止概念混淆。
- 所有实验与笔记关联到 issue/任务单，形成可追踪链路。

## 13. 测试与验证
- 单元测试：覆盖自定义模块的 forward/backward，使用 `torch.autograd.gradcheck` 验证数值正确性。
- 冒烟测试：对每个实验脚本运行一次短周期训练，确保日常改动不破坏主流程。
- 性能基准：记录训练吞吐量、显存占用，为不同实现比较提供依据。
- 兼容性测试：在 CPU/GPU、不同 PyTorch 版本上验证示例是否可执行。

## 14. 工具与资源
- 官方文档与源码导航：PyTorch Doc、Torchvision、Torchtext。
- 可视化：TensorBoard、Weights & Biases、Matplotlib/Seaborn。
- 调试：`torchviz`, `torchsummary`, `pdb`、`ipdb`、`torch.utils.benchmark`。
- 数据处理：TorchData、Albumentations、OpenCV。
- 代码质量：Black、isort、ruff、pytest、mypy（针对类型提示的示例）。

## 15. 常见问题处理
- **梯度消失/爆炸**：检查初始化、激活函数、梯度裁剪、归一化层布置。
- **损失不下降**：确认标签格式、学习率、数据增益是否损坏样本。
- **推理慢**：定位瓶颈（数据加载/模型计算/IO），尝试批量化、半精度或算子融合。
- **显存不足**：换用梯度累积、checkpointing、减少 batch size。

## 16. 版本与变更管理
- 文档采用语义化版本：`MAJOR.MINOR.PATCH`。
- 每次更新在 `docs/changelog.md` 记录新增示例、修复、废弃内容。
- 若实验脚手架有破坏性调整，提供迁移指南与旧版本保留链接。

## 17. 快速检查清单
- 目标方法是否已有最小示例并通过梯度校验？
- 是否记录参数、输入形状、关键约束与异常行为？
- 是否完成与同类方法或超参的对比实验？
- 评估与可视化结果是否归档，便于后续复现？
- 知识库文档、变更日志是否同步更新？

> 按照此规范持续迭代，可系统化掌握深度学习方法的原理与实现细节，并为团队积累可复用的 PyTorch 学习资产。
