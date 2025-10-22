"""
深入理解优化器(Optimizer)
========================
核心问题:
1. 优化器是什么?有什么用?
2. 各种优化器(SGD/Adam/RMSprop等)有什么区别?
3. 参数(lr/momentum/weight_decay等)是什么意思?
4. 如何选择合适的优化器?
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def explain_optimizer_basics():
    """解释优化器的基础概念"""
    print("="*70)
    print("Part 1: 优化器是什么?")
    print("="*70 + "\n")

    print("💡 优化器的核心作用:")
    print("  输入: 参数的梯度 (param.grad)")
    print("  输出: 参数的更新量 (如何调整参数)")
    print("  目标: 让 loss 尽快下降到最小值")
    print()

    print("基础公式:")
    print("  参数更新 = 参数_old - 学习率 × 梯度")
    print("  param_new = param_old - lr × grad")
    print()

    # 创建简单示例
    print("="*70)
    print("示例:手动实现优化器")
    print("="*70 + "\n")

    # 一个简单的参数
    param = torch.tensor([2.0], requires_grad=True)

    print(f"初始参数: {param.item():.4f}")

    # 模拟梯度
    loss = param ** 2
    loss.backward()

    print(f"计算得到梯度: {param.grad.item():.4f}")

    # 手动更新
    lr = 0.1
    with torch.no_grad():  # 更新时不需要计算梯度
        param -= lr * param.grad

    print(f"学习率: {lr}")
    print(f"更新后参数: {param.item():.4f}")
    print()

    print("💡 优化器做的就是这件事!")
    print("  只不过更聪明:不是简单地减去梯度,而是用各种技巧加速收敛")


def compare_sgd_variants():
    """对比不同的SGD变体"""
    print("\n\n" + "="*70)
    print("Part 2: SGD家族优化器详解")
    print("="*70 + "\n")

    # 创建一个测试问题
    def rosenbrock(x, y):
        """Rosenbrock函数:经典的优化测试函数"""
        return (1 - x)**2 + 100 * (y - x**2)**2

    # 初始点
    start_x, start_y = -1.0, -1.0

    print("测试函数: Rosenbrock函数")
    print(f"起始点: ({start_x}, {start_y})")
    print(f"目标最小值点: (1, 1)")
    print()

    # ========== 1. 基础SGD ==========
    print("="*70)
    print("1. 基础SGD (Stochastic Gradient Descent)")
    print("="*70)
    print()
    print("原理:")
    print("  param_new = param_old - lr × grad")
    print()
    print("特点:")
    print("  - 最简单的优化器")
    print("  - 严格按照梯度方向更新")
    print("  - 容易陷入局部最优")
    print("  - 震荡较大")
    print()

    # 模拟SGD
    x1 = torch.tensor([start_x], requires_grad=True)
    y1 = torch.tensor([start_y], requires_grad=True)
    optimizer1 = optim.SGD([x1, y1], lr=0.001)

    print("代码:")
    print("  optimizer = optim.SGD(model.parameters(), lr=0.001)")
    print()

    for step in range(5):
        optimizer1.zero_grad()
        loss = rosenbrock(x1, y1)
        loss.backward()
        optimizer1.step()

        if step == 0 or step == 4:
            print(f"  步骤{step}: 位置=({x1.item():.4f}, {y1.item():.4f}), "
                  f"loss={loss.item():.4f}")

    print()
    print("📊 观察:")
    print("  - 每步严格按梯度方向移动")
    print("  - 没有'记忆',每步独立决策")
    print()

    # ========== 2. SGD + Momentum ==========
    print("="*70)
    print("2. SGD + Momentum (动量)")
    print("="*70)
    print()
    print("原理:")
    print("  velocity = momentum × velocity_old + grad")
    print("  param_new = param_old - lr × velocity")
    print()
    print("💡 类比:")
    print("  想象一个球从山坡滚下来:")
    print("  - 不仅受当前坡度影响(梯度)")
    print("  - 还保持之前的速度(动量)")
    print("  - 能冲过小山丘(避免局部最优)")
    print()

    x2 = torch.tensor([start_x], requires_grad=True)
    y2 = torch.tensor([start_y], requires_grad=True)
    optimizer2 = optim.SGD([x2, y2], lr=0.001, momentum=0.9)

    print("代码:")
    print("  optimizer = optim.SGD(model.parameters(),")
    print("                        lr=0.001,")
    print("                        momentum=0.9)  # 保持90%的历史速度")
    print()

    for step in range(5):
        optimizer2.zero_grad()
        loss = rosenbrock(x2, y2)
        loss.backward()
        optimizer2.step()

        if step == 0 or step == 4:
            print(f"  步骤{step}: 位置=({x2.item():.4f}, {y2.item():.4f}), "
                  f"loss={loss.item():.4f}")

    print()
    print("参数解释:")
    print("  momentum ∈ [0, 1]:")
    print("    - 0.0: 退化为基础SGD,无动量")
    print("    - 0.9: 保持90%的历史速度(常用值)")
    print("    - 0.99: 保持99%的历史速度(用于大batch)")
    print()
    print("优点:")
    print("  ✓ 加速收敛(利用历史信息)")
    print("  ✓ 减少震荡(平滑梯度)")
    print("  ✓ 更容易跳出局部最优")
    print()

    # ========== 3. SGD + Nesterov Momentum ==========
    print("="*70)
    print("3. SGD + Nesterov Momentum (Nesterov加速梯度)")
    print("="*70)
    print()
    print("原理:")
    print("  1. 先按动量移动到'预测位置'")
    print("  2. 在预测位置计算梯度")
    print("  3. 根据预测位置的梯度修正方向")
    print()
    print("💡 类比:")
    print("  普通动量: 看当前位置的路标")
    print("  Nesterov: 先往前看一步,看前面的路标(更聪明!)")
    print()

    x3 = torch.tensor([start_x], requires_grad=True)
    y3 = torch.tensor([start_y], requires_grad=True)
    optimizer3 = optim.SGD([x3, y3], lr=0.001, momentum=0.9, nesterov=True)

    print("代码:")
    print("  optimizer = optim.SGD(model.parameters(),")
    print("                        lr=0.001,")
    print("                        momentum=0.9,")
    print("                        nesterov=True)  # 启用Nesterov")
    print()

    # ========== 4. Weight Decay ==========
    print("="*70)
    print("4. Weight Decay (权重衰减 = L2正则化)")
    print("="*70)
    print()
    print("原理:")
    print("  grad_new = grad + weight_decay × param")
    print("  param_new = param_old - lr × grad_new")
    print()
    print("💡 作用:")
    print("  - 防止权重过大")
    print("  - 相当于给损失函数加上 λ||w||²")
    print("  - 防止过拟合")
    print()

    print("代码:")
    print("  optimizer = optim.SGD(model.parameters(),")
    print("                        lr=0.001,")
    print("                        weight_decay=1e-4)  # L2正则化系数")
    print()

    print("参数解释:")
    print("  weight_decay ∈ [0, ∞):")
    print("    - 0: 无正则化")
    print("    - 1e-5 ~ 1e-3: 常用范围")
    print("    - 过大: 权重被压制得太小,欠拟合")
    print()


def compare_adaptive_optimizers():
    """对比自适应学习率优化器"""
    print("\n\n" + "="*70)
    print("Part 3: 自适应学习率优化器")
    print("="*70 + "\n")

    print("💡 核心思想:")
    print("  不同参数应该用不同的学习率!")
    print("  - 梯度大的参数 → 用小学习率(防止震荡)")
    print("  - 梯度小的参数 → 用大学习率(加速收敛)")
    print()

    # ========== 1. Adagrad ==========
    print("="*70)
    print("1. Adagrad (Adaptive Gradient)")
    print("="*70)
    print()
    print("原理:")
    print("  sum_squared_grad += grad²")
    print("  adjusted_lr = lr / sqrt(sum_squared_grad + ε)")
    print("  param_new = param_old - adjusted_lr × grad")
    print()
    print("💡 特点:")
    print("  - 累积历史梯度的平方")
    print("  - 学习率会不断减小")
    print("  - 适合稀疏梯度(如NLP)")
    print()

    print("代码:")
    print("  optimizer = optim.Adagrad(model.parameters(), lr=0.01)")
    print()

    print("优点:")
    print("  ✓ 自动调整学习率")
    print("  ✓ 适合处理稀疏数据")
    print()
    print("缺点:")
    print("  ✗ 学习率单调递减")
    print("  ✗ 训练后期可能太慢")
    print()

    # ========== 2. RMSprop ==========
    print("="*70)
    print("2. RMSprop (Root Mean Square Propagation)")
    print("="*70)
    print()
    print("原理:")
    print("  squared_avg = α × squared_avg + (1-α) × grad²")
    print("  adjusted_lr = lr / sqrt(squared_avg + ε)")
    print("  param_new = param_old - adjusted_lr × grad")
    print()
    print("💡 改进:")
    print("  - 用指数移动平均代替累积和")
    print("  - 学习率不会无限减小")
    print("  - 适合RNN训练")
    print()

    print("代码:")
    print("  optimizer = optim.RMSprop(model.parameters(),")
    print("                            lr=0.01,")
    print("                            alpha=0.99)  # 移动平均系数")
    print()

    print("参数解释:")
    print("  alpha ∈ [0, 1]:")
    print("    - 0.99: 保留99%的历史信息(常用)")
    print("    - 0.9:  更快适应新梯度")
    print()

    # ========== 3. Adam ==========
    print("="*70)
    print("3. Adam (Adaptive Moment Estimation) ⭐最常用⭐")
    print("="*70)
    print()
    print("原理:结合Momentum和RMSprop")
    print("  m = β₁ × m + (1-β₁) × grad           # 一阶矩(动量)")
    print("  v = β₂ × v + (1-β₂) × grad²          # 二阶矩(方差)")
    print("  m_hat = m / (1 - β₁^t)                # 偏差修正")
    print("  v_hat = v / (1 - β₂^t)")
    print("  param_new = param_old - lr × m_hat / (sqrt(v_hat) + ε)")
    print()
    print("💡 集大成者:")
    print("  - 有动量(利用历史梯度方向)")
    print("  - 有自适应学习率(根据梯度大小调整)")
    print("  - 有偏差修正(训练初期更准确)")
    print()

    print("代码:")
    print("  optimizer = optim.Adam(model.parameters(),")
    print("                         lr=0.001,")
    print("                         betas=(0.9, 0.999),  # (β₁, β₂)")
    print("                         eps=1e-8,")
    print("                         weight_decay=0)")
    print()

    print("参数解释:")
    print("  lr: 学习率")
    print("    - 0.001: 默认值,适合大多数情况")
    print("    - 0.0001: 微调时使用")
    print()
    print("  betas = (β₁, β₂):")
    print("    - β₁=0.9: 一阶矩(动量)衰减率")
    print("    - β₂=0.999: 二阶矩(方差)衰减率")
    print("    - 通常不需要改")
    print()
    print("  eps: 数值稳定性")
    print("    - 防止除零")
    print("    - 默认1e-8即可")
    print()

    print("优点:")
    print("  ✓ 收敛快")
    print("  ✓ 鲁棒性好")
    print("  ✓ 超参数默认值就很好用")
    print("  ✓ 适用范围广")
    print()
    print("缺点:")
    print("  ✗ 可能过拟合")
    print("  ✗ 有时泛化性不如SGD+Momentum")
    print()

    # ========== 4. AdamW ==========
    print("="*70)
    print("4. AdamW (Adam with Weight Decay) ⭐推荐⭐")
    print("="*70)
    print()
    print("原理:")
    print("  - Adam的改进版本")
    print("  - 修正了weight_decay的实现方式")
    print("  - 更好的泛化性能")
    print()

    print("代码:")
    print("  optimizer = optim.AdamW(model.parameters(),")
    print("                          lr=0.001,")
    print("                          weight_decay=0.01)  # 常用值")
    print()

    print("💡 AdamW vs Adam:")
    print("  Adam:   grad = grad + weight_decay × param  (错误的L2)")
    print("  AdamW:  param = param × (1 - weight_decay)  (正确的L2)")
    print()
    print("📊 现代最佳实践:")
    print("  - 大多数情况优先选择AdamW")
    print("  - weight_decay=0.01~0.1")
    print()


def demo_optimizer_comparison():
    """实战对比不同优化器"""
    print("\n\n" + "="*70)
    print("Part 4: 实战对比 - 训练简单模型")
    print("="*70 + "\n")

    # 创建一个简单的分类任务
    torch.manual_seed(42)

    # 模型
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 3)
    )

    # 数据
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))

    criterion = nn.CrossEntropyLoss()

    # 测试不同优化器
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=0.01),
        'SGD+Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'RMSprop': optim.RMSprop(model.parameters(), lr=0.01),
        'Adam': optim.Adam(model.parameters(), lr=0.01),
        'AdamW': optim.AdamW(model.parameters(), lr=0.01),
    }

    print("任务: 100个样本,10维输入,3分类")
    print("模型: 10→50→3的全连接网络")
    print()

    for name, optimizer in optimizers.items():
        # 重新初始化模型
        for layer in model:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # 训练10步
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"{name:15s}: 初始loss={losses[0]:.4f}, "
              f"最终loss={losses[-1]:.4f}, "
              f"下降={losses[0]-losses[-1]:.4f}")

    print()
    print("📊 观察:")
    print("  - Adam/AdamW通常收敛最快")
    print("  - SGD+Momentum比纯SGD快")
    print("  - 但SGD+Momentum长期训练可能泛化更好")


def explain_optimizer_parameters():
    """详解优化器的各种参数"""
    print("\n\n" + "="*70)
    print("Part 5: 优化器参数完全指南")
    print("="*70 + "\n")

    print("【1. 学习率 (lr / learning_rate)】")
    print("  最重要的超参数!")
    print()
    print("  作用: 控制参数更新的步长")
    print("  param_new = param_old - lr × grad")
    print()
    print("  选择指南:")
    print("    - 太大: 震荡,不收敛,loss爆炸")
    print("    - 太小: 收敛慢,容易卡住")
    print("    - 合适: 稳定下降")
    print()
    print("  常用范围:")
    print("    SGD:     0.01 ~ 0.1")
    print("    SGD+Momentum: 0.01 ~ 0.1")
    print("    Adam:    0.0001 ~ 0.001")
    print("    AdamW:   0.0001 ~ 0.001")
    print()
    print("  调参技巧:")
    print("    1. 从小开始(如1e-4)")
    print("    2. 观察loss曲线")
    print("    3. 逐步增大,直到出现震荡")
    print("    4. 选择震荡前的最大值")
    print()

    print("【2. 动量 (momentum)】")
    print("  用于SGD")
    print()
    print("  作用: 保留历史梯度信息,加速收敛")
    print()
    print("  常用值:")
    print("    - 0.9: 标准选择")
    print("    - 0.95~0.99: 大batch size时")
    print()

    print("【3. 权重衰减 (weight_decay)】")
    print("  L2正则化")
    print()
    print("  作用: 防止过拟合,限制权重大小")
    print()
    print("  常用范围:")
    print("    - 0: 无正则化")
    print("    - 1e-5 ~ 1e-3: 小数据集")
    print("    - 0.01 ~ 0.1: 大模型(如Transformer)")
    print()
    print("  注意:")
    print("    - Adam配合weight_decay效果不好")
    print("    - 推荐用AdamW")
    print()

    print("【4. Betas (β₁, β₂)】")
    print("  用于Adam/AdamW")
    print()
    print("  β₁: 一阶矩(动量)衰减率")
    print("  β₂: 二阶矩(方差)衰减率")
    print()
    print("  默认值: (0.9, 0.999)")
    print("  通常不需要调整!")
    print()
    print("  特殊情况:")
    print("    - NLP/大batch: β₁=0.9, β₂=0.98")
    print("    - 噪声大的任务: β₁=0.5")
    print()

    print("【5. Epsilon (eps)】")
    print("  数值稳定性参数")
    print()
    print("  作用: 防止除零错误")
    print("  adjusted_lr = lr / (sqrt(variance) + eps)")
    print()
    print("  默认值: 1e-8")
    print("  通常不需要改!")
    print()


def provide_practical_guide():
    """提供实用选择指南"""
    print("\n\n" + "="*70)
    print("Part 6: 优化器选择实用指南")
    print("="*70 + "\n")

    print("🎯 快速决策树:")
    print()
    print("1. 你在做什么任务?")
    print()
    print("   【计算机视觉 (CNN)】")
    print("     首选: SGD + Momentum")
    print("       optimizer = optim.SGD(model.parameters(),")
    print("                             lr=0.1,")
    print("                             momentum=0.9,")
    print("                             weight_decay=1e-4)")
    print("     原因: 泛化性能最好,业界验证")
    print()
    print("     备选: AdamW (快速原型)")
    print("       optimizer = optim.AdamW(model.parameters(),")
    print("                               lr=0.001,")
    print("                               weight_decay=0.01)")
    print()

    print("   【自然语言处理 (Transformer)】")
    print("     首选: AdamW")
    print("       optimizer = optim.AdamW(model.parameters(),")
    print("                               lr=1e-4,")
    print("                               betas=(0.9, 0.98),")
    print("                               weight_decay=0.01)")
    print("     原因: 处理稀疏梯度好,训练稳定")
    print()

    print("   【强化学习】")
    print("     首选: Adam 或 RMSprop")
    print("       optimizer = optim.Adam(model.parameters(), lr=1e-4)")
    print("     原因: 梯度噪声大,需要自适应学习率")
    print()

    print("   【GAN】")
    print("     首选: Adam")
    print("       optimizer_G = optim.Adam(generator.parameters(),")
    print("                                lr=0.0002, betas=(0.5, 0.999))")
    print("       optimizer_D = optim.Adam(discriminator.parameters(),")
    print("                                lr=0.0002, betas=(0.5, 0.999))")
    print("     注意: β₁=0.5 (降低动量,增加稳定性)")
    print()

    print("="*70)
    print("📊 优化器对比总结表")
    print("="*70)
    print()
    print("优化器          收敛速度  泛化性能  超参数敏感度  适用场景")
    print("-" * 70)
    print("SGD             慢        优        高           CV大模型")
    print("SGD+Momentum    中        优        中           CV通用,推荐")
    print("Adagrad         中        中        低           稀疏数据")
    print("RMSprop         快        中        低           RNN")
    print("Adam            快        中        低           通用原型")
    print("AdamW           快        优        低           NLP,大模型")
    print()

    print("="*70)
    print("🔧 调参建议")
    print("="*70)
    print()
    print("1. 学习率调整策略:")
    print("   - 使用学习率调度器(lr_scheduler)")
    print("   - 常用: StepLR, CosineAnnealingLR, ReduceLROnPlateau")
    print()
    print("   示例:")
    print("     optimizer = optim.SGD(model.parameters(), lr=0.1)")
    print("     scheduler = optim.lr_scheduler.StepLR(optimizer,")
    print("                                            step_size=30,")
    print("                                            gamma=0.1)")
    print("     # 每30个epoch,学习率×0.1")
    print()

    print("2. 学习率预热(Warmup):")
    print("   - 训练初期用小学习率")
    print("   - 逐步增大到目标学习率")
    print("   - 对大batch size很重要")
    print()

    print("3. 梯度裁剪(Gradient Clipping):")
    print("   - 防止梯度爆炸")
    print("   - 特别是RNN/Transformer")
    print()
    print("   代码:")
    print("     loss.backward()")
    print("     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
    print("     optimizer.step()")
    print()


def demo_complete_training_loop():
    """完整训练循环示例"""
    print("\n\n" + "="*70)
    print("Part 7: 完整训练循环示例")
    print("="*70 + "\n")

    code = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 1. 创建模型
model = MyModel()

# 2. 选择优化器
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,           # 初始学习率
    weight_decay=0.01   # L2正则化
)

# 3. 学习率调度器
scheduler = StepLR(
    optimizer,
    step_size=10,  # 每10个epoch
    gamma=0.5      # 学习率×0.5
)

# 4. 损失函数
criterion = nn.CrossEntropyLoss()

# 5. 训练循环
for epoch in range(100):

    for batch_images, batch_labels in train_loader:
        # 5.1 前向传播
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        # 5.2 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()         # 计算梯度

        # 5.3 梯度裁剪(可选)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # 5.4 参数更新
        optimizer.step()

    # 5.5 学习率调整
    scheduler.step()

    # 5.6 打印信息
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: loss={loss.item():.4f}, lr={current_lr:.6f}")
'''
    print(code)

    print("="*70)
    print("关键步骤解释:")
    print("="*70)
    print()
    print("1. optimizer.zero_grad()")
    print("   - 清空上一次的梯度")
    print("   - 必须在每次backward前调用!")
    print()
    print("2. loss.backward()")
    print("   - 计算所有参数的梯度")
    print("   - 梯度累积在param.grad中")
    print()
    print("3. optimizer.step()")
    print("   - 根据梯度更新参数")
    print("   - 实现具体的优化算法")
    print()
    print("4. scheduler.step()")
    print("   - 调整学习率")
    print("   - 在每个epoch结束后调用")
    print()


if __name__ == "__main__":
    print("\n\n")
    print("█" * 70)
    print(" " * 20 + "优化器完全指南")
    print("█" * 70)

    # Part 1: 基础概念
    explain_optimizer_basics()

    # Part 2: SGD家族
    compare_sgd_variants()

    # Part 3: 自适应优化器
    compare_adaptive_optimizers()

    # Part 4: 实战对比
    demo_optimizer_comparison()

    # Part 5: 参数详解
    explain_optimizer_parameters()

    # Part 6: 选择指南
    provide_practical_guide()

    # Part 7: 完整示例
    demo_complete_training_loop()

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
🎯 记住这些关键点:

1. 优化器的本质
   - 输入: 梯度
   - 输出: 参数更新量
   - 目标: 快速找到loss最小值

2. 快速选择
   - CV任务: SGD + Momentum
   - NLP任务: AdamW
   - 快速原型: Adam
   - 不确定: 试试AdamW

3. 关键参数
   - lr: 最重要,需要仔细调
   - momentum: SGD用0.9
   - weight_decay: 0.01左右
   - Adam的betas: 用默认值

4. 训练技巧
   - 用学习率调度器
   - 大模型用warmup
   - RNN用梯度裁剪
   - 监控loss曲线

5. 记忆口诀
   "梯度告诉方向,优化器决定步法"
""")
    print("="*70 + "\n")
