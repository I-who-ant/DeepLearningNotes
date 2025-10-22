"""
深入理解反向传播与梯度
======================
核心问题:
1. loss.backward() 是如何计算梯度的?
2. 梯度为什么是"斜率"?
3. 为什么能通过梯度更新参数来减小loss?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 尝试导入matplotlib,如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib未安装,将跳过可视化部分")


def demo_gradient_as_slope():
    """演示:梯度就是斜率"""
    print("="*70)
    print("Part 1: 梯度 = 斜率的直观理解")
    print("="*70 + "\n")

    # 简单函数: y = x²
    print("考虑简单函数: L(w) = w²")
    print()

    # 创建一个需要梯度的变量
    w = torch.tensor([2.0], requires_grad=True)

    # 计算损失
    L = w ** 2

    print(f"当 w = {w.item():.1f} 时:")
    print(f"  L(w) = w² = {L.item():.1f}")

    # 反向传播计算梯度
    L.backward()

    print(f"\n反向传播计算得到:")
    print(f"  梯度 dL/dw = {w.grad.item():.1f}")
    print()
    print(f"手动计算验证:")
    print(f"  dL/dw = d(w²)/dw = 2w = 2×{w.item():.1f} = {2*w.item():.1f} ✅")
    print()

    # 可视化
    if not HAS_MATPLOTLIB:
        print("\n(跳过可视化部分,需要安装matplotlib)")
        return

    print("="*70)
    print("可视化:梯度是曲线的斜率")
    print("="*70)

    # 绘制函数曲线
    w_values = np.linspace(-3, 3, 100)
    L_values = w_values ** 2

    plt.figure(figsize=(12, 5))

    # 左图:损失函数曲线
    plt.subplot(1, 2, 1)
    plt.plot(w_values, L_values, 'b-', linewidth=2, label='L(w) = w²')

    # 标注当前点
    w_point = 2.0
    L_point = w_point ** 2
    gradient = 2 * w_point

    plt.plot(w_point, L_point, 'ro', markersize=10, label=f'当前点 (w={w_point})')

    # 绘制切线(斜率 = 梯度)
    tangent_x = np.linspace(w_point-1, w_point+1, 10)
    tangent_y = L_point + gradient * (tangent_x - w_point)
    plt.plot(tangent_x, tangent_y, 'r--', linewidth=2,
             label=f'切线斜率={gradient:.1f}')

    plt.xlabel('权重 w', fontsize=12)
    plt.ylabel('损失 L(w)', fontsize=12)
    plt.title('损失函数与梯度(斜率)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图:梯度随w的变化
    plt.subplot(1, 2, 2)
    gradients = 2 * w_values
    plt.plot(w_values, gradients, 'g-', linewidth=2, label='梯度 dL/dw = 2w')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.plot(w_point, gradient, 'ro', markersize=10)

    plt.xlabel('权重 w', fontsize=12)
    plt.ylabel('梯度 dL/dw', fontsize=12)
    plt.title('梯度的变化', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('artifacts/gradient_as_slope.png', dpi=100, bbox_inches='tight')
    print("\n📊 图表已保存到 artifacts/gradient_as_slope.png")
    plt.close()

    print("\n💡 关键理解:")
    print("  1. 梯度 = 函数在某点的斜率")
    print("  2. 正梯度 → 函数在增长 → 需要减小w")
    print("  3. 负梯度 → 函数在下降 → 需要增大w")
    print("  4. 梯度为0 → 到达极值点(最小值或最大值)")
    print()


def demo_simple_network_backprop():
    """演示简单网络的反向传播过程"""
    print("\n" + "="*70)
    print("Part 2: 简单网络的反向传播详解")
    print("="*70 + "\n")

    print("网络结构: 输入(2) → Linear(2,1) → 输出(1)")
    print()

    # 创建一个超简单的网络
    torch.manual_seed(42)
    model = nn.Linear(2, 1, bias=True)

    print("初始参数:")
    print(f"  权重 w: {model.weight.data}")
    print(f"  偏置 b: {model.bias.data}")
    print()

    # 输入和标签
    x = torch.tensor([[1.0, 2.0]])  # 输入
    y_true = torch.tensor([[5.0]])   # 真实值

    print("数据:")
    print(f"  输入 x: {x}")
    print(f"  真实值 y_true: {y_true.item()}")
    print()

    # === 前向传播 ===
    print("="*70)
    print("【前向传播】从输入到损失")
    print("="*70)

    y_pred = model(x)
    print(f"\n步骤1: 计算预测值")
    print(f"  y_pred = w·x + b")
    print(f"         = {model.weight.data} · {x[0]} + {model.bias.data.item():.4f}")
    w1, w2 = model.weight.data[0, 0].item(), model.weight.data[0, 1].item()
    x1, x2 = x[0, 0].item(), x[0, 1].item()
    b = model.bias.data.item()
    print(f"         = ({w1:.4f}×{x1:.1f} + {w2:.4f}×{x2:.1f}) + {b:.4f}")
    print(f"         = {y_pred.item():.4f}")

    loss = F.mse_loss(y_pred, y_true)
    print(f"\n步骤2: 计算损失(均方误差)")
    print(f"  loss = (y_pred - y_true)²")
    print(f"       = ({y_pred.item():.4f} - {y_true.item():.1f})²")
    print(f"       = {loss.item():.4f}")

    # === 反向传播 ===
    print("\n" + "="*70)
    print("【反向传播】从损失到参数梯度")
    print("="*70)

    print("\n执行: loss.backward()")
    loss.backward()

    print("\n自动计算得到的梯度:")
    print(f"  dL/dw₁ = {model.weight.grad[0, 0].item():.4f}")
    print(f"  dL/dw₂ = {model.weight.grad[0, 1].item():.4f}")
    print(f"  dL/db  = {model.bias.grad.item():.4f}")

    # === 手动验证梯度 ===
    print("\n" + "="*70)
    print("【手动验证】链式法则计算梯度")
    print("="*70)

    print("\n链式法则:")
    print("  dL/dw = dL/dy_pred × dy_pred/dw")
    print()

    # 计算中间梯度
    error = y_pred.item() - y_true.item()
    dL_dy = 2 * error  # MSE的导数

    print(f"步骤1: 计算 dL/dy_pred")
    print(f"  dL/dy_pred = d/dy_pred[(y_pred - y_true)²]")
    print(f"             = 2(y_pred - y_true)")
    print(f"             = 2 × ({y_pred.item():.4f} - {y_true.item():.1f})")
    print(f"             = {dL_dy:.4f}")

    print(f"\n步骤2: 计算 dy_pred/dw")
    print(f"  因为 y_pred = w₁×x₁ + w₂×x₂ + b")
    print(f"  所以:")
    print(f"    dy_pred/dw₁ = x₁ = {x1:.1f}")
    print(f"    dy_pred/dw₂ = x₂ = {x2:.1f}")
    print(f"    dy_pred/db  = 1")

    print(f"\n步骤3: 应用链式法则")
    manual_grad_w1 = dL_dy * x1
    manual_grad_w2 = dL_dy * x2
    manual_grad_b = dL_dy * 1

    print(f"  dL/dw₁ = dL/dy_pred × dy_pred/dw₁")
    print(f"         = {dL_dy:.4f} × {x1:.1f}")
    print(f"         = {manual_grad_w1:.4f}")
    print(f"  PyTorch计算: {model.weight.grad[0, 0].item():.4f} ✅")
    print()
    print(f"  dL/dw₂ = {dL_dy:.4f} × {x2:.1f} = {manual_grad_w2:.4f}")
    print(f"  PyTorch计算: {model.weight.grad[0, 1].item():.4f} ✅")
    print()
    print(f"  dL/db  = {dL_dy:.4f} × 1 = {manual_grad_b:.4f}")
    print(f"  PyTorch计算: {model.bias.grad.item():.4f} ✅")

    # === 参数更新 ===
    print("\n" + "="*70)
    print("【参数更新】梯度下降")
    print("="*70)

    lr = 0.01
    print(f"\n学习率 lr = {lr}")
    print("\n更新公式: 参数_new = 参数_old - lr × 梯度")
    print()

    w1_old = model.weight.data[0, 0].item()
    w1_new = w1_old - lr * model.weight.grad[0, 0].item()
    print(f"w₁: {w1_old:.4f} - {lr} × {model.weight.grad[0, 0].item():.4f}")
    print(f"  = {w1_new:.4f}")

    w2_old = model.weight.data[0, 1].item()
    w2_new = w2_old - lr * model.weight.grad[0, 1].item()
    print(f"\nw₂: {w2_old:.4f} - {lr} × {model.weight.grad[0, 1].item():.4f}")
    print(f"  = {w2_new:.4f}")

    print("\n💡 为什么减去梯度?")
    print(f"  梯度 = {model.weight.grad[0, 0].item():.4f} > 0")
    print(f"  说明:增大w会增大loss")
    print(f"  因此:减小w才能减小loss")
    print(f"  即:w_new = w_old - lr×grad(往斜率反方向移动)")


def demo_cifar_backprop():
    """演示CIFAR模型的反向传播"""
    print("\n\n" + "="*70)
    print("Part 3: CIFAR模型的反向传播")
    print("="*70 + "\n")

    # 简化的CIFAR模型
    class SimpleCIFAR(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.fc1 = nn.Linear(6*28*28, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    model = SimpleCIFAR()
    criterion = nn.CrossEntropyLoss()

    # 模拟数据
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([3, 7])

    print("模型结构:")
    print("  Conv2d(3→6) + ReLU")
    print("  Flatten")
    print("  Linear(4704→10)")
    print()

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")
    print()

    for name, param in model.named_parameters():
        print(f"  {name:20s}: {str(param.shape):20s} "
              f"({param.numel():>6,} 个参数)")
    print()

    # 前向传播
    print("="*70)
    print("执行前向传播...")
    print("="*70)
    logits = model(images)
    print(f"输出 logits: {logits.shape}")
    print(f"样本1的logits: {logits[0].detach().numpy()}")
    print()

    # 计算损失
    loss = criterion(logits, labels)
    print(f"交叉熵损失: {loss.item():.4f}")
    print()

    # 反向传播
    print("="*70)
    print("执行反向传播: loss.backward()")
    print("="*70)

    print("\n这一步PyTorch自动完成了:")
    print("  1. 从损失开始,逐层计算梯度")
    print("  2. 使用链式法则传播梯度")
    print("  3. 为每个参数计算 dL/d(参数)")
    print()

    loss.backward()

    print("反向传播完成!查看各层梯度:")
    print()

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"  {name:20s}:")
            print(f"    梯度形状: {param.grad.shape}")
            print(f"    梯度范围: [{param.grad.min().item():.6f}, "
                  f"{param.grad.max().item():.6f}]")
            print(f"    梯度平均: {grad_mean:.6f}")
            print()

    print("💡 关键理解:")
    print("  1. 每个参数都有一个梯度 (与参数形状相同)")
    print("  2. 梯度告诉我们:'调整这个参数多少,loss会减小多少'")
    print("  3. 梯度大 → 这个参数对loss影响大 → 需要大幅调整")
    print("  4. 梯度小 → 这个参数对loss影响小 → 微调即可")


def visualize_gradient_descent():
    """可视化梯度下降过程"""
    print("\n\n" + "="*70)
    print("Part 4: 可视化梯度下降")
    print("="*70 + "\n")

    if not HAS_MATPLOTLIB:
        print("(跳过可视化部分,需要安装matplotlib)\n")
        return

    # 创建一个简单的优化问题
    def loss_function(w1, w2):
        """一个二元函数作为损失"""
        return (w1 - 2)**2 + (w2 - 3)**2 + 1

    # 初始化参数
    w1 = torch.tensor([0.0], requires_grad=True)
    w2 = torch.tensor([0.0], requires_grad=True)

    lr = 0.2
    n_steps = 20

    # 记录轨迹
    trajectory_w1 = [w1.item()]
    trajectory_w2 = [w2.item()]
    trajectory_loss = [loss_function(w1.item(), w2.item())]

    print(f"初始点: w1={w1.item():.2f}, w2={w2.item():.2f}")
    print(f"初始损失: {trajectory_loss[0]:.4f}")
    print()

    # 梯度下降
    for step in range(n_steps):
        # 计算损失
        loss = loss_function(w1, w2)

        # 反向传播
        if w1.grad is not None:
            w1.grad.zero_()
            w2.grad.zero_()
        loss.backward()

        # 更新参数
        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad

        # 记录
        trajectory_w1.append(w1.item())
        trajectory_w2.append(w2.item())
        trajectory_loss.append(loss.item())

        if step % 5 == 0:
            print(f"步骤 {step:2d}: w1={w1.item():6.3f}, w2={w2.item():6.3f}, "
                  f"loss={loss.item():8.4f}")

    print(f"\n最终点: w1={w1.item():.2f}, w2={w2.item():.2f}")
    print(f"最终损失: {trajectory_loss[-1]:.4f}")
    print(f"最优点: w1=2.00, w2=3.00 (理论值)")
    print()

    # 绘图
    fig = plt.figure(figsize=(15, 5))

    # 左图:等高线图 + 梯度下降轨迹
    ax1 = plt.subplot(1, 3, 1)
    w1_range = np.linspace(-1, 4, 100)
    w2_range = np.linspace(-1, 5, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = (W1 - 2)**2 + (W2 - 3)**2 + 1

    contour = ax1.contour(W1, W2, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(trajectory_w1, trajectory_w2, 'r.-', linewidth=2,
             markersize=8, label='梯度下降轨迹')
    ax1.plot(2, 3, 'g*', markersize=20, label='全局最优点')
    ax1.set_xlabel('w₁')
    ax1.set_ylabel('w₂')
    ax1.set_title('梯度下降轨迹')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 中图:损失随迭代次数的变化
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(trajectory_loss, 'b.-', linewidth=2, markersize=8)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失值')
    ax2.set_title('损失下降曲线')
    ax2.grid(True, alpha=0.3)

    # 右图:参数收敛过程
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(trajectory_w1, 'r.-', label='w₁', linewidth=2, markersize=6)
    ax3.plot(trajectory_w2, 'b.-', label='w₂', linewidth=2, markersize=6)
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='w₁最优值=2')
    ax3.axhline(y=3, color='b', linestyle='--', alpha=0.5, label='w₂最优值=3')
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('参数值')
    ax3.set_title('参数收敛过程')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('artifacts/gradient_descent.png', dpi=100, bbox_inches='tight')
    print("📊 图表已保存到 artifacts/gradient_descent.png")
    plt.close()


if __name__ == "__main__":
    # 创建输出目录
    import os
    os.makedirs('artifacts', exist_ok=True)

    print("\n\n")
    print("█" * 70)
    print(" " * 18 + "反向传播与梯度完全解析")
    print("█" * 70)

    # Part 1: 梯度是斜率
    demo_gradient_as_slope()

    # Part 2: 简单网络反向传播
    demo_simple_network_backprop()

    # Part 3: CIFAR模型反向传播
    demo_cifar_backprop()

    # Part 4: 可视化梯度下降
    visualize_gradient_descent()

    print("\n" + "="*70)
    print("总结:反向传播的完整理解")
    print("="*70)
    print("""
1. 梯度 = 斜率
   - 告诉我们函数在某点的变化率
   - 正梯度 → 参数增大会增大loss → 需要减小参数
   - 负梯度 → 参数增大会减小loss → 需要增大参数

2. 反向传播 = 自动微分
   - PyTorch自动计算每个参数的梯度
   - 使用链式法则从输出层传播到输入层
   - loss.backward() 一行代码完成所有梯度计算

3. 为什么叫"反向"?
   - 前向:输入 → 层1 → 层2 → ... → 输出 → loss
   - 反向:loss → ... → 层2的梯度 → 层1的梯度 → 输入的梯度
   - 梯度从loss开始,沿着网络反向传播

4. 参数更新
   - 参数_new = 参数_old - 学习率 × 梯度
   - 学习率控制步长大小
   - 重复"前向→损失→反向→更新"直到收敛

5. 核心公式
   - 链式法则: dL/dw = dL/dy × dy/dw
   - 梯度下降: w := w - α·(dL/dw)
   - 其中 α 是学习率
""")
    print("="*70 + "\n")
