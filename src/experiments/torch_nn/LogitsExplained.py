"""
深入理解:从图像到 logits 的完整过程
======================================
解答:层层卷积如何得到10类打分(logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARNetDetailed(nn.Module):
    """带详细注释的CIFAR-10模型"""

    def __init__(self):
        super().__init__()

        # 特征提取部分:从图像中提取抽象特征
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)    # 3通道 → 32通道
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)   # 32通道 → 32通道
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)   # 32通道 → 64通道

        self.pool = nn.MaxPool2d(2, 2)

        # 分类部分:将特征映射到10个类别的打分
        self.fc1 = nn.Linear(64 * 4 * 4, 64)           # 提取高级特征
        self.fc2 = nn.Linear(64, 10)                   # 👈 关键!映射到10类

    def forward(self, x):
        # 输入: (batch, 3, 32, 32) - RGB图像
        print(f"\n{'='*70}")
        print("前向传播详细过程")
        print(f"{'='*70}")
        print(f"输入图像: {x.shape} - 3通道32×32的RGB图像")

        # 第一层卷积块
        x = F.relu(self.conv1(x))
        print(f"  ↓ Conv1+ReLU: {x.shape} - 提取32种低级特征(边缘/纹理)")
        x = self.pool(x)
        print(f"  ↓ Pool1:      {x.shape} - 降低空间分辨率")

        # 第二层卷积块
        x = F.relu(self.conv2(x))
        print(f"  ↓ Conv2+ReLU: {x.shape} - 组合特征(角点/简单形状)")
        x = self.pool(x)
        print(f"  ↓ Pool2:      {x.shape}")

        # 第三层卷积块
        x = F.relu(self.conv3(x))
        print(f"  ↓ Conv3+ReLU: {x.shape} - 抽象特征(物体部件)")
        x = self.pool(x)
        print(f"  ↓ Pool3:      {x.shape} - 64个特征图,每个4×4")

        # 展平:从2D特征图变成1D向量
        print(f"\n  展平前: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"  展平后: {x.shape} - 将所有特征连接成向量")
        print(f"          这是64个特征图×16个位置 = 1024维特征向量")

        # 全连接层1:进一步提取高级语义特征
        x = F.relu(self.fc1(x))
        print(f"  ↓ FC1+ReLU:   {x.shape} - 压缩到64维语义特征")
        print(f"          (这64个数编码了图像的高级语义信息)")

        # 全连接层2:映射到10个类别的打分
        x = self.fc2(x)
        print(f"  ↓ FC2(输出):  {x.shape} - 10类打分(logits)")
        print(f"{'='*70}\n")

        return x


def visualize_logits_generation():
    """可视化 logits 生成过程"""

    print("="*70)
    print("理解 logits:从1024维特征到10类打分")
    print("="*70)

    # 模拟最后一层全连接层的权重
    # fc2: Linear(64, 10)
    print("\n【最后一层的权重矩阵】")
    print("形状: (10, 64) - 10行代表10个类别,64列对应64个输入特征")
    print()

    # 创建一个简化的例子
    torch.manual_seed(42)
    fc2_weight = torch.randn(10, 64) * 0.1  # 10个类别 × 64个特征
    fc2_bias = torch.randn(10) * 0.1        # 10个偏置

    # 模拟输入特征(fc1的输出)
    features = torch.randn(1, 64)

    print("输入特征向量 (64维):")
    print(f"  形状: {features.shape}")
    print(f"  前5个值: {features[0, :5].tolist()}")
    print()

    # 计算 logits
    print("计算过程: logits = 权重矩阵 @ 特征向量 + 偏置")
    print()

    logits = features @ fc2_weight.T + fc2_bias

    print(f"输出 logits (10维):")
    print(f"  形状: {logits.shape}")
    print()

    # 显示每个类别的详细计算
    cifar10_classes = ['飞机', '汽车', '鸟', '猫', '鹿',
                       '狗', '青蛙', '马', '船', '卡车']

    print("各类别的打分(logits):")
    for i, (cls_name, score) in enumerate(zip(cifar10_classes, logits[0])):
        print(f"  类别{i} ({cls_name:>3}): {score.item():>8.4f}")

    print(f"\n{'='*70}")
    print("关键理解:")
    print(f"{'='*70}")
    print("1. 每个 logit 值是该类别的'原始评分'")
    print("2. 评分通过特征向量和该类别权重的点积计算:")
    print("   logit[i] = Σ(feature[j] * weight[i,j]) + bias[i]")
    print("3. 正值 → 模型认为'可能是'这个类别")
    print("   负值 → 模型认为'不太可能是'这个类别")
    print("4. logits 之间可以比较:分数越高,模型越倾向该类别")
    print(f"{'='*70}\n")

    # 转换为概率
    probs = F.softmax(logits, dim=1)
    print("通过 softmax 转换为概率:")
    for i, (cls_name, prob) in enumerate(zip(cifar10_classes, probs[0])):
        bar = '█' * int(prob.item() * 50)
        print(f"  {cls_name:>3}: {prob.item():.4f} ({prob.item()*100:5.2f}%) {bar}")

    print(f"\n概率和: {probs.sum().item():.6f} (必定为1.0)")


def explain_fc_weight_meaning():
    """解释全连接层权重的含义"""

    print("\n" + "="*70)
    print("深度解析:权重如何编码'类别特征'")
    print("="*70 + "\n")

    print("假设简化场景:只有3个特征,3个类别")
    print()

    # 简化例子:3个特征 → 3个类别
    features = torch.tensor([[0.8, 0.2, 0.1]])  # 示例特征
    weights = torch.tensor([
        [1.0, -0.5, -0.3],   # 类别0的权重:喜欢特征0,不喜欢特征1,2
        [-0.2, 1.5, -0.1],   # 类别1的权重:喜欢特征1
        [-0.3, -0.2, 2.0],   # 类别2的权重:喜欢特征2
    ])
    bias = torch.tensor([0.0, 0.0, 0.0])

    print("输入特征: [0.8, 0.2, 0.1]")
    print("  特征0 = 0.8 (强) - 可能表示'有翅膀'")
    print("  特征1 = 0.2 (弱) - 可能表示'有轮子'")
    print("  特征2 = 0.1 (弱) - 可能表示'有鳍'")
    print()

    print("权重矩阵:")
    print("         特征0  特征1  特征2")
    print(f"  类别0: [ 1.0, -0.5, -0.3]  (飞机:需要'翅膀',不要'轮子/鳍')")
    print(f"  类别1: [-0.2,  1.5, -0.1]  (汽车:需要'轮子',不要'翅膀/鳍')")
    print(f"  类别2: [-0.3, -0.2,  2.0]  (鱼  :需要'鳍',  不要'翅膀/轮子')")
    print()

    # 计算
    logits = features @ weights.T + bias

    print("计算过程:")
    for i in range(3):
        calc = features[0] * weights[i]
        print(f"  类别{i} logit = {features[0,0]:.1f}×{weights[i,0]:5.1f} + "
              f"{features[0,1]:.1f}×{weights[i,1]:5.1f} + "
              f"{features[0,2]:.1f}×{weights[i,2]:5.1f}")
        print(f"              = {calc[0]:.3f} + {calc[1]:.3f} + {calc[2]:.3f}")
        print(f"              = {logits[0,i]:.3f}")
        print()

    print("结果解释:")
    print(f"  类别0(飞机) = {logits[0,0]:.3f} ← 最高分!因为'翅膀'特征强")
    print(f"  类别1(汽车) = {logits[0,1]:.3f}")
    print(f"  类别2(鱼)   = {logits[0,2]:.3f}")
    print()
    print("💡 权重学习的本质:")
    print("  - 训练过程中,权重会自动调整")
    print("  - 让能代表某类别的特征获得高权重")
    print("  - 让不相关的特征获得低/负权重")
    print("  - 这样 logit = 特征·权重 就能衡量'相似度'")


def demonstrate_real_example():
    """用真实模型演示"""
    print("\n" + "="*70)
    print("真实模型演示")
    print("="*70)

    model = CIFARNetDetailed()
    model.eval()

    # 创建一个假的CIFAR-10图像
    fake_image = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        logits = model(fake_image)

    # 分析输出
    print("输出分析:")
    print(f"  Logits: {logits[0].tolist()}")
    print()

    # 找到最大值
    max_logit = logits.max().item()
    max_idx = logits.argmax().item()

    cifar10_classes = ['飞机', '汽车', '鸟', '猫', '鹿',
                       '狗', '青蛙', '马', '船', '卡车']

    print(f"  最高分: {max_logit:.4f} (类别{max_idx}: {cifar10_classes[max_idx]})")
    print()

    # 转换为概率
    probs = F.softmax(logits, dim=1)
    print("  转换为概率后:")
    for i, (cls, prob) in enumerate(zip(cifar10_classes, probs[0])):
        marker = "👈" if i == max_idx else ""
        print(f"    {cls:>3}: {prob.item()*100:5.2f}% {marker}")

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("Logits 的本质:")
    print("  1. 是模型的'原始判断'")
    print("  2. 通过学习到的权重,将特征映射为类别评分")
    print("  3. 未归一化,但可以比较大小")
    print("  4. 通过 softmax 转为概率,用于交叉熵计算")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n\n")
    print("█" * 70)
    print(" " * 15 + "从图像到 Logits 的完整解析")
    print("█" * 70)

    # 1. 可视化生成过程
    visualize_logits_generation()

    # 2. 解释权重含义
    explain_fc_weight_meaning()

    # 3. 真实模型演示
    demonstrate_real_example()