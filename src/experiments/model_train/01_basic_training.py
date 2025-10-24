"""
示例1: 最基础的模型训练循环

这是最简单的训练示例，展示：
- 如何定义简单的CNN模型
- 如何加载CIFAR-10数据集
- 如何使用backward()和step()优化
- 如何打印每个epoch的loss

运行: python src/experiments/model_train/01_basic_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. 定义一个简单的CNN模型
# ============================================================
class SimpleCNN(nn.Module):
    """简单的3层卷积神经网络"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)      # 32x32x3 -> 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)     # 16x16x32 -> 16x16x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    # 8x8x64 -> 8x8x128
        self.pool = nn.MaxPool2d(2, 2)                   # 下采样
        self.fc1 = nn.Linear(128 * 4 * 4, 256)           # 全连接层
        self.fc2 = nn.Linear(256, 10)                    # 输出层 (10类)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))    # -> 16x16x32
        x = self.pool(self.relu(self.conv2(x)))    # -> 8x8x64
        x = self.pool(self.relu(self.conv3(x)))    # -> 4x4x128
        x = x.view(x.size(0), -1)                  # 展平
        x = self.relu(self.fc1(x)) # -> 256
        x = self.fc2(x) # -> 10
        return x


# ============================================================
# 2. 准备数据
# ============================================================
def prepare_data():
    """加载CIFAR-10数据集"""
    print("正在加载 CIFAR-10 数据集...")

    # 数据预处理：转为Tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 创建DataLoader (批量加载)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,      # 每批64张图片
        shuffle=True,       # 打乱数据
        num_workers=2       # 多线程加载
    )

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"批次数: {len(train_loader)} 批")

    return train_loader


# ============================================================
# 3. 训练函数 - 核心部分！
# ============================================================
def train(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    训练模型

    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
    """
    print("\n开始训练...")
    print("=" * 60)

    # 设置设备 (自动检测GPU,如果没有GPU则使用CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    print("=" * 60)

    model.train()  # 设置为训练模式, 意思是将模型设置为训练状态, 启用 dropout 和 batch normalization 等层的训练行为
                    # 这些训练行为可以确保模型在训练过程中能够正常工作,normalization 层会在训练过程中根据每个 batch 的统计信息进行归一化,
                    # 而不是使用整个数据集的统计信息, 这可以防止模型在训练过程中对某些特征的过拟合, 提高模型的泛化能力

    # 外层循环: 遍历每个epoch
    for epoch in range(num_epochs):
        running_loss = 0.0  # 累计loss

        # 内层循环: 遍历每个batch
        for batch_idx, (images, labels) in enumerate(train_loader):

            # 将数据移动到设备 (CPU或GPU)
            images = images.to(device)
            labels = labels.to(device)

            # images: [64, 3, 32, 32] - 64张32x32的RGB图片
            # labels: [64] - 64个标签

            # ========================================
            # 训练的4个关键步骤
            # ========================================

            # 步骤1: 前向传播 (Forward)
            outputs = model(images)           # 模型预测
            loss = criterion(outputs, labels)  # 计算损失

            # 步骤2: 清空梯度 (非常重要!)
            optimizer.zero_grad()

            # 步骤3: 反向传播 (Backward)
            loss.backward()  # 计算梯度

            # 步骤4: 更新参数 (Step)
            optimizer.step()  # 根据梯度更新权重

            # ========================================

            # 累计loss (用于显示)
            running_loss += loss.item()

            # 每100个batch打印一次
            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # 每个epoch结束后打印分隔线
        print("-" * 60)

    print("训练完成!")


# ============================================================
# 4. 主函数
# ============================================================
def main():
    """主函数：串联所有步骤"""
    print("=" * 60)
    print("示例1: 最基础的模型训练")
    print("=" * 60)

    # 步骤1: 准备数据
    train_loader = prepare_data()

    # 步骤2: 创建模型
    print("\n创建模型...")
    model = SimpleCNN()

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 步骤3: 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.SGD(              # SGD优化器
        model.parameters(),
        lr=0.01,                        # 学习率
        momentum=0.9                    # 动量
    )

    print(f"损失函数: CrossEntropyLoss")
    print(f"优化器: SGD (lr=0.01, momentum=0.9)")

    # 步骤4: 训练模型
    train(model, train_loader, criterion, optimizer, num_epochs=5)

    print("\n" + "=" * 60)
    print("✅ 训练示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
