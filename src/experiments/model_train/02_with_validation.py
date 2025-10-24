"""
示例2: 添加验证集和模型评估

在示例1的基础上添加:
- 训练集/验证集划分
- 验证集准确率评估
- model.train() 和 model.eval() 模式切换
- TensorBoard 可视化训练过程
- 过拟合检测

运行: python src/experiments/model_train/02_with_validation.py
查看TensorBoard: tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os


# ============================================================
# 1. 定义相同的CNN模型 (复用示例1的模型)
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
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# 2. 准备数据 - 新增: 训练集/验证集划分
# ============================================================
def prepare_data(val_ratio=0.2):
    """
    加载CIFAR-10数据集并划分训练集和验证集

    参数:
        val_ratio: 验证集比例 (默认20%)

    返回:
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
    """
    print("正在加载 CIFAR-10 数据集...")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载完整的训练集
    full_train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 计算训练集和验证集的大小
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    # 随机划分训练集和验证集
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子,保证可复现
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,  # 验证集不需要打乱
        num_workers=2
    )

    print(f"训练集: {len(train_dataset)} 张图片 ({len(train_loader)} 批)")
    print(f"验证集: {len(val_dataset)} 张图片 ({len(val_loader)} 批)")

    return train_loader, val_loader


# ============================================================
# 3. 训练函数 - 新增: 返回平均loss
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    """
    训练一个epoch

    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epoch: 当前epoch
        writer: TensorBoard writer

    返回:
        avg_loss: 平均训练损失
    """
    model.train()  # 设置为训练模式 (重要! 启用dropout和BN)

    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # ========================================
        # 训练的4个关键步骤
        # ========================================
        # 步骤1: 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 步骤2: 清空梯度
        optimizer.zero_grad()

        # 步骤3: 反向传播
        loss.backward()

        # 步骤4: 更新参数
        optimizer.step()

        # ========================================

        # 累计loss
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # 每100个batch打印一次
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / total_samples
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")

            # 写入TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

    # 计算整个epoch的平均loss
    avg_loss = running_loss / total_samples
    return avg_loss


# ============================================================
# 4. 验证函数 - 新增!
# ============================================================
def validate(model, val_loader, criterion):
    """
    在验证集上评估模型

    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数

    返回:
        avg_loss: 平均验证损失
        accuracy: 验证准确率
    """
    model.eval()  # 设置为评估模式 (重要! 关闭dropout和BN)

    running_loss = 0.0
    correct = 0
    total = 0

    # 验证时不需要计算梯度
    with torch.no_grad():
        for images, labels in val_loader:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 累计loss
            running_loss += loss.item() * images.size(0)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)  # 获取预测类别 (dim=1 表示按行取最大值)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算平均指标
    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# ============================================================
# 5. 主训练循环
# ============================================================
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    完整的训练+验证循环

    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
    """
    print("\n开始训练...")
    print("=" * 70)

    # 创建TensorBoard writer
    log_dir = '/home/seeback/PycharmProjects/DeepLearning/tensor_board/02_with_validation'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志保存到: {log_dir}")
    print(f"查看命令: tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board")
    print("=" * 70)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 外层循环: 遍历每个epoch
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)

        # 训练阶段
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        train_losses.append(train_loss)

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 打印本epoch的统计信息
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] 总结:")
        print(f"  训练Loss: {train_loss:.4f}")
        print(f"  验证Loss: {val_loss:.4f}")
        print(f"  验证准确率: {val_acc:.2f}%")

        # 写入TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # 同时绘制训练和验证loss对比
        writer.add_scalars('Loss/Comparison', {
            'Train': train_loss,
            'Validation': val_loss
        }, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ⭐ 新的最佳验证准确率! 保存模型...")

        # 检测过拟合
        if epoch > 0 and val_loss > train_loss * 1.2:
            print(f"  ⚠️ 警告: 可能出现过拟合 (验证loss明显高于训练loss)")

        print("-" * 70)

    writer.close()
    print("\n✅ 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")


# ============================================================
# 6. 主函数
# ============================================================
def main():
    """主函数:串联所有步骤"""
    print("=" * 70)
    print("示例2: 添加验证集和模型评估")
    print("=" * 70)

    # 步骤1: 准备数据 (新增: 训练/验证划分)
    train_loader, val_loader = prepare_data(val_ratio=0.2)

    # 步骤2: 创建模型
    print("\n创建模型...")
    model = SimpleCNN()

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 步骤3: 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    print(f"损失函数: CrossEntropyLoss")
    print(f"优化器: SGD (lr=0.01, momentum=0.9)")

    # 步骤4: 训练模型
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    print("\n" + "=" * 70)
    print("✅ 训练示例完成!")
    print("=" * 70)
    print("\n💡 新增内容:")
    print("  1. ✅ 训练集/验证集划分 (80%/20%)")
    print("  2. ✅ model.train() 和 model.eval() 模式切换")
    print("  3. ✅ 验证集准确率计算")
    print("  4. ✅ TensorBoard 可视化")
    print("  5. ✅ 过拟合检测")
    print("\n📊 查看训练可视化:")
    print("  tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board")
    print("=" * 70)


if __name__ == '__main__':
    main()
