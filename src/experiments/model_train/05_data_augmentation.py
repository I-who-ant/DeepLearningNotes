"""
示例5: 数据增强 (Data Augmentation)

在示例4的基础上添加:
- 常用数据增强技术 (翻转、旋转、裁剪、颜色变换等)
- 对比有无数据增强的训练效果
- 可视化数据增强效果
- TensorBoard 可视化增强后的图像
- 防止过拟合

运行: python src/experiments/model_train/05_data_augmentation.py
查看TensorBoard: tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


# ============================================================
# 1. 定义相同的CNN模型
# ============================================================
class SimpleCNN(nn.Module):
    """简单的3层卷积神经网络"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 新增: Dropout防止过拟合

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 新增: 在FC层之间应用Dropout
        x = self.fc2(x)
        return x


# ============================================================
# 2. 创建数据增强变换 - 新增!
# ============================================================
def get_transforms(use_augmentation=True):
    """
    获取数据预处理和增强的transforms

    参数:
        use_augmentation: 是否使用数据增强

    返回:
        train_transform: 训练集transform
        val_transform: 验证集transform
    """
    if use_augmentation:
        # 训练集: 应用数据增强
        train_transform = transforms.Compose([
            # 1. 随机水平翻转 (50%概率)
            transforms.RandomHorizontalFlip(p=0.5),

            # 2. 随机裁剪并调整大小
            transforms.RandomCrop(32, padding=4),

            # 3. 随机旋转 (±15度)
            transforms.RandomRotation(15),

            # 4. 颜色抖动 (随机改变亮度、对比度、饱和度、色调)
            transforms.ColorJitter(
                brightness=0.2,    # 亮度变化±20%
                contrast=0.2,      # 对比度变化±20%
                saturation=0.2,    # 饱和度变化±20%
                hue=0.1            # 色调变化±10%
            ),

            # 5. 随机擦除 (模拟遮挡)
            transforms.RandomErasing(
                p=0.3,              # 30%概率
                scale=(0.02, 0.1),  # 擦除区域占2%-10%
                ratio=(0.3, 3.3),   # 长宽比
                value='random'      # 随机填充
            ),

            # 6. 转为Tensor
            transforms.ToTensor(),

            # 7. 标准化
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print("✅ 启用数据增强:")
        print("   - 随机水平翻转 (50%)")
        print("   - 随机裁剪 (padding=4)")
        print("   - 随机旋转 (±15°)")
        print("   - 颜色抖动 (亮度/对比度/饱和度/色调)")
        print("   - 随机擦除 (30%概率)")

    else:
        # 训练集: 不使用数据增强
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        print("❌ 不使用数据增强")

    # 验证集: 始终不使用数据增强
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return train_transform, val_transform


# ============================================================
# 3. 准备数据 - 修改: 支持数据增强
# ============================================================
def prepare_data(use_augmentation=True, val_ratio=0.2):
    """
    加载CIFAR-10数据集并划分训练集和验证集

    参数:
        use_augmentation: 是否使用数据增强
        val_ratio: 验证集比例

    返回:
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
    """
    print(f"\n正在加载 CIFAR-10 数据集 (数据增强: {use_augmentation})...")

    # 获取transforms
    train_transform, val_transform = get_transforms(use_augmentation)

    # 加载完整的训练集
    full_train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=None  # 先不应用transform
    )

    # 划分训练集和验证集
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建子数据集并应用不同的transform
    train_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform),
        train_indices.indices
    )

    val_dataset = torch.utils.data.Subset(
        datasets.CIFAR10(root='./data', train=True, download=False, transform=val_transform),
        val_indices.indices
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")

    return train_loader, val_loader


# ============================================================
# 4. 训练一个epoch
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / total_samples
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")

    avg_loss = running_loss / total_samples
    return avg_loss


# ============================================================
# 5. 验证函数
# ============================================================
def validate(model, val_loader, criterion):
    """在验证集上评估模型"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# ============================================================
# 6. 主训练循环
# ============================================================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, experiment_name, num_epochs=20):
    """
    完整的训练+验证循环

    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        experiment_name: 实验名称
        num_epochs: 训练轮数
    """
    print(f"\n开始训练 ({experiment_name})...")
    print("=" * 70)

    # 创建TensorBoard writer
    log_dir = f'/home/seeback/PycharmProjects/DeepLearning/tensor_board/05_data_augmentation/{experiment_name}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志: {log_dir}")
    print("=" * 70)

    best_val_acc = 0.0
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)

        # 训练阶段
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion)

        # 计算训练集准确率 (用于检测过拟合)
        _, train_acc = validate(model, train_loader, criterion)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 打印统计信息
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] 总结:")
        print(f"  训练Loss: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证Loss: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")

        # 过拟合检测
        if train_acc - val_acc > 10:
            print(f"  ⚠️ 警告: 可能过拟合 (训练准确率比验证准确率高{train_acc - val_acc:.1f}%)")

        # 写入TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Accuracy/Gap', train_acc - val_acc, epoch)  # 过拟合指标

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ⭐ 新的最佳验证准确率!")

        scheduler.step()
        print("-" * 70)

    writer.close()
    print(f"\n✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

    return best_val_acc, train_accs, val_accs


# ============================================================
# 7. 可视化数据增强效果 - 新增!
# ============================================================
def visualize_augmentation():
    """可视化数据增强的效果"""
    print("\n" + "=" * 70)
    print("可视化数据增强效果")
    print("=" * 70)

    # 加载一张原始图片
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    image, label = dataset[0]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 创建数据增强transform
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    # 生成8个增强版本
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'数据增强效果展示 - 原始类别: {class_names[label]}',
                 fontsize=16, fontweight='bold')

    # 第一个显示原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')

    # 其他8个显示增强后的图像
    for i in range(1, 9):
        row = i // 3
        col = i % 3

        # 应用数据增强
        aug_image = augmentation(image)

        # 转换为numpy用于显示
        aug_image_np = aug_image.permute(1, 2, 0).numpy()
        aug_image_np = np.clip(aug_image_np, 0, 1)

        axes[row, col].imshow(aug_image_np)
        axes[row, col].set_title(f'增强版本 {i}', fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()

    # 保存图片
    output_path = 'artifacts/data_augmentation_demo.png'
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 数据增强可视化保存到: {output_path}")

    plt.close()


# ============================================================
# 8. 对比有无数据增强 - 新增!
# ============================================================
def compare_augmentation():
    """对比有无数据增强的训练效果"""
    print("\n" + "=" * 70)
    print("对比实验: 有无数据增强")
    print("=" * 70)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    results = {}

    # 实验1: 不使用数据增强
    print("\n" + "="*70)
    print("实验1: 不使用数据增强")
    print("="*70)

    train_loader, val_loader = prepare_data(use_augmentation=False)
    model1 = SimpleCNN()
    torch.manual_seed(42)

    optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=20)

    best_acc1, train_accs1, val_accs1 = train(
        model1, train_loader, val_loader,
        criterion, optimizer1, scheduler1,
        experiment_name='without_augmentation',
        num_epochs=20
    )
    results['不使用增强'] = best_acc1

    # 实验2: 使用数据增强
    print("\n" + "="*70)
    print("实验2: 使用数据增强")
    print("="*70)

    train_loader, val_loader = prepare_data(use_augmentation=True)
    model2 = SimpleCNN()
    torch.manual_seed(42)

    optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20)

    best_acc2, train_accs2, val_accs2 = train(
        model2, train_loader, val_loader,
        criterion, optimizer2, scheduler2,
        experiment_name='with_augmentation',
        num_epochs=20
    )
    results['使用增强'] = best_acc2

    # 打印对比结果
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)
    print(f"\n{'实验':<15} {'最佳验证准确率':>15} {'提升':>10}")
    print("-" * 42)
    print(f"{'不使用增强':<15} {best_acc1:>14.2f}% {'-':>10}")
    print(f"{'使用增强':<15} {best_acc2:>14.2f}% {f'+{best_acc2-best_acc1:.2f}%':>10}")
    print("=" * 70)

    # 分析过拟合情况
    final_gap1 = train_accs1[-1] - val_accs1[-1]
    final_gap2 = train_accs2[-1] - val_accs2[-1]

    print(f"\n📈 过拟合分析 (训练准确率 - 验证准确率):")
    print(f"  不使用增强: {final_gap1:.2f}% (越大越过拟合)")
    print(f"  使用增强:   {final_gap2:.2f}% (越小越好)")
    print(f"\n💡 结论: 数据增强{'有效' if final_gap2 < final_gap1 else '未明显'}缓解了过拟合")


# ============================================================
# 9. 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 70)
    print("示例5: 数据增强")
    print("=" * 70)

    print("\n📖 数据增强技术说明:")
    print("-" * 70)
    print("""
    1. 几何变换:
       - RandomHorizontalFlip: 随机水平翻转
       - RandomRotation: 随机旋转
       - RandomCrop: 随机裁剪
       - RandomResizedCrop: 随机裁剪并调整大小

    2. 颜色变换:
       - ColorJitter: 亮度/对比度/饱和度/色调抖动
       - RandomGrayscale: 随机转灰度图
       - RandomInvert: 随机反色

    3. 遮挡:
       - RandomErasing: 随机擦除区域 (模拟遮挡)

    4. 作用:
       - 增加训练数据多样性
       - 提高模型泛化能力
       - 防止过拟合
       - 提升验证/测试准确率
    """)

    # 步骤1: 可视化数据增强效果
    visualize_augmentation()

    # 步骤2: 对比有无数据增强的训练效果
    print("\n\n对比实验将训练2个模型,需要一些时间...")
    response = input("是否执行对比实验? (y/n): ")
    if response.lower() == 'y':
        compare_augmentation()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 新增内容:")
    print("  1. ✅ 随机水平翻转")
    print("  2. ✅ 随机旋转")
    print("  3. ✅ 随机裁剪")
    print("  4. ✅ 颜色抖动 (亮度/对比度/饱和度/色调)")
    print("  5. ✅ 随机擦除")
    print("  6. ✅ Dropout防止过拟合")
    print("  7. ✅ 可视化增强效果")
    print("  8. ✅ 对比有无增强的训练效果")
    print("  9. ✅ 过拟合检测和分析")
    print("\n📊 查看训练可视化:")
    print("  tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/05_data_augmentation")
    print("=" * 70)


if __name__ == '__main__':
    main()
