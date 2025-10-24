"""
示例4: 学习率调度 (Learning Rate Scheduling)

在示例3的基础上添加:
- 学习率调度器 (StepLR, CosineAnnealingLR, ReduceLROnPlateau)
- 学习率可视化
- 对比不同学习率策略的效果
- TensorBoard 可视化学习率变化

运行: python src/experiments/model_train/04_lr_scheduler.py
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

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# 2. 准备数据
# ============================================================
def prepare_data(val_ratio=0.2):
    """加载CIFAR-10数据集并划分训练集和验证集"""
    print("正在加载 CIFAR-10 数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")

    return train_loader, val_loader


# ============================================================
# 3. 训练一个epoch
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
# 4. 验证函数
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
# 5. 主训练循环 - 新增: 学习率调度器
# ============================================================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, scheduler_name, num_epochs=20):
    """
    完整的训练+验证循环,包含学习率调度

    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器 (新增!)
        scheduler_name: 调度器名称 (用于日志)
        num_epochs: 训练轮数
    """
    print(f"\n开始训练 (学习率调度器: {scheduler_name})...")
    print("=" * 70)

    # 创建TensorBoard writer
    log_dir = f'/home/seeback/PycharmProjects/DeepLearning/tensor_board/04_lr_scheduler/{scheduler_name}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志: {log_dir}")
    print("=" * 70)

    best_val_acc = 0.0

    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  当前学习率: {current_lr:.6f}")

        # 训练阶段
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion)

        # 打印统计信息
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] 总结:")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  训练Loss: {train_loss:.4f}")
        print(f"  验证Loss: {val_loss:.4f}")
        print(f"  验证准确率: {val_acc:.2f}%")

        # 写入TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)  # 新增: 记录学习率

        # 更新最佳准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ⭐ 新的最佳验证准确率!")

        # ========================================
        # 关键: 更新学习率调度器
        # ========================================
        if scheduler_name == 'ReduceLROnPlateau':
            # ReduceLROnPlateau 需要 metric 参数
            scheduler.step(val_loss)
        else:
            # StepLR, CosineAnnealingLR 等不需要参数
            scheduler.step()

        print("-" * 70)

    writer.close()
    print(f"\n✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

    return best_val_acc


# ============================================================
# 6. 创建学习率调度器 - 新增!
# ============================================================
def create_scheduler(optimizer, scheduler_type):
    """
    创建学习率调度器

    参数:
        optimizer: 优化器
        scheduler_type: 调度器类型

    返回:
        scheduler: 学习率调度器
    """
    if scheduler_type == 'StepLR':
        # 每 step_size 个epoch降低学习率
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,    # 每5个epoch
            gamma=0.5       # 学习率乘以0.5
        )
        print("  StepLR: 每5个epoch, 学习率 × 0.5")

    elif scheduler_type == 'MultiStepLR':
        # 在指定的epoch降低学习率
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[6, 12, 18],  # 在第6, 12, 18个epoch
            gamma=0.5                # 学习率乘以0.5
        )
        print("  MultiStepLR: 在epoch [6, 12, 18], 学习率 × 0.5")

    elif scheduler_type == 'CosineAnnealingLR':
        # 余弦退火: 学习率按余弦曲线变化
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,      # 周期长度
            eta_min=1e-6   # 最小学习率
        )
        print("  CosineAnnealingLR: T_max=20, eta_min=1e-6")

    elif scheduler_type == 'ReduceLROnPlateau':
        # 当指标停止改善时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # 监控指标越小越好 (loss)
            factor=0.5,      # 学习率乘以0.5
            patience=3,      # 容忍3个epoch不改善
            verbose=True     # 打印信息
        )
        print("  ReduceLROnPlateau: patience=3, factor=0.5")

    elif scheduler_type == 'ExponentialLR':
        # 指数衰减
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95  # 每个epoch学习率乘以0.95
        )
        print("  ExponentialLR: gamma=0.95")

    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")

    return scheduler


# ============================================================
# 7. 对比不同调度器 - 新增!
# ============================================================
def compare_schedulers():
    """对比不同学习率调度器的效果"""
    print("\n" + "=" * 70)
    print("对比不同学习率调度器")
    print("=" * 70)

    # 准备数据 (所有实验使用相同数据)
    train_loader, val_loader = prepare_data(val_ratio=0.2)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 要对比的调度器
    scheduler_types = [
        'StepLR',
        'CosineAnnealingLR',
        'ReduceLROnPlateau'
    ]

    results = {}

    for scheduler_type in scheduler_types:
        print(f"\n{'='*70}")
        print(f"测试调度器: {scheduler_type}")
        print(f"{'='*70}")

        # 创建新模型 (确保每个实验从相同初始状态开始)
        model = SimpleCNN()
        torch.manual_seed(42)  # 固定随机种子

        # 创建优化器 (初始学习率都是0.01)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # 创建调度器
        scheduler = create_scheduler(optimizer, scheduler_type)

        # 训练
        best_acc = train(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            scheduler_name=scheduler_type,
            num_epochs=20
        )

        results[scheduler_type] = best_acc

    # 打印对比结果
    print("\n" + "=" * 70)
    print("📊 调度器对比结果")
    print("=" * 70)
    print(f"\n{'调度器':<25} {'最佳验证准确率':>15}")
    print("-" * 42)
    for scheduler_type, acc in results.items():
        print(f"{scheduler_type:<25} {acc:>14.2f}%")
    print("=" * 70)


# ============================================================
# 8. 可视化学习率变化 - 新增!
# ============================================================
def visualize_lr_schedules():
    """可视化不同学习率调度器的变化曲线"""
    print("\n" + "=" * 70)
    print("可视化学习率调度")
    print("=" * 70)

    # 创建一个简单模型 (只是为了创建optimizer)
    model = SimpleCNN()

    scheduler_types = ['StepLR', 'CosineAnnealingLR', 'ExponentialLR']

    for scheduler_type in scheduler_types:
        # 创建optimizer和scheduler
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = create_scheduler(optimizer, scheduler_type)

        # 创建TensorBoard writer
        log_dir = f'/home/seeback/PycharmProjects/DeepLearning/tensor_board/04_lr_scheduler/viz_{scheduler_type}'
        writer = SummaryWriter(log_dir)

        # 模拟20个epoch,记录学习率变化
        print(f"\n{scheduler_type} 学习率变化:")
        for epoch in range(20):
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', lr, epoch)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch:2d}: lr = {lr:.6f}")

            # 更新学习率
            if scheduler_type != 'ReduceLROnPlateau':
                scheduler.step()
            else:
                # ReduceLROnPlateau需要metric,这里模拟一个递减的loss
                fake_loss = 2.0 - epoch * 0.05
                scheduler.step(fake_loss)

        writer.close()

    print("\n✅ 学习率可视化完成!")
    print("=" * 70)


# ============================================================
# 9. 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 70)
    print("示例4: 学习率调度")
    print("=" * 70)

    print("\n📖 学习率调度器说明:")
    print("-" * 70)
    print("""
    1. StepLR
       - 每隔固定epoch降低学习率
       - 简单稳定,常用于基础训练
       - 示例: 每5个epoch, lr × 0.5

    2. CosineAnnealingLR
       - 学习率按余弦曲线变化
       - 平滑衰减,常用于fine-tuning
       - 先快后慢,有利于收敛

    3. ReduceLROnPlateau
       - 当验证loss不再下降时降低学习率
       - 自适应,无需手动设置epoch
       - 适合不确定训练轮数的情况

    4. ExponentialLR
       - 指数衰减
       - 每个epoch学习率乘以固定系数
       - 衰减速度可控

    5. MultiStepLR
       - 在指定的epoch降低学习率
       - 灵活,可以自定义降低时机
    """)

    # 选项1: 可视化学习率变化曲线
    print("\n选项1: 可视化学习率变化曲线")
    visualize_lr_schedules()

    # 选项2: 对比不同调度器的训练效果
    print("\n\n选项2: 对比不同调度器的训练效果")
    print("(这将训练3个模型,需要一些时间...)")
    response = input("是否执行对比实验? (y/n): ")
    if response.lower() == 'y':
        compare_schedulers()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 新增内容:")
    print("  1. ✅ StepLR - 固定步长降低学习率")
    print("  2. ✅ CosineAnnealingLR - 余弦退火")
    print("  3. ✅ ReduceLROnPlateau - 自适应降低学习率")
    print("  4. ✅ ExponentialLR - 指数衰减")
    print("  5. ✅ MultiStepLR - 多阶段降低学习率")
    print("  6. ✅ 学习率可视化")
    print("  7. ✅ 对比不同调度器效果")
    print("\n📊 查看训练可视化:")
    print("  tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board/04_lr_scheduler")
    print("=" * 70)


if __name__ == '__main__':
    main()
