"""
示例3: 模型保存和加载

在示例2的基础上添加:
- 训练过程中保存checkpoint
- 保存最佳模型
- 从checkpoint恢复训练
- TensorBoard 可视化
- 模型加载和推理

运行: python src/experiments/model_train/03_save_load_model.py
查看TensorBoard: tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
from datetime import datetime


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
# 3. 训练一个epoch, 并在每个batch打印损失
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader): # 遍历每个batch
        outputs = model(images)
        loss = criterion(outputs, labels) # 计算损失

        optimizer.zero_grad() # 清空梯度
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / total_samples
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

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
# 5. 保存checkpoint - 新增!
# ============================================================
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, checkpoint_dir, is_best=False):
    """
    保存训练checkpoint , 包含模型状态、优化器状态、损失和准确率

    参数:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        train_loss: 训练损失
        val_loss: 验证损失
        val_acc: 验证准确率
        checkpoint_dir: 保存目录
        is_best: 是否是最佳模型
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 构建checkpoint字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'date': datetime.now().isoformat(),
        'model_architecture': 'SimpleCNN',
        'num_classes': 10
    }

    # 保存最新的checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"  💾 保存checkpoint: {latest_path}")

    # 如果是最佳模型,额外保存一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  ⭐ 保存最佳模型: {best_path}")

    # 每5个epoch保存一个带编号的checkpoint
    if (epoch + 1) % 5 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"  📌 保存epoch checkpoint: {epoch_path}")


# ============================================================
# 6. 加载checkpoint - 新增!
# ============================================================
def load_checkpoint(model, optimizer, checkpoint_path):
    """
    从checkpoint恢复训练

    参数:
        model: 模型
        optimizer: 优化器
        checkpoint_path: checkpoint文件路径

    返回:
        start_epoch: 应该从哪个epoch开始训练
        best_val_acc: 之前的最佳验证准确率
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint 不存在: {checkpoint_path}")
        return 0, 0.0

    print(f"\n📂 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复训练状态
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['val_acc']

    print(f"  ✅ 恢复成功!")
    print(f"     上次训练到 Epoch {checkpoint['epoch']}")
    print(f"     验证准确率: {checkpoint['val_acc']:.2f}%")
    print(f"     验证Loss: {checkpoint['val_loss']:.4f}")
    print(f"     保存时间: {checkpoint['date']}")

    return start_epoch, best_val_acc


# ============================================================
# 7. 主训练循环 - 新增: checkpoint保存
# ============================================================
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, resume_from=None):
    """
    完整的训练+验证循环,支持checkpoint保存和恢复

    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        resume_from: 从哪个checkpoint恢复 (None表示从头开始)
    """
    print("\n开始训练...")
    print("=" * 70)

    # 创建checkpoint保存目录
    checkpoint_dir = 'artifacts/checkpoints/03_save_load_model'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint 保存目录: {checkpoint_dir}")

    # 创建TensorBoard writer
    log_dir = '/home/seeback/PycharmProjects/DeepLearning/tensor_board/03_save_load_model'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志: {log_dir}")
    print("=" * 70)

    # 尝试从checkpoint恢复
    start_epoch = 0
    best_val_acc = 0.0

    if resume_from is not None:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, resume_from)
        print(f"\n▶️ 从 Epoch {start_epoch} 继续训练...")
    else:
        print(f"\n▶️ 从头开始训练...")

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 70)

        # 训练阶段
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion)

        # 打印统计信息
        print(f"\n📊 Epoch [{epoch+1}/{num_epochs}] 总结:")
        print(f"  训练Loss: {train_loss:.4f}")
        print(f"  验证Loss: {val_loss:.4f}")
        print(f"  验证准确率: {val_acc:.2f}%")

        # 写入TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # 判断是否是最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  ⭐ 新的最佳验证准确率!")

        # 保存checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            train_loss, val_loss, val_acc,
            checkpoint_dir, is_best=is_best
        )

        print("-" * 70)

    writer.close()
    print("\n✅ 训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    return best_val_acc


# ============================================================
# 8. 演示加载模型进行推理 - 新增!
# ============================================================
def demo_inference(checkpoint_path):
    """
    演示如何加载保存的模型进行推理

    参数:
        checkpoint_path: 模型checkpoint路径
    """
    print("\n" + "=" * 70)
    print("演示: 加载模型进行推理")
    print("=" * 70)

    # 1. 创建模型实例
    model = SimpleCNN() # 创建模型实例
    model.eval()

    # 2. 加载模型权重
    print(f"\n1️⃣ 加载模型: {checkpoint_path}") # 加载模型checkpoint
    checkpoint = torch.load(checkpoint_path) # 加载checkpoint文件
    model.load_state_dict(checkpoint['model_state_dict']) # 加载模型状态字典
    print(f"   ✅ 加载成功! (验证准确率: {checkpoint['val_acc']:.2f}%)") # 打印加载成功信息

    # 3. 准备测试数据
    print(f"\n2️⃣ 加载测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化, 使像素值在[-1, 1]之间
    ])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. 在测试集上评估
    print(f"\n3️⃣ 在测试集上评估...")
    criterion = nn.CrossEntropyLoss() # 定义损失函数
    test_loss, test_acc = validate(model, test_loader, criterion) # 在测试集上评估模型

    print(f"\n📊 测试集结果:")
    print(f"  测试Loss: {test_loss:.4f}")
    print(f"  测试准确率: {test_acc:.2f}%")

    # 5. 单张图片推理示例
    print(f"\n4️⃣ 单张图片推理示例...")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 获取一张图片
    image, true_label = test_dataset[0]
    image_batch = image.unsqueeze(0)  # 添加batch维度

    # 推理
    with torch.no_grad():
        output = model(image_batch)
        _, predicted = torch.max(output, 1)

    print(f"  真实标签: {class_names[true_label]}")
    print(f"  预测标签: {class_names[predicted.item()]}")
    print(f"  预测正确: {'✅' if predicted.item() == true_label else '❌'}")

    print("=" * 70)


# ============================================================
# 9. 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 70)
    print("示例3: 模型保存和加载")
    print("=" * 70)

    # 步骤1: 准备数据
    train_loader, val_loader = prepare_data(val_ratio=0.2)

    # 步骤2: 创建模型
    print("\n创建模型...")
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 步骤3: 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 步骤4: 训练模型 (可以指定 resume_from 来恢复训练)
    # resume_from = 'artifacts/checkpoints/03_save_load_model/latest_checkpoint.pth'  # 从上次中断处继续
    resume_from = None  # 从头开始

    best_acc = train(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=15,
        resume_from=resume_from
    )

    # 步骤5: 演示加载模型进行推理
    best_model_path = 'artifacts/checkpoints/03_save_load_model/best_model.pth'
    if os.path.exists(best_model_path):
        demo_inference(best_model_path)

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 新增内容:")
    print("  1. ✅ 保存checkpoint (模型+优化器+训练状态)")
    print("  2. ✅ 保存最佳模型")
    print("  3. ✅ 每5个epoch保存一个checkpoint")
    print("  4. ✅ 从checkpoint恢复训练")
    print("  5. ✅ 加载模型进行推理")
    print("\n💾 保存的checkpoint:")
    print("  - latest_checkpoint.pth  (最新)")
    print("  - best_model.pth         (最佳)")
    print("  - checkpoint_epoch_X.pth (每5个epoch)")
    print("\n📊 查看训练可视化:")
    print("  tensorboard --logdir=/home/seeback/PycharmProjects/DeepLearning/tensor_board")
    print("=" * 70)


if __name__ == '__main__':
    main()
