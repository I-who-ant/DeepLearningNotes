"""
示例7: CPU/GPU训练详解

详细说明:
- 如何检测GPU是否可用
- 如何将模型和数据移动到GPU/CPU
- CPU vs GPU训练速度对比
- 多GPU训练基础
- 混合精度训练 (AMP)
- 最佳实践和常见错误

运行: python src/experiments/model_train/07_cpu_gpu_training.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import os


# ============================================================
# 1. 检测和选择设备
# ============================================================
def get_device(prefer_gpu=True):
    """
    智能选择训练设备

    参数:
        prefer_gpu: 如果GPU可用,是否优先使用GPU

    返回:
        device: torch.device对象
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("=" * 70)
        print("🎮 GPU 信息")
        print("=" * 70)
        print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"PyTorch 版本: {torch.__version__}")

        # 显示GPU内存信息
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 总内存: {total_memory:.2f} GB")

        print("=" * 70)
    else:
        device = torch.device('cpu')
        print("=" * 70)
        print("💻 CPU 信息")
        print("=" * 70)
        if not torch.cuda.is_available():
            print("⚠️ 未检测到 GPU,使用 CPU 训练")
            print("💡 提示: CPU训练会比GPU慢很多")
        else:
            print("ℹ️ GPU 可用但选择使用 CPU")

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CPU 线程数: {torch.get_num_threads()}")
        print("=" * 70)

    return device


# ============================================================
# 2. 定义CNN模型
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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 3. 准备数据
# ============================================================
def prepare_data():
    """加载CIFAR-10数据集"""
    print("\n正在加载 CIFAR-10 数据集...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    # 只使用一部分数据来快速演示
    train_subset, _ = random_split(
        train_dataset, [5000, 45000],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_subset, batch_size=128, shuffle=True, num_workers=2
    )

    print(f"训练集: {len(train_subset)} 张图片")

    return train_loader


# ============================================================
# 4. 训练函数 (正确的CPU/GPU处理)
# ============================================================
def train_with_device(model, train_loader, criterion, optimizer, device, num_epochs=3):
    """
    在指定设备上训练模型

    参数:
        model: 模型
        train_loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备 (cpu 或 cuda)
        num_epochs: 训练轮数

    返回:
        avg_time_per_epoch: 每个epoch的平均时间
    """
    print(f"\n开始在 {device} 上训练...")
    print("=" * 70)

    # ========================================
    # 关键步骤1: 将模型移动到指定设备
    # ========================================
    model = model.to(device)
    print(f"✅ 模型已移动到 {device}")

    total_time = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # ========================================
            # 关键步骤2: 将数据移动到指定设备
            # ========================================
            images = images.to(device)  # 推荐使用 .to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每20个batch打印一次
            if (batch_idx + 1) % 20 == 0:
                avg_loss = running_loss / 20
                print(f"  Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0

        epoch_time = time.time() - epoch_start
        total_time += epoch_time

        print(f"  Epoch {epoch+1} 完成,耗时: {epoch_time:.2f} 秒")
        print("-" * 70)

    avg_time = total_time / num_epochs
    print(f"✅ 训练完成! 平均每个epoch: {avg_time:.2f} 秒")

    return avg_time


# ============================================================
# 5. CPU vs GPU 速度对比
# ============================================================
def compare_cpu_gpu():
    """对比CPU和GPU的训练速度"""
    print("\n" + "=" * 70)
    print("⚡ CPU vs GPU 训练速度对比")
    print("=" * 70)

    # 准备数据
    train_loader = prepare_data()
    criterion = nn.CrossEntropyLoss()

    results = {}

    # 测试CPU
    print("\n" + "="*70)
    print("测试 CPU 训练速度")
    print("="*70)

    model_cpu = SimpleCNN()
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01, momentum=0.9)
    device_cpu = torch.device('cpu')

    cpu_time = train_with_device(
        model_cpu, train_loader, criterion, optimizer_cpu,
        device_cpu, num_epochs=2
    )
    results['CPU'] = cpu_time

    # 测试GPU (如果可用)
    if torch.cuda.is_available():
        print("\n" + "="*70)
        print("测试 GPU 训练速度")
        print("="*70)

        model_gpu = SimpleCNN()
        optimizer_gpu = optim.SGD(model_gpu.parameters(), lr=0.01, momentum=0.9)
        device_gpu = torch.device('cuda')

        # 清空GPU缓存
        torch.cuda.empty_cache()

        gpu_time = train_with_device(
            model_gpu, train_loader, criterion, optimizer_gpu,
            device_gpu, num_epochs=2
        )
        results['GPU'] = gpu_time

        # 打印对比结果
        print("\n" + "=" * 70)
        print("📊 速度对比结果")
        print("=" * 70)
        print(f"\nCPU 平均每个epoch: {cpu_time:.2f} 秒")
        print(f"GPU 平均每个epoch: {gpu_time:.2f} 秒")
        print(f"\n🚀 GPU 比 CPU 快: {cpu_time / gpu_time:.2f} 倍")
        print("=" * 70)
    else:
        print("\n⚠️ GPU 不可用,无法进行对比测试")
        print("=" * 70)


# ============================================================
# 6. 常见错误示例
# ============================================================
def common_mistakes():
    """演示常见的CPU/GPU错误"""
    print("\n" + "=" * 70)
    print("⚠️ 常见错误示例")
    print("=" * 70)

    model = SimpleCNN()

    print("\n错误1: 模型在CPU,数据在GPU (或相反)")
    print("-" * 70)
    print("代码示例:")
    print("""
    model = SimpleCNN()  # 模型在CPU
    images = images.cuda()  # 数据在GPU
    outputs = model(images)  # ❌ 错误! RuntimeError
    """)
    print("错误信息: RuntimeError: Expected all tensors to be on the same device")

    print("\n正确做法:")
    print("""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # 模型移到设备
    images = images.to(device)  # 数据移到设备
    outputs = model(images)  # ✅ 正确!
    """)

    print("\n错误2: 直接使用 .cuda() 而不检查GPU是否可用")
    print("-" * 70)
    print("代码示例:")
    print("""
    model = model.cuda()  # ❌ 如果没有GPU会报错
    images = images.cuda()  # ❌ RuntimeError: CUDA not available
    """)

    print("\n正确做法:")
    print("""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # ✅ 自动适配
    images = images.to(device)  # ✅ 自动适配
    """)

    print("\n错误3: 忘记将损失函数的输入移到GPU")
    print("-" * 70)
    print("代码示例:")
    print("""
    model = model.cuda()
    images = images.cuda()
    labels = labels  # ❌ 忘记移到GPU
    outputs = model(images)
    loss = criterion(outputs, labels)  # ❌ 错误!
    """)

    print("\n正确做法:")
    print("""
    model = model.to(device)
    images = images.to(device)
    labels = labels.to(device)  # ✅ 标签也要移到设备
    outputs = model(images)
    loss = criterion(outputs, labels)  # ✅ 正确!
    """)


# ============================================================
# 7. 最佳实践
# ============================================================
def best_practices():
    """CPU/GPU使用的最佳实践"""
    print("\n" + "=" * 70)
    print("💡 CPU/GPU 使用最佳实践")
    print("=" * 70)

    print("""
1. 始终使用 device 对象
   ✅ 推荐:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = model.to(device)
      data = data.to(device)

   ❌ 不推荐:
      model = model.cuda()  # 不够灵活
      data = data.cuda()

2. 在训练开始时移动模型,在循环内移动数据
   ✅ 推荐:
      model = model.to(device)  # 只移动一次
      for images, labels in loader:
          images = images.to(device)  # 每个batch移动
          labels = labels.to(device)

3. 使用 .to(device, non_blocking=True) 加速数据传输
   ✅ 推荐:
      images = images.to(device, non_blocking=True)
      # 允许CPU和GPU异步执行

4. GPU训练完成后,将结果移回CPU (用于保存或可视化)
   ✅ 推荐:
      predictions = model(images)  # GPU上计算
      predictions = predictions.cpu()  # 移回CPU
      predictions = predictions.numpy()  # 转为numpy

5. 定期清空GPU缓存 (避免内存泄漏)
   ✅ 推荐:
      torch.cuda.empty_cache()

6. 使用多个worker加载数据 (CPU预处理,GPU训练)
   ✅ 推荐:
      DataLoader(dataset, batch_size=64, num_workers=4)
      # CPU并行加载数据,GPU专注训练

7. 混合精度训练 (GPU专用,更快更省内存)
   ✅ 推荐 (如果有GPU):
      from torch.cuda.amp import autocast, GradScaler
      scaler = GradScaler()

      with autocast():  # 自动使用FP16
          outputs = model(images)
          loss = criterion(outputs, labels)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
    """)


# ============================================================
# 8. 检查当前环境
# ============================================================
def check_environment():
    """检查当前环境的GPU配置"""
    print("\n" + "=" * 70)
    print("🔍 环境检查")
    print("=" * 70)

    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
    else:
        print("\n⚠️ 当前环境没有可用的GPU")
        print("💡 你正在使用 CPU 进行训练")
        print("\n如何获得GPU支持:")
        print("  1. 使用云平台: Google Colab, Kaggle, AWS, Azure")
        print("  2. 本地安装: 需要NVIDIA显卡 + CUDA + cuDNN")
        print("  3. 购买GPU服务器: 阿里云、腾讯云等")

    print("\n当前默认设备:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  {device}")

    print("=" * 70)


# ============================================================
# 9. 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 70)
    print("示例7: CPU/GPU 训练详解")
    print("=" * 70)

    # 1. 检查环境
    check_environment()

    # 2. 演示正确的设备选择
    device = get_device(prefer_gpu=True)

    # 3. 常见错误
    common_mistakes()

    # 4. 最佳实践
    best_practices()

    # 5. 速度对比 (可选)
    print("\n" + "=" * 70)
    response = input("是否进行 CPU vs GPU 速度对比测试? (y/n): ")
    if response.lower() == 'y':
        compare_cpu_gpu()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 关键要点:")
    print("  1. ✅ 使用 torch.device 自动适配 CPU/GPU")
    print("  2. ✅ 模型和数据都要移到同一设备")
    print("  3. ✅ 优先使用 .to(device) 而不是 .cuda()")
    print("  4. ✅ GPU 训练速度通常是 CPU 的 10-100 倍")
    print("  5. ✅ 没有 GPU 时,CPU 训练完全可行,只是慢一些")
    print("\n📖 你当前的情况:")
    if torch.cuda.is_available():
        print("  ✅ 你有可用的 GPU,建议使用 GPU 训练")
    else:
        print("  💻 你没有 GPU,使用 CPU 训练")
        print("     CPU 训练完全没问题,只是需要更多时间")
        print("     建议: 减小模型大小或数据量来加快训练")
    print("=" * 70)


if __name__ == '__main__':
    main()
