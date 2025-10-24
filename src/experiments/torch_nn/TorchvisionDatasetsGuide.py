"""
torchvision.datasets 完全使用指南

这个模块详细演示如何使用 torchvision.datasets 下载和使用各种数据集：
1. 常用数据集的下载和加载
2. 数据预处理和增强 (transforms)
3. DataLoader 的使用
4. 自定义数据集
5. 数据可视化

重要说明:
- torchvision.datasets 内置了数据下载功能,不需要 scipy
- scipy 是科学计算库,主要用于数学运算,不用于下载数据集
- torchvision.datasets 会自动从官方源下载数据

作者: Seeback
日期: 2025-10-23
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def demo_available_datasets():
    """演示 torchvision.datasets 中可用的数据集"""
    print("=" * 70)
    print("torchvision.datasets 中可用的数据集")
    print("=" * 70)

    dataset_categories = {
        "图像分类数据集": {
            "MNIST": "手写数字 (60K训练 + 10K测试, 28x28 灰度)",
            "FashionMNIST": "时尚物品 (60K训练 + 10K测试, 28x28 灰度)",
            "CIFAR10": "10类物体 (50K训练 + 10K测试, 32x32 彩色)",
            "CIFAR100": "100类物体 (50K训练 + 10K测试, 32x32 彩色)",
            "ImageNet": "1000类物体 (需手动下载, 需要注册)",
            "STL10": "10类物体 (5K训练 + 8K测试, 96x96 彩色)",
        },
        "目标检测数据集": {
            "VOCDetection": "PASCAL VOC 目标检测",
            "VOCSegmentation": "PASCAL VOC 语义分割",
            "CocoDetection": "COCO 目标检测 (需手动下载)",
        },
        "其他数据集": {
            "SVHN": "街景门牌号 (73K训练 + 26K测试)",
            "EMNIST": "扩展的 MNIST 数据集",
            "KMNIST": "日文平假名手写字符",
            "Omniglot": "多语言手写字符",
        }
    }

    for category, datasets_dict in dataset_categories.items():
        print(f"\n📦 {category}:")
        print("-" * 70)
        for name, description in datasets_dict.items():
            print(f"   {name:<20} - {description}")

    print("\n" + "=" * 70)
    print("💡 提示: 大多数数据集都支持自动下载 (download=True)")
    print("=" * 70)


def demo_download_and_load_datasets():
    """演示下载和加载数据集"""
    print("\n" + "=" * 70)
    print("下载和加载数据集")
    print("=" * 70)

    print("\n1️⃣ 基本用法 - 下载 MNIST 数据集:")
    print("-" * 70)
    print("""
    from torchvision import datasets

    # 下载训练集
    train_dataset = datasets.MNIST(
        root='./data',          # 数据保存路径
        train=True,             # 加载训练集
        download=True,          # 如果不存在则下载
        transform=None          # 数据预处理 (可选)
    )

    # 下载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,            # 加载测试集
        download=True
    )
    """)

    print("\n2️⃣ 实际加载 MNIST 数据集:")
    print("-" * 70)

    # 实际加载 (只加载一小部分演示)
    try:
        train_dataset = datasets.MNIST( # 加载 MNIST 训练集, 并将图像转换为 Tensor
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        print(f"✅ MNIST 训练集加载成功!")
        print(f"   样本数量: {len(train_dataset)}")
        print(f"   图像形状: {train_dataset[0][0].shape}")
        print(f"   标签范围: 0-9 (10个类别)")

        # 查看第一个样本
        image, label = train_dataset[0]
        print(f"\n   第一个样本:")
        print(f"   - 图像形状: {image.shape}")
        print(f"   - 标签: {label}")
        print(f"   - 像素值范围: [{image.min():.3f}, {image.max():.3f}]")

    except Exception as e:
        print(f"❌ 加载失败: {e}")

    print("\n3️⃣ 加载 CIFAR-10 数据集:")
    print("-" * 70)

    try:
        cifar10_train = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

        print(f"✅ CIFAR-10 训练集加载成功!")
        print(f"   样本数量: {len(cifar10_train)}")
        print(f"   图像形状: {cifar10_train[0][0].shape}")
        print(f"   类别: {cifar10_train.classes}")

    except Exception as e:
        print(f"❌ 加载失败: {e}")


def demo_transforms():
    """演示数据预处理和增强"""
    print("\n" + "=" * 70)
    print("数据预处理和增强 (transforms)")
    print("=" * 70)

    print("\n1️⃣ 基本转换:")
    print("-" * 70)
    print("""
    from torchvision import transforms

    # 转换为 Tensor
    transform = transforms.ToTensor()

    # 归一化 (mean, std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 灰度图
    ])

    # RGB 图像归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),  # ImageNet mean
                            (0.229, 0.224, 0.225))   # ImageNet std
    ])
    """)

    print("\n2️⃣ 数据增强 (Data Augmentation):")
    print("-" * 70)
    print("""
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),    # 随机水平翻转
        transforms.RandomRotation(10),              # 随机旋转 ±10度
        transforms.RandomCrop(32, padding=4),       # 随机裁剪
        transforms.ColorJitter(                     # 颜色抖动
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 测试集不需要增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    """)

    print("\n3️⃣ 实际应用 transforms:")
    print("-" * 70)

    # 定义 transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform  # 应用训练集增强
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform  # 应用测试集转换
    )

    print(f"✅ 数据集加载完成!")
    print(f"   训练集: {len(train_dataset)} 样本 (带增强)")
    print(f"   测试集: {len(test_dataset)} 样本 (无增强)")

    return train_dataset, test_dataset


def demo_dataloader():
    """演示 DataLoader 的使用"""
    print("\n" + "=" * 70)
    print("DataLoader - 批量数据加载")
    print("=" * 70)

    print("\n1️⃣ DataLoader 基本用法:")
    print("-" * 70)
    print("""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,          # 批次大小
        shuffle=True,           # 打乱数据
        num_workers=2,          # 并行加载进程数
        pin_memory=True         # 固定内存 (GPU训练时加速)
    )

    # 迭代数据
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: [batch_size, channels, height, width]
        # labels: [batch_size]
        print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
    """)

    print("\n2️⃣ 实际创建 DataLoader:")
    print("-" * 70)

    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # 设置为 0 避免多进程问题
    )

    print(f"✅ DataLoader 创建成功!")
    print(f"   总样本数: {len(train_dataset)}")
    print(f"   批次大小: {train_loader.batch_size}")
    print(f"   总批次数: {len(train_loader)}")

    # 获取一个批次
    images, labels = next(iter(train_loader))
    print(f"\n   单个批次:")
    print(f"   - 图像形状: {images.shape}")
    print(f"   - 标签形状: {labels.shape}")
    print(f"   - 标签内容: {labels[:10].tolist()}")

    print("\n3️⃣ 不同配置的 DataLoader:")
    print("-" * 70)

    configs = [
        {"batch_size": 16, "shuffle": True, "name": "小批次 + 打乱"},
        {"batch_size": 128, "shuffle": True, "name": "大批次 + 打乱"},
        {"batch_size": 32, "shuffle": False, "name": "中批次 + 不打乱"},
    ]

    for config in configs:
        loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=0
        )
        print(f"   {config['name']:<20} - 批次数: {len(loader)}")

    return train_loader


def demo_dataset_split():
    """演示数据集划分"""
    print("\n" + "=" * 70)
    print("数据集划分 - 训练集 / 验证集 / 测试集")
    print("=" * 70)

    print("\n1️⃣ 划分训练集和验证集:")
    print("-" * 70)
    print("""
    from torch.utils.data import random_split

    # 加载完整训练集
    full_train = datasets.CIFAR10(root='./data', train=True, ...)

    # 划分为训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size]
    )
    """)

    print("\n2️⃣ 实际划分数据集:")
    print("-" * 70)

    # 加载数据
    full_train = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # 划分
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])# random_split: 随机划分数据集

    print(f"✅ 数据集划分完成!")
    print(f"   原始训练集: {len(full_train)} 样本")
    print(f"   新训练集: {len(train_dataset)} 样本 (90%)")
    print(f"   验证集: {len(val_dataset)} 样本 (10%)")

    # 创建对应的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"\n   训练 DataLoader: {len(train_loader)} 批次")
    print(f"   验证 DataLoader: {len(val_loader)} 批次")

    return train_loader, val_loader


def demo_custom_dataset():
    """演示自定义数据集"""
    print("\n" + "=" * 70)
    print("自定义数据集 - 继承 Dataset 类")
    print("=" * 70)

    print("\n1️⃣ 自定义数据集的基本结构:")
    print("-" * 70)
    print("""
    from torch.utils.data import Dataset

    class CustomDataset(Dataset):
        def __init__(self, data, labels, transform=None):
            '''
            初始化数据集
            '''
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            '''
            返回数据集大小
            '''
            return len(self.data)

        def __getitem__(self, idx):
            '''
            根据索引返回一个样本
            '''
            image = self.data[idx]
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
    """)

    print("\n2️⃣ 实际创建自定义数据集:")
    print("-" * 70)

    class SimpleDataset(Dataset):
        """简单的随机数据集"""
        def __init__(self, num_samples, image_size, num_classes):
            self.num_samples = num_samples
            self.image_size = image_size
            self.num_classes = num_classes

            # 生成随机数据
            self.data = torch.randn(num_samples, 3, image_size, image_size)
            self.labels = torch.randint(0, num_classes, (num_samples,))

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # 创建数据集
    custom_dataset = SimpleDataset(num_samples=1000, image_size=32, num_classes=10)

    print(f"✅ 自定义数据集创建成功!")
    print(f"   样本数量: {len(custom_dataset)}")
    print(f"   图像形状: {custom_dataset[0][0].shape}")
    print(f"   标签范围: 0-{custom_dataset.num_classes-1}")

    # 创建 DataLoader
    custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
    print(f"   批次数: {len(custom_loader)}")

    return custom_dataset


def demo_visualization():
    """演示数据可视化"""
    print("\n" + "=" * 70)
    print("数据可视化")
    print("=" * 70)

    # 加载数据
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    print(f"\n正在可视化 CIFAR-10 数据集...")
    print(f"类别: {train_dataset.classes}")

    # 创建图表
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('CIFAR-10 样本展示', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        # 获取样本
        image, label = train_dataset[i]

        # 转换为 numpy 并调整通道顺序
        image_np = image.permute(1, 2, 0).numpy()

        # 显示图像
        ax.imshow(image_np)
        ax.set_title(f'类别: {train_dataset.classes[label]}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    # 保存图像
    output_path = 'artifacts/cifar10_samples.png'
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化完成! 保存到: {output_path}")

    plt.close()


def demo_batch_visualization():
    """演示批次数据可视化"""
    print("\n" + "=" * 70)
    print("批次数据可视化")
    print("=" * 70)

    # 加载数据
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 获取一个批次
    images, labels = next(iter(train_loader))

    print(f"\n批次信息:")
    print(f"   图像形状: {images.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   标签: {labels.tolist()}")

    # 可视化批次
    fig, axes = plt.subplots(4, 4, figsize=(12, 12)) #subplots: 创建一个 4x4 的子图网格
    fig.suptitle('一个批次的数据 (batch_size=16)', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat): #enumerate : 遍历子图网格的每个子图
        # 转换图像
        image_np = images[i].permute(1, 2, 0).numpy() #permute: 改变张量的维度顺序, 从 (C, H, W) 转换为 (H, W, C)

        # 显示
        ax.imshow(image_np) #imshow: 显示图像
        ax.set_title(f'{train_dataset.classes[labels[i]]}', fontsize=10)#set_title: 设置子图的标题
        ax.axis('off')

    plt.tight_layout()

    # 保存
    output_path = 'artifacts/cifar10_batch.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 批次可视化完成! 保存到: {output_path}")

    plt.close()


def demo_scipy_vs_torchvision():
    """演示 scipy 和 torchvision 的区别"""
    print("\n" + "=" * 70)
    print("❓ scipy vs torchvision - 澄清常见误解")
    print("=" * 70)

    print("\n1️⃣ torchvision.datasets - 数据集下载和加载工具")
    print("-" * 70)
    print("""
    用途: 下载、加载、预处理计算机视觉数据集
    功能:
    - 自动下载常用数据集 (MNIST, CIFAR-10, ImageNet等)
    - 数据预处理和增强 (transforms)
    - 与 PyTorch DataLoader 无缝集成

    示例:
    from torchvision import datasets
    dataset = datasets.CIFAR10(root='./data', download=True)
    """)

    print("\n2️⃣ scipy - 科学计算库")
    print("-" * 70)
    print("""
    用途: 数学运算、科学计算、信号处理
    功能:
    - 优化算法 (scipy.optimize)
    - 线性代数 (scipy.linalg)
    - 信号处理 (scipy.signal)
    - 图像处理 (scipy.ndimage) - 基础的图像操作
    - 统计分析 (scipy.stats)

    示例:
    from scipy import ndimage
    from scipy.optimize import minimize
    """)

    print("\n3️⃣ 关键区别:")
    print("-" * 70)
    print("""
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │     功能         │   torchvision        │      scipy           │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ 数据集下载       │   ✅ 支持            │   ❌ 不支持          │
    │ 深度学习集成     │   ✅ 完美集成        │   ❌ 无集成          │
    │ 数据增强         │   ✅ 丰富            │   ❌ 无              │
    │ 图像处理         │   ✅ (transforms)    │   ✅ (ndimage)       │
    │ 数学计算         │   ❌ 有限            │   ✅ 强大            │
    │ 优化算法         │   ❌ 无              │   ✅ 丰富            │
    └─────────────────┴──────────────────────┴──────────────────────┘
    """)

    print("\n4️⃣ 正确的使用方式:")
    print("-" * 70)
    print("""
    ✅ 正确: 使用 torchvision.datasets 下载数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True  # 自动从官方源下载
    )

    ❌ 错误: 试图用 scipy 下载数据集
    # scipy 没有数据集下载功能!
    """)

    print("\n5️⃣ scipy 的实际用途示例:")
    print("-" * 70)
    print("""
    # 图像滤波
    from scipy import ndimage
    smoothed = ndimage.gaussian_filter(image, sigma=2)

    # 优化问题
    from scipy.optimize import minimize
    result = minimize(loss_function, initial_params)

    # 统计分析
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(group1, group2)
    """)


def main():
    """主函数 - 运行所有演示"""
    print("\n" + "=" * 70)
    print("🎓 torchvision.datasets 完全使用指南")
    print("=" * 70)

    # 1. 显示可用数据集
    demo_available_datasets()

    # 2. 下载和加载数据集
    demo_download_and_load_datasets()

    # 3. 数据预处理
    train_dataset, test_dataset = demo_transforms()

    # 4. DataLoader 使用
    train_loader = demo_dataloader()

    # 5. 数据集划分
    train_loader, val_loader = demo_dataset_split()

    # 6. 自定义数据集
    custom_dataset = demo_custom_dataset()

    # 7. 数据可视化
    demo_visualization()

    # 8. 批次可视化
    demo_batch_visualization()

    # 9. scipy vs torchvision
    demo_scipy_vs_torchvision()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 关键要点:")
    print("   1. ✅ 使用 torchvision.datasets 下载数据集 (download=True)")
    print("   2. ✅ transforms 用于数据预处理和增强")
    print("   3. ✅ DataLoader 用于批量加载和迭代数据")
    print("   4. ✅ random_split 用于划分训练集和验证集")
    print("   5. ❌ scipy 不用于下载数据集,它是科学计算库")
    print("   6. ✅ 自定义数据集需要实现 __len__ 和 __getitem__")
    print("=" * 70)


if __name__ == "__main__":
    main()
