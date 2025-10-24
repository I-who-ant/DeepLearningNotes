"""
原始数据集格式解析和读取

这个模块详细讲解如何直接读取数据集的原始格式：
1. MNIST 的 ubyte 格式解析
2. CIFAR-10 的 pickle 格式解析
3. 数据来源和格式说明
4. 手动读取和可视化

作者: Seeback
日期: 2025-10-23
"""

import struct
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from pathlib import Path


def explain_mnist_format():
    """解释 MNIST 数据集的原始格式"""
    print("=" * 70)
    print("MNIST 数据集原始格式详解")
    print("=" * 70)

    print("\n📦 1. 数据来源和下载")
    print("-" * 70)
    print("""
    数据来源: Yann LeCun 的 MNIST 官方网站
    原始URL: http://yann.lecun.com/exdb/mnist/
    备用URL: https://ossci-datasets.s3.amazonaws.com/mnist/

    torchvision.datasets.MNIST 做了什么:
    1. 自动从上述URL下载 .gz 压缩文件
    2. 解压缩到 data/MNIST/raw/ 目录
    3. 得到 .ubyte 格式的原始数据文件
    """)

    print("\n📁 2. 文件结构")
    print("-" * 70)
    print("""
    data/MNIST/raw/
    ├── train-images-idx3-ubyte    (45 MB)  - 训练集图像
    ├── train-labels-idx1-ubyte    (59 KB)  - 训练集标签
    ├── t10k-images-idx3-ubyte     (7.5 MB) - 测试集图像
    └── t10k-labels-idx1-ubyte     (9.8 KB) - 测试集标签
    """)

    print("\n🔍 3. IDX 文件格式 (ubyte 格式)")
    print("-" * 70)
    print("""
    IDX 格式是 MNIST 使用的二进制文件格式

    图像文件格式 (idx3-ubyte):
    ┌────────────────┬─────────┬──────────────────────────┐
    │     字段       │  字节数  │         说明              │
    ├────────────────┼─────────┼──────────────────────────┤
    │ Magic Number   │   4     │ 0x00000803 (2051)        │
    │ 图像数量        │   4     │ 60000 (训练) / 10000 (测试)│
    │ 图像高度        │   4     │ 28                       │
    │ 图像宽度        │   4     │ 28                       │
    │ 像素数据        │  N*784  │ 每个像素 0-255 (1字节)    │
    └────────────────┴─────────┴──────────────────────────┘

    标签文件格式 (idx1-ubyte):
    ┌────────────────┬─────────┬──────────────────────────┐
    │     字段       │  字节数  │         说明              │
    ├────────────────┼─────────┼──────────────────────────┤
    │ Magic Number   │   4     │ 0x00000801 (2049)        │
    │ 标签数量        │   4     │ 60000 (训练) / 10000 (测试)│
    │ 标签数据        │   N     │ 每个标签 0-9 (1字节)      │
    └────────────────┴─────────┴──────────────────────────┘

    Magic Number 说明:
    - 前2字节总是 0x0000
    - 第3字节: 数据类型 (0x08 = unsigned byte)
    - 第4字节: 维度数 (0x01 = 1D, 0x03 = 3D)
    """)

    print("\n💡 4. 为什么使用 ubyte 格式?")
    print("-" * 70)
    print("""
    优点:
    ✅ 高效: 二进制格式,占用空间小
    ✅ 简单: 格式简单,易于解析
    ✅ 标准: IDX 格式是机器学习领域的标准格式
    ✅ 跨平台: 大端序存储,所有平台通用

    缺点:
    ❌ 不直观: 需要专门的程序才能查看
    ❌ 不通用: 不像 PNG/JPEG 那样有现成查看器
    """)


def read_mnist_images(file_path):
    """
    手动读取 MNIST 图像文件 (idx3-ubyte 格式)

    参数:
        file_path: 图像文件路径

    返回:
        images: numpy数组,形状为 [N, 28, 28]
    """
    print(f"\n📖 读取图像文件: {file_path}")

    with open(file_path, 'rb') as f:
        # 读取文件头 (前16字节)
        magic = struct.unpack('>I', f.read(4))[0]  # > 表示大端序, I 表示4字节整数
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        print(f"   Magic Number: {magic} (0x{magic:08X})")
        print(f"   图像数量: {num_images}")
        print(f"   图像尺寸: {num_rows}x{num_cols}")

        # 验证 magic number
        if magic != 2051:
            raise ValueError(f"无效的 magic number: {magic}, 期望 2051")

        # 读取所有像素数据
        buffer = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

        print(f"   数据形状: {images.shape}")
        print(f"   数据类型: {images.dtype}")
        print(f"   像素值范围: [{images.min()}, {images.max()}]")

    return images


def read_mnist_labels(file_path):
    """
    手动读取 MNIST 标签文件 (idx1-ubyte 格式)

    参数:
        file_path: 标签文件路径

    返回:
        labels: numpy数组,形状为 [N]
    """
    print(f"\n📖 读取标签文件: {file_path}")

    with open(file_path, 'rb') as f:
        # 读取文件头 (前8字节)
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        print(f"   Magic Number: {magic} (0x{magic:08X})")
        print(f"   标签数量: {num_labels}")

        # 验证 magic number
        if magic != 2049:
            raise ValueError(f"无效的 magic number: {magic}, 期望 2049")

        # 读取所有标签数据
        buffer = f.read(num_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)

        print(f"   数据形状: {labels.shape}")
        print(f"   标签范围: [{labels.min()}, {labels.max()}]")
        print(f"   前10个标签: {labels[:10].tolist()}")

    return labels


def explain_cifar10_format():
    """解释 CIFAR-10 数据集的原始格式"""
    print("\n" + "=" * 70)
    print("CIFAR-10 数据集原始格式详解")
    print("=" * 70)

    print("\n📦 1. 数据来源和下载")
    print("-" * 70)
    print("""
    数据来源: University of Toronto - CIFAR 官方网站
    原始URL: https://www.cs.toronto.edu/~kriz/cifar.html

    torchvision.datasets.CIFAR10 做了什么:
    1. 下载 cifar-10-python.tar.gz (约170MB)
    2. 解压缩到 data/ 目录
    3. 得到 cifar-10-batches-py/ 文件夹
    4. 里面包含 Python pickle 格式的数据文件
    """)

    print("\n📁 2. 文件结构")
    print("-" * 70)
    print("""
    data/cifar-10-batches-py/
    ├── data_batch_1        (30 MB) - 训练集 batch 1 (10000张)
    ├── data_batch_2        (30 MB) - 训练集 batch 2 (10000张)
    ├── data_batch_3        (30 MB) - 训练集 batch 3 (10000张)
    ├── data_batch_4        (30 MB) - 训练集 batch 4 (10000张)
    ├── data_batch_5        (30 MB) - 训练集 batch 5 (10000张)
    ├── test_batch          (30 MB) - 测试集 (10000张)
    ├── batches.meta        (158 B) - 元数据 (类别名称)
    └── readme.html         (88 B)  - 说明文件

    总计: 50000 训练图像 + 10000 测试图像 = 60000 张
    """)

    print("\n🔍 3. Pickle 文件格式")
    print("-" * 70)
    print("""
    CIFAR-10 使用 Python 的 pickle 格式存储数据

    每个 batch 文件是一个 Python 字典,包含:
    ┌─────────────┬──────────────┬──────────────────────────┐
    │     键      │    类型      │         说明              │
    ├─────────────┼──────────────┼──────────────────────────┤
    │ 'data'      │ numpy数组    │ 形状 [10000, 3072]       │
    │             │              │ 3072 = 32*32*3 (RGB)     │
    │             │              │ 数据排列: RRRR...GGGG...BBBB│
    │ 'labels'    │ 列表         │ 10000个标签 (0-9)        │
    │ 'batch_label'│ 字符串      │ batch名称                │
    │ 'filenames' │ 列表         │ 10000个文件名            │
    └─────────────┴──────────────┴──────────────────────────┘

    batches.meta 文件包含:
    {
        'label_names': ['airplane', 'automobile', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'num_cases_per_batch': 10000,
        'num_vis': 3072
    }
    """)

    print("\n💡 4. 为什么使用 Pickle 格式?")
    print("-" * 70)
    print("""
    优点:
    ✅ Python 原生: 可以直接用 pickle.load() 读取
    ✅ 灵活: 可以存储任意 Python 对象
    ✅ 包含元数据: 文件名、batch名称等额外信息
    ✅ 方便: 不需要解析复杂的二进制格式

    缺点:
    ❌ Python 专用: 其他语言需要额外工具
    ❌ 安全问题: pickle 可能执行恶意代码
    ❌ 版本兼容: Python 2/3 可能不兼容
    """)


def read_cifar10_batch(file_path):
    """
    手动读取 CIFAR-10 batch 文件 (pickle 格式)

    参数:
        file_path: batch 文件路径

    返回:
        data_dict: 包含图像和标签的字典
    """
    print(f"\n📖 读取 CIFAR-10 batch: {file_path}")

    with open(file_path, 'rb') as f:
        # 使用 pickle 加载数据
        data_dict = pickle.load(f, encoding='bytes')

    # 显示字典键
    print(f"   字典键: {list(data_dict.keys())}")

    # 获取数据
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    filenames = data_dict[b'filenames']
    batch_label = data_dict[b'batch_label']

    print(f"   Batch 名称: {batch_label}")
    print(f"   图像数据形状: {images.shape}")
    print(f"   图像数据类型: {images.dtype}")
    print(f"   标签数量: {len(labels)}")
    print(f"   前10个标签: {labels[:10]}")

    # 重塑图像数据: [10000, 3072] -> [10000, 3, 32, 32]
    images = images.reshape(-1, 3, 32, 32)
    # 转换通道顺序: [N, C, H, W] -> [N, H, W, C]
    images = images.transpose(0, 2, 3, 1)

    print(f"   重塑后形状: {images.shape} (N, H, W, C)")

    return {
        'images': images,
        'labels': labels,
        'filenames': filenames,
        'batch_label': batch_label
    }


def read_cifar10_meta(file_path):
    """读取 CIFAR-10 元数据"""
    print(f"\n📖 读取 CIFAR-10 元数据: {file_path}")

    with open(file_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')

    label_names = [name.decode('utf-8') for name in meta[b'label_names']]

    print(f"   类别名称: {label_names}")
    print(f"   每个batch的样本数: {meta[b'num_cases_per_batch']}")

    return label_names


def visualize_mnist_raw():
    """可视化手动读取的 MNIST 数据"""
    print("\n" + "=" * 70)
    print("可视化 MNIST 原始数据")
    print("=" * 70)

    # 读取数据
    images = read_mnist_images('data/MNIST/raw/train-images-idx3-ubyte')
    labels = read_mnist_labels('data/MNIST/raw/train-labels-idx1-ubyte')

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('MNIST 原始数据 (直接从 ubyte 文件读取)',
                 fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()

    output_path = 'artifacts/mnist_raw_data.png'
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ MNIST 可视化完成! 保存到: {output_path}")

    plt.close()


def visualize_cifar10_raw():
    """可视化手动读取的 CIFAR-10 数据"""
    print("\n" + "=" * 70)
    print("可视化 CIFAR-10 原始数据")
    print("=" * 70)

    # 读取数据
    batch = read_cifar10_batch('data/cifar-10-batches-py/data_batch_1')
    label_names = read_cifar10_meta('data/cifar-10-batches-py/batches.meta')

    images = batch['images']
    labels = batch['labels']

    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('CIFAR-10 原始数据 (直接从 pickle 文件读取)',
                 fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f'{label_names[labels[i]]}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()

    output_path = 'artifacts/cifar10_raw_data.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ CIFAR-10 可视化完成! 保存到: {output_path}")

    plt.close()


def compare_with_torchvision():
    """对比手动读取和 torchvision 读取的结果"""
    print("\n" + "=" * 70)
    print("验证: 手动读取 vs torchvision 读取")
    print("=" * 70)

    # 1. 手动读取 MNIST
    print("\n1️⃣ MNIST 数据验证:")
    print("-" * 70)
    manual_images = read_mnist_images('data/MNIST/raw/train-images-idx3-ubyte')
    manual_labels = read_mnist_labels('data/MNIST/raw/train-labels-idx1-ubyte')

    # 2. torchvision 读取
    from torchvision import datasets
    torch_dataset = datasets.MNIST(root='./data', train=True, download=False)

    print(f"\n✅ 数据一致性检查:")
    print(f"   手动读取: {manual_images.shape}, {manual_labels.shape}")
    print(f"   torchvision: {len(torch_dataset)} 样本")

    # 对比前10个样本
    all_match = True
    for i in range(10):
        torch_img, torch_label = torch_dataset[i]
        torch_img_np = np.array(torch_img)

        if not np.array_equal(manual_images[i], torch_img_np):
            all_match = False
            break
        if manual_labels[i] != torch_label:
            all_match = False
            break

    if all_match:
        print(f"   ✅ 前10个样本完全一致!")
    else:
        print(f"   ❌ 数据不一致!")


def demo_hex_dump():
    """演示查看二进制文件的前几个字节"""
    print("\n" + "=" * 70)
    print("二进制文件 Hex Dump 演示")
    print("=" * 70)

    print("\n1️⃣ MNIST 图像文件的前32字节:")
    print("-" * 70)

    with open('data/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
        header = f.read(32)

    print("   偏移量  |  十六进制                               |  ASCII")
    print("   " + "-" * 66)

    for i in range(0, len(header), 16):
        chunk = header[i:i+16]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        print(f"   0x{i:04X}   |  {hex_str:<47} |  {ascii_str}")

    print("\n   解析:")
    print(f"   - 前4字节: 0x{header[0]:02X} {header[1]:02X} {header[2]:02X} {header[3]:02X}")
    print(f"     → Magic Number = {struct.unpack('>I', header[0:4])[0]} (期望 2051)")
    print(f"   - 第5-8字节: 图像数量 = {struct.unpack('>I', header[4:8])[0]}")
    print(f"   - 第9-12字节: 图像高度 = {struct.unpack('>I', header[8:12])[0]}")
    print(f"   - 第13-16字节: 图像宽度 = {struct.unpack('>I', header[12:16])[0]}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🎓 原始数据集格式解析和读取")
    print("=" * 70)

    # 1. 解释 MNIST 格式
    explain_mnist_format()

    # 2. 解释 CIFAR-10 格式
    explain_cifar10_format()

    # 3. 二进制文件 Hex Dump
    demo_hex_dump()

    # 4. 可视化 MNIST
    visualize_mnist_raw()

    # 5. 可视化 CIFAR-10
    visualize_cifar10_raw()

    # 6. 验证一致性
    compare_with_torchvision()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 关键要点:")
    print("   1. ✅ MNIST 使用 IDX (ubyte) 格式 - 高效的二进制格式")
    print("   2. ✅ CIFAR-10 使用 Python pickle 格式 - Python 原生支持")
    print("   3. ✅ 可以直接手动读取原始数据,不依赖 torchvision")
    print("   4. ✅ torchvision 会自动下载并解压数据到 data/ 目录")
    print("   5. ✅ 手动读取和 torchvision 读取的结果完全一致")
    print("=" * 70)


if __name__ == "__main__":
    main()
