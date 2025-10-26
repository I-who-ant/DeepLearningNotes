"""
自定义数据集模块

提供自定义PyTorch数据集的示例实现,用于加载本地图像数据

核心概念:
- 继承torch.utils.data.Dataset
- 实现__init__, __getitem__, __len__三个方法
- 支持与DataLoader配合使用
"""

import os
from typing import Tuple

from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):
    """
    自定义图像数据集

    目录结构示例:
        root_dir/
        ├── label1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── label2/
        │   ├── img1.jpg
        │   └── ...
        └── ...

    Args:
        root_dir: 数据集根目录
        label_dir: 标签目录(子文件夹名称)
    """

    def __init__(self, root_dir: str, label_dir: str):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """
        获取指定索引的样本

        Args:
            idx: 样本索引

        Returns:
            (image, label): PIL.Image对象和标签字符串的元组
        """
        image_name = self.img_path[idx]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        image = Image.open(image_item_path)
        label = self.label_dir
        return image, label

    def __len__(self) -> int:
        """
        返回数据集大小

        Returns:
            数据集中的样本数量
        """
        return len(self.img_path)


def demo_custom_dataset():
    """
    演示自定义数据集的使用

    注意:需要提供有效的数据目录
    """
    # 示例:假设有以下目录结构
    # dataset/
    # └── arch_linux/
    #     ├── img1.png
    #     ├── img2.png
    #     └── ...

    root_dir = 'dataset'
    label_dir = 'arch_linux'

    # 创建数据集对象
    dataset = MyData(root_dir, label_dir)

    # 方式1: 获取元组
    # sample = dataset[0]  # sample是元组: (image, label)
    # image = sample[0]
    # label = sample[1]

    # 方式2: 直接解包(推荐)
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"✅ 成功加载第1个样本")
        print(f"  图像类型: {type(image)}")
        print(f"  图像尺寸: {image.size if hasattr(image, 'size') else 'N/A'}")
        print(f"  标签: {label}")
        print(f"  数据集大小: {len(dataset)}")
    else:
        print("⚠️  数据集为空")


if __name__ == "__main__":
    print("📖 自定义数据集模块")
    print("\n核心要点:")
    print("  1. 继承 torch.utils.data.Dataset")
    print("  2. 实现 __init__: 初始化数据路径列表")
    print("  3. 实现 __getitem__: 加载单个样本")
    print("  4. 实现 __len__: 返回数据集大小")
    print("\n使用方式:")
    print("  dataset = MyData(root_dir, label_dir)")
    print("  image, label = dataset[0]  # 获取第一个样本")
    print("  DataLoader(dataset, ...)   # 配合DataLoader使用")
