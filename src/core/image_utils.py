"""
图像处理工具模块

提供图像加载、转换、保存等常用操作,包括:
- PIL与NumPy格式转换
- 图像添加到TensorBoard
- HuggingFace数据集加载
"""

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset


def load_image_as_array(image_path: str) -> np.ndarray:
    """
    加载图像并转换为NumPy数组

    Args:
        image_path: 图像文件路径

    Returns:
        HWC格式的numpy数组,shape为(H, W, C)
    """
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil)
    return img_array


def add_image_to_tensorboard(
    writer: SummaryWriter,
    tag: str,
    image_path: str,
    global_step: int = 0,
    dataformats: str = 'HWC'
) -> None:
    """
    将图像添加到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter对象
        tag: 图像标签
        image_path: 图像文件路径
        global_step: 全局步数
        dataformats: 数据格式,'HWC'表示(高度, 宽度, 通道)
    """
    img_array = load_image_as_array(image_path)
    writer.add_image(tag, img_array, global_step, dataformats=dataformats)


def load_huggingface_dataset(dataset_name: str, **kwargs):
    """
    加载HuggingFace数据集

    Args:
        dataset_name: 数据集名称,如 "Bingsu/Cat_and_Dog"
        **kwargs: 传递给load_dataset的其他参数

    Returns:
        HuggingFace DatasetDict对象

    Example:
        >>> ds = load_huggingface_dataset("Bingsu/Cat_and_Dog")
        >>> print(ds)
    """
    return load_dataset(dataset_name, **kwargs)


def demo_tensorboard_image():
    """
    演示:将图像添加到TensorBoard

    注意:需要提供有效的图像路径
    """
    # 示例代码(需要根据实际路径修改)
    writer = SummaryWriter("logs")

    # 假设有一个图像文件
    # image_path = "path/to/your/image.png"
    # add_image_to_tensorboard(writer, "test_image", image_path, global_step=1)

    writer.close()
    print("✅ 图像已记录到TensorBoard")
    print("💡 运行 'tensorboard --logdir=logs' 查看效果")


if __name__ == "__main__":
    # 演示图像加载
    # img = load_image_as_array("your_image.png")
    # print(f"图像shape: {img.shape}")

    print("📖 图像工具模块加载成功")
    print("💡 主要功能:")
    print("  - load_image_as_array: 加载图像为NumPy数组")
    print("  - add_image_to_tensorboard: 添加图像到TensorBoard")
    print("  - load_huggingface_dataset: 加载HuggingFace数据集")
