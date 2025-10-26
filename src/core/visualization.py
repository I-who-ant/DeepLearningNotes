"""
可视化工具模块

提供TensorBoard和图像可视化相关的工具函数,包括:
- TensorBoard标量曲线记录
- 艺术图案生成(爱心、花朵、螺旋线等)
- 图像添加到TensorBoard
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def log_scalar_patterns(writer: SummaryWriter) -> None:
    """
    记录基础标量曲线,展示平滑与异常点效果

    Args:
        writer: TensorBoard SummaryWriter对象
    """
    for step in range(100):
        writer.add_scalar("demo/linear", step, step)
    writer.add_scalar("demo/spike", 250, 20)


def add_image(writer: SummaryWriter, tag: str, image: np.ndarray, global_step: int) -> None:
    """
    将HWC格式的numpy图像写入TensorBoard

    Args:
        writer: TensorBoard SummaryWriter对象
        tag: 图像标签
        image: HWC格式的numpy数组
        global_step: 全局步数
    """
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    writer.add_image(tag, tensor, global_step, dataformats="CHW")


def generate_heart(size: int = 128) -> np.ndarray:
    """
    基于极坐标方程渲染爱心效果

    Args:
        size: 图像尺寸(size x size)

    Returns:
        HWC格式的numpy数组,值范围[0, 1]
    """
    y, x = np.ogrid[-1.2:1.2:size * 1j, -1.2:1.2:size * 1j]
    r = np.sqrt(x**2 + y**2) + 1e-8
    theta = np.arctan2(y, x)
    mask = (r - 1 + np.sin(theta)) < -0.2 * np.sin(2 * theta)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = np.clip(0.2 + 0.8 * mask.astype(np.float32), 0.0, 1.0)
    image[..., 1] = 0.1 * mask.astype(np.float32)
    return image


def generate_flower(size: int = 128, petals: int = 6) -> np.ndarray:
    """
    使用正弦调制绘制花朵轮廓,展示多模态分布

    Args:
        size: 图像尺寸(size x size)
        petals: 花瓣数量

    Returns:
        HWC格式的numpy数组,值范围[0, 1]
    """
    y, x = np.ogrid[-1.1:1.1:size * 1j, -1.1:1.1:size * 1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    petal_wave = 0.4 * np.cos(petals * theta)
    mask = r < (0.6 + petal_wave)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 1] = np.clip(0.2 + 0.7 * mask.astype(np.float32), 0.0, 1.0)
    image[..., 0] = 0.1 * mask.astype(np.float32)
    image[..., 2] = 0.3 * (1 - r / r.max())
    return image


def generate_spiral(size: int = 128, turns: int = 3) -> np.ndarray:
    """
    绘制简单螺旋线,展示非对称图案

    Args:
        size: 图像尺寸(size x size)
        turns: 螺旋圈数

    Returns:
        HWC格式的numpy数组,值范围[0, 1]
    """
    y, x = np.ogrid[-1:1:size * 1j, -1:1:size * 1j]
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    spiral = np.mod((theta + math.pi) * turns / math.pi, 1.0)
    mask = np.abs(r - spiral / 1.5) < 0.03
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 2] = np.clip(mask.astype(np.float32), 0.0, 1.0)
    image[..., 1] = 0.2 * mask.astype(np.float32)
    return image


def log_pattern(writer: SummaryWriter, tag: str, generator: Callable[[], np.ndarray], step: int) -> None:
    """
    记录艺术图案到TensorBoard

    Args:
        writer: TensorBoard SummaryWriter对象
        tag: 图案标签
        generator: 图案生成函数
        step: 全局步数
    """
    add_image(writer, tag, generator(), step)


def demo_pattern_gallery() -> None:
    """
    演示:记录基础标量曲线和艺术图案到TensorBoard

    运行后可以通过 tensorboard --logdir=runs/pattern_gallery 查看效果
    """
    writer = SummaryWriter(log_dir="runs/pattern_gallery")
    log_scalar_patterns(writer)
    log_pattern(writer, "art/heart", generate_heart, 0)
    log_pattern(writer, "art/flower", generate_flower, 1)
    log_pattern(writer, "art/spiral", generate_spiral, 2)
    writer.close()
    print("✅ 图案已记录到 runs/pattern_gallery")
    print("💡 运行 'tensorboard --logdir=runs/pattern_gallery' 查看效果")


if __name__ == "__main__":
    demo_pattern_gallery()
