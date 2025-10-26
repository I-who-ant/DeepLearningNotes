"""
核心工具模块

提供深度学习项目常用的工具函数:
- visualization: TensorBoard可视化和艺术图案生成
- image_utils: 图像加载、转换、保存
"""

from .visualization import (
    log_scalar_patterns,
    add_image,
    generate_heart,
    generate_flower,
    generate_spiral,
    log_pattern,
    demo_pattern_gallery,
)

from .image_utils import (
    load_image_as_array,
    add_image_to_tensorboard,
    load_huggingface_dataset,
)

__all__ = [
    # visualization
    'log_scalar_patterns',
    'add_image',
    'generate_heart',
    'generate_flower',
    'generate_spiral',
    'log_pattern',
    'demo_pattern_gallery',
    # image_utils
    'load_image_as_array',
    'add_image_to_tensorboard',
    'load_huggingface_dataset',
]
