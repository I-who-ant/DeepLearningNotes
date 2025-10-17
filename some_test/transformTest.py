from pathlib import Path

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT_DIR / "dataset" / "arch_linux" / "cobalt-2.png"
LOG_DIR = ROOT_DIR / "some_test" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 1. 读取图像并查看模式；转换为 RGB 以匹配三通道模型输入
img_pil = Image.open(IMAGE_PATH)
print(img_pil)  # RGBA -> 说明原图带透明通道
img_rgb = img_pil.convert("RGB") # 转换为 RGB 模式，以匹配三通道模型输入

# 2. 将图像转为张量（值域 0~1，通道由转换后的模式决定）
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_rgb)  # shape: [3, H, W]

# 3. 写入原始 RGB 张量
writer = SummaryWriter(str(LOG_DIR))
writer.add_image("raw/original", img_tensor)

# 4. 按通道归一化：提供与 RGB 对应的均值/标准差
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
img_norm = normalize(img_tensor)

print("Normalized sample pixel:", img_norm[0, 0, 0])

# 5. 为日志展示反归一化到 0~1，确保 TensorBoard 显示正常
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
img_norm_vis = img_norm * std + mean
img_norm_vis = img_norm_vis.clamp(0.0, 1.0)
writer.add_image("normalized/recovered", img_norm_vis)

# 6. Resize 返回 PIL，需要再转张量
resize = transforms.Resize((1080, 1920))
img_resized = resize(img_rgb)
img_resized_tensor = to_tensor(img_resized)
writer.add_image("resized/resize", img_resized_tensor, global_step=0)

print("Original size:", img_pil.size)
print("Resized size:", img_resized.size)

# Compose -resize
trans_resize = transforms.Resize(1080) # 1080 是目标高度，宽度会按比例缩放
    # 组合多个变换, 先resize, 再转换为张量,转换为张量的目的是为了后续的归一化等操作
trans_compose = transforms.Compose([trans_resize, transforms.ToTensor()])

img_resized_2 = trans_compose(img_rgb)
writer.add_image("resized/compose", img_resized_2, global_step=0)

writer.close()
