import numpy as np
import torch
from datasets import load_from_disk
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# 该文件展示torch.nn.Conv2d的使用 :(将图片张量转换为特征图张量展示在tensorboard中对比理解)
# 卷积层的参数：
# in_channels: 输入通道数（如RGB图像为3通道）
# out_channels: 输出通道数（即卷积核的数量）
# kernel_size: 卷积核大小（如3x3卷积核为(3,3)）
# stride: 步幅（默认1）
# padding: 填充（默认0）
# dilation: 膨胀（默认1）
# groups: 分组卷积（默认1）
# bias: 是否添加偏置项（默认True）




def to_tensor(pil_image, size=(224, 224)):
    pil_image = pil_image.convert("RGB").resize(size)
    array = np.array(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


class CatDogDataset(Dataset):
    def __init__(self, hf_split):
        self.split = hf_split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        sample = self.split[idx]
        tensor = to_tensor(sample["image"])
        result = {"pixel_values": tensor}
        if "label" in sample:
            result["label"] = sample["label"]
        return result


class Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, bias=False)

    def forward(self, x):
        return self.conv(x)


def normalize_batch(tensor):
    min_val = tensor.amin(dim=(2, 3), keepdim=True)
    max_val = tensor.amax(dim=(2, 3), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + 1e-6)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    dataset = load_from_disk(project_root / "dataset" / "Cat_and_Dog")
    train_dataset = CatDogDataset(dataset["train"])

    loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = Transfer().eval()

    log_dir = project_root / "artifacts" / "some_test" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    with torch.no_grad():
        for step, batch in enumerate(loader):
            images = batch["pixel_values"]
            input_images = images.clone()
            features = model(images)

            writer.add_images("input/batch", input_images, global_step=step)
            feature_slice = features[:, :3]
            writer.add_images(
                "conv/features",
                normalize_batch(feature_slice),
                global_step=step,
            )
            break

    writer.close()


