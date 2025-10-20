import numpy as np
import torch
from datasets import load_from_disk
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# 该文件展示torch.nn中的池化层使用：(将特征图经过池化层处理，展示在tensorboard中对比理解)
# 池化层的类型和参数：
# MaxPool2d: 最大池化（保留区域内的最大值）
# AvgPool2d: 平均池化（计算区域内的平均值）
# kernel_size: 池化窗口大小（如2x2池化为(2,2)）
# stride: 步幅（默认等于kernel_size）
# padding: 填充（默认0）
# dilation: 膨胀（默认1）
# ceil_mode: 向上取整模式（默认False）


def to_tensor(pil_image, size=(224, 224)): # 将图片转换为张量
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


class SimpleCNN(nn.Module): # 简单的卷积神经网络，用于对比池化层的效果
    def __init__(self):
        super().__init__()
        # 先使用卷积层提取特征
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, bias=False)
        
        # 定义不同的池化层进行对比
        self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2最大池化
        self.maxpool_3x3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 3x3最大池化，步长2
        self.avgpool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2平均池化
        self.avgpool_3x3 = nn.AvgPool2d(kernel_size=3, stride=2)  # 3x3平均池化，步长2

    def forward(self, x):
        # 卷积提取特征
        features = self.conv(x) # 卷积提取特征
        
        # 应用不同的池化操作
        maxpool_2x2_out = self.maxpool_2x2(features)
        maxpool_3x3_out = self.maxpool_3x3(features)
        avgpool_2x2_out = self.avgpool_2x2(features)
        avgpool_3x3_out = self.avgpool_3x3(features)
        
        return {
            'features': features,  # 原始特征图
            'maxpool_2x2': maxpool_2x2_out,
            'maxpool_3x3': maxpool_3x3_out,
            'avgpool_2x2': avgpool_2x2_out,
            'avgpool_3x3': avgpool_3x3_out
        }


def normalize_batch(tensor):
    """归一化批次张量，使值在[0,1]范围内"""
    min_val = tensor.amin(dim=(2, 3), keepdim=True)
    max_val = tensor.amax(dim=(2, 3), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + 1e-6)


def calculate_output_size(input_size, kernel_size, stride, padding=0):
    """计算池化层输出尺寸"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    dataset = load_from_disk(project_root / "dataset" / "Cat_and_Dog")
    train_dataset = CatDogDataset(dataset["train"])

    loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    model = SimpleCNN().eval()

    log_dir = project_root / "artifacts" / "pool_test" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    with torch.no_grad(): # 不计算梯度，因为只是可视化
        for step, batch in enumerate(loader):
            images = batch["pixel_values"]# 从批次中获取图像张量
            input_images = images.clone() # 复制输入图像，用于可视化
            
            # 前向传播获取各种池化结果
            outputs = model(images) # 前向传播，获取池化结果
            
            # 写入TensorBoard进行可视化对比
            writer.add_images("1_input/original", input_images, global_step=step)
            
            # 原始特征图
            feature_slice = outputs['features'][:, :3]  # 取前3个通道
            writer.add_images("2_features/conv_output", normalize_batch(feature_slice), global_step=step)
            
            # 最大池化结果
            maxpool_2x2_slice = outputs['maxpool_2x2'][:, :3]
            writer.add_images("3_maxpool/2x2_stride2", normalize_batch(maxpool_2x2_slice), global_step=step)
            
            maxpool_3x3_slice = outputs['maxpool_3x3'][:, :3]
            writer.add_images("3_maxpool/3x3_stride2", normalize_batch(maxpool_3x3_slice), global_step=step)
            
            # 平均池化结果
            avgpool_2x2_slice = outputs['avgpool_2x2'][:, :3]
            writer.add_images("4_avgpool/2x2_stride2", normalize_batch(avgpool_2x2_slice), global_step=step)
            
            avgpool_3x3_slice = outputs['avgpool_3x3'][:, :3]
            writer.add_images("4_avgpool/3x3_stride2", normalize_batch(avgpool_3x3_slice), global_step=step)
            
            # 打印尺寸信息
            print(f"输入尺寸: {input_images.shape}")
            print(f"卷积后特征图尺寸: {outputs['features'].shape}")
            print(f"2x2最大池化后尺寸: {outputs['maxpool_2x2'].shape}")
            print(f"3x3最大池化后尺寸: {outputs['maxpool_3x3'].shape}")
            print(f"2x2平均池化后尺寸: {outputs['avgpool_2x2'].shape}")
            print(f"3x3平均池化后尺寸: {outputs['avgpool_3x3'].shape}")
            
            # 验证尺寸计算公式
            input_h, input_w = input_images.shape[2], input_images.shape[3]
            print(f"\n尺寸计算验证:")
            print(f"2x2池化理论尺寸: {calculate_output_size(input_h, 2, 2)}x{calculate_output_size(input_w, 2, 2)}")
            print(f"3x3池化理论尺寸: {calculate_output_size(input_h, 3, 2)}x{calculate_output_size(input_w, 3, 2)}")
            
            break

    writer.close()
    print(f"\nTensorBoard日志已保存到: {log_dir}")
    print("运行以下命令查看可视化结果:")
    print(f"tensorboard --logdir={log_dir}")