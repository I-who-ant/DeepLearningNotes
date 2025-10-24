"""understand.py
提供一个高度注释的极简 Transformer 预训练示例，帮助理解 DeepSeek 等大模型的基础训练流程。
"""  # 顶部文档字符串说明文件用途和整体背景

import math  # 引入数学函数库，后续用于注意力缩放计算
from dataclasses import dataclass  # dataclass 简化配置对象定义
from typing import Optional  # Optional 用于提示可选类型，便于理解函数签名

import torch  # 引入 PyTorch 主库，负责张量计算与自动求导
from torch import nn  # nn 子模块提供神经网络层构建工具
from torch.nn import functional as F  # F 提供函数式 API，例如交叉熵
from torch.utils.data import DataLoader, Dataset  # Dataset/DataLoader 负责数据组织与批处理

# ========================================================================================
# 模块一：数据准备
# ========================================================================================

class ToyTextDataset(Dataset):  # 自定义数据集，模拟语言模型的序列样本
    """将一长串 token 切分成输入/标签对，模拟自回归语言建模数据。"""  # 文档字符串说明类目的

    def __init__(self, tokens: torch.Tensor, context_len: int) -> None:  # 初始化时接收整段 token 序列与上下文长度
        super().__init__()  # 调用父类构造，符合 PyTorch Dataset 规范
        self.tokens = tokens  # 保存完整 token 序列，供 __getitem__ 切片
        self.context_len = context_len  # 保存上下文窗口长度，决定模型一次能看到多少 token

    def __len__(self) -> int:  # 返回数据集长度，供 DataLoader 枚举
        return self.tokens.size(0) - self.context_len  # 每个样本需要 context_len+1 token，减去窗口保证索引合法

    def __getitem__(self, idx: int):  # 根据索引返回单个样本
        x = self.tokens[idx : idx + self.context_len]  # 取长度为 context_len 的输入序列
        y = self.tokens[idx + 1 : idx + self.context_len + 1]  # 标签是右移一位的序列，用于预测下一个 token
        return x, y  # 返回 (输入, 标签) 元组，供训练循环使用

# 数据模块总结：ToyTextDataset 将原始 token 串拆分成模型训练需要的滑动窗口样本，为下一步 DataLoader 批量加载打好基础。  # 模块说明，强调对下一步的作用

# ========================================================================================
# 模块二：模型配置
# ========================================================================================

@dataclass  # dataclass 装饰器自动生成 __init__ 等方法，便于管理配置
class TransformerConfig:  # 保存模型超参数的配置对象
    vocab_size: int  # 词表大小，决定 Embedding 与输出层维度
    context_len: int  # 单条样本长度，控制位置编码与注意力遮罩大小
    embed_dim: int = 768  # 隐藏维度，决定特征表达能力
    n_heads: int = 12  # 注意力头数，用于多头拆分
    n_layers: int = 12  # Transformer 层数，影响模型深度
    dropout: float = 0.1  # Dropout 比例，缓解过拟合

# 配置模块总结：TransformerConfig 集中保存超参，便于模型、注意力层等组件读取统一设置。  # 模块说明

# ========================================================================================
# 模块三：注意力与 Transformer Block
# ========================================================================================

class CausalSelfAttention(nn.Module):  # 自回归掩码自注意力层，实现核心的上下文建模
    def __init__(self, cfg: TransformerConfig) -> None:  # 构造函数读取配置初始化各参数
        super().__init__()  # 初始化 nn.Module 父类部分
        self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias=False)  # 通过线性层一次性生成 Query/Key/Value
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)  # 注意力输出后再映射回原维度
        self.n_heads = cfg.n_heads  # 记录注意力头数量，用于拆分
        self.head_dim = cfg.embed_dim // cfg.n_heads  # 单头维度，决定每个注意力头的特征宽度
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len))  # 构造下三角矩阵，实现因果遮罩
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))  # 将遮罩注册为 buffer，推理时无需梯度

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向计算接收形状(B,T,C)的特征
        B, T, C = x.shape  # 读取 batch 大小、序列长度、通道数
        qkv = self.qkv(x)  # 线性层生成拼接后的 QKV 特征
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 重塑维度后拆分成三个张量
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别取出 Query/Key/Value，形状(B, heads, T, head_dim)
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力分数并按头维度缩放
        attn_logits = attn_logits.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # 利用遮罩阻止模型看到未来信息
        attn = attn_logits.softmax(dim=-1)  # 对最后一维做 softmax，得到注意力权重
        out = attn @ v  # 加权求和 Value，获得上下文融合信息
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 拼回原始维度，准备投影
        return self.proj(out)  # 通过线性层融合多头信息并返回

# 注意力模块总结：CausalSelfAttention 实现“看到过去预测未来”的核心逻辑，为后续 Transformer Block 提供基础的上下文建模能力。  # 模块说明

class TransformerBlock(nn.Module):  # 单个 Transformer Block，包含注意力与前馈网络
    def __init__(self, cfg: TransformerConfig) -> None:  # 构造函数读取配置
        super().__init__()  # 初始化父类
        self.ln1 = nn.LayerNorm(cfg.embed_dim)  # 第一层归一化，稳定注意力输入
        self.attn = CausalSelfAttention(cfg)  # 堆叠注意力子模块
        self.ln2 = nn.LayerNorm(cfg.embed_dim)  # 第二层归一化，稳定 MLP 输入
        self.mlp = nn.Sequential(  # 前馈网络，实现非线性变换
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),  # 扩展维度提高表达力
            nn.GELU(),  # GELU 激活提供平滑非线性
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),  # 映射回原维度方便残差连接
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Block 前向计算
        x = x + self.attn(self.ln1(x))  # 归一化后进入注意力，并与输入做残差相加
        x = x + self.mlp(self.ln2(x))  # 归一化后进入前馈网络，并残差相加
        return x  # 返回处理后的特征

# Block 模块总结：TransformerBlock 在注意力基础上引入残差和前馈网络，为堆叠多个 Block 构建深层模型打下单元结构。  # 模块说明

# ========================================================================================
# 模块四：语言模型主体
# ========================================================================================

class MiniGPT(nn.Module):  # 极简 GPT 模型骨架
    def __init__(self, cfg: TransformerConfig) -> None:  # 构造函数读取配置
        super().__init__()  # 初始化父类
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)  # 将 token id 映射到向量空间
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.embed_dim)  # 添加可学习位置编码，保留顺序信息
        self.drop = nn.Dropout(cfg.dropout)  # Dropout 缓解过拟合
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])  # 堆叠多层 Transformer Block
        self.ln_f = nn.LayerNorm(cfg.embed_dim)  # 输出层归一化，提升数值稳定性
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)  # 输出层投射到词表维度，给出每个 token 的 logits
        self.context_len = cfg.context_len  # 保存上下文长度，供位置编码与遮罩校验

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # 前向计算
        B, T = idx.shape  # 读取 batch 和序列长度
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)  # 生成位置索引，形状(1, T)
        x = self.token_emb(idx) + self.pos_emb(positions)  # 词嵌入与位置嵌入相加形成输入特征
        x = self.drop(x)  # 应用 Dropout
        for block in self.blocks:  # 依次通过每个 Transformer Block
            x = block(x)  # 更新特征表示
        x = self.ln_f(x)  # 最后一层 LayerNorm
        logits = self.head(x)  # 线性层输出每个位置的 vocab logits
        return logits  # 返回 logits，供交叉熵计算

# 模型主体总结：MiniGPT 将嵌入、位置编码与多层 Block 组合，生成自回归 logits，为训练循环提供预测依据。  # 模块说明

# ========================================================================================
# 模块五：训练循环
# ========================================================================================

def train_language_model() -> None:  # 训练主函数，串联所有模块
    cfg = TransformerConfig(vocab_size=5000, context_len=128)  # 定义模型配置，确定后续组件超参
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备，优先使用 GPU
    model = MiniGPT(cfg).to(device)  # 初始化模型并搬到设备
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)  # AdamW 优化器，常用于大模型训练
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)  # 余弦退火学习率调度，帮助后期收敛
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")  # 混合精度梯度缩放，减少数值溢出

    tokens = torch.randint(0, cfg.vocab_size, (100_000,), dtype=torch.long)  # 构造随机 token 序列，实际训练需换成清洗后的海量语料
    dataset = ToyTextDataset(tokens, cfg.context_len)  # 基于 token 序列构建数据集
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)  # DataLoader 批量加载并打乱样本

    model.train()  # 设置模型为训练模式，启用 Dropout 等机制
    for epoch in range(3):  # 演示性训练多个 epoch，真实训练可能需要数万步
        for inputs, targets in loader:  # 遍历每个批次
            inputs = inputs.to(device, non_blocking=True)  # 将输入搬到训练设备
            targets = targets.to(device, non_blocking=True)  # 将标签搬到训练设备
            optimizer.zero_grad(set_to_none=True)  # 梯度清零，避免累计
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):  # 启用自动混合精度，减少显存占用
                logits = model(inputs)  # 前向计算得到预测结果
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))  # 计算交叉熵，自回归任务常用
            scaler.scale(loss).backward()  # 在缩放空间反向传播
            scaler.unscale_(optimizer)  # 恢复真实梯度，便于梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)  # 应用缩放后的优化器更新
            scaler.update()  # 根据情况调整缩放因子
            scheduler.step()  # 更新学习率，实现退火

        print(f"epoch {epoch} | loss {loss.item():.4f}")  # 打印当前 epoch 的损失，监控训练进展

# 训练模块总结：train_language_model 串联配置、数据、模型与优化器，演示自回归语言模型的完整训练闭环，为理解大模型预训练流程提供实践参考。  # 模块说明


if __name__ == "__main__":  # 入口条件，确保模块以脚本运行时执行训练
    train_language_model()  # 启动示例训练流程，帮助实地观察 loss 下降趋势




# ========================================================================================
# 模块六：综述总结
# ========================================================================================
# - 数据准备（ToyTextDataset + DataLoader）：负责把原始 token 流切成模型可接受的批次，确保训练循环每次都能取到“输入 → 目标”对。
# - 模型构建（TransformerConfig + MiniGPT）：通过配置统一超参，再把嵌入、注意力、前馈、输出层组合成可前向预测的网络。
# - 训练驱动（train_language_model）：把数据和模型连接起来，执行前向、损失计算、反向传播和参数更新，形成完整的闭环。
# △ 只要掌握这三块协作逻辑，就能将任意任务的数据换进来、按需求替换模型模块，并利用相同的训练骨架完成模型优化。
