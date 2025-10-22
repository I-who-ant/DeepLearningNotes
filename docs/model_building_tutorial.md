# 模型构建全流程教程

> 目标：把“任务/数据 → 数据管线 → 模型骨架 → 训练策略 → Encoder/Decoder 角色 → 综合练习”整合成一条清晰路径，可直接对照仓库中的 `HowToStartAndUnderstand.ipynb` 与 `understand.py` 实践。

## 1. 明确任务与数据形态

1. **确认业务目标**：是分类、回归、序列预测还是生成任务？
2. **盘点原始数据**：输入是图片、文本、语音还是结构化表格？标签是什么类型？
3. **预估约束条件**：可用算力、时延要求、数据规模将影响模型大小与训练策略。

示例：假设要做影评情感分类，输入为中文短文本，标签为正/负。

```python
# 假设已经有 CSV，包含列: text, label (0/1)
import pandas as pd
raw = pd.read_csv("data/movie_reviews.csv")
print(raw.head())
```

- 这一步不是为了训练，而是检查输入格式、缺失值、类别分布，为数据清洗做好准备。
## 2. 数据准备：从原始样本到 DataLoader

### 2.1 文本数据清洗与编码

```python
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }
```

- `tokenizer(...)` 把文本转成模型可识别的 token id；`attention_mask` 用来告诉模型哪些位置是有效 token。
- `ReviewDataset` 封装成 PyTorch Dataset，供下游 DataLoader 批量取数据。

```python
train_loader = DataLoader(
    ReviewDataset(raw.iloc[:-1000]),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

- `DataLoader` 负责批量化、随机打乱以及多进程加载，是连接数据与模型的关键。
## 3. 模型骨架：从最小可行到可扩展

### 3.1 轻量基线（EmbeddingBag + 线性层）

```python
import torch.nn as nn

class FastTextLike(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.classifier = nn.Linear(embed_dim, num_class)
    def forward(self, input_ids, offsets):
        pooled = self.embedding(input_ids, offsets)
        return self.classifier(pooled)
```

- `EmbeddingBag` 通过对不等长序列做平均/求和，快速得到句子向量，作为轻量基线可迅速验证可行性。
- `forward` 返回 logits，后续可直接喂入交叉熵损失。

### 3.2 预训练骨架（BERT Encoder + 分类头）

```python
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, pretrained_name: str, num_class: int):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_class)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0]  # [CLS] 向量
        return self.classifier(cls_repr)
```

- `BertModel` 在内部就是一个 **Encoder 堆栈**，负责理解整段文本。
- 只需要外接一个线性分类层，就能把语义特征映射到标签空间。
- 这体现了“通用骨架 + 任务特定头”的模式，便于替换和扩展
- 
## 4. 训练循环与策略

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier("bert-base-chinese", num_class=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

- **损失函数**：交叉熵衡量预测分布与真实标签差距。
- **优化器**：AdamW 在大模型微调中表现稳定；`weight_decay` 控制 L2 正则。
- **学习率调度**：余弦退火帮助后期收敛；真实场景可配合 warmup。
- **混合精度**：`autocast + GradScaler` 降低显存占用的同时保持稳定性。
- **梯度裁剪**：防止梯度爆炸，尤其是 RNN/Transformer 中常用。
## 5. Encoder 与 Decoder 的角色

### 5.1 Encoder-only（理解输入）

```python
from torch import nn

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

x = torch.randn(16, 120, 512)            # (batch, seq_len, hidden)
mask = torch.zeros(16, 120, dtype=torch.bool)
encoded = encoder(x, src_key_padding_mask=mask)
```

- `TransformerEncoder` 连续堆叠自注意力层，输出与输入同长的序列表示。
- 常用于分类（取 [CLS] 或平均）、检索（拿整个序列特征）、编码器-解码器的“编码”阶段。

### 5.2 Decoder-only（生成输出）

```python
from torch import nn

class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
    def forward(self, tgt_ids):
        positions = torch.arange(tgt_ids.size(1), device=tgt_ids.device)
        tgt = self.token_emb(tgt_ids) + self.pos_emb(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        decoded = self.decoder(tgt, memory=None, tgt_mask=causal_mask)
        return self.output(decoded)
```

- 此结构与 `understand.py` 中的 `MiniGPT` 类似，属于纯 Decoder，靠因果遮罩确保“只看过去”。
- 若用于翻译，可把 Encoder 输出作为 `memory` 传入 `self.decoder`，即可实现 Encoder-Decoder 架构。
## 6. 综合练习：套用流程到语音识别

### 6.1 数据 → 特征

```python
import torchaudio

train_ds = torchaudio.datasets.LIBRISPEECH(root="data", url="train-clean-100", download=True)
mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)

def collate_fn(batch):
    waveforms, labels, input_lengths, label_lengths = [], [], [], []
    for waveform, _, transcript, _, _, _ in batch:
        waveforms.append(mel(waveform).squeeze(0).transpose(0, 1))  # (time, mel)
        labels.append(text_to_int(transcript))
        input_lengths.append(waveforms[-1].size(0))
        label_lengths.append(len(labels[-1]))
    return pad_features(waveforms), pad_labels(labels), torch.tensor(input_lengths), torch.tensor(label_lengths)

loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
```

- 将波形转成梅尔频谱，再配合 `collate_fn` 对齐动态长度，是语音模型的常规做法。

### 6.2 模型 → 损失

```python
class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(256, 512, num_layers=3, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512 * 2, num_classes)
    def forward(self, features):
        x = self.cnn(features.transpose(1, 2))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.classifier(x)

criterion = nn.CTCLoss(blank=0)
```

- CNN 压缩时间维，LSTM 捕捉上下文，线性层输出每个时间步的字符概率，配合 CTC 损失对齐不同长度的输入与标签。

### 6.3 训练循环

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for wave_batch, label_batch, input_lengths, label_lengths in loader:
    wave_batch = wave_batch.to(device)
    label_batch = label_batch.to(device)
    logits = model(wave_batch)                 # (batch, time, num_classes)
    log_probs = logits.log_softmax(dim=-1)
    loss = criterion(log_probs.transpose(0, 1), label_batch, input_lengths, label_lengths)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    optimizer.zero_grad()
```

- 这段代码与前面训练循环的结构完全一致，只是损失函数和数据形态不同。
- 通过替换数据/模型组件，就能把通用流程迁移到语音识别任务。

---

> 记忆口诀：**先认清任务和数据 → 决定数据管线 → 套最小模型骨架 → 用合适的损失与优化器训练 → 观察指标再迭代 → 根据需求切换 Encoder/Decoder or 混合结构**。配合 `understand.py` 的注释代码，可以快速定位每一步在真实工程中的落点。
