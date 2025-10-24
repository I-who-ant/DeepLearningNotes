import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



class seeback(nn.Module):
    """CIFAR-10 示例：卷积→池化→全连接，实现 3×32×32 到 10 类打分的完整流程。"""

    def __init__(self):# 作用 : 初始化模型, 定义卷积层, 池化层, 全连接层
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2), # 32@32x32
            nn.ReLU(inplace=True), #
            nn.MaxPool2d(kernel_size=2, stride=2), # 32@16x16
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),#作用 :提取特征
            nn.ReLU(inplace=True), # 32@16x16
            nn.MaxPool2d(kernel_size=2, stride=2), # 32@8x8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),# 作用 :提取特征, 从32@8x8 到 64@8x8
            nn.ReLU(inplace=True), # 64@8x8
            nn.MaxPool2d(kernel_size=2, stride=2), # 64@4x4, 作用 : 池化, 从64@8x8 到 64@4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 作用 : 展平, 从64@4x4 到 64*4*4 个特征图
            nn.Linear(in_features=64 * 4 * 4, out_features=64),  # 作用 : 全连接层, 从64*4*4 到 64 维
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=10),  # 作用 : 全连接层, 从64 到 10 类打分
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x) # 作用 : 提取特征, 从3@32x32 到 64@4x4, 提取出64个特征图
        x = self.classifier(x) # 作用 : 分类, 从64@4x4 到 10 类打分
        return x


def describe_flow():
    """打印各阶段张量形状，帮助理解通道与尺寸的演化。"""
    model = seeback()
    x = torch.randn(1, 3, 32, 32) # 作用 : 随机生成一个样本, 3@32x32
    print("输入:", x.shape)  # 3@32x32, 作用 : 输入样本, 3通道, 32x32 尺寸

    for layer in model.features: # 作用 : 提取特征, 从3@32x32 到 64@4x4, 提取出64个特征图
        x = layer(x)
        print(f"{layer.__class__.__name__:>12}: {x.shape}") # 作用 : 打印当前层的输出形状, 帮助理解通道与尺寸的演化


    x = model.classifier[0](x) # 作用 : 展平, 从64@4x4 到 64*4*4
    print(f"{model.classifier[0].__class__.__name__:>12}: {x.shape}")  # flatten -> 64*4*4

    x = model.classifier[1](x)# 作用 : 全连接层, 从64*4*4 到 64 维
    print(f"{model.classifier[1].__class__.__name__:>12}: {x.shape}")  # Linear -> 64, 作用 : 全连接层, 从64*4*4 到 64 维

    x = model.classifier[2](x)
    print(f"{model.classifier[2].__class__.__name__:>12}: {x.shape}")  # ReLU 保持 64 维

    x = model.classifier[3](x)# 作用 : 全连接层, 从64 到 10 类打分
    print(f"{model.classifier[3].__class__.__name__:>12}: {x.shape}")  # Linear -> 10, 作用 : 全连接层, 从64 到 10 类打分
    #结果是: 10 类打分, 每个类别的打分范围在 [-inf, inf] 之间
    # 作用 : 打印模型的输出形状, 帮助理解模型的输出 :




def demo_cross_entropy():
    """演示交叉熵损失的计算过程"""
    import torch.nn.functional as F

    print("\n" + "="*70)
    print("交叉熵损失计算演示")
    print("="*70)

    # 1. 创建模型和损失函数
    model = seeback()
    criterion = nn.CrossEntropyLoss()

    # 2. 模拟一个批次的数据 (batch_size=4)
    batch_images = torch.randn(4, 3, 32, 32)  # 4张CIFAR-10图像, 3通道, 32x32 尺寸
    batch_labels = torch.tensor([3, 7, 2, 5])  # 真实标签 (0-9之间), 4个样本

    print(f"\n输入形状: {batch_images.shape}")# 4张CIFAR-10图像, 3通道, 32x32 尺寸
    print(f"标签: {batch_labels}")# 真实标签 (0-9之间), 4个样本

    # 3. 前向传播得到 logits
    logits = model(batch_images) # 作用 : 前向传播, 从3@32x32 到 10 类打分
    print(f"\n模型输出(logits)形状: {logits.shape}")  # (4, 10), 作用 : 模型的输出, 4个样本, 每个样本有10个类别的打分
    print(f"第一个样本的logits:\n{logits[0].detach()}")# 第一个样本的10个类别的打分, 范围在 [-inf, inf] 之间

    # 4. 查看 softmax 后的概率分布
    probs = F.softmax(logits, dim=1)
    print(f"\n第一个样本的概率分布:")
    for i in range(10):
        marker = "✓" if i == batch_labels[0] else " "
        print(f"  [{marker}] 类别{i}: {probs[0, i].item():.4f} ({probs[0, i].item()*100:.2f}%)")

    # 5. 计算交叉熵损失
    loss = criterion(logits, batch_labels) #
    print(f"\n交叉熵损失: {loss.item():.4f}")

    # 6. 手动验证第一个样本的损失
    print(f"\n手动计算验证:")
    correct_class = batch_labels[0].item()
    correct_prob = probs[0, correct_class].item()
    manual_loss = -torch.log(probs[0, correct_class])
    print(f"  样本1的正确类别: {correct_class}")
    print(f"  模型给正确类别的概率: {correct_prob:.4f}")
    print(f"  单样本损失 -log({correct_prob:.4f}): {manual_loss.item():.4f}")

    # 7. 计算所有样本的损失并求平均
    print(f"\n各样本的损失详情:")
    individual_losses = []
    for i in range(len(batch_labels)):
        correct_cls = batch_labels[i].item()
        prob = probs[i, correct_cls].item()
        loss_val = -torch.log(probs[i, correct_cls]).item()
        individual_losses.append(loss_val)
        print(f"  样本{i+1} | 真实类别:{correct_cls} | "
              f"预测概率:{prob:.4f} | 损失:{loss_val:.4f}")

    avg_loss = sum(individual_losses) / len(individual_losses)
    print(f"\n手动计算的平均损失: {avg_loss:.4f}")
    print(f"PyTorch计算的损失: {loss.item():.4f}")
    print(f"两者差异: {abs(avg_loss - loss.item()):.6f} (应该非常接近0)")

    # 8. 解释
    print(f"\n" + "="*70)
    print("关键理解:")
    print("="*70)
    print("1. 模型输出的是 logits (未归一化的分数)")
    print("2. CrossEntropyLoss 内部自动做 softmax 转换为概率")
    print("3. 损失值 = -log(正确类别的预测概率)")
    print("4. 预测概率越高 → 损失越小 → 模型越好")
    print("5. 预测概率越低 → 损失越大 → 模型需要改进")
    print("="*70 + "\n")


def demo_sgd_training():
    """演示使用SGD优化器训练模型"""
    import torch.nn.functional as F

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("⚠️  matplotlib未安装,将跳过可视化部分")

    print("\n" + "="*70)
    print("SGD优化器训练演示")
    print("="*70)

    # 1. 创建模型
    model = seeback()

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数总数: {total_params:,}")

    # 2. 创建优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print(f"使用优化器: SGD (lr=0.01, momentum=0)")

    # 3. 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 准备训练数据 (模拟CIFAR-10数据)
    num_samples = 100
    batch_size = 16
    num_batches = (num_samples + batch_size - 1) // batch_size

    # 生成随机数据
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))

    print(f"训练数据: {num_samples}个样本, batch_size={batch_size}, 总批次={num_batches}")

    # 5. 训练循环
    num_epochs = 50
    loss_history = []  # 记录每个batch的loss

    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("-" * 70)

    for epoch in range(num_epochs):
        epoch_losses = []

        # 分批训练
        for batch_idx in range(num_batches):
            # 获取当前批次数据
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            # 记录loss
            current_loss = loss.item()
            epoch_losses.append(current_loss)
            loss_history.append(current_loss)

            # 打印loss (每个batch)
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
                  f"Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {current_loss:.4f}")

        # 每个epoch结束后打印平均loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  → Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
        print()

    print("-" * 70)
    print("训练完成!")
    print(f"初始loss: {loss_history[0]:.4f}")
    print(f"最终loss: {loss_history[-1]:.4f}")
    print(f"总下降: {loss_history[0] - loss_history[-1]:.4f}")

    # 6. 可视化loss曲线
    if HAS_MATPLOTLIB:
        import os
        os.makedirs('artifacts', exist_ok=True)

        plt.figure(figsize=(12, 6))

        # 绘制loss曲线
        plt.plot(loss_history, 'b-', linewidth=1, alpha=0.6, label='Batch Loss')

        # 计算并绘制移动平均 (平滑曲线)
        window_size = 10
        if len(loss_history) >= window_size:
            moving_avg = []
            for i in range(len(loss_history) - window_size + 1):
                avg = sum(loss_history[i:i+window_size]) / window_size
                moving_avg.append(avg)
            plt.plot(range(window_size-1, len(loss_history)), moving_avg,
                    'r-', linewidth=2, label=f'{window_size}-Batch Moving Average')

        # 标注关键点
        plt.scatter([0], [loss_history[0]], color='green', s=100,
                   zorder=5, label=f'Start: {loss_history[0]:.4f}')
        plt.scatter([len(loss_history)-1], [loss_history[-1]], color='red',
                   s=100, zorder=5, label=f'End: {loss_history[-1]:.4f}')

        plt.xlabel('训练步数 (Batch)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('SGD训练Loss下降曲线', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 保存图片
        save_path = 'artifacts/sgd_training_loss.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\n📊 Loss曲线已保存到: {save_path}")
        plt.close()

    # 7. 测试最终模型性能
    print("\n" + "="*70)
    print("测试最终模型性能")
    print("="*70)

    model.eval()  # 切换到评估模式
    with torch.no_grad():
        # 在训练数据上测试
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y).sum().item()
        accuracy = 100 * correct / num_samples

        print(f"训练数据准确率: {correct}/{num_samples} = {accuracy:.2f}%")

        # 显示概率分布示例
        probs = F.softmax(outputs[0:1], dim=1)
        print(f"\n第一个样本预测概率分布:")
        print(f"  真实类别: {y[0].item()}")
        print(f"  预测类别: {predicted[0].item()}")
        for i in range(10):
            marker = "✓" if i == y[0].item() else " "
            pred_marker = "👉" if i == predicted[0].item() else "  "
            print(f"  [{marker}] {pred_marker} 类别{i}: {probs[0, i].item():.4f} "
                  f"({probs[0, i].item()*100:.2f}%)")


def demo_sgd_training_real_data():
    """演示使用真实CIFAR-10数据的SGD优化器训练"""
    import torch.nn.functional as F

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("⚠️  matplotlib未安装,将跳过可视化部分")

    print("\n" + "="*70)
    print("SGD优化器训练演示 (真实CIFAR-10数据)")
    print("="*70)

    # 1. 创建模型
    model = seeback()

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数总数: {total_params:,}")

    # 2. 创建优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print(f"使用优化器: SGD (lr=0.01, momentum=0)")

    # 3. 创建损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 准备真实CIFAR-10数据
    print("\n正在加载CIFAR-10数据集...")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
    ])

    # 加载训练数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 为了演示,只使用前1000个样本
    subset_size = 1000
    train_subset = torch.utils.data.Subset(train_dataset, range(subset_size))

    batch_size = 32
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    print(f"训练数据: {subset_size}个样本, batch_size={batch_size}, 总批次={len(train_loader)}")

    # CIFAR-10类别名称
    classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    print(f"CIFAR-10类别: {classes}")

    # 5. 训练循环
    num_epochs = 10  # 减少epoch数,因为真实数据训练更有效
    loss_history = []  # 记录每个batch的loss

    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("-" * 70)

    for epoch in range(num_epochs):
        epoch_losses = []

        # 使用DataLoader遍历数据
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            # 记录loss
            current_loss = loss.item()
            epoch_losses.append(current_loss)
            loss_history.append(current_loss)

            # 打印loss (每个batch)
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
                  f"Batch [{batch_idx+1:2d}/{len(train_loader)}] "
                  f"Loss: {current_loss:.4f}")

        # 每个epoch结束后打印平均loss
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  → Epoch {epoch+1} 平均Loss: {avg_loss:.4f}")
        print()

    print("-" * 70)
    print("训练完成!")
    print(f"初始loss: {loss_history[0]:.4f}")
    print(f"最终loss: {loss_history[-1]:.4f}")
    print(f"总下降: {loss_history[0] - loss_history[-1]:.4f}")

    # 6. 可视化loss曲线
    if HAS_MATPLOTLIB:
        import os
        os.makedirs('artifacts', exist_ok=True)

        plt.figure(figsize=(12, 6))

        # 绘制loss曲线
        plt.plot(loss_history, 'b-', linewidth=1, alpha=0.6, label='Batch Loss')

        # 计算并绘制移动平均 (平滑曲线)
        window_size = 10
        if len(loss_history) >= window_size:
            moving_avg = []
            for i in range(len(loss_history) - window_size + 1):
                avg = sum(loss_history[i:i+window_size]) / window_size
                moving_avg.append(avg)
            plt.plot(range(window_size-1, len(loss_history)), moving_avg,
                    'r-', linewidth=2, label=f'{window_size}-Batch Moving Average')

        # 标注关键点
        plt.scatter([0], [loss_history[0]], color='green', s=100,
                   zorder=5, label=f'Start: {loss_history[0]:.4f}')
        plt.scatter([len(loss_history)-1], [loss_history[-1]], color='red',
                   s=100, zorder=5, label=f'End: {loss_history[-1]:.4f}')

        plt.xlabel('训练步数 (Batch)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('SGD训练Loss下降曲线 (真实CIFAR-10数据)', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # 保存图片
        save_path = 'artifacts/sgd_training_real_cifar10_loss.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\n📊 Loss曲线已保存到: {save_path}")
        plt.close()

    # 7. 测试最终模型性能
    print("\n" + "="*70)
    print("测试最终模型性能")
    print("="*70)

    model.eval()  # 切换到评估模式
    with torch.no_grad():
        # 在训练数据子集上测试
        correct = 0
        total = 0
        sample_outputs = None
        sample_labels = None

        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # 保存第一个batch用于展示
            if sample_outputs is None:
                sample_outputs = outputs
                sample_labels = batch_y

        accuracy = 100 * correct / total
        print(f"训练数据准确率: {correct}/{total} = {accuracy:.2f}%")

        # 显示第一个样本的概率分布
        probs = F.softmax(sample_outputs[0:1], dim=1)
        true_class = sample_labels[0].item()
        pred_class = torch.argmax(sample_outputs[0]).item()

        print(f"\n第一个样本预测概率分布:")
        print(f"  真实类别: {true_class} ({classes[true_class]})")
        print(f"  预测类别: {pred_class} ({classes[pred_class]})")
        for i in range(10):
            marker = "✓" if i == true_class else " "
            pred_marker = "👉" if i == pred_class else "  "
            print(f"  [{marker}] {pred_marker} 类别{i} ({classes[i]:>2}): "
                  f"{probs[0, i].item():.4f} ({probs[0, i].item()*100:.2f}%)")


#科普：反向传播意思就是，
#尝试如何调整网络过程中的参数才会导致最终的loss变小（因为是从loss开始推导参数，和网络的顺序相反，所以叫反向传播），以及梯度的理解可以直接当成"斜率"

if __name__ == "__main__":
    describe_flow() # 作用 : 打印各阶段张量形状, 帮助理解通道与尺寸的演化
    demo_cross_entropy()  # 演示交叉熵损失计算
    demo_sgd_training()  # 演示SGD优化器训练 (模拟数据)
    demo_sgd_training_real_data()  # 演示SGD优化器训练 (真实CIFAR-10数据)














"""
PyTorch Module 调用链补充
=========================

1. 实例化模型触发 __init__
   model = seeback() 时，先调用 nn.Module.__init__，随后执行自定义 __init__，
   将卷积层、池化层、全连接层注册为子模块。只要写成 self.xxx = nn.Module()，
   PyTorch 就会自动把其参数加入可训练列表。

2. model(x) 调用 __call__
   执行 model(inputs) 时，nn.Module.__call__ 会：
     - 处理前向/后向 hook
     - 确认训练或评估模式 self.training
     - 调用 forward(inputs)
   因此 classifier[0](x) 实际等价于 classifier[0].forward(x)。

3. Sequential 支持索引访问
   nn.Sequential 实现了 __getitem__，可以用下标取出其中的子层。
   比如 classifier[0] 对应 Flatten，classifier[1] 对应第一个 Linear。

4. 自动求导流程
   loss.backward() 会沿着前向构建的计算图回传梯度；optimizer.step()
   根据梯度更新参数，无需手动求导。

典型训练迭代：
   model = seeback()
   logits = model(batch_images)
   loss = criterion(logits, labels)
   loss.backward()
   optimizer.step()
   optimizer.zero_grad()
"""
