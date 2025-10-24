"""
示例6: 详细的模型评估与分析

在前面示例的基础上添加:
- 使用argmax获取预测结果的详细解释
- 测试集上的完整评估
- 每个类别的准确率、精确率、召回率、F1分数
- 混淆矩阵可视化
- Top-K准确率
- 预测概率分布分析
- 错误样本可视化
- 置信度分析

运行: python src/experiments/model_train/06_detailed_evaluation.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ============================================================
# 1. 定义相同的CNN模型
# ============================================================
class SimpleCNN(nn.Module):
    """简单的3层卷积神经网络"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================
# 2. 准备数据
# ============================================================
def prepare_data(val_ratio=0.2):
    """加载CIFAR-10数据集"""
    print("正在加载 CIFAR-10 数据集...")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 测试集不使用数据增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 训练集
    full_train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )

    # 划分训练集和验证集
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 测试集
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")

    return train_loader, val_loader, test_loader


# ============================================================
# 3. 简单训练函数 (复用之前的逻辑)
# ============================================================
def train_one_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def validate(model, val_loader, criterion):
    """验证函数"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total


# ============================================================
# 4. 详细的测试集评估 - 新增!
# ============================================================
def detailed_evaluation(model, test_loader, class_names):
    """
    在测试集上进行详细评估

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        class_names: 类别名称列表

    返回:
        all_labels: 所有真实标签
        all_preds: 所有预测标签
        all_probs: 所有预测概率
    """
    print("\n" + "=" * 70)
    print("📊 详细的测试集评估")
    print("=" * 70)

    model.eval()

    all_labels = []      # 存储所有真实标签
    all_preds = []       # 存储所有预测标签
    all_probs = []       # 存储所有预测概率

    correct = 0
    total = 0

    print("\n正在评估测试集...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # ========================================
            # 前向传播获取输出
            # ========================================
            outputs = model(images)  # shape: [batch_size, num_classes]

            # ========================================
            # 方法1: 使用 torch.max 获取预测类别
            # ========================================
            # torch.max返回两个值: (最大值, 最大值的索引)
            # dim=1 表示在类别维度上找最大值
            max_values, predicted = torch.max(outputs, dim=1)

            # ========================================
            # 方法2: 使用 argmax 获取预测类别 (等价于torch.max)
            # ========================================
            # predicted_argmax = torch.argmax(outputs, dim=1)
            # 注意: torch.max 和 torch.argmax 结果相同
            # torch.max 额外返回最大值,而 argmax 只返回索引

            # ========================================
            # 获取预测概率 (使用softmax)
            # ========================================
            probabilities = torch.softmax(outputs, dim=1)  # shape: [batch_size, num_classes]

            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 批次")

    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 计算总体准确率
    accuracy = 100 * correct / total

    print(f"\n✅ 评估完成!")
    print(f"📊 总体准确率: {accuracy:.2f}% ({correct}/{total})")

    return all_labels, all_preds, all_probs


# ============================================================
# 5. 每个类别的详细指标 - 新增!
# ============================================================
def per_class_metrics(all_labels, all_preds, class_names):
    """
    计算每个类别的详细指标

    参数:
        all_labels: 真实标签
        all_preds: 预测标签
        class_names: 类别名称
    """
    print("\n" + "=" * 70)
    print("📈 每个类别的详细指标")
    print("=" * 70)

    # 使用sklearn计算分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )

    # 打印表头
    print(f"\n{'类别':<12} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>8}")
    print("-" * 55)

    # 打印每个类别的指标
    for i, class_name in enumerate(class_names):
        metrics = report[class_name]
        print(f"{class_name:<12} {metrics['precision']:>9.2%} {metrics['recall']:>9.2%} "
              f"{metrics['f1-score']:>9.2%} {int(metrics['support']):>8}")

    # 打印总体指标
    print("-" * 55)
    print(f"{'总体准确率':<12} {report['accuracy']:>9.2%}")
    print(f"{'宏平均':<12} {report['macro avg']['precision']:>9.2%} "
          f"{report['macro avg']['recall']:>9.2%} {report['macro avg']['f1-score']:>9.2%}")
    print(f"{'加权平均':<12} {report['weighted avg']['precision']:>9.2%} "
          f"{report['weighted avg']['recall']:>9.2%} {report['weighted avg']['f1-score']:>9.2%}")

    print("\n💡 指标说明:")
    print("  - 精确率 (Precision): 预测为该类的样本中,真正属于该类的比例")
    print("  - 召回率 (Recall): 真正属于该类的样本中,被正确预测的比例")
    print("  - F1分数: 精确率和召回率的调和平均数")


# ============================================================
# 6. 混淆矩阵可视化 - 新增!
# ============================================================
def plot_confusion_matrix(all_labels, all_preds, class_names, save_path='artifacts/confusion_matrix.png'):
    """
    绘制混淆矩阵

    参数:
        all_labels: 真实标签
        all_preds: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    print("\n" + "=" * 70)
    print("🔥 混淆矩阵")
    print("=" * 70)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 创建图表
    plt.figure(figsize=(12, 10))

    # 绘制混淆矩阵
    sns.heatmap(
        cm,
        annot=True,           # 显示数值
        fmt='d',              # 整数格式
        cmap='Blues',         # 颜色映射
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': '样本数量'}
    )

    plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 混淆矩阵已保存到: {save_path}")

    plt.close()

    # 分析混淆矩阵
    print("\n💡 混淆矩阵分析:")
    print("  - 对角线: 正确预测的样本数量")
    print("  - 非对角线: 错误预测的样本数量")

    # 找出最容易混淆的类别对
    np.fill_diagonal(cm, 0)  # 忽略对角线
    max_confusion_idx = np.unravel_index(cm.argmax(), cm.shape)
    true_class = class_names[max_confusion_idx[0]]
    pred_class = class_names[max_confusion_idx[1]]
    confusion_count = cm[max_confusion_idx]

    print(f"\n⚠️ 最容易混淆的类别对:")
    print(f"  真实类别: {true_class}")
    print(f"  预测为: {pred_class}")
    print(f"  混淆次数: {confusion_count} 次")


# ============================================================
# 7. Top-K准确率 - 新增!
# ============================================================
def top_k_accuracy(all_labels, all_probs, class_names, k_values=[1, 3, 5]):
    """
    计算Top-K准确率

    参数:
        all_labels: 真实标签
        all_probs: 预测概率
        class_names: 类别名称
        k_values: K值列表
    """
    print("\n" + "=" * 70)
    print("🎯 Top-K 准确率")
    print("=" * 70)

    print("\n💡 Top-K准确率说明:")
    print("  Top-1: 预测概率最高的类别是正确类别的比例")
    print("  Top-3: 预测概率前3的类别中包含正确类别的比例")
    print("  Top-5: 预测概率前5的类别中包含正确类别的比例")

    print(f"\n{'K值':<6} {'准确率':>10} {'说明'}")
    print("-" * 40)

    for k in k_values:
        # 获取Top-K预测
        top_k_preds = np.argsort(all_probs, axis=1)[:, -k:]  # 每个样本的Top-K预测

        # 计算Top-K准确率
        correct = 0
        for i, true_label in enumerate(all_labels):
            if true_label in top_k_preds[i]:
                correct += 1

        accuracy = 100 * correct / len(all_labels)
        print(f"Top-{k:<3} {accuracy:>9.2f}%  预测概率前{k}中包含正确类别")


# ============================================================
# 8. 预测置信度分析 - 新增!
# ============================================================
def confidence_analysis(all_labels, all_preds, all_probs, save_path='artifacts/confidence_analysis.png'):
    """
    分析预测置信度

    参数:
        all_labels: 真实标签
        all_preds: 预测标签
        all_probs: 预测概率
        save_path: 保存路径
    """
    print("\n" + "=" * 70)
    print("📊 预测置信度分析")
    print("=" * 70)

    # 获取预测的最大概率 (即置信度)
    confidences = np.max(all_probs, axis=1)

    # 判断预测是否正确
    correct_mask = (all_labels == all_preds)

    # 分别获取正确和错误预测的置信度
    correct_confidences = confidences[correct_mask]
    wrong_confidences = confidences[~correct_mask]

    # 打印统计信息
    print(f"\n正确预测的置信度统计:")
    print(f"  平均值: {np.mean(correct_confidences):.4f}")
    print(f"  中位数: {np.median(correct_confidences):.4f}")
    print(f"  最小值: {np.min(correct_confidences):.4f}")
    print(f"  最大值: {np.max(correct_confidences):.4f}")

    print(f"\n错误预测的置信度统计:")
    print(f"  平均值: {np.mean(wrong_confidences):.4f}")
    print(f"  中位数: {np.median(wrong_confidences):.4f}")
    print(f"  最小值: {np.min(wrong_confidences):.4f}")
    print(f"  最大值: {np.max(wrong_confidences):.4f}")

    # 绘制置信度分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 置信度直方图
    axes[0].hist(correct_confidences, bins=50, alpha=0.7, label='正确预测', color='green')
    axes[0].hist(wrong_confidences, bins=50, alpha=0.7, label='错误预测', color='red')
    axes[0].set_xlabel('预测置信度', fontsize=12)
    axes[0].set_ylabel('样本数量', fontsize=12)
    axes[0].set_title('预测置信度分布', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: 置信度箱线图
    data = [correct_confidences, wrong_confidences]
    axes[1].boxplot(data, labels=['正确预测', '错误预测'])
    axes[1].set_ylabel('预测置信度', fontsize=12)
    axes[1].set_title('预测置信度箱线图', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 置信度分析图已保存到: {save_path}")

    plt.close()


# ============================================================
# 9. 错误样本可视化 - 新增!
# ============================================================
def visualize_errors(model, test_dataset, all_labels, all_preds, all_probs, class_names,
                     num_samples=16, save_path='artifacts/error_samples.png'):
    """
    可视化错误预测的样本

    参数:
        model: 模型
        test_dataset: 测试数据集
        all_labels: 真实标签
        all_preds: 预测标签
        all_probs: 预测概率
        class_names: 类别名称
        num_samples: 显示的样本数量
        save_path: 保存路径
    """
    print("\n" + "=" * 70)
    print("🔍 错误样本可视化")
    print("=" * 70)

    # 找出所有错误预测的索引
    error_indices = np.where(all_labels != all_preds)[0]

    print(f"\n总错误数: {len(error_indices)}")
    print(f"显示前 {num_samples} 个错误样本...")

    # 随机选择一些错误样本
    np.random.seed(42)
    selected_indices = np.random.choice(error_indices, min(num_samples, len(error_indices)), replace=False)

    # 创建图表
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    fig.suptitle('错误预测样本分析', fontsize=16, fontweight='bold')

    for idx, error_idx in enumerate(selected_indices):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        # 获取图片和标签
        image, _ = test_dataset[error_idx]

        # 反归一化用于显示
        image_np = image.numpy().transpose(1, 2, 0)
        image_np = image_np * 0.5 + 0.5  # 反归一化
        image_np = np.clip(image_np, 0, 1)

        # 获取预测信息
        true_label = all_labels[error_idx]
        pred_label = all_preds[error_idx]
        confidence = all_probs[error_idx, pred_label]

        # 显示图片
        axes[row, col].imshow(image_np)
        axes[row, col].axis('off')

        # 添加标题
        title = f'真实: {class_names[true_label]}\n' \
                f'预测: {class_names[pred_label]}\n' \
                f'置信度: {confidence:.2%}'
        axes[row, col].set_title(title, fontsize=10, color='red')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 错误样本可视化已保存到: {save_path}")

    plt.close()


# ============================================================
# 10. argmax详细解释 - 新增!
# ============================================================
def explain_argmax():
    """详细解释argmax的原理和使用"""
    print("\n" + "=" * 70)
    print("📚 argmax 详细解释")
    print("=" * 70)

    print("""
🎯 什么是 argmax?
  argmax 返回数组中最大值的索引(位置)

🔍 示例演示:
""")

    # 示例1: 简单的1维数组
    scores = np.array([0.1, 0.3, 0.8, 0.2, 0.5])
    max_idx = np.argmax(scores)

    print("示例1: 一维数组")
    print(f"  输入: {scores}")
    print(f"  argmax结果: {max_idx} (第{max_idx}个位置,值为{scores[max_idx]})")

    # 示例2: 神经网络输出
    print("\n示例2: 神经网络输出 (CIFAR-10分类)")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 模拟一个输出 (logits)
    output = np.array([2.1, -0.5, 1.8, 0.3, -1.2, 0.8, 3.5, 1.1, 0.2, -0.8])

    print(f"\n  网络输出 (logits):")
    for i, (name, score) in enumerate(zip(class_names, output)):
        print(f"    {i}. {name:<12}: {score:>6.2f}")

    # 使用argmax获取预测类别
    pred_class = np.argmax(output)
    print(f"\n  argmax结果: {pred_class} -> {class_names[pred_class]}")
    print(f"  最大值: {output[pred_class]:.2f}")

    # 示例3: Batch预测
    print("\n示例3: Batch预测 (多个样本)")

    # 模拟3个样本的输出
    batch_output = np.array([
        [2.1, -0.5, 1.8, 0.3, -1.2, 0.8, 3.5, 1.1, 0.2, -0.8],  # 样本1
        [1.2, 2.8, 0.3, 0.5, 1.1, 0.2, 0.7, 0.9, 1.5, 0.4],      # 样本2
        [0.1, 0.2, 3.2, 1.1, 0.8, 1.5, 0.9, 1.2, 0.5, 0.3],      # 样本3
    ])

    print(f"  Batch输出 shape: {batch_output.shape} (3个样本, 10个类别)")

    # 对每个样本使用argmax
    batch_preds = np.argmax(batch_output, axis=1)  # axis=1表示在类别维度上取最大值

    print(f"\n  Batch预测结果:")
    for i, pred in enumerate(batch_preds):
        print(f"    样本{i+1}: 类别{pred} ({class_names[pred]})")

    # PyTorch vs NumPy
    print("\n" + "=" * 70)
    print("🔧 PyTorch vs NumPy")
    print("=" * 70)

    print("""
NumPy:
  pred = np.argmax(output, axis=1)

PyTorch (方法1 - 使用argmax):
  pred = torch.argmax(output, dim=1)

PyTorch (方法2 - 使用max):
  max_values, pred = torch.max(output, dim=1)
  # max返回两个值: (最大值, 最大值的索引)

💡 注意:
  - NumPy使用 axis 参数
  - PyTorch使用 dim 参数
  - torch.max 额外返回最大值本身
  - torch.argmax 只返回索引
""")


# ============================================================
# 11. 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 70)
    print("示例6: 详细的模型评估与分析")
    print("=" * 70)

    # 首先解释argmax
    explain_argmax()

    # 准备数据
    train_loader, val_loader, test_loader = prepare_data(val_ratio=0.2)

    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 选项: 是否训练新模型
    print("\n" + "=" * 70)
    response = input("是否训练新模型? (y/n, 直接回车使用预训练模型): ")

    if response.lower() == 'y':
        # 训练模型
        print("\n开始训练模型...")
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

        best_val_acc = 0.0
        num_epochs = 20

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # 保存最佳模型
                os.makedirs('artifacts/checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 'artifacts/checkpoints/best_model_eval.pth')

            scheduler.step()

        print(f"\n✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

    else:
        # 尝试加载预训练模型
        model = SimpleCNN()
        model_path = 'artifacts/checkpoints/best_model_eval.pth'

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"\n✅ 已加载预训练模型: {model_path}")
        else:
            print(f"\n⚠️ 未找到预训练模型,将使用随机初始化的模型")
            print("   (建议选择 'y' 训练新模型以获得更好的评估结果)")

    # ========================================
    # 详细评估
    # ========================================

    # 1. 测试集评估
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )

    all_labels, all_preds, all_probs = detailed_evaluation(model, test_loader, class_names)

    # 2. 每个类别的详细指标
    per_class_metrics(all_labels, all_preds, class_names)

    # 3. 混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # 4. Top-K准确率
    top_k_accuracy(all_labels, all_probs, class_names, k_values=[1, 3, 5])

    # 5. 置信度分析
    confidence_analysis(all_labels, all_preds, all_probs)

    # 6. 错误样本可视化
    visualize_errors(model, test_dataset, all_labels, all_preds, all_probs, class_names)

    print("\n" + "=" * 70)
    print("✅ 所有评估完成!")
    print("=" * 70)
    print("\n💡 新增内容:")
    print("  1. ✅ argmax详细解释和示例")
    print("  2. ✅ 测试集完整评估")
    print("  3. ✅ 每个类别的精确率/召回率/F1分数")
    print("  4. ✅ 混淆矩阵可视化")
    print("  5. ✅ Top-K准确率")
    print("  6. ✅ 预测置信度分析")
    print("  7. ✅ 错误样本可视化")
    print("\n📊 生成的可视化文件:")
    print("  - artifacts/confusion_matrix.png      (混淆矩阵)")
    print("  - artifacts/confidence_analysis.png   (置信度分析)")
    print("  - artifacts/error_samples.png         (错误样本)")
    print("=" * 70)


if __name__ == '__main__':
    main()
