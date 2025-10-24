"""
深度学习模型保存与读取完全指南

这个模块详细介绍各种框架的模型保存与读取方法：
1. PyTorch 模型保存 (state_dict, 完整模型, checkpoint)
2. Keras/TensorFlow 模型保存 (SavedModel, HDF5, weights)
3. ONNX 格式 (跨框架部署)
4. TorchScript (PyTorch 部署)
5. 最佳实践和常见陷阱

作者: Seeback
日期: 2025-10-23
"""

import torch
import torch.nn as nn
from torchvision import models
import os
import json
from datetime import datetime


def explain_pytorch_save_methods():
    """解释 PyTorch 的保存方法"""
    print("=" * 70)
    print("PyTorch 模型保存方法详解")
    print("=" * 70)

    print("\n📦 1. 三种主要保存方式")
    print("-" * 70)
    print("""
    方式一: 仅保存参数 (state_dict) ⭐ 推荐
    ─────────────────────────────────────────────
    优点:
    ✅ 文件小 - 只保存权重
    ✅ 灵活 - 可以加载到不同架构
    ✅ 安全 - 不包含代码,不执行任意代码
    ✅ 版本兼容 - PyTorch 版本升级友好

    缺点:
    ❌ 需要模型定义 - 必须先创建模型实例

    使用场景:
    - 训练完成后保存最佳模型
    - 分享模型权重给他人
    - 生产环境部署
    ─────────────────────────────────────────────

    方式二: 保存完整模型
    ─────────────────────────────────────────────
    优点:
    ✅ 方便 - 不需要模型定义代码

    缺点:
    ❌ 文件大 - 包含整个模型结构
    ❌ 不灵活 - 依赖保存时的代码
    ❌ 版本问题 - PyTorch 版本升级可能失败
    ❌ 安全风险 - 使用 pickle, 可能执行恶意代码

    使用场景:
    - 快速原型和实验
    - 个人项目
    ─────────────────────────────────────────────

    方式三: 保存 Checkpoint (训练状态)
    ─────────────────────────────────────────────
    优点:
    ✅ 完整 - 包含所有训练信息
    ✅ 可恢复 - 可以从中断处继续训练

    内容:
    - 模型参数 (model.state_dict())
    - 优化器状态 (optimizer.state_dict())
    - 当前 epoch
    - 当前 loss
    - 学习率调度器状态
    - 随机数种子

    使用场景:
    - 长时间训练 (定期保存)
    - 分布式训练
    - 超参数搜索
    """)


def demo_pytorch_save_state_dict():
    """演示 PyTorch state_dict 保存方法"""
    print("\n" + "=" * 70)
    print("方法一: 保存 state_dict (推荐)")
    print("=" * 70)

    print("\n1️⃣ 创建并训练模型:")
    print("-" * 70)

    # 创建简单模型
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(16 * 32 * 32, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleNet()
    print(f"   模型创建完成: {sum(p.numel() for p in model.parameters())} 参数")

    print("\n2️⃣ 保存模型参数:")
    print("-" * 70)
    print("""
    # 方法 A: 只保存参数
    torch.save(model.state_dict(), 'model_weights.pth')

    # 方法 B: 保存参数 + 额外信息 (推荐)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 10,
        'accuracy': 0.95,
        'model_architecture': 'SimpleNet',
        'date': datetime.now().isoformat()
    }, 'model_checkpoint.pth')
    """)

    # 实际保存
    os.makedirs('artifacts/models', exist_ok=True)

    # 方法 A
    torch.save(model.state_dict(), 'artifacts/models/simple_weights.pth')
    size_a = os.path.getsize('artifacts/models/simple_weights.pth') / 1024
    print(f"   ✅ 方法A 已保存: simple_weights.pth ({size_a:.1f} KB)")

    # 方法 B
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 10,
        'accuracy': 0.95,
        'model_architecture': 'SimpleNet',
        'date': datetime.now().isoformat()
    }, 'artifacts/models/simple_checkpoint.pth')
    size_b = os.path.getsize('artifacts/models/simple_checkpoint.pth') / 1024
    print(f"   ✅ 方法B 已保存: simple_checkpoint.pth ({size_b:.1f} KB)")

    print("\n3️⃣ 加载模型参数:")
    print("-" * 70)
    print("""
    # 方法 A: 加载纯参数
    model = SimpleNet()  # 先创建模型实例
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # 设置为评估模式

    # 方法 B: 加载 checkpoint
    model = SimpleNet()
    checkpoint = torch.load('model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    """)

    # 实际加载
    model_loaded = SimpleNet()
    model_loaded.load_state_dict(torch.load('artifacts/models/simple_weights.pth'))
    model_loaded.eval()
    print(f"   ✅ 方法A 加载成功")

    checkpoint = torch.load('artifacts/models/simple_checkpoint.pth')
    model_loaded2 = SimpleNet()
    model_loaded2.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ 方法B 加载成功: epoch={checkpoint['epoch']}, accuracy={checkpoint['accuracy']}")

    print("\n4️⃣ 验证加载正确:")
    print("-" * 70)
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out_original = model(dummy_input)
        out_loaded = model_loaded(dummy_input)
        diff = (out_original - out_loaded).abs().max().item()

    print(f"   原始模型输出: {out_original[0][:3].tolist()}")
    print(f"   加载模型输出: {out_loaded[0][:3].tolist()}")
    print(f"   最大差异: {diff:.10f}")
    if diff < 1e-6:
        print(f"   ✅ 加载完全正确!")
    else:
        print(f"   ❌ 加载有误!")


def demo_pytorch_save_full_model():
    """演示保存完整模型"""
    print("\n" + "=" * 70)
    print("方法二: 保存完整模型 (不推荐)")
    print("=" * 70)

    print("\n⚠️ 警告: 这种方法存在安全风险和兼容性问题")
    print("-" * 70)

    print("\n1️⃣ 保存完整模型:")
    print("-" * 70)
    print("""
    torch.save(model, 'model_full.pth')

    注意: 由于 pickle 的限制,局部定义的类无法序列化。
    这正是不推荐使用完整模型保存的原因之一!
    """)

    print("\n2️⃣ 加载完整模型:")
    print("-" * 70)
    print("""
    model = torch.load('model_full.pth')
    model.eval()
    """)

    print("\n❌ 这种方法的问题:")
    print("-" * 70)
    print("""
    1. 依赖保存时的代码 - 如果类定义改变,加载会失败
    2. 安全风险 - pickle 可以执行任意代码
    3. 版本兼容 - PyTorch 版本升级可能导致失败
    4. 文件更大 - 包含不必要的信息

    结论: 除非快速实验,否则不推荐使用!
    """)


def demo_pytorch_checkpoint():
    """演示完整训练 checkpoint"""
    print("\n" + "=" * 70)
    print("方法三: 保存训练 Checkpoint")
    print("=" * 70)

    print("\n1️⃣ 完整的 Checkpoint 内容:")
    print("-" * 70)

    # 创建模型和优化器
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("""
    checkpoint = {
        # 必需信息
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,

        # 可选信息
        'scheduler_state_dict': scheduler.state_dict(),
        'best_accuracy': best_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'random_seed': random_seed,

        # 元信息
        'model_architecture': 'ResNet18',
        'num_classes': 10,
        'date': datetime.now().isoformat(),
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'weight_decay': 1e-4
        }
    }
    """)

    print("\n2️⃣ 保存 Checkpoint:")
    print("-" * 70)

    checkpoint = {
        'epoch': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': 0.523,
        'best_accuracy': 0.92,
        'model_architecture': 'ResNet18',
        'num_classes': 10,
        'date': datetime.now().isoformat(),
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 32,
        }
    }

    torch.save(checkpoint, 'artifacts/models/resnet18_checkpoint.pth')
    size = os.path.getsize('artifacts/models/resnet18_checkpoint.pth') / (1024 * 1024)
    print(f"   ✅ 已保存: resnet18_checkpoint.pth ({size:.1f} MB)")

    print("\n3️⃣ 恢复训练:")
    print("-" * 70)
    print("""
    # 加载 checkpoint
    checkpoint = torch.load('checkpoint.pth')

    # 恢复模型
    model = ResNet18()
    model.load_state_dict(checkpoint['model_state_dict'])

    # 恢复优化器
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 恢复训练状态
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_accuracy']

    # 继续训练
    for epoch in range(start_epoch, num_epochs):
        train(...)
    """)

    # 实际加载
    checkpoint_loaded = torch.load('artifacts/models/resnet18_checkpoint.pth')
    print(f"\n   ✅ Checkpoint 信息:")
    print(f"      Epoch: {checkpoint_loaded['epoch']}")
    print(f"      Loss: {checkpoint_loaded['loss']}")
    print(f"      Best Accuracy: {checkpoint_loaded['best_accuracy']}")
    print(f"      Date: {checkpoint_loaded['date']}")


def demo_save_pretrained_model():
    """演示保存预训练模型"""
    print("\n" + "=" * 70)
    print("实战: 保存和加载预训练模型")
    print("=" * 70)

    print("\n1️⃣ 加载预训练模型并修改:")
    print("-" * 70)

    # 加载预训练 ResNet
    model = models.resnet18(weights='DEFAULT')
    num_classes = 10

    # 修改最后一层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    print(f"   ✅ ResNet18 加载完成")
    print(f"   ✅ 修改输出层: {model.fc}")

    print("\n2️⃣ 保存微调后的模型:")
    print("-" * 70)

    # 保存完整信息
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_name': 'resnet18',
        'num_classes': num_classes,
        'pretrained': True,
        'modified_layers': ['fc'],
        'date': datetime.now().isoformat()
    }

    torch.save(save_dict, 'artifacts/models/resnet18_finetuned.pth')
    size = os.path.getsize('artifacts/models/resnet18_finetuned.pth') / (1024 * 1024)
    print(f"   ✅ 已保存: resnet18_finetuned.pth ({size:.1f} MB)")

    print("\n3️⃣ 加载微调后的模型:")
    print("-" * 70)

    # 加载
    checkpoint = torch.load('artifacts/models/resnet18_finetuned.pth')

    # 重建模型
    model_loaded = models.resnet18(weights=None)
    model_loaded.fc = nn.Linear(model_loaded.fc.in_features, checkpoint['num_classes'])
    model_loaded.load_state_dict(checkpoint['model_state_dict'])
    model_loaded.eval()

    print(f"   ✅ 加载成功")
    print(f"      模型: {checkpoint['model_name']}")
    print(f"      类别数: {checkpoint['num_classes']}")
    print(f"      预训练: {checkpoint['pretrained']}")


def explain_onnx_format():
    """解释 ONNX 格式"""
    print("\n" + "=" * 70)
    print("ONNX 格式 - 跨框架部署")
    print("=" * 70)

    print("\n📦 1. 什么是 ONNX?")
    print("-" * 70)
    print("""
    ONNX (Open Neural Network Exchange) 是跨框架的模型格式

    优点:
    ✅ 跨框架 - PyTorch → TensorFlow → ONNX Runtime
    ✅ 优化 - 针对推理优化
    ✅ 部署友好 - 支持多种硬件 (CPU, GPU, 移动端)
    ✅ 标准化 - 工业界广泛支持

    使用场景:
    - 模型部署到生产环境
    - 跨框架模型转换
    - 移动端和边缘设备部署
    """)

    print("\n💻 2. PyTorch 导出到 ONNX:")
    print("-" * 70)
    print("""
    import torch
    import torch.onnx

    # 创建模型
    model = MyModel()
    model.eval()

    # 准备示例输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出为 ONNX
    torch.onnx.export(
        model,                          # 模型
        dummy_input,                    # 示例输入
        'model.onnx',                   # 输出文件
        export_params=True,             # 导出参数
        opset_version=11,               # ONNX 版本
        do_constant_folding=True,       # 优化常量折叠
        input_names=['input'],          # 输入名称
        output_names=['output'],        # 输出名称
        dynamic_axes={                  # 动态维度
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    """)

    print("\n3️⃣ 使用 ONNX Runtime 推理:")
    print("-" * 70)
    print("""
    import onnxruntime as ort

    # 加载 ONNX 模型
    session = ort.InferenceSession('model.onnx')

    # 准备输入
    input_data = numpy_array

    # 推理
    outputs = session.run(
        None,
        {'input': input_data}
    )
    """)


def explain_torchscript():
    """解释 TorchScript"""
    print("\n" + "=" * 70)
    print("TorchScript - PyTorch 生产部署")
    print("=" * 70)

    print("\n📦 1. 什么是 TorchScript?")
    print("-" * 70)
    print("""
    TorchScript 是 PyTorch 模型的中间表示,用于生产部署

    优点:
    ✅ 无 Python 依赖 - C++ 环境可运行
    ✅ 优化 - JIT 编译优化
    ✅ 序列化 - 保存为 .pt 文件
    ✅ 跨平台 - 移动端、服务器、边缘设备

    两种转换方式:
    - Tracing: 跟踪执行路径 (推荐)
    - Scripting: 编译 Python 代码
    """)

    print("\n💻 2. Tracing (跟踪):")
    print("-" * 70)
    print("""
    import torch

    model = MyModel()
    model.eval()

    # 准备示例输入
    example_input = torch.randn(1, 3, 224, 224)

    # Tracing
    traced_model = torch.jit.trace(model, example_input)

    # 保存
    traced_model.save('model_traced.pt')

    # 加载和推理
    loaded_model = torch.jit.load('model_traced.pt')
    output = loaded_model(input_tensor)
    """)

    print("\n3️⃣ Scripting (脚本化):")
    print("-" * 70)
    print("""
    # 方法 A: 装饰器
    @torch.jit.script
    class MyModule(nn.Module):
        def forward(self, x):
            return x * 2

    # 方法 B: 函数调用
    scripted_model = torch.jit.script(model)

    # 保存
    scripted_model.save('model_scripted.pt')
    """)


def best_practices():
    """最佳实践"""
    print("\n" + "=" * 70)
    print("模型保存与加载 - 最佳实践")
    print("=" * 70)

    print("\n✅ 推荐做法:")
    print("-" * 70)
    print("""
    1. 使用 state_dict 方式保存
       torch.save(model.state_dict(), 'model.pth')

    2. 保存额外元信息
       torch.save({
           'model_state_dict': model.state_dict(),
           'epoch': epoch,
           'optimizer_state_dict': optimizer.state_dict(),
           'loss': loss,
           'accuracy': acc,
           'hyperparameters': {...},
           'date': datetime.now().isoformat()
       }, 'checkpoint.pth')

    3. 定期保存 checkpoint (每 N 个 epoch)
       if epoch % 5 == 0:
           torch.save(...)

    4. 保存最佳模型
       if val_acc > best_acc:
           torch.save(model.state_dict(), 'best_model.pth')
           best_acc = val_acc

    5. 加载时使用 map_location (CPU/GPU 兼容)
       checkpoint = torch.load('model.pth', map_location='cpu')

    6. 设置评估模式
       model.eval()
       with torch.no_grad():
           predictions = model(input)
    """)

    print("\n❌ 避免的错误:")
    print("-" * 70)
    print("""
    1. ❌ 不要保存完整模型
       torch.save(model, 'model.pth')  # 不推荐!

    2. ❌ 不要忘记 model.eval()
       model.load_state_dict(...)
       # 忘记 model.eval()
       predictions = model(input)  # 错误! dropout 和 BN 还在训练模式

    3. ❌ 不要硬编码设备
       model.to('cuda')  # 如果没有 GPU 会报错
       # 应该:
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model.to(device)

    4. ❌ 不要覆盖唯一的模型文件
       # 应该保存多个版本:
       torch.save(..., f'model_epoch_{epoch}.pth')
       torch.save(..., 'model_best.pth')
       torch.save(..., 'model_latest.pth')

    5. ❌ 不要忘记保存优化器状态
       # 继续训练时需要:
       checkpoint = {
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),  # 重要!
       }
    """)


def comparison_table():
    """格式对比表"""
    print("\n" + "=" * 70)
    print("保存格式对比")
    print("=" * 70)

    print("""
    ┌─────────────────┬──────────┬────────┬────────┬──────────────┐
    │     格式        │  推荐度  │ 文件大小│ 灵活性 │  使用场景     │
    ├─────────────────┼──────────┼────────┼────────┼──────────────┤
    │ state_dict      │  ⭐⭐⭐⭐⭐│  小     │  高    │ 生产部署      │
    │ 完整模型         │  ⭐⭐     │  大     │  低    │ 快速实验      │
    │ checkpoint      │  ⭐⭐⭐⭐⭐│  中     │  高    │ 训练恢复      │
    │ ONNX            │  ⭐⭐⭐⭐  │  中     │  中    │ 跨框架部署    │
    │ TorchScript     │  ⭐⭐⭐⭐  │  中     │  中    │ C++ 部署      │
    └─────────────────┴──────────┴────────┴────────┴──────────────┘

    文件扩展名惯例:
    .pth / .pt     - PyTorch 模型
    .h5            - Keras/TensorFlow HDF5
    .pb            - TensorFlow SavedModel
    .onnx          - ONNX 格式
    .pkl / .pickle - Pickle 格式 (不推荐)
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🎓 深度学习模型保存与读取完全指南")
    print("=" * 70)

    # 1. PyTorch 保存方法说明
    explain_pytorch_save_methods()

    # 2. state_dict 演示
    demo_pytorch_save_state_dict()

    # 3. 完整模型演示
    demo_pytorch_save_full_model()

    # 4. Checkpoint 演示
    demo_pytorch_checkpoint()

    # 5. 预训练模型演示
    demo_save_pretrained_model()

    # 6. ONNX 格式
    explain_onnx_format()

    # 7. TorchScript
    explain_torchscript()

    # 8. 最佳实践
    best_practices()

    # 9. 对比表
    comparison_table()

    print("\n" + "=" * 70)
    print("✅ 教程完成!")
    print("=" * 70)
    print("\n💡 快速参考:")
    print("   训练时保存:    torch.save(model.state_dict(), 'model.pth')")
    print("   加载模型:      model.load_state_dict(torch.load('model.pth'))")
    print("   评估模式:      model.eval()")
    print("   保存 checkpoint: torch.save({'model': ..., 'optimizer': ...}, 'ckpt.pth')")
    print("=" * 70)

    print("\n📁 已生成的模型文件:")
    model_dir = 'artifacts/models'
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith('.pth'):
                path = os.path.join(model_dir, file)
                size = os.path.getsize(path) / 1024
                if size > 1024:
                    print(f"   {file:<35} {size/1024:>8.2f} MB")
                else:
                    print(f"   {file:<35} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
