"""
torchvision.models 使用完全指南

这个模块演示了如何使用 torchvision.models 中的预训练模型：
1. 加载预训练模型的多种方式
2. 模型结构查看和修改
3. 特征提取 vs 微调
4. 迁移学习的实际应用
5. 不同模型架构的对比

作者: Seeback
日期: 2025-10-23
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def demo_available_models():
    """演示 torchvision 中可用的模型"""
    print("=" * 70)
    print("torchvision.models 中可用的主流模型")
    print("=" * 70)

    model_categories = {
        "经典CNN架构": [
            "alexnet", "vgg11", "vgg16", "vgg19",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ],
        "轻量级模型": [
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
            "shufflenet_v2_x0_5", "shufflenet_v2_x1_0"
        ],
        "高性能模型": [
            "efficientnet_b0", "efficientnet_b4", "efficientnet_b7",
            "resnext50_32x4d", "wide_resnet50_2"
        ],
        "视觉Transformer": [
            "vit_b_16", "vit_b_32", "vit_l_16",
            "swin_t", "swin_s", "swin_b"
        ],
        "目标检测": [
            "fasterrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn",
            "retinanet_resnet50_fpn"
        ],
        "语义分割": [
            "fcn_resnet50", "deeplabv3_resnet50",
            "lraspp_mobilenet_v3_large"
        ]
    }

    for category, model_list in model_categories.items():
        print(f"\n📦 {category}:")
        for model_name in model_list:
            print(f"   - {model_name}")

    print("\n" + "=" * 70)
    print("💡 提示: 所有模型都可以通过 models.模型名(weights='DEFAULT') 加载")
    print("=" * 70)


def demo_load_pretrained_models():
    """演示加载预训练模型的不同方式"""
    print("\n" + "=" * 70)
    print("加载预训练模型的方式")
    print("=" * 70)

    # 方式1: 旧版API (PyTorch < 1.13)
    print("\n1️⃣ 旧版API (不推荐,但仍可用):")
    print("   model = models.resnet18(pretrained=True)")

    # 方式2: 新版API (PyTorch >= 1.13) - 推荐
    print("\n2️⃣ 新版API (推荐):")
    print("   from torchvision.models import ResNet18_Weights")
    print("   model = models.resnet18(weights=ResNet18_Weights.DEFAULT)")
    print("   或者简化写法:")
    print("   model = models.resnet18(weights='DEFAULT')")

    # 方式3: 不加载预训练权重
    print("\n3️⃣ 不使用预训练权重:")
    print("   model = models.resnet18(weights=None)")

    # 实际加载一个模型
    print("\n📥 实际加载 ResNet-18 模型...")
    model = models.resnet18(weights='DEFAULT')

    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数统计:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    return model


def demo_model_structure(model):
    """演示如何查看和理解模型结构"""
    print("\n" + "=" * 70)
    print("模型结构分析")
    print("=" * 70)

    print("\n1️⃣ ResNet-18 整体结构:")
    print("-" * 70)
    print(model)

    print("\n" + "=" * 70)
    print("2️⃣ 关键层解析:")
    print("-" * 70)
    print(f"   输入层: conv1 - {model.conv1}")
    print(f"   BatchNorm: bn1 - {model.bn1}")
    print(f"   激活函数: relu - {model.relu}")
    print(f"   池化层: maxpool - {model.maxpool}")
    print(f"   残差块1: layer1 - {len(model.layer1)} 个BasicBlock")
    print(f"   残差块2: layer2 - {len(model.layer2)} 个BasicBlock")
    print(f"   残差块3: layer3 - {len(model.layer3)} 个BasicBlock")
    print(f"   残差块4: layer4 - {len(model.layer4)} 个BasicBlock")
    print(f"   全局平均池化: avgpool - {model.avgpool}")
    print(f"   分类器: fc - {model.fc}")

    # 查看输入输出形状
    print("\n" + "=" * 70)
    print("3️⃣ 数据流动演示 (输入: 224x224 RGB图像):")
    print("-" * 70)

    # 创建一个测试输入
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        # 逐层查看形状变化
        x = dummy_input
        print(f"   输入形状: {x.shape}")

        x = model.conv1(x) # conv1: 卷积层, 输入: 3通道, 输出: 64通道, 核大小: 7x7, 步长: 2, 填充: 3
        print(f"   conv1 后: {x.shape}")

        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        print(f"   maxpool 后: {x.shape}")

        x = model.layer1(x)
        print(f"   layer1 后: {x.shape}")

        x = model.layer2(x)
        print(f"   layer2 后: {x.shape}")

        x = model.layer3(x)
        print(f"   layer3 后: {x.shape}")

        x = model.layer4(x)
        print(f"   layer4 后: {x.shape}")

        x = model.avgpool(x)
        print(f"   avgpool 后: {x.shape}")

        x = torch.flatten(x, 1)
        print(f"   flatten 后: {x.shape}")

        x = model.fc(x)
        print(f"   fc 后(输出): {x.shape}")


def demo_modify_model_for_transfer_learning():
    """演示如何修改模型用于迁移学习"""
    print("\n" + "=" * 70)
    print("迁移学习 - 修改模型用于自定义任务")
    print("=" * 70)

    # 场景: 用 ResNet-18 做 10 分类任务 (如CIFAR-10)
    print("\n🎯 任务: 将 ImageNet(1000类) 模型改为 CIFAR-10(10类) 模型")

    # 1. 加载预训练模型
    model = models.resnet18(weights='DEFAULT')
    print("\n1️⃣ 原始模型的最后一层:")
    print(f"   {model.fc}")
    print(f"   输出维度: 1000 (ImageNet类别数)")

    # 2. 方法一: 直接替换最后一层
    print("\n2️⃣ 方法一: 只替换最后一层 (特征提取)")
    num_classes = 10

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"   新的fc层: {model.fc}")
    print(f"   输出维度: {num_classes}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # 3. 方法二: 微调整个模型
    print("\n3️⃣ 方法二: 微调整个模型 (Fine-tuning)")
    model2 = models.resnet18(weights='DEFAULT')
    model2.fc = nn.Linear(model2.fc.in_features, num_classes)

    # 不冻结任何层,但可以对不同层使用不同学习率
    trainable2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    total2 = sum(p.numel() for p in model2.parameters())
    print(f"   可训练参数: {trainable2:,} / {total2:,} ({trainable2/total2*100:.2f}%)")

    # 4. 方法三: 添加自定义分类器
    print("\n4️⃣ 方法三: 添加自定义分类器层")
    model3 = models.resnet18(weights='DEFAULT')

    # 冻结特征提取层
    for param in model3.parameters():
        param.requires_grad = False

    # 添加自定义分类器
    model3.fc = nn.Sequential(
        nn.Linear(model3.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    print(f"   新的分类器:")
    print(f"   {model3.fc}")

    trainable3 = sum(p.numel() for p in model3.parameters() if p.requires_grad)
    total3 = sum(p.numel() for p in model3.parameters())
    print(f"   可训练参数: {trainable3:,} / {total3:,} ({trainable3/total3*100:.2f}%)")

    return model, model2, model3


def demo_feature_extraction():
    """演示如何使用预训练模型提取特征"""
    print("\n" + "=" * 70)
    print("特征提取 - 使用预训练模型作为特征提取器")
    print("=" * 70)

    # 加载模型
    model = models.resnet18(weights='DEFAULT')

    # 方法1: 移除最后的分类层
    print("\n1️⃣ 方法一: 移除最后的全连接层")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    print("   特征提取器结构:")
    print(f"   {feature_extractor}")

    # 测试
    dummy_input = torch.randn(4, 3, 224, 224) # 随机生成的 4 张 224x224  RGB 图像
    with torch.no_grad():
        features = feature_extractor(dummy_input)

    print(f"\n   输入形状: {dummy_input.shape}")
    print(f"   输出特征形状: {features.shape}")
    print(f"   特征向量维度: {features.shape[1]}")

    # 方法2: 使用 hook 提取中间层特征
    print("\n2️⃣ 方法二: 使用 Hook 提取中间层特征")

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 注册 hook
    model.layer4.register_forward_hook(get_activation('layer4')) # layer4: 最后一个卷积层, 输出: 512通道
    model.avgpool.register_forward_hook(get_activation('avgpool')) # avgpool: 全局平均池化层, 输出: 512通道

    # 前向传播
    with torch.no_grad():
        _ = model(dummy_input)# 前向传播, 提取特征

    print(f"   layer4 输出形状: {activation['layer4'].shape}")
    print(f"   avgpool 输出形状: {activation['avgpool'].shape}")

    return feature_extractor


def demo_different_models_comparison():
    """演示不同模型的对比"""
    print("\n" + "=" * 70)
    print("不同模型架构对比")
    print("=" * 70)

    models_to_compare = {
        "ResNet-18": models.resnet18(weights='DEFAULT'),
        "MobileNetV2": models.mobilenet_v2(weights='DEFAULT'),
        "EfficientNet-B0": models.efficientnet_b0(weights='DEFAULT'),
    }

    print("\n" + "-" * 70)
    print(f"{'模型':<20} {'参数量':<15} {'模型大小':<15} {'输入尺寸':<15}")
    print("-" * 70)

    dummy_input = torch.randn(1, 3, 224, 224)

    for name, model in models_to_compare.items():
        model.eval()

        # 计算参数量
        params = sum(p.numel() for p in model.parameters())
        size_mb = params * 4 / 1024 / 1024

        # 测试推理速度
        import time
        with torch.no_grad():
            start = time.time()
            _ = model(dummy_input)
            inference_time = (time.time() - start) * 1000

        print(f"{name:<20} {params:>12,}   {size_mb:>10.2f} MB   224x224")

    print("-" * 70)


def demo_training_setup():
    """演示如何设置训练循环"""
    print("\n" + "=" * 70)
    print("训练配置示例")
    print("=" * 70)

    # 加载模型并修改
    model = models.resnet18(weights='DEFAULT')
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    print("\n1️⃣ 特征提取模式 (只训练最后一层):")
    print("-" * 70)

    # 冻结所有层除了最后一层
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    # 优化器只优化可训练参数
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    print("   optimizer = torch.optim.Adam(")
    print("       filter(lambda p: p.requires_grad, model.parameters()),")
    print("       lr=0.001")
    print("   )")
    print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n2️⃣ 微调模式 (所有层都训练,使用不同学习率):")
    print("-" * 70)

    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    # 为不同层设置不同学习率
    optimizer2 = torch.optim.SGD([
        {'params': model.conv1.parameters(), 'lr': 0.0001},
        {'params': model.layer1.parameters(), 'lr': 0.0001},
        {'params': model.layer2.parameters(), 'lr': 0.0005},
        {'params': model.layer3.parameters(), 'lr': 0.0005},
        {'params': model.layer4.parameters(), 'lr': 0.001},
        {'params': model.fc.parameters(), 'lr': 0.01}
    ], momentum=0.9)

    print("   optimizer = torch.optim.SGD([")
    print("       {'params': model.conv1.parameters(), 'lr': 0.0001},")
    print("       {'params': model.layer1.parameters(), 'lr': 0.0001},")
    print("       ...")
    print("       {'params': model.fc.parameters(), 'lr': 0.01}")
    print("   ], momentum=0.9)")
    print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n3️⃣ 完整训练循环示例:")
    print("-" * 70)
    print("""
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个epoch后验证
        model.eval()
        with torch.no_grad():
            val_loss = evaluate(model, val_loader)
        model.train()
    """)


def demo_practical_usage():
    """演示实际使用场景"""
    print("\n" + "=" * 70)
    print("实际应用场景示例")
    print("=" * 70)

    print("\n📱 场景1: 移动端部署 - 使用轻量级模型")
    print("-" * 70)
    print("   model = models.mobilenet_v2(weights='DEFAULT')")
    print("   # MobileNetV2: 3.5M参数, 适合移动端")

    print("\n🎯 场景2: 高精度任务 - 使用大型模型")
    print("-" * 70)
    print("   model = models.resnet152(weights='DEFAULT')")
    print("   # ResNet-152: 60M参数, 更高精度")

    print("\n⚡ 场景3: 实时推理 - 使用高效模型")
    print("-" * 70)
    print("   model = models.efficientnet_b0(weights='DEFAULT')")
    print("   # EfficientNet-B0: 5.3M参数, 效率与精度平衡")

    print("\n🔬 场景4: 迁移学习 - 小数据集")
    print("-" * 70)
    print("""
    # 1. 加载预训练模型
    model = models.resnet18(weights='DEFAULT')

    # 2. 冻结特征提取层
    for param in model.parameters():
        param.requires_grad = False

    # 3. 替换分类器
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 4. 只训练分类器 (更快, 避免过拟合)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    """)

    print("\n🚀 场景5: 特征提取 - 用于下游任务")
    print("-" * 70)
    print("""
    # 1. 移除分类层,只保留特征提取器
    model = models.resnet50(weights='DEFAULT')
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    # 2. 提取特征
    features = feature_extractor(images)  # shape: [batch, 2048, 1, 1]
    features = features.flatten(1)        # shape: [batch, 2048]

    # 3. 用提取的特征训练简单分类器 (如 SVM, KNN)
    """)


def main():
    """主函数 - 运行所有演示"""
    print("\n" + "=" * 70)
    print("🎓 torchvision.models 完全使用指南")
    print("=" * 70)

    # 1. 显示可用模型
    demo_available_models()

    # 2. 加载预训练模型
    model = demo_load_pretrained_models()

    # 3. 查看模型结构
    demo_model_structure(model)

    # 4. 迁移学习 - 修改模型
    model1, model2, model3 = demo_modify_model_for_transfer_learning()

    # 5. 特征提取
    feature_extractor = demo_feature_extraction()

    # 6. 模型对比
    demo_different_models_comparison()

    # 7. 训练配置
    demo_training_setup()

    # 8. 实际应用场景
    demo_practical_usage()

    print("\n" + "=" * 70)
    print("✅ 所有演示完成!")
    print("=" * 70)
    print("\n💡 关键要点:")
    print("   1. 使用 weights='DEFAULT' 加载最新的预训练权重")
    print("   2. 迁移学习时冻结特征提取层可以加速训练")
    print("   3. 小数据集适合特征提取,大数据集适合微调")
    print("   4. 不同层使用不同学习率可以提高微调效果")
    print("   5. 选择模型时要权衡精度、速度和模型大小")
    print("=" * 70)


if __name__ == "__main__":
    main()
