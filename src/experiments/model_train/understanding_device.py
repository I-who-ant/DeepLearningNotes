"""
深入理解 torch.device() 和 GPU 选择机制

详细解释:
1. torch.device() 是什么
2. 为什么定义device就能自动使用GPU
3. 如何选择特定的GPU
4. 多GPU环境下的设备管理
5. 实际演示和最佳实践
"""

import torch
import os


# ============================================================
# 1. torch.device() 基础概念
# ============================================================
def explain_device_concept():
    """解释device的基本概念"""
    print("=" * 70)
    print("📚 torch.device() 基础概念")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                 torch.device() 是什么?                          │
└─────────────────────────────────────────────────────────────────┘

torch.device 是 PyTorch 中表示"计算设备"的抽象对象。

可以把它理解为一个"地址标签",告诉 PyTorch:
  "请在这个设备上进行计算"

┌─────────────────────────────────────────────────────────────────┐
│                 创建 device 对象的方式                          │
└─────────────────────────────────────────────────────────────────┘

方法1: 使用字符串
  device = torch.device('cpu')           # CPU
  device = torch.device('cuda')          # 默认GPU (cuda:0)
  device = torch.device('cuda:0')        # 第0个GPU
  device = torch.device('cuda:1')        # 第1个GPU

方法2: 动态选择 (推荐!)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # 如果有GPU就用GPU,没有就用CPU

方法3: 使用环境变量 (高级)
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 只使用GPU 0和1
  device = torch.device('cuda')

┌─────────────────────────────────────────────────────────────────┐
│                 device 对象包含什么信息?                        │
└─────────────────────────────────────────────────────────────────┘
""")

    # 演示device对象的属性
    print("实际演示:")
    print("-" * 70)

    # CPU设备
    cpu_device = torch.device('cpu')
    print(f"\nCPU设备:")
    print(f"  device对象: {cpu_device}")
    print(f"  设备类型: {cpu_device.type}")
    print(f"  设备索引: {cpu_device.index}")

    # GPU设备 (如果可用)
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda:0')
        print(f"\nGPU设备:")
        print(f"  device对象: {gpu_device}")
        print(f"  设备类型: {gpu_device.type}")
        print(f"  设备索引: {gpu_device.index}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️ 当前环境没有GPU")

    print("\n💡 关键理解:")
    print("  device 只是一个'标签',告诉PyTorch在哪里计算")
    print("  就像快递地址一样!")


# ============================================================
# 2. 为什么定义device就能自动使用GPU?
# ============================================================
def explain_how_device_works():
    """解释device是如何工作的"""
    print("\n" + "=" * 70)
    print("🔍 为什么定义device就能自动使用GPU?")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                 工作原理详解                                    │
└─────────────────────────────────────────────────────────────────┘

关键: device 不是"挑选显卡",而是"告诉数据和模型去哪里"

类比: 快递系统
  1️⃣ device = 收件地址
  2️⃣ model.to(device) = 把模型(包裹)送到这个地址
  3️⃣ data.to(device) = 把数据(包裹)送到这个地址

┌─────────────────────────────────────────────────────────────────┐
│                 完整流程                                        │
└─────────────────────────────────────────────────────────────────┘

步骤1: 创建device对象
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  PyTorch 做了什么:
    ✅ 检查系统是否有GPU
    ✅ 如果有: device = cuda:0 (默认第0个GPU)
    ✅ 如果没有: device = cpu

步骤2: 将模型移动到设备
  model = model.to(device)

  PyTorch 做了什么:
    ✅ 将模型的所有参数(权重、偏置)复制到指定设备的内存
    ✅ CPU → CPU内存
    ✅ CUDA → GPU显存
    ✅ 返回一个在新设备上的模型引用

步骤3: 将数据移动到设备
  images = images.to(device)

  PyTorch 做了什么:
    ✅ 将tensor从当前位置复制到指定设备
    ✅ 例如: CPU内存 → GPU显存

步骤4: 自动在正确设备上计算
  outputs = model(images)

  PyTorch 做了什么:
    ✅ 检测到 model 在 GPU, images 也在 GPU
    ✅ 自动调用 GPU 上的计算核心
    ✅ 结果 outputs 也在 GPU 上

┌─────────────────────────────────────────────────────────────────┐
│                 内存分布示意图                                  │
└─────────────────────────────────────────────────────────────────┘

CPU设备:
  ┌──────────────────────────────┐
  │       CPU 内存 (RAM)         │
  │  ┌────────┐  ┌────────┐     │
  │  │ 模型   │  │ 数据   │     │
  │  └────────┘  └────────┘     │
  └──────────────────────────────┘

GPU设备 (cuda:0):
  ┌──────────────────────────────┐
  │      GPU 显存 (VRAM)         │
  │  ┌────────┐  ┌────────┐     │
  │  │ 模型   │  │ 数据   │     │
  │  └────────┘  └────────┘     │
  └──────────────────────────────┘

关键:
  - model.to(device) 把模型复制到指定设备的内存
  - data.to(device) 把数据复制到指定设备的内存
  - 计算自动在数据所在的设备上进行
""")


# ============================================================
# 3. GPU选择机制
# ============================================================
def explain_gpu_selection():
    """解释如何选择特定的GPU"""
    print("\n" + "=" * 70)
    print("🎯 GPU选择机制详解")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                 单GPU环境 (最常见)                              │
└─────────────────────────────────────────────────────────────────┘

如果只有一个GPU:
  device = torch.device('cuda')          # 自动使用GPU 0
  device = torch.device('cuda:0')        # 明确指定GPU 0
  # 效果相同!

PyTorch 会自动使用系统中唯一可用的GPU。

┌─────────────────────────────────────────────────────────────────┐
│                 多GPU环境                                       │
└─────────────────────────────────────────────────────────────────┘

假设你有4个GPU: [GPU 0, GPU 1, GPU 2, GPU 3]

方法1: 直接指定GPU编号
  device = torch.device('cuda:0')        # 使用GPU 0
  device = torch.device('cuda:1')        # 使用GPU 1
  device = torch.device('cuda:2')        # 使用GPU 2
  device = torch.device('cuda:3')        # 使用GPU 3

方法2: 使用环境变量 (推荐!)
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'      # 只使用GPU 0
  os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'    # 只使用GPU 1和3
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'      # 只使用GPU 2

  device = torch.device('cuda')  # 会使用CUDA_VISIBLE_DEVICES指定的第一个

  ⚠️ 注意: 要在 import torch 之前设置!

方法3: 动态选择空闲GPU
  def get_free_gpu():
      '''选择显存使用最少的GPU'''
      import subprocess
      result = subprocess.check_output(
          ['nvidia-smi', '--query-gpu=memory.free',
           '--format=csv,nounits,noheader'],
          encoding='utf-8'
      )
      gpu_memory = [int(x) for x in result.strip().split('\\n')]
      return gpu_memory.index(max(gpu_memory))

  free_gpu = get_free_gpu()
  device = torch.device(f'cuda:{free_gpu}')

┌─────────────────────────────────────────────────────────────────┐
│                 默认GPU vs 显式指定                             │
└─────────────────────────────────────────────────────────────────┘

情况A: 使用 'cuda' (默认)
  device = torch.device('cuda')
  → 使用 GPU 0 (第一个GPU)

情况B: 显式指定
  device = torch.device('cuda:1')
  → 使用 GPU 1 (第二个GPU)

示例代码:
  # 假设有2个GPU
  device0 = torch.device('cuda:0')  # GPU 0
  device1 = torch.device('cuda:1')  # GPU 1

  # 可以把不同模型放在不同GPU上
  model1 = Model1().to(device0)  # 模型1在GPU 0
  model2 = Model2().to(device1)  # 模型2在GPU 1
""")


# ============================================================
# 4. 实际演示
# ============================================================
def demo_device_usage():
    """实际演示device的使用"""
    print("\n" + "=" * 70)
    print("🔬 实际演示")
    print("=" * 70)

    print("\n示例1: 基础使用")
    print("-" * 70)

    # 检查GPU
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 创建device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n当前使用设备: {device}")

    # 创建tensor并移动到设备
    print("\n示例2: 移动tensor到设备")
    print("-" * 70)

    # 在CPU上创建
    x = torch.randn(3, 3)
    print(f"创建tensor x:")
    print(f"  数据: {x}")
    print(f"  设备: {x.device}")
    print(f"  内存位置: {'CPU内存' if x.device.type == 'cpu' else 'GPU显存'}")

    # 移动到目标设备
    x = x.to(device)
    print(f"\n移动到 {device} 后:")
    print(f"  设备: {x.device}")
    print(f"  内存位置: {'CPU内存' if x.device.type == 'cpu' else 'GPU显存'}")

    # 创建简单模型
    print("\n示例3: 移动模型到设备")
    print("-" * 70)

    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    print(f"创建模型:")
    print(f"  模型类型: {type(model).__name__}")

    # 检查参数位置
    first_param = next(model.parameters())
    print(f"  参数设备 (移动前): {first_param.device}")

    # 移动模型
    model = model.to(device)  # 移动模型到目标设备
    first_param = next(model.parameters())
    print(f"  参数设备 (移动后): {first_param.device}")

    print("\n示例4: 计算会自动在正确设备上进行")
    print("-" * 70)

    # 创建输入数据
    input_data = torch.randn(2, 10).to(device)
    print(f"输入数据设备: {input_data.device}") # 输入数据在GPU上
    print(f"模型参数设备: {next(model.parameters()).device}") # 模型参数也在GPU上

    # 前向传播
    output = model(input_data) # 模型在GPU上,输入数据也在GPU上,输出数据也在GPU上
    print(f"输出数据设备: {output.device}")
    print(f"\n✅ 计算自动在 {output.device} 上完成!")


# ============================================================
# 5. 常见错误和解决方法
# ============================================================
def common_mistakes():
    """常见错误"""
    print("\n" + "=" * 70)
    print("⚠️ 常见错误和解决方法")
    print("=" * 70)

    print("""
错误1: 模型和数据不在同一设备
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 错误代码:
  model = model.to('cuda')
  data = torch.randn(10, 10).to('cuda')  # 在CPU上创建, 然后移到GPU
  output = model(data)  # RuntimeError! # 模型在GPU上,数据在CPU上,会报错

错误信息:
  RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cuda:0 and cpu!

✅ 正确代码:
  device = torch.device('cuda')
  model = model.to(device)
  data = torch.randn(10, 10).to(device)  # 也移到GPU
  output = model(data)  # ✅ 正确


错误2: 直接使用cuda()不检查GPU是否可用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 错误代码:
  model = model.cuda()  # 如果没有GPU会报错

错误信息:
  RuntimeError: CUDA error: no kernel image is available

✅ 正确代码:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)  # 自动适配


错误3: 在多GPU环境未指定使用哪个GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 问题:
  # 假设有4个GPU,但都被占用
  device = torch.device('cuda')  # 默认使用GPU 0,可能显存不足

✅ 解决方案1: 显式指定GPU
  device = torch.device('cuda:2')  # 使用GPU 2

✅ 解决方案2: 使用环境变量
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 只使用GPU 3
  device = torch.device('cuda')


错误4: 忘记将损失值移回CPU (无法转numpy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 错误代码:
  loss = criterion(output, labels)  # loss在GPU上
  loss_value = loss.numpy()  # TypeError!

错误信息:
  TypeError: can't convert cuda:0 device type tensor to numpy

✅ 正确代码:
  loss = criterion(output, labels)
  loss_value = loss.cpu().numpy()  # 先移到CPU


错误5: 多次移动tensor (浪费时间)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 低效代码:
  for epoch in range(100):
      model = model.to(device)  # 每个epoch都移动,浪费!
      for data, labels in dataloader:
          ...

✅ 高效代码:
  model = model.to(device)  # 只移动一次
  for epoch in range(100):
      for data, labels in dataloader:
          data = data.to(device)  # 每个batch移动
          ...
""")


# ============================================================
# 6. 最佳实践
# ============================================================
def best_practices():
    """最佳实践"""
    print("\n" + "=" * 70)
    print("💡 最佳实践")
    print("=" * 70)

    print("""
实践1: 始终使用device对象
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 推荐:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  data = data.to(device)

❌ 不推荐:
  if torch.cuda.is_available():
      model = model.cuda()
      data = data.cuda()
  # 代码重复,不优雅


实践2: 模型移动一次,数据每个batch移动
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 标准模式:
  # 训练开始前移动模型
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)

  # 训练循环
  for epoch in range(epochs):
      for images, labels in train_loader:
          # 每个batch移动数据
          images = images.to(device)
          labels = labels.to(device)

          # 训练代码
          outputs = model(images)
          loss = criterion(outputs, labels)
          ...


实践3: 使用非阻塞传输加速 (高级)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 更快的数据传输:
  images = images.to(device, non_blocking=True)
  labels = labels.to(device, non_blocking=True)

  # non_blocking=True 允许CPU和GPU异步执行
  # 数据传输和GPU计算可以重叠


实践4: 多GPU环境使用环境变量
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 在脚本开头设置:
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'  # 只使用GPU 0和2
  import torch

  device = torch.device('cuda')  # 会使用GPU 0


实践5: 打印设备信息 (调试)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 训练开始时打印:
  def print_device_info():
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print(f"使用设备: {device}")

      if torch.cuda.is_available():
          print(f"GPU名称: {torch.cuda.get_device_name(0)}")
          print(f"GPU数量: {torch.cuda.device_count()}")

  print_device_info()  # 在训练前调用


实践6: 保存/加载模型时注意设备
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 保存:
  torch.save(model.state_dict(), 'model.pth')  # 自动保存到CPU

✅ 加载:
  # 方法1: 直接加载到指定设备
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.load_state_dict(torch.load('model.pth', map_location=device))

  # 方法2: 先加载再移动
  model.load_state_dict(torch.load('model.pth'))
  model = model.to(device)
""")


# ============================================================
# 7. 主函数
# ============================================================
def main():
    """主函数"""
    print("=" * 70)
    print("🎯 深入理解 torch.device() 和 GPU 选择机制")
    print("=" * 70)

    # 1. 基础概念
    explain_device_concept()

    # 2. 工作原理
    explain_how_device_works()

    # 3. GPU选择
    explain_gpu_selection()

    # 4. 实际演示
    demo_device_usage()

    # 5. 常见错误
    common_mistakes()

    # 6. 最佳实践
    best_practices()

    print("\n" + "=" * 70)
    print("✅ 所有解释完成!")
    print("=" * 70)

    print("""
🎯 核心总结:

1. torch.device() 是什么?
   → 一个"地址标签",告诉PyTorch在哪个设备上计算

2. 为什么定义device就能用GPU?
   → device 本身不"挑选"GPU
   → model.to(device) 把模型移到指定设备
   → data.to(device) 把数据移到指定设备
   → PyTorch自动在数据所在设备上计算

3. 单GPU vs 多GPU
   → 单GPU: device = torch.device('cuda') 自动使用GPU 0
   → 多GPU: device = torch.device('cuda:1') 指定使用GPU 1

4. 最佳实践
   → 始终用 device 对象 (不要用 .cuda())
   → 模型移动一次,数据每batch移动
   → 使用 map_location 正确加载模型

💡 记住:
   device = 地址标签
   model.to(device) = 把模型送到这个地址
   data.to(device) = 把数据送到这个地址
   → PyTorch 自动在正确地址计算!
""")


if __name__ == '__main__':
    main()
