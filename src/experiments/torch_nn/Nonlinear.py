import torch



# 1. 使用ReLU激活函数 : 会将reshaped中小于0的元素设为0


# 对reshape后的数据进行ReLU激活
input_data = [[1, 2], [-1, 3]]
input_tensor = torch.tensor(input_data, dtype=torch.float32)
reshaped = torch.reshape(input_tensor, (-1, 1, 2, 2))

# 应用ReLU激活函数 : 会将reshaped中小于0的元素设为0
relu_output = torch.relu(reshaped, inplace=False  )# 不直接在reshaped上操作，创建新的tensor
torch.relu(reshaped, inplace=True  )# 直接在reshaped上操作，不创建新的tensor

print("ReLU激活后的输出:", relu_output)



# 2. 使用sigmoid激活函数 : 会将reshaped中的元素映射到(0, 1)范围

sigmoid_output = torch.sigmoid(reshaped)
print("Sigmoid激活后的输出:", sigmoid_output)


# • ReLU（Rectified Linear Unit，线性修正单元）是现代神经网络里使用最广的非线性激活函数之一。核心定义非常简单：
#   ReLU(x) = max(0, x)——输入为正就保留原值，输入为负就置零。
#
#   怎么理解这个“非线性”？
#
#   - 打破线性限制：如果网络所有层都是线性变换，堆再多层也只等同于一层线性层，无法拟合复杂的数据分布。ReLU 把负
#     值截断，产生不同区间的分段行为，使网络具备逼近任意非线性函数的能力。
#   - 稀疏响应：一旦输入为负，输出就变成 0。大量神经元会在不同输入下“沉默”，形成稀疏表示。稀疏特征更容易解析，也
#     能减轻权值更新时的相互干扰。
#   - 梯度传递顺畅：在正区间，导数恒为 1，不会像 Sigmoid/Tanh 那样出现梯度饱和，故在深层网络中更利于反向传播。
#     只有输入落在负区间时梯度为 0（著名的“死 ReLU”问题，通常可用合理初始化、较小学习率或替代激活如 LeakyReLU
#     规避）。
#   - 计算高效：实现只需一次比较操作，GPU 上可高度并行，几乎没有额外开销。
#
#   可以把 ReLU 类比成“电流开关”：信号高于零就导通，信号小于零就断开。这种“硬拐点”使得每层神经元对输入空间分块响
#   应，组合起来就构筑了复杂的决策边界，从而让深度网络具备强大的表达力。