import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)

targets = torch.tensor([1, 1, 2], dtype=torch.float32)

inputs = torch.reshape(inputs, (1,1,1,3)) #作用是将inputs转换为(1,1,1,3)的张量


targets = torch.reshape(targets, (1,1,1,3)) #作用是将targets转换为(1,1,1,3)的张量

criterion = nn.MSELoss() #作用是创建一个均方误差损失函数的实例
loss = criterion(inputs, targets) #作用是计算inputs和targets之间的均方误差损失
print(loss)


loss = L1Loss()(inputs, targets) #作用是计算inputs和targets之间的平均绝对误差损失
print(loss)


#参数意义 :
# size_average : 是否对损失进行平均，默认值为 None
# reduce : 是否对损失进行.reduce()操作，默认值为 None
# reduction : 损失的 reduction 方式，可选值为 'none'、'mean'、'sum'，默认值为 'mean'
L1Loss(size_average=None, reduce=None, reduction='mean')

#sum : 对损失进行求和操作
#mean : 对损失进行求平均操作




#---------------------------------------------
# 多分类交叉熵损失 (Cross Entropy Loss)

# L = -∑(i=1 to C) y_i * log(p_i)

    # 其中:
    # C = 类别总数
    # y_i = 真实标签的one-hot编码 (只有正确类别为1,其他为0)
    # p_i = 模型预测的概率分布 (softmax后的输出)


    # 交叉熵：logits 形状应为 (N, C)，targets 形状为 (N,) 或 (N, 1)
criterion_ce = nn.CrossEntropyLoss()#作用是创建一个交叉熵损失函数的实例,交叉熵原理:
    # 1. 对 logits 进行 softmax 归一化，将其转换为概率分布
    # 2. 计算每个样本的交叉熵损失
    # 3. 对所有样本的损失取平均（或求和），得到最终损失
#简单来说，这个式子是前半部分是识别是否准确贡献的
#后半部分是三个类别概率是否平均贡献的，想要减少损失函数，既要让识别到的各个类别概率相差大，又要识别的概率在标签设定的那种类别上大


#例子 :

# 假设是3分类任务:猫(0)、狗(1)、鸟(2)
num_classes = 3

# 模型的原始输出(logits,未经softmax)
logits = torch.tensor([[2.0, 1.0, 0.1]])  # 形状: (1, 3) ,表示模型对这只样本的原始输出,每个元素对应一个类别的未归一化分数
# 解读: 模型认为是猫的原始分数为2.0, 是狗的原始分数为1.0, 是鸟的原始分数为0.1

# 真实标签: 这是一只猫(类别0)
target = torch.tensor([0])

# ===== 手动计算交叉熵 =====

# 步骤1: 对logits应用softmax,得到概率分布
probabilities = torch.nn.functional.softmax(logits, dim=1) #作用是对logits进行softmax归一化,将其转换为概率分布, dim=1表示对每个样本的3个类别进行归一化
print(f"预测概率: {probabilities}")
# 输出: tensor([[0.6590, 0.2424, 0.0986]])
# 解读: 模型认为是猫的概率65.9%, 是狗24.24%, 是鸟9.86%

# 步骤2: 真实标签的one-hot编码
#作用是将target转换为one-hot编码, num_classes=num_classes表示有3个类别
y_true = torch.nn.functional.one_hot(target, num_classes=num_classes).float()
# 解读: 真实标签是猫(类别0), 所以one-hot编码为[1, 0, 0]



print(f"真实标签(one-hot): {y_true}") # 解读: 真实标签是猫(类别0), 所以one-hot编码为[1, 0, 0]
# 输出: tensor([[1., 0., 0.]])  # 表示真实是猫(类别0),也就是 (类别0是正确答案)


# 步骤3: 计算交叉熵 : L = -∑(i=1 to C) y_i * log(p_i)
# L = -(y0*log(p0) + y1*log(p1) + y2*log(p2))
#   = -(1*log(0.659) + 0*log(0.242) + 0*log(0.099))
#   = -log(0.659)
#   = 0.417

#(log(0.659)是对模型认为是猫的概率65.9%取自然对数,
# 因为log函数是单调递增的,所以log(0.659)是一个负数,负号是为了使损失函数单调递减,即损失函数越小,模型的预测越准确)

#为什么后面要加上其他两个类别的概率的对数 :
# 因为交叉熵损失函数是对每个样本的每个类别进行计算的,所以要对所有类别进行求和,才能得到最终的损失


manual_loss = -torch.sum(y_true * torch.log(probabilities)) #作用是计算手动计算的交叉熵损失
#为什么要加上 - :
# 因为交叉熵损失函数的定义是对每个样本的每个类别进行计算的,所以要对所有类别进行求和,才能得到最终的损失
# 而手动计算的损失是对每个样本的每个类别进行计算的,所以要对所有类别进行求和,才能得到最终的损失
# 所以要加上 - 号,使手动计算的损失与PyTorch计算的损失保持一致
#但问题是 ,如果其他的概率不为0 , 通过 -log(0.242) 和 -log(0.099) 也会得到一个负数,
# 所以手动计算的损失会比PyTorch计算的损失小,这是因为PyTorch计算的损失函数在计算时,会对所有类别进行归一化,
# 而手动计算的损失函数在计算时,不会对所有类别进行归一化,所以手动计算的损失函数会比PyTorch计算的损失函数小
# 所以要注意,手动计算的损失函数只是一个近似值,与PyTorch计算的损失函数有一定的差异
#


print(f"手动计算损失: {manual_loss.item():.4f}") # .item()作用是将张量转换为Python标量,使打印结果更清晰



# 步骤4: 使用PyTorch的CrossEntropyLoss验证
criterion = nn.CrossEntropyLoss()  #作用是创建一个交叉熵损失函数的实例
pytorch_loss = criterion(logits, target)#参数: logits(模型的原始输出,未经softmax), target(真实标签)

print(f"PyTorch计算损失: {pytorch_loss.item():.4f}")# 输出: 0.4170

#怎么理解这个0.4170 :
# 解读: 模型认为是猫的概率65.9%, 是狗24.24%, 是鸟9.86%
# 真实标签是猫(类别0), 所以one-hot编码为[1, 0, 0]
# 所以交叉熵损失函数的计算为 -(1*log(0.659) + 0*log(0.242) + 0*log(0.099)) = -log(0.659) = 0.417


#损失的意思:
# 损失函数的意思是,模型的预测与真实标签之间的差异程度,
# 损失函数越小,模型的预测越准确,损失函数越大,模型的预测越不准确
# 所以损失函数的作用是,通过优化模型的参数,使损失函数最小化,从而使模型的预测更准确



























