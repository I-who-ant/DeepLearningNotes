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


