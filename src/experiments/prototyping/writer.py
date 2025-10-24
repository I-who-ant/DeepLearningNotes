from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image

writer =SummaryWriter("logs")

image_path = "../dataset/arch_linux/base16-default-dark.png"

img_PIL=Image.open(image_path) # 打开图像文件,返回一个PIL.Image对象

img_array = np.array(img_PIL) # 将PIL.Image对象转换为NumPy数组

print(type(img_array)) # <class 'numpy.ndarray'>

print(img_array.shape) # (512, 512, 3)


writer.add_image("test",img_array, 1, dataformats='HWC') # 添加图像到TensorBoard,指定数据格式为HWC
writer.close()
