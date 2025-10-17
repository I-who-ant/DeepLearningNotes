from torch.utils.data import Dataset # 导入torch.utils.data模块中的Dataset类,用于自定义数据集
from PIL import Image # 导入PIL库中的Image类,用于处理图像

import os  # 导入os模块,用于操作文件和目录



class MyData(Dataset):

    def __init__(self, root_dir,label_dir) :
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        image_name = self.img_path[idx]
        image_item_path = os.path.join(self.root_dir, self.label_dir,image_name)
        image = Image.open(image_item_path)
        label = self.label_dir
        return image,label


    def __len__(self):
        return len(self.img_path)

root_dir = 'dataset'
label_dir = 'arch_linux'
dataset = MyData(root_dir,label_dir)
# 获取数据集的第一个样本 , image 是 PIL.Image.Image 类型, label 是字符串类型 'arch_linux'

# 如果只写img=dataset[0] 则 img 是元组类型 (image,label)
# 可以使用 img[0] 或 img[1] 来获取 image 或 label

#那么,使用 image,label = dataset[0] 来获取第一个样本的 image 和 label,
# 可以使用 image 来查看图像,使用 label 来查看标签

image,label = dataset[0]

