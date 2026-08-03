import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class NeuralNetwork(nn.Module):
    def __init__(self):  #继承nn.Module类，固定初始化方法
        super().__init__()
        self.flatten = nn.Flatten()     # 将输入的图片展平成一维向量
        self.linear_relu_stack = nn.Sequential(     #顺序容器，按顺序执行子模块
            nn.Linear(28*28, 512),  #第一个全连接层，输入28*28，输出512个神经元
            nn.ReLU(),              #ReLU激活函数，将输入的负值设为0，正值设为本身，用于引入非线性关系
            nn.Linear(512, 512),  #第二个全连接层，输入512个神经元，输出512个神经元
            nn.ReLU(),              #ReLU激活函数，将输入的负值设为0，正值设为本身
            nn.Linear(512, 10),  # 输出层，10个类别
        )  

    def forward(self, x):
        x = self.flatten(x)  #将输入的图片展平成一维向量
        logits = self.linear_relu_stack(x)  #按顺序执行子模块，得到输出
        return logits  #返回输出，用于计算损失函数

model = NeuralNetwork().to(device) #将模型移动到指定设备上
print(model)