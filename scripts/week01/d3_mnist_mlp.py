import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#1.数据
transform = transforms.Compose([
    transforms.ToTensor(),          # 转张量与初步缩放，把普通图片（PIL或NumPy数组等）转换为张量

    # 数据变化：
    #     数值缩放：把像素值从 [0, 255] 转换为 [0, 1]，方便模型训练。
    #     形状改变：把图片形状从[高度，宽度，通道数](H, W, C) 转换为[通道数，高度，宽度](C, H, W)

    #实验C：去掉标准化处理，观察模型训练效果：模型训练效果下降，损失值增加
     transforms.Normalize((0.1307,), (0.3081,)) #对数据进行标准化处理，使数据分布更符合正态分布，方便模型训练，公式为：(x - mean) / std
    #0.1307 是灰度图像的均值，0.3081 是灰度图像的标准差，因为MNIST是单通道灰度图，所以元组只有一个值，如果是rgb图像，需要提供3个值，分别对应r,g,b通道的均值和标准差
    ])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform) # 训练集
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform) # 测试集

#实验B：batch_size=512
train_loader = DataLoader(train_set, batch_size=512, shuffle=True)   # 训练集数据加载器
test_loader = DataLoader(test_set, batch_size=512, shuffle=True)   # 测试集数据加载器

#2.模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss(reduction="mean") # 交叉熵损失函数，隐藏了参数：reduction="mean"，表示对每个样本的损失值取平均，得到一个标量损失值
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam优化器
# optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam优化器,实验A：lr=0.01
# optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam优化器,实验A：lr=0.0001

#3.训练一个epoch

def train_one_epoch(model,loader,criterion,optimizer): # 训练一个epoch
    model.train() # 训练模式
    total_loss,correct,total = 0.0,0,0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # 清空梯度
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step() # 更新模型参数

        total_loss += loss.item() * labels.size(0) # 计算当前batch的损失值，乘以样本数量，得到当前epoch的总损失
        correct += (logits.argmax(dim=1) == labels).sum().item()    # 计算当前batch的正确预测数量，累加到当前epoch的正确预测数量
        # """
        # logits.argmax(dim=1):在类别维度（dim=1）上找最大值，得出模型预测的类别编号。
        # == labels:把预测的编号和真实的标签进行逐个对比，相等的返回 True（相当于 1），不相等的返回 False（相当于 0）。
        # .sum().item()：把所有 True 加起来，就得到了当前这个 Batch 里猜对的图片数量。
        # """
        total += images.size(0)  # 累加当前batch的样本数量，得到当前epoch的样本数量
    return total_loss / total, correct / total # 返回当前epoch的平均损失和正确预测率

@torch.no_grad()  # 不计算梯度，节省内存，评估时不需要计算梯度
def evaluate(model,loader,criterion):
    model.eval() # 评估模式
    total_loss,correct,total = 0.0,0,0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0) # 计算当前batch的损失值，乘以样本数量，得到当前epoch的总损失
        correct += (logits.argmax(dim=1) == labels).sum().item()    # 计算当前batch的正确预测数量，累加到当前epoch的正确预测数量
        total += images.size(0)  # 累加当前batch的样本数量，得到当前epoch的样本数量
    return total_loss / total, correct / total # 返回当前epoch的平均损失和正确预测率

#准备空列表来收集每个epoch的损失值和正确预测率
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []


for epoch in range(5):
    train_loss,train_acc = train_one_epoch(model,train_loader,criterion,optimizer)
    test_loss,test_acc = evaluate(model,test_loader,criterion)
    #将本轮结果添加到列表中
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    # 打印本轮结果
    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
    )

#4.可视化训练过程
plt.figure(figsize=(10, 6))  # 设置图表大小
plt.plot(train_loss_list, label='Train Loss', color='blue',linewidth=2)
plt.plot(test_loss_list, label='Test Loss', color='red',linewidth=2,linestyle='--') # 测试集损失值用虚线表示
plt.xlabel('Epoch',fontsize=12) # 设置x轴标签,12号字体
plt.ylabel('Loss',fontsize=12)
plt.title('Training Loss',fontsize=14)
plt.legend(fontsize=12)
plt.grid(True,alpha=0.3)

#保存图表
plt.savefig('./pic/training_loss.png',dpi=300,bbox_inches='tight')
# 显示图表
plt.show()

    
