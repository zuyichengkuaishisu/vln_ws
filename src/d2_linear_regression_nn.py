import torch
import torch.nn as nn #导入神经网络模块
import matplotlib.pyplot as plt
#用nn.Parameter + optimizer 实现线性回归

torch.manual_seed(42) #set random seed
N = 100
X = torch.randn(N, 1)  #input tensor
Y = 3.0 * X - 1.5 + 0.1 * torch.randn(N, 1)  #true output tensor with noise

model = nn.Linear(1, 1)  #内含weight和bias
criterion_mse = nn.MSELoss()  #mean squared error loss function,计算均方误差损失
criterion_l1 = nn.L1Loss()  #mean absolute error loss function,计算均绝对误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  #stochastic gradient descent,随机梯度下降

#准备空列表，用于存储损失值
loss_mse_list = []
loss_l1_list = []

for epoch in range(200):
    pred = model(X)  #forward pass
    loss = criterion_mse(pred, Y)  #compute loss
    loss_mse_list.append(loss.item()) #store loss value in list for plotting


    optimizer.zero_grad()  #zero out gradient for next epoch
    loss.backward()  #backpropagate
    optimizer.step()  #update parameters

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, w: {model.weight.item():.4f}, b: {model.bias.item():.4f}")

for name, param in model.named_parameters():    #打印模型参数
    print(f"{name}: {param.item():.4f}")

#将loss换成L1损失函数
for epoch in range(200):
    pred = model(X)  #forward pass
    loss = criterion_l1(pred, Y)  #compute loss
    loss_l1_list.append(loss.item()) #store loss value in list for plotting


    optimizer.zero_grad()  #zero out gradient for next epoch
    loss.backward()  #backpropagate
    optimizer.step()  #update parameters

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, w: {model.weight.item():.4f}, b: {model.bias.item():.4f}")

for name, param in model.named_parameters():    #打印模型参数
    print(f"{name}: {param.item():.4f}")



'''             效果分析总结
MSE：收敛快，对大误差极其敏感，是回归任务的默认首选。
L1 Loss：收敛稍慢，但对异常值更鲁棒。
'''

# 绘制损失函数曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_mse_list, label='MSE', color='blue')
plt.plot(loss_l1_list, label='L1 Loss', color='red')

#设置图标标题和坐标轴标签
plt.title('Loss Function')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
#
