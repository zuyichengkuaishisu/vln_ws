import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 1. 构建模型并移动到同一设备
model = NeuralNetwork().to(device)
print(model)

# 2. 单个样本的前向传播
x = torch.randn(1, 28, 28, device=device)
logits = model(x)
print(logits)
print(logits.shape)

pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
print(pred_probab.shape)

y_pred = torch.argmax(pred_probab, dim=1)
print(y_pred)
print(y_pred.shape)

# 3. 小批量输入（mini-batch）
input_image = torch.randn(3, 28, 28, device=device)
print(input_image.size())

# 4. nn.Flatten：把每张图片展平为一维向量
flatten = nn.Flatten()
flatten_image = flatten(input_image)
print(flatten_image.size())

# 5. nn.Linear：把 784 维输入映射到 20 维隐藏表示
layer1 = nn.Linear(in_features=28 * 28, out_features=20).to(device)
hidden1 = layer1(flatten_image)
print(hidden1.size())

# 6. nn.ReLU：把负数截断为 0，引入非线性
print(f"before ReLU: {hidden1}")
hidden1 = nn.ReLU()(hidden1)
print(f"after ReLU: {hidden1}")

# 7. nn.Sequential：按顺序组合多个子模块
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
).to(device)

input_image = torch.randn(3, 28, 28, device=device)
logits = seq_modules(input_image)
print(logits)

# 8. Softmax：把 logits 转成概率分布
pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
print(pred_probab.shape)

# 9. 查看模型参数
print(f"Model structure: {model}\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Value: {param[:2]}\n")
