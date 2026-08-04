# 使用 `nn.Module` 构建神经网络

在 PyTorch 中，无论是简单的层（如全连接层、卷积层）还是复杂的层（如注意力层、Transformer 层），本质上都可以看作 `nn.Module` 的子类。

神经网络本身也可以是一个更大的 `nn.Module`，通过把多个子层按一定连接关系组合起来，就能构建出复杂模型。这就是 PyTorch 中模块化搭建网络的核心思想。

## 通过继承 `nn.Module` 自定义网络层

通过继承 `nn.Module`，我们可以定义自己的神经网络层。下面是一个简单的全连接层封装示例：

```python
import torch
import torch.nn as nn


class MyLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
```

这个类里有两个关键部分：

1. `__init__()`：定义并初始化层中的参数或子模块。
2. `forward()`：定义前向传播逻辑，也就是输入数据如何经过这一层得到输出。
3. `nn.Linear` 默认 `bias=True`，表示该层默认带有偏置项。

例如，上面这段代码中：

```python
self.linear = nn.Linear(in_features, out_features)
```

表示定义了一个线性层，其中：

- `in_features`：输入特征维度。
- `out_features`：输出特征维度。

而在 `forward()` 中调用：

```python
self.linear(x)
```

PyTorch 会自动完成线性变换：

```math
y = Wx + b
```

其中：

- $W$ 是权重矩阵
- $x$ 是输入特征
- $b$ 是偏置
- $y$ 是输出结果

## 什么是全连接层

全连接层（Fully Connected Layer）是指当前层中的每个神经元都与前一层的所有神经元相连。

在 PyTorch 中，全连接层通常由 `nn.Linear` 实现，因此也常被称为线性层（Linear Layer）。

它的核心作用是对输入特征做一次线性变换，把输入映射到新的特征空间。

## 全连接层在神经网络中的作用

全连接层常见的作用包括：

1. 特征变换：把输入特征映射到新的表示空间。
2. 分类输出：在分类任务中，将高维特征映射到类别数对应的输出维度。
3. 回归输出：在回归任务中，将特征映射到连续值。

## 全连接层和线性回归的区别

### 共同点

从数学形式上看，全连接层和线性回归都可以写成：

```math
y = Wx + b
```

也就是说，它们本质上都包含线性变换（更准确地说是仿射变换）。

### 不同点

1. 角色不同  
   线性回归通常是一个完整的机器学习模型，用于回归任务；全连接层通常只是神经网络中的一个组成模块。

2. 是否通常接非线性激活函数  
   线性回归的输出一般直接作为最终预测值；全连接层后面通常会接激活函数（如 ReLU、Sigmoid），从而让整个网络具备表示非线性关系的能力。

3. 所处系统不同  
   线性回归往往单独作为模型使用；全连接层通常嵌入在更深的网络结构中，与卷积层、归一化层、注意力层等一起工作。

4. 训练方式的语境不同  
   线性回归常作为传统机器学习模型来讨论，通常以最小化均方误差为目标；全连接层则作为神经网络的一部分，通过反向传播和优化器（如 SGD、Adam）与整个网络一起训练。

## 一个容易混淆的点

“全连接层的特征空间是多维，而线性回归的特征空间是 1 维” 这种说法并不严谨。

更准确地说：

- 线性回归最常见的是输出一个连续值，但也可以扩展到多输出回归。
- 全连接层的输出维度由 `out_features` 决定，可以是一维，也可以是多维。

因此，两者更本质的区别不在于“输出一定是几维”，而在于它们在模型中的角色和使用场景不同。

## 如何调用模型

使用模型时，通常直接写：

```python
output = model(x)
```

而不是手动写：

```python
output = model.forward(x)
```

原因不是“手动调用一定会导致模型状态错误”，而是：

1. `model(x)` 会调用 `nn.Module` 中的 `__call__()`。
2. `__call__()` 内部再去调用 `forward()`。
3. 在这个过程中，PyTorch 还会正确处理钩子（hook）等机制。

所以，`forward()` 当然可以被直接调用，但在实际使用中不推荐这样做；标准写法始终是 `model(x)`。

## 模型输出、logits 和概率

假设最后一层是：

```python
nn.Linear(512, 10)
```

那么对于一个 batch 的输入，模型输出通常是一个二维张量，形状类似：

```python
[batch_size, 10]
```

这里：

- 第 0 维（`dim=0`）表示 batch 中的样本数。
- 第 1 维（`dim=1`）表示每个样本对应 10 个类别的输出值。

这 10 个输出值通常叫做 `logits`，也可以理解为原始预测值。

## 什么是 logits

`logits` 是模型最后一层直接输出的数值，它们还不是概率。

例如：

```python
tensor([2.1, 0.3, -1.2, ...])
```

这些值只是线性层或网络最后一层计算得到的结果，还没有经过归一化，因此：

- 它们的取值范围不一定在 `0` 到 `1` 之间。
- 它们的总和也不一定等于 `1`。

所以，`logits` 不能直接当作概率解释。

## 什么是 `nn.Softmax`

`nn.Softmax` 的作用是把一组 `logits` 转换为概率分布。

它会把输出映射到 `0` 到 `1` 之间，并让这些值在指定维度上的总和等于 `1`。

例如在分类任务中，若输出形状是 `[batch_size, num_classes]`，通常会在类别维度上做 `Softmax`：

```python
import torch.nn as nn

softmax = nn.Softmax(dim=1)
probs = softmax(logits)
```

这里 `dim=1` 表示对每个样本的各类别分数做归一化，得到该样本属于各类别的概率分布。

## 一个常见注意点

如果训练时使用的是 `nn.CrossEntropyLoss`，通常不要在模型输出后手动再接 `Softmax`，因为：

1. `CrossEntropyLoss` 期望输入的是 `logits`。
2. 它内部已经包含了 `Softmax` 相关计算。

也就是说：

- 训练时：通常直接把 `logits` 传给 `nn.CrossEntropyLoss`。
- 推理时：如果你想把输出解释为概率，再额外使用 `Softmax`。

## 什么是 `nn.Flatten`

图像数据常见的形状是：

```python
[batch_size, height, width]
```

例如 MNIST 单张图片大小为 `28 x 28`，如果输入一个 batch，张量形状可能是：

```python
[3, 28, 28]
```

而全连接层 `nn.Linear` 期望输入的最后一维是特征维度，因此通常需要先把图片展平成一维向量：

```python
flatten = nn.Flatten()
flatten_image = flatten(input_image)
```

展平后，形状会变成：

```python
[3, 784]
```

这里的 `784 = 28 x 28`。

## 什么是 `nn.ReLU`

`nn.ReLU` 是神经网络中最常见的激活函数之一，它的规则很简单：

```math
\text{ReLU}(x) = \max(0, x)
```

也就是说：

- 输入大于 0 时，输出保持不变。
- 输入小于等于 0 时，输出变为 0。

它的作用是为网络引入非线性能力。如果网络里只有线性层，那么无论叠多少层，本质上仍然等价于一个线性变换。

## 什么是 `nn.Sequential`

`nn.Sequential` 是一个顺序容器，它会按照定义顺序依次执行各个子模块。

例如：

```python
seq_modules = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
)
```

它等价于“上一层的输出作为下一层的输入”这一串操作，因此非常适合用来搭建结构简单、按顺序传播的网络。

## 为什么要关注设备一致性

在使用 GPU 时，模型参数和输入张量必须放在同一个设备上。

例如：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_image = input_image.to(device)
```

如果输入在 `cuda:0`，而某一层的参数还留在 CPU 上，就会报类似这样的错误：

```python
RuntimeError: Expected all tensors to be on the same device
```

因此，一个很重要的习惯是：

- 要么把整个模型统一 `.to(device)`。
- 要么确保模型中的每一层和输入数据都在同一设备上。

## 如何查看模型参数

神经网络中的线性层、卷积层等通常都带有可学习参数，例如：

- 权重（weight）
- 偏置（bias）

这些参数会在训练过程中不断更新。PyTorch 会自动追踪 `nn.Module` 中注册的参数，我们可以用：

```python
model.parameters()
```

或：

```python
model.named_parameters()
```

来查看它们。

例如：

```python
for name, param in model.named_parameters():
    print(name, param.size())
```

这样可以看到每一层参数的名字和形状，有助于理解模型结构，也方便调试。

Loss function 损失函数:给模型的输出和真实标签之间的差异进行量化，用于指导模型的训练过程。
    最常用的Loss:交叉熵损失函数(Cross Entropy Loss)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels) #传入logits和真实标签
    
Optimizer 优化器:用于更新模型参数，使损失函数最小化,核心算法是梯度下降(Gradient Descent)
    最常用的优化器:Adam优化器(Adam)
    如何工作的：
        1. 计算模型参数的梯度
        2. 更新参数，步长为lr（学习率，learning rate）
        3.反复重复以上步骤，直到模型的损失函数最小化

    先定义优化器，再在训练循环中使用它
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad() #调用优化器的zero_grad()方法，将所有参数的梯度设为0,不然每次更新参数时，梯度会累加，导致参数更新错误
    loss.backward() #计算梯度
    optimizer.step() #更新参数
