训练神经网络时最常用的算法是反向传播（Back propagation）
（1）反向传播的基本思想是：在每个训练样本上，计算模型输出与真实标签之间的误差，然后根据误差更新模型参数，以最小化误差。
（2）反向传播的实现是通过自动微分（Autograd）来计算梯度的。

![alt text](image.png)

 PyTorch 自动求导（Autograd）机制的底层实现原理
 function类：运算和梯度的结合
    对张量（Tensor）进行的每一个操作（比如加法、乘法、矩阵运算），本质上都不是简单的数学计算，而是创建了一个 torch.autograd.Function 类的实例对象。
    内部封装了两套逻辑：
    前向传播逻辑 (forward)：负责执行具体的数学运算，计算出结果张量。
    反向传播逻辑 (backward)：负责定义该运算的导数计算公式（即梯度如何从输出传回输入）。

grad_fn：梯度函数
    对张量进行操作时，pytorch会自动记录下操作的function对象，绑定到新张量的grad_fn属性上。

如何构建计算图（Computational Graph）
    每个张量进行操作时，pytorch会自动记录下操作的function对象，绑定到新张量的grad_fn属性上。
    这样，通过遍历计算图，就可以计算出所有张量的梯度。

只能对叶子节点量进行反向传播，其他张量的梯度只能通过链式法则计算。
出于性能考虑，在给定的计算图中，调用loss.backward()方法，Pytorch计算完梯度后，自动销毁和释放计算图。
需要对同一个计算图反向传播多次:loss.backward(retain_graph=True)

禁用梯度计算
    对张量进行操作时，pytorch会自动记录下操作的function对象，绑定到新张量的grad_fn属性上。
    但是，在训练完毕后，要将其应用于某些输入数据，而不是对模型参数进行反向传播。
    可以通过torch.no_grad()上下文管理器来禁用梯度计算。

torch.no_grad()用法：
    with torch.no_grad():
        # 代码块中的张量不会计算梯度
        pass

也可以使用detach()方法：断开梯度计算
    该方法会返回一个新的张量，与原张量共享内存，但是不会记录下操作的function对象。
    这样，对新张量进行操作时，不会触发梯度计算。

    例如：
    z = torch.matmul(x, w) + b  #linear combination
    z = z.detach()
    print(z.requires_grad)

禁用梯度计算的核心应用场景
    冻结参数（Transfer Learning/迁移学习）
    加速推理（Inference）
    