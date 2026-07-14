import torch

x = torch.ones(5)  #input tensor
y = torch.zeros(3)  #expected output tensor
w = torch.randn(5, 3, requires_grad=True)  #weight tensor
b = torch.randn(3, requires_grad=True)  #bias tensor
z = torch.matmul(x, w) + b  #linear combination
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  #mean squared error loss function

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(f"Gradient of loss with respect to w = {w.grad}")
print(f"Gradient of loss with respect to b = {b.grad}")

z = torch.matmul(x, w) + b  #linear combination
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b  #linear combination
    print(z.requires_grad)

#detach()方法：断开梯度计算
z = torch.matmul(x, w) + b  #linear combination
z = z.detach()
print(z.requires_grad)

#对标量进行梯度计算，在x=2.0处计算梯度
x = torch.tensor(2.0, requires_grad=True)
y = x** 2 + 3 * x
y.backward()
print(x.grad)

#向量求导
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x** 2).sum()  #sum of squares
y.backward()
print(x.grad)

#关闭梯度计算
with torch.no_grad():
    z = x * 2
    print(z.requires_grad)

#线性回归
torch.manual_seed(42) #set random seed
N = 100
X = torch.randn(N, 1)  #input tensor
true_w, true_b = 3.0,-1.5  #true weight and bias
Y = true_w * X + true_b + 0.1 * torch.randn(N, 1)  #true output tensor with noise

w = torch.randn(1, requires_grad=True)  #weight tensor
b = torch.randn(1, requires_grad=True)  #bias tensor

lr = 0.1  #learning rate
for epoch in range(200):
    pred = torch.matmul(X, w) + b  #linear combination
    loss = (pred - Y).pow(2).mean()  #mean squared error loss function

    loss.backward()

    with torch.no_grad():
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

print(f"True w: {true_w:.4f}, True b: {true_b:.4f}")

