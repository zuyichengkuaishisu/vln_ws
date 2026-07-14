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
