import torch
import numpy as np

#Tensors can be created directly from data. The data type is automatically inferred.
data = [6,7.5,8,0,1]
x_data = torch.tensor(data)

#Tensors can be created from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#Tensors can be created from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: {x_ones}")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: {x_rand}")

#Tensors can be created from shape
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: {rand_tensor}")
print(f"Ones Tensor: {ones_tensor}")
print(f"Zeros Tensor: {zeros_tensor}")

#Tensors can be created from shape
shape = (2,3,)
rand_tensor = torch.rand(shape)

#Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#We need to explicitly move tensors to the accelerator using .to method (after checking for accelerator availability)
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

#standard indexing and slicing  
tensor = torch.ones(4,4)
print(f"First row:{tensor[0]}")
print(f"First column:{tensor[:,0]}")
print(f"Last column:{tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

#use torch.cat to concatenate a sequence of tensors along a given dimension. 
t1 = torch.cat([tensor,tensor,tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T          #   这里使用了 @ 运算符。tensor.T 是获取 tensor 的转置矩阵。这行代码执行的是标准的线性代数矩阵乘法。
y2 = tensor.matmul(tensor.T)    #   这是上一行代码的面向对象写法（调用张量的方法）。它的作用与 tensor @ tensor.T 完全一模一样。
y3 = torch.rand_like(y1)        #   创建一个和y1形状相同的随机张量y3（作为预分配的内存空间）
torch.matmul(tensor,tensor.T,out=y3)    #使用 torch.matmul() 函数进行矩阵乘法，并通过 out=y3 将计算结果直接存入 y3 中。

#逐项自乘
z1 = tensor * tensor
z2 = tensor.mul(tensor)         #注意矩阵乘法与自乘不同
z3 = torch.rand_like(z1)
torch.mul(tensor,tensor,out=z3)
print(z1)

#把一个“张量（Tensor）”变成普通的 Python 数字。
print(tensor)
agg = tensor.sum()      #agg 的本质依然是一个 PyTorch 张量（只不过是一个形状为 () 的零维标量张量）。
agg_item = agg.item()   #.item把单元素张量的值提取出来，转换成一个标准的python原生数字
print(agg_item,type(agg_item))

#In-place operations（就地操作）。
"""
解释：普通的操作（比如 a + b）会创建一个新的张量来存放结果，原来的张量不变。而“就地操作”会直接修改原来的张量，把结果存回它自己的内存里。
标志：在 PyTorch 中，所有就地操作的方法名末尾都会带一个下划线 _。
"""
print(f"{tensor}\n")
tensor.add_(5)      #把 tensor 里的每一个元素都加上 5，并且直接把结果写回 tensor 自己。
print(tensor)

#Bridge with NumPy

#Tensor to NumPy array
t = torch.ones(5)
print(t)
n = t.numpy()
print(n)

"""重要理念：numpy和tensor的内存共享"""

t.add_(1)
print(t)
print(n)

#numpy to tensor

n = np.ones(5)
t = torch.from_numpy(n)

print(n)
print(t)


#每日跟敲补充
rand = torch.randn(2,3)  #元素符合标准正态
eye = torch.eye(3)      #单位阵
print(rand)
print(eye)

a = torch.tensor([1.,2,3])
b = torch.tensor([4.,5,6])
print("dot:",torch.dot(a,b))   #点积

v = torch.arange(12)
print(v)
m = v.view(3,4)   #view()方法可以改变张量的形状
print(m)
print(m.shape)
print(m[1,2])   #索引


#slam:2D旋转+平移变换
def make_se2(theta: float | torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    """
    Create a 2D SE(2) transformation matrix.

    Args:
        theta (float | torch.Tensor): Rotation angle in radians.
        tx (float): Translation in x direction.
        ty (float): Translation in y direction.

    Returns:
        torch.Tensor: A 3x3 SE(2) transformation matrix.
    """
    theta = torch.tensor(theta) if not isinstance(theta, torch.Tensor) else theta
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)   

    se2_matrix = torch.tensor([
        [cos_theta, -sin_theta, tx],
        [sin_theta, cos_theta, ty],
        [0.0, 0.0, 1.0]
    ])

    return se2_matrix

def transform_point(se2_matrix: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Transform a 2D point using an SE(2) transformation matrix.

    Args:
        point:[x,y,1]
        se2_matrix (torch.Tensor): A 3x3 SE(2) transformation matrix.
        point (torch.Tensor): A 2D point represented as a tensor of shape (2,).

    Returns:
        torch.Tensor: The transformed 2D point as a tensor of shape (2,).
    """

    # Apply the transformation
    transformed_homogeneous_point = se2_matrix @ point

    return transformed_homogeneous_point 

T = make_se2(torch.pi/4, 1.0, 2.0)  # 旋转45度，平移(1,2)
point = torch.tensor([1.0, 0.0, 1.0])
transformed_point = transform_point(T, point)
print("Transformed Point:", transformed_point)

#3D旋转+平移变换
def rot_z(yaw: float) -> torch.Tensor:
    """
    Create a 3D rotation matrix for rotation around the Z-axis.

    Args:
        yaw (float): Rotation angle in radians.

    Returns:
        torch.Tensor: A 3x3 rotation matrix.
    """
    yaw = torch.tensor(yaw) if not isinstance(yaw, torch.Tensor) else yaw
    cos_yaw = torch.cos(yaw) #结果保留在计算图中
    sin_yaw = torch.sin(yaw)

    rot_matrix = torch.tensor([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return rot_matrix

R = rot_z(torch.pi/4)  # 旋转45度
T = torch.tensor([1.0, 2.0, 3.0])  # 平移向量
point_3d = torch.tensor([1.0, 0.0, 0.0])
transformed_point_3d = R @ point_3d + T
print("Transformed 3D Point:", transformed_point_3d)

#练习:3D旋转+平移变换,齐次坐标表示
def make_se3(yaw: float | torch.Tensor, pitch: float | torch.Tensor, roll: float | torch.Tensor, tx: float, ty: float, tz: float) -> torch.Tensor:
    """
    Create a 3D SE(3) transformation matrix.

    Args:
        yaw (float | torch.Tensor): Rotation angle around Z-axis in radians.
        pitch (float | torch.Tensor): Rotation angle around Y-axis in radians.
        roll (float | torch.Tensor): Rotation angle around X-axis in radians.
        tx (float): Translation in x direction.
        ty (float): Translation in y direction.
        tz (float): Translation in z direction.

    Returns:
        torch.Tensor: A 4x4 SE(3) transformation matrix.
    """    
    # 统一设备、数据类型
    device = yaw.device  # 获取设备信息
    dtype = yaw.dtype    # 获取数据类型

    # 改用 torch.as_tensor：保留device、dtype、requires_grad，不断梯度
    yaw = torch.as_tensor(yaw,device=device, dtype=dtype)
    pitch = torch.as_tensor(pitch,device=device, dtype=dtype)
    roll = torch.as_tensor(roll,device=device, dtype=dtype)



    Rz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0.0],
        [torch.sin(yaw), torch.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    Ry = torch.tensor([
        [torch.cos(pitch), 0.0, torch.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-torch.sin(pitch), 0.0, torch.cos(pitch)]
    ], device=device, dtype=dtype)

    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(roll), -torch.sin(roll)],
        [0.0, torch.sin(roll), torch.cos(roll)]
    ], device=device, dtype=dtype)

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Create the SE(3) transformation matrix
    se3_matrix = torch.eye(4, device=device, dtype=dtype)
    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = torch.tensor([tx, ty, tz], device=device, dtype=dtype)

    return se3_matrix


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

yaw = torch.tensor(torch.pi/4, device=device, dtype=torch.float32)
pitch = torch.tensor(torch.pi/6, device=device, dtype=torch.float32)
roll = torch.tensor(torch.pi/3, device=device, dtype=torch.float32)

point_3d_homogeneous = torch.tensor([3.0, 2.0, 0.0, 1.0], device=device, dtype=torch.float32)# 齐次坐标表示的点
T_se3 = make_se3(yaw, pitch, roll, 1.0, 2.0, 3.0)  # 旋转45度、30度、60度，平移(1,2,3)
transformed_point_3d_homogeneous = T_se3 @ point_3d_homogeneous
print("Transformed 3D Point (Homogeneous):", transformed_point_3d_homogeneous)


