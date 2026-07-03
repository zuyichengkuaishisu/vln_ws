# W1 D1 笔记 — PyTorch 张量基础

> **日期**：2026-07-03  
> **代码**：`src/week01_d1.py`  
> **跟敲清单**：[`plan/week01_pytorch.md`](../plan/week01_pytorch.md) D1

---

## 1. PyTorch 四大核心

| 模块 | 作用 |
|------|------|
| **Tensor** | 核心数据结构，多维数组，支持 CPU/GPU |
| **Autograd** | 自动求导，构建计算图，反向传播（D2 学） |
| **nn.Module** | 神经网络模块，定义模型与前向传播（D3 学） |
| **optim** | 优化器（SGD、Adam），更新可学习参数（D2 学） |

### 生态系统

```text
┌─────────────────────────────────────────────────────────────┐
│  torchvision  │  torchtext  │  torchaudio  │  其他专业库     │
├─────────────────────────────────────────────────────────────┤
│                     PyTorch 核心                            │
├───────────────┬─────────────────┬───────────────────────────┤
│   torch.nn    │   torch.optim   │      torch.utils          │
├───────────────┼─────────────────┼───────────────────────────┤
│  torch 核心   │  autograd       │  torch.utils.data         │
│  (张量计算)   │  (自动微分)     │  (DataLoader, Dataset)    │
└───────────────┴─────────────────┴───────────────────────────┘
```

---

## 2. Tensor 核心属性

| 属性 | 含义 | 示例 |
|------|------|------|
| **shape** | 各维长度 | 标量 `()`、向量 `(N,)`、矩阵 `(H,W)`、图像 batch `(B,C,H,W)` |
| **dtype** | 元素数据类型 | 默认 `float32`；标签常用 `int64` |
| **device** | 存放设备 | `cpu` / `cuda:0` |
| **requires_grad** | 是否追踪梯度 | 默认 `False`；可学习参数设 `True`（D2） |

### shape 与 dim 编号（2D 矩阵）

```text
shape = [行, 列]
         ↑    ↑
       dim=0 dim=1
```

- `torch.cat(..., dim=0)`：行方向叠（竖拼），行数变多  
- `torch.cat(..., dim=1)`：列方向拼（横拼），列数变多  

---

## 3. dtype 常用类型

**浮点（深度学习主力）**

| dtype | 说明 |
|-------|------|
| `torch.float32` / `float` | 默认，训练最常用 |
| `torch.float64` / `double` | 双精度，数值更稳但更慢 |
| `torch.float16` / `half` | 半精度，省显存 |
| `torch.bfloat16` | 大模型训练常用 |

**整数**

| dtype | 说明 |
|-------|------|
| `torch.int64` / `long` | 分类标签、索引 |
| `torch.int32` / `int` | 一般整数 |

**其他**：`torch.bool`

---

## 4. GPU 使用（重要）

**张量默认创建在 CPU**，不会自动上 GPU。

```python
# 创建时指定
x = torch.rand(3, 4, device="cuda")

# 或创建后迁移
x = x.to("cuda")
# x = x.to(torch.accelerator.current_accelerator())  # PyTorch 2.11+ 新 API
```

训练时 **model 和每个 batch 的数据** 必须在同一 device，否则报错。

本机：RTX 5090 → PyTorch 需 **cu128**（见 [`conda.md`](conda.md)）。

---

## 5. 训练术语（预习）

| 术语 | 含义 |
|------|------|
| **Batch Size** | 一次喂给模型的样本数（常用 8/16/32/64） |
| **Iteration** | 跑完 1 个 batch = 1 次迭代 |
| **Epoch** | 整个训练集所有 batch 各跑一遍 = 1 个 epoch |

---

## 6. 常用操作速查

### 创建

```python
torch.tensor(data)       # 从 Python 列表
torch.from_numpy(arr)    # 从 NumPy（共享内存）
torch.ones_like(x)       # 同 shape/dtype
torch.rand / ones / zeros / eye / arange
```

### 运算

```python
A @ B              # 矩阵乘
A * B              # 逐元素乘（不是矩阵乘！）
torch.dot(a, b)    # 向量点积
A.T                # 转置
v.view(3, 4)       # 改 shape（元素总数不变）
```

### 索引

```python
tensor[0]          # 第一行
tensor[:, 0]       # 第一列
tensor[..., -1]    # 最后一列
tensor[:, 1] = 0   # 赋值（in-place 改列）
```

### 聚合与转换

```python
tensor.sum()       # 求和，仍是 tensor
tensor.sum().item()  # 提取为 Python 标量
tensor.add_(5)     # in-place，方法名带 _ 表示改自身
```

### NumPy 互转

```python
n = t.numpy()           # tensor → numpy，共享内存
t = torch.from_numpy(n) # numpy → tensor，共享内存
```

`t.add_(1)` 后 `n` 也会变——**同一块内存**。

---

## 7. Tensor vs NumPy vs Eigen

| | NumPy | Eigen (C++) | PyTorch Tensor |
|--|-------|-------------|----------------|
| 多维数组 | ✅ | ✅ | ✅ |
| 几何/矩阵运算 | ✅ | ✅（SLAM 常用） | ✅ |
| GPU 加速 | ❌ | ❌ | ✅ |
| 自动微分 | ❌ | ❌ | ✅ |
| 与深度学习框架集成 | 需转换 | 需转换 | 原生 |

**SLAM 工程师视角**：SE(2)/SE(3) 变换在 Eigen 里怎么写，用 Tensor 同样用 `@` 做矩阵乘；区别是 Tensor 可以 `requires_grad=True` 参与端到端训练。

---

## 8. SLAM ↔ PyTorch（今日实践）

### SE(2) — 2D 旋转 + 平移

```python
T = make_se2(theta, tx, ty)   # 3×3 齐次矩阵
p = torch.tensor([x, y, 1.0]) # 齐次坐标
p_new = T @ p
```

### SE(3) — 3D 刚体变换

```python
T = make_se3(yaw, pitch, roll, tx, ty, tz)  # 4×4
p = torch.tensor([x, y, z, 1.0])
p_new = T @ p
```

### 复合变换

```python
# (T1 @ T2) @ p  ==  T1 @ (T2 @ p)
```

### device 一致性

SE(3) 练习中：yaw/pitch/roll、点、矩阵都在同一 `device`（`cuda:0`），否则 `@` 报错。

---

## 9. requires_grad（D2 预告）

| 值 | 行为 |
|----|------|
| `False`（默认） | 普通常量，不参与反向传播 |
| `True` | 记录运算到计算图，可 `.backward()` 求梯度 |

可用于优化：待估计的位姿、网络权重等。

---

## 10. 易错点

1. **默认 CPU**：`torch.rand()` 不会自动上 GPU  
2. **矩阵乘 vs 逐元素乘**：`@` / `matmul` ≠ `*`  
3. **dim 方向**：`cat` 的 dim 是「变长的那一维」  
4. **NumPy 共享内存**：`from_numpy` / `.numpy()` 后改一个另一个也变  
5. **device 混用**：CPU tensor 和 CUDA tensor 不能直接 `@`  

---

## 11. D1 验收自评

- [x] 能解释 shape / dtype / device
- [x] 能写 SE(2) / SE(3) 变换
- [x] GPU 上完成矩阵运算（SE(3) @ point on cuda:0）
- [x] 理解 Tensor 与 NumPy / Eigen 异同
- [ ] SE(2) 复合变换 `T1 @ T2` 验证（可选补）

**D1 结论：通过，可进入 D2 Autograd。**
