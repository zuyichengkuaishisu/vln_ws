# W1：PyTorch 入门 — 每日跟敲清单

> **跟敲清单**：配合 [`learning_plan.md`](learning_plan.md) 使用  
> **个人笔记**：写到 [`notes/`](../notes/)（如 `week01.md`、`week01_review.md`）  
> **本周目标**：理解 tensor / autograd / nn / DataLoader / GPU 训练，独立完成 MNIST 训练脚本  
> **时间**：每天 3h · **代码目录**：`scripts/week01/` · **环境**：Python 3.10+，PyTorch cu128，RTX 5090

---

## 环境准备（D1 开始前完成）

```bash
# 1. 创建 conda 环境
conda create -n vln python=3.10 -y
conda activate vln

# 2. 安装 PyTorch（RTX 5090 需 cu128，含 sm_120 支持；不要用 cu124）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 验证
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 4. 可选：jupyter / matplotlib
pip install matplotlib jupyter ipython
```

**验收**：输出 `True`，设备名含 `5090`。

```bash
mkdir -p /home/zy/.wzy/vln_ws/scripts/week01
```

---

## D1：张量基础 + SLAM 里的 SE(3)

### Hour 1 — 理论

- [ ] 阅读 [PyTorch 60-min blitz — Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [ ] 理解：`shape`、`dtype`、`device`、`requires_grad`
- [ ] 对照 SLAM：旋转矩阵 `R`、平移 `t`、齐次变换 `T` 都是数值矩阵 → 天然适合 tensor

### Hour 2 — 跟敲 `scripts/week01/d1_tensors.py`

```python
import torch

# ---- 1. 创建与属性 ----
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("shape:", x.shape, "dtype:", x.dtype, "device:", x.device)

zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
rand = torch.randn(2, 3)          # 标准正态
eye = torch.eye(3)

# ---- 2. 运算 ----
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print("a+b:", a + b)
print("dot:", torch.dot(a, b))
print("matmul:", torch.matmul(x, x.T))

# ---- 3. reshape / view / squeeze ----
v = torch.arange(12)
m = v.view(3, 4)                  # 3x4
print(m.shape)
print(m[1, 2])                    # 索引

# ---- 4. numpy 互转（注意共享内存）----
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(np_arr)
np_back = t_from_np.numpy()

# ---- 5. SLAM：2D 旋转 + 平移 ----
def make_se2(theta: float, tx: float, ty: float) -> torch.Tensor:
    """SE(2) 齐次变换矩阵 3x3"""
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    T = torch.tensor([
        [c, -s, tx],
        [s,  c, ty],
        [0., 0., 1.],
    ])
    return T

def transform_point(T: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """p: [x, y, 1]"""
    return T @ p

T = make_se2(theta=0.5, tx=1.0, ty=2.0)
p = torch.tensor([1.0, 0.0, 1.0])
p_new = transform_point(T, p)
print("transformed point:", p_new[:2])

# ---- 6. 3D 旋转：Rodrigues 思路（仅练习矩阵运算）----
# 绕 z 轴旋转
def rot_z(yaw: float) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
    R = torch.tensor([
        [c, -s, 0.],
        [s,  c, 0.],
        [0., 0., 1.],
    ])
    return R

R = rot_z(0.3)
t = torch.tensor([1., 2., 3.])
p3 = torch.tensor([1., 0., 0.])
p3_new = R @ p3 + t
print("R @ p + t:", p3_new)
```

运行：

```bash
cd /home/zy/.wzy/vln_ws/scripts/week01
python d1_tensors.py
```

### Hour 3 — 巩固

- [ ] **练习 1**：不用 copy，手写 `make_se2`，输入 `(theta, tx, ty)`，变换点 `(3, 2)`
- [ ] **练习 2**：两个 SE(2) 矩阵 `T1 @ T2`，验证与「先 T2 再 T1」复合变换一致
- [ ] **练习 3**：把 `x` 移到 GPU（若可用）：`x_cuda = x.to("cuda")`，对 `x_cuda` 做一次 `@`
- [ ] 在 `notes/week01.md` 写 D1 笔记：tensor 与 Eigen/numpy 的异同（3 句话）

**D1 验收**

- [ ] 能解释 `shape` 每一维含义
- [ ] 能写 SE(2) 变换而不查代码

---

## D2：Autograd + 线性回归 from scratch

### Hour 1 — 理论

- [ ] 阅读 [Autograd 教程](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [ ] 理解：计算图、`requires_grad`、`backward()`、`.grad`、`with torch.no_grad()`
- [ ] 类比 SLAM：BA 里对位姿/路标求 Jacobian → 这里是自动求导

### Hour 2 — 跟敲 `scripts/week01/d2_autograd.py`

```python
import torch

# ---- 1. 标量求导 ----
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print("dy/dx at x=2:", x.grad)   # 2*2+3 = 7

# ---- 2. 向量求导 ----
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print("grad:", x.grad)           # [2, 4, 6]

# ---- 3. 关闭梯度追踪 ----
with torch.no_grad():
    z = x * 2
print("z requires_grad:", z.requires_grad)

# ---- 4. 线性回归 y = wx + b ----
torch.manual_seed(42)
N = 100
X = torch.randn(N, 1)
true_w, true_b = 3.0, -1.5
Y = true_w * X + true_b + 0.1 * torch.randn(N, 1)

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for epoch in range(200):
    pred = X * w + b
    loss = ((pred - Y) ** 2).mean()   # MSE

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  w={w.item():.3f}  b={b.item():.3f}")

print(f"final: w≈{true_w}, b≈{true_b}")
```

### Hour 2 续 — `scripts/week01/d2_linear_regression_nn.py`

用 `nn.Parameter` + `optim.SGD` 重写（更接近真实训练）：

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
N = 100
X = torch.randn(N, 1)
Y = 3.0 * X - 1.5 + 0.1 * torch.randn(N, 1)

model = nn.Linear(1, 1)           # 内含 w, b
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    pred = model(X)
    loss = criterion(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}")

for name, param in model.named_parameters():
    print(name, param.data)
```

### Hour 3 — 巩固

- [ ] **练习 1**：把 MSE(均方误差) 改成 `L1 loss(MAE 绝对值误差)` ，观察收敛差异
- [ ] **练习 2**：画 loss 曲线（matplotlib），保存 `d2_loss.png`
- [ ] **练习 3**：解释 `optimizer.zero_grad()` 为什么必须在 `backward()` 前调用
- [ ] 对比：手写梯度下降 vs `nn.Linear` + `SGD` 各有什么优劣

**D2 验收**

- [ ] 能口述 autograd 四步：`zero_grad → forward → backward → step`
- [ ] 线性回归 w/b 接近真值（误差 < 0.2）

---

## D3：nn.Module + MNIST MLP

### Hour 1 — 理论

- [ ] 阅读 [Build Model 教程](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [ ] 理解：`nn.Module`、`forward()`、`nn.Sequential`、loss、optimizer
- [ ] MNIST：28×28 灰度图 → 784 维向量 → 10 类分类

### Hour 2 — 跟敲 `scripts/week01/d3_mnist_mlp.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---- 1. 数据 ----
transform = transforms.Compose([
    transforms.ToTensor(),                          # [0,1], shape [1,28,28]
    transforms.Normalize((0.1307,), (0.3081,)),    # MNIST 均值/方差
])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2)

# ---- 2. 模型：784 -> 256 -> 128 -> 10 ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- 3. 训练 1 个 epoch（先跑通）----
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    print(f"epoch {epoch+1}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
```

### Hour 3 — 巩固

- [ ] **实验 A**：`lr=1e-2` vs `1e-4`，记录 5 epoch 后 val_acc
- [ ] **实验 B**：`batch_size=64` vs `512`，观察每 epoch 耗时与 loss 曲线
- [ ] **实验 C**：去掉 `Normalize`，acc 是否下降？
- [ ] 画训练/验证 loss 曲线 → `d3_loss_curve.png`
- [ ] 记录实验结果到 `notes/week01.md` 下方表格

| 实验 | 参数 | val_acc (5 epoch) | 备注 |
|------|------|-------------------|------|
| baseline | lr=1e-3, bs=64 | | |
| A | lr=1e-2 | | |
| B | bs=512 | | |

**D3 验收**

- [ ] 5 epoch 后 test acc > 95%
- [ ] 能解释 `CrossEntropyLoss` 与 `softmax` 的关系

---

## D4：Dataset / DataLoader 深入

### Hour 1 — 理论

- [ ] 阅读 [Data 教程](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [ ] 理解：`Dataset` 协议、`DataLoader` 批处理、`collate_fn`、`num_workers`
- [ ] VLN 预告：以后 Dataset 返回 `(rgb_image, instruction_tokens, action_label)`

### Hour 2 — 跟敲 `scripts/week01/d4_custom_dataset.py`

```python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---- 1. 最小自定义 Dataset ----
class SimpleImageDataset(Dataset):
    """假设目录结构：
    data/
      cat/  xxx.jpg
      dog/  yyy.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []   # [(path, label), ...]
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ---- 2. 用 MNIST 模拟「自定义」逻辑 ----
from torchvision import datasets

class MNISTSubset(Dataset):
    """只取前 1000 张，练习 Dataset 包装"""
    def __init__(self, mnist_dataset):
        self.ds = mnist_dataset

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.ds[idx]

transform = transforms.ToTensor()
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)
subset = MNISTSubset(mnist)
loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)

images, labels = next(iter(loader))
print("batch images shape:", images.shape)   # [32, 1, 28, 28]
print("batch labels shape:", labels.shape)   # [32]

# ---- 3. collate_fn 示例：变长序列（VLN 以后会用到）----
def pad_collate(batch):
    """batch: list of (tensor[L_i], label)"""
    seqs, labels = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded, torch.tensor(labels)

fake_batch = [(torch.randn(5), 0), (torch.randn(8), 1), (torch.randn(3), 2)]
padded, labs = pad_collate(fake_batch)
print("padded shape:", padded.shape)
```

### Hour 3 — 巩固

- [ ] **练习 1**：手写 `__len__` 和 `__getitem__` 框架（不看代码）
- [ ] **练习 2**：`num_workers=0` vs `2` vs `4`，记录 DataLoader 首个 batch 加载时间
- [ ] **练习 3**：设计一个「伪 VLN Dataset」返回 dict：`{"image": ..., "instruction": ..., "action": ...}`
- [ ] 总结 Dataset 三要素到 `notes/week01.md`

**D4 验收**

- [ ] 能独立写一个返回 `(image, label)` 的 Dataset 类
- [ ] 能解释 `shuffle=True` 为什么只应在 train 使用

---

## D5：GPU 训练 + 5090 吞吐测试

### Hour 1 — 理论

- [ ] 阅读 [Training 教程](https://pytorch.org/tutorials/beginner/basics/optimizeloop_tutorial.html)
- [ ] 理解：`.to(device)`、`model.train()` / `model.eval()`、`torch.cuda.synchronize()`
- [ ] 混合精度预告：`torch.cuda.amp`（W9+ 训练 DUET 时会用）

### Hour 2 — 跟敲 `scripts/week01/d5_gpu_benchmark.py`

在 D3 的 MNIST 脚本基础上增加 benchmark：

```python
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ... 复用 D3 的 MLP 定义 ...

device = torch.device("cuda")
print(torch.cuda.get_device_name(0))

def benchmark(batch_size: int, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # warmup
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        break

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_batches = 100
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    samples = n_batches * batch_size
    print(f"bs={batch_size:4d}  workers={num_workers}  "
          f"{samples/elapsed:,.0f} samples/s  {elapsed:.2f}s for {n_batches} batches")

for bs in [64, 128, 256, 512, 1024]:
    try:
        benchmark(bs)
    except RuntimeError as e:
        print(f"bs={bs} OOM:", e)
```

### Hour 2 续 — 模型保存与加载 `scripts/week01/d5_save_load.py`

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters())

# 保存
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 5,
}
torch.save(checkpoint, "mnist_mlp.ckpt")

# 加载
ckpt = torch.load("mnist_mlp.ckpt", map_location="cpu")
model2 = MLP()
model2.load_state_dict(ckpt["model_state_dict"])
model2.eval()
print("loaded epoch:", ckpt["epoch"])
```

### Hour 3 — 巩固

- [ ] 跑完 benchmark，填下方表格
- [ ] 完整训练 MNIST 10 epoch（GPU），目标 test acc > 97%
- [ ] 保存最佳 checkpoint 到 `scripts/week01/checkpoints/mnist_mlp_best.ckpt`

| batch_size | samples/s | 是否 OOM |
|------------|-----------|----------|
| 64 | | |
| 128 | | |
| 256 | | |
| 512 | | |
| 1024 | | |

**D5 验收**

- [ ] 训练全程在 GPU 上，无 device 报错
- [ ] 能正确 save / load 模型并继续推理

---

## D6：综合复习 — 闭卷重写 MNIST

### Hour 1 — 复习

- [ ] 重读 `notes/week01.md`，列出还不清楚的 3 个问题并查文档解决
- [ ] 画出 MNIST 训练流程图：Data → Model → Loss → Optimizer → Metrics

### Hour 2 — 闭卷实现 `scripts/week01/d6_mnist_from_scratch.py`

**规则**：不打开 D3 代码，凭记忆写完。允许查 PyTorch 文档，不允许 copy 旧代码。

必须包含：

- [ ] `MLP` 类（或 `nn.Sequential`）
- [ ] `train_one_epoch()` 和 `evaluate()`
- [ ] `main()`：至少 5 epoch，打印 train/val loss 和 acc
- [ ] 模型保存到 `checkpoints/`

### Hour 3 — 对比与 debug

- [ ] 与 D3 代码 diff，找出遗漏（常见：`zero_grad` 位置、`model.eval()`、`.to(device)`）
- [ ] 修复直到 val_acc 与 D3 接近（差距 < 1%）

**D6 验收**

- [ ] 闭卷脚本可运行，5 epoch val_acc > 95%

---

## D7：Weekly Review + W1 复盘

### Hour 1 — 查漏补缺

- [ ] 重做一次 D2 线性回归（计时：15 分钟内写完）
- [ ] 口述 autograd 流程（录音或写文字，1 分钟版）

### Hour 2 — 小测验（自测）

1. `torch.tensor([1,2,3])` 与 `torch.tensor([1.,2.,3.])` 的 dtype 区别？
2. `(logits.argmax(1) == labels).float().mean()` 在算什么？
3. 为什么 eval 时要 `torch.no_grad()`？
4. `pin_memory=True` 有什么用？
5. CrossEntropyLoss 的输入 shape 要求？

<details>
<summary>参考答案（做完再看）</summary>

1. 前者 int64，后者 float32；深度学习一般用 float。
2. 当前 batch 的分类准确率。
3. 不构建计算图，省显存、加速推理。
4. 锁页内存，加速 CPU→GPU 异步拷贝（配合 `non_blocking=True`）。
5. logits: `[N, C]`，labels: `[N]` 且为 class index（long），内部含 log-softmax。

</details>

### Hour 3 — 写 W1 复盘

写到 `notes/week01_review.md`，并更新 `plan/learning_plan.md` 进度表 W1 行。

建议包含：

- 本周最大收获
- 还不清楚、W2 要继续学的
- 与 SLAM 工作的联系（2 点）

---

## W1 总验收 checklist

完成以下全部项，即可进入 W2：

- [ ] conda 环境 `vln` 可用，5090 识别正常
- [ ] `scripts/week01/` 下有 d1–d6 全部脚本
- [ ] 能 30 分钟内闭卷写出 MNIST 训练循环
- [ ] 理解 tensor / autograd / nn.Module / DataLoader / GPU
- [ ] 完成 W1 复盘 → `notes/week01_review.md`

---

## 常见问题速查

| 报错 | 原因 | 解决 |
|------|------|------|
| `Expected object of scalar type Long` | labels 不是 int64 | `labels.long()` |
| `CUDA out of memory` | batch 太大 | 减小 batch_size |
| `gradients are None` | 忘记 `requires_grad` 或用了 `no_grad` | 检查上下文 |
| loss 不变 | lr 太小或忘记 `step()` | 检查 optimizer |
| acc 随机猜 | 忘记 `model.train()` 或 label 错位 | 逐行 debug 一个 batch |

---

## 下一周预告（W2）

CNN、ResNet 迁移学习、CIFAR-10、为 VLN 准备「房间类型分类」小项目。W1 的 MNIST 流程会直接复用到 W2，只需把 MLP 换成 CNN。
