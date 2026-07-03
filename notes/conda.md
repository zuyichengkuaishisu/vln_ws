# 环境配置

## Conda

```bash
# 创建 VLN 学习环境
conda create -n vln python=3.10 -y
conda activate vln
```

安装路径：`~/miniconda3`，环境列表 `conda env list`。

---

## PyTorch（RTX 5090）

5090 为 Blackwell 架构（sm_120），**必须用 cu128**，不要用 cu124。

```bash
conda activate vln
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 验证

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('cuda build:', torch.version.cuda)
print('available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
print('matmul ok:', (x @ x.T).shape)
"
```

期望：`cuda build: 12.8`，设备名含 `5090`，matmul 不报错。

---

## 本机硬件

| 项 | 值 |
|----|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| 驱动 | 580.x（支持 CUDA 13.0） |
| 说明 | 驱动够新即可；PyTorch 自带 CUDA 12.8 运行时，无需单独装 nvcc |

---

## 常用命令

```bash
conda activate vln          # 进入环境
conda deactivate            # 退出
conda env list              # 查看所有环境
pip list | grep torch       # 查看 PyTorch 版本
```
