# VLN 学习计划

> **背景**：SLAM 算法工程师，转岗 / 机器人落地 / 兴趣学习  
> **硬件**：RTX 5090  
> **基础**：PyTorch、Transformer 不熟悉（需从基础补起）  
> **时间**：每天 3h（≈ 21h/周，共 20 周 ≈ 420h）  
> **工作区**：`/home/zy/.wzy/vln_ws`（ROS2 colcon 结构，后期用于落地 demo）

---

## 一、画像与策略

| 项 | 情况 | 策略 |
|----|------|------|
| GPU | RTX 5090 | 后期可开大 batch、跑 CLIP/小 VLM、Habitat 多进程渲染 |
| 基础 | PyTorch/Transformer 不熟 | 前 6 周专补，占总量约 30% |
| 背景 | SLAM 工程师 | 第 10 周起走「地图/分层导航」路线，差异化竞争力 |
| 目标 | 转岗 + 落地 + 兴趣 | 中期 benchmark 复现（转岗），后期 ROS2 demo（落地） |
| 时间 | 3h/天 ≈ 21h/周 | 按「学-练-记」固定节奏，避免只看不写 |

### 三条目标如何兼顾

- **转岗**：第 8–14 周完成 1 个可写进简历的 VLN 复现 + 实验报告
- **落地**：第 15–20 周做「语言 → 语义目标 → Nav2」分层 demo
- **兴趣**：每周留 2–3h 读 1 篇论文 + 可视化 agent 轨迹，保持动力

### SLAM 背景 vs VLN 要求

| 已有 (SLAM) | 需加强 | VLN 特有 |
|-------------|--------|----------|
| 几何/位姿 | Transformer/VLM | 语言 grounding |
| 建图/定位 | IL/RL 基础 | R2R 评测协议 |
| ROS/导航栈 | Habitat 生态 | Teacher forcing 问题 |
| 传感器模型 | PyTorch 训练 | 多模态预训练 |
| 路径规划 | 论文复现能力 | Unseen 泛化 |

---

## 二、每日 3 小时固定模板

```text
Hour 1  理论：视频/论文/笔记（输入）
Hour 2  动手：跟敲代码 / 跑实验（输出）
Hour 3  巩固：改参数、写总结、整理到 notes/（沉淀）
```

周末可选：其中 1 天做 **Weekly Review**（2h 复盘 + 1h 自由探索）。

---

## 三、总路线图（20 周）

```text
W1–W4   PyTorch + 深度学习基础
W5–W6   Transformer + 多模态入门（CLIP）
W7–W8   VLN 问题定义 + Habitat 跑通
W9–W12  复现 DUET（转岗核心项目）
W13–W14 VLN-CE 连续导航 + 失败案例分析
W15–W17 ROS2 分层导航 demo（落地项目）
W18–W20 简历/portfolio + 面试准备 + 1 个小改进
```

---

## 四、分阶段详细计划

### 阶段 0：VLN 心智模型（阅读即可，穿插 W7）

**VLN 定义**：智能体根据自然语言指令，在环境中利用视觉观测自主导航到目标。

**与 SLAM/经典导航的差异**：

| 维度 | SLAM / 经典导航 | VLN |
|------|----------------|-----|
| 输入 | 传感器 + 地图/代价图 | RGB(D) + 语言指令 |
| 输出 | 位姿/轨迹/路径点 | 离散动作或连续控制 |
| 核心难点 | 几何一致性、漂移、回环 | 语言 grounding、长程推理、泛化 |
| 评测 | ATE/RPE、定位误差 | SR、SPL、NE、nDTW |
| 训练 | 大多无大规模标注轨迹 | 依赖仿真器 + 语言标注数据 |

**核心指标**：

- **SR (Success Rate)**：是否到达目标（通常 NE < 3m）
- **SPL (Success weighted by Path Length)**：成功且路径效率（惩罚绕路）
- **NE (Navigation Error)**：终点到 goal 的距离误差

---

### 阶段 1：PyTorch 基础（W1–W4）

**目标**：能独立写训练循环，理解 tensor/autograd/DataLoader。

#### W1：Python 数值计算 + PyTorch 入门

| 天 | Hour 1 | Hour 2 | Hour 3 |
|----|--------|--------|--------|
| D1 | 张量概念：shape/dtype/device | 安装 PyTorch，tensor 创建与运算 | 把 SLAM 里一个公式（如 SE3）用 tensor 实现 |
| D2 | autograd 原理 | 手动求导 vs `backward()` | 写线性回归 from scratch |
| D3 | `nn.Module` / `optim` | 搭 2 层 MLP 分类 MNIST | 改 lr、batch size，画 loss 曲线 |
| D4 | Dataset / DataLoader | 自定义 Dataset（读图片+标签） | 总结：Dataset 三要素 `__len__` `__getitem__` |
| D5 | GPU 训练流程 | 把 MNIST 训练迁到 CUDA | 测 5090 上不同 batch size 的吞吐 |
| D6–D7 | 复习 W1 | 不看教程独立完成 MNIST 训练脚本 | 跟敲 `plan/week01_pytorch.md`，复盘 → `notes/week01_review.md` |

**资源**：[PyTorch 官方 Tutorial](https://pytorch.org/tutorials/)

**验收**：不看代码，30 分钟内写出「读数据 → 训练 → 保存模型 → 推理」完整脚本。

#### W2：CNN + 视觉基础

| 天 | 内容 |
|----|------|
| D1–D2 | Conv/Pool/ResNet 结构；用 `torchvision.models.resnet18` 做迁移学习 |
| D3 | 数据增强：`RandomResizedCrop`, `Normalize` |
| D4 | 训练/验证 split，early stopping 概念 |
| D5 | 在 CIFAR-10 或小型分类任务上达到合理 accuracy |
| D6–D7 | **小项目**：训练「房间类型分类器」（kitchen/bedroom/...），为 VLN 预热 |

#### W3：序列模型入门（为 Transformer 铺垫）

| 天 | 内容 |
|----|------|
| D1–D2 | RNN/LSTM 基本原理（理解「序列编码」即可） |
| D3 | Embedding + 序列 padding / pack |
| D4 | 用 LSTM 做简单文本分类或命名实体 |
| D5 | Attention 直觉：Q/K/V 手算一个小例子 |
| D6–D7 | 读 [Attention Is All You Need](https://arxiv.org/abs/1706.0372) 前半，对照图解 |

**验收**：能口头解释「为什么 RNN 长序列会梯度消失，Attention 如何缓解」。

#### W4：PyTorch 工程化 + 调试

| 天 | 内容 |
|----|------|
| D1 | `tensorboard` / `wandb` 日志 |
| D2 | checkpoint 保存加载，`state_dict` vs 整模型 |
| D3 | 常见 bug：shape 不匹配、device 不一致、忘记 `model.train()` |
| D4 | 读一个开源 repo 的训练脚本结构（`train.py` / `config.yaml`） |
| D5–D7 | **综合练习**：Fork 简单 CV 项目，改 config 跑通训练 |

**阶段结束标志**：看到 `loss.backward(); optimizer.step()` 知道每一行在干什么。

---

### 阶段 2：Transformer + 多模态（W5–W6）

**目标**：理解 VLN 模型的「语言编码 + 视觉编码 + 跨模态融合」三板斧。

#### W5：Transformer 精读 + 实现

| 天 | 内容 |
|----|------|
| D1 | Multi-Head Attention、Positional Encoding 公式推导 |
| D2 | Encoder/Decoder 结构；BERT vs GPT 区别 |
| D3 | 用 PyTorch 实现 scaled dot-product attention（~30 行） |
| D4 | 使用 `transformers` 库：`AutoTokenizer`, `AutoModel` |
| D5 | BERT 提文本特征：`input_ids` → `[CLS]` embedding |
| D6–D7 | 读 **VLN-BERT** 论文 Abstract + Method 图，标注各模块对应 Transformer 哪部分 |

**资源**：李宏毅 Transformer 视频 + [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

#### W6：CLIP + 视觉-语言对齐

| 天 | 内容 |
|----|------|
| D1 | 读 CLIP 论文（对比学习、image-text pair） |
| D2 | 跑 OpenAI CLIP 或 `open_clip`：给定图片+文本，算相似度 |
| D3 | 实验：同一场景不同描述（"kitchen" vs "bedroom"）的 score 差异 |
| D4 | ViT 简介：patch embedding 与 CNN 的差异 |
| D5 | 小结：VLN 里「语言 grounding」≈ CLIP 式跨模态匹配 |
| D6–D7 | 写 `notes/week06_transformer_clip.md`：画 VLN 模型通用架构图 |

**验收**：能解释 VLN-BERT 里 language/vision encoder、cross-modal fusion 的输入输出。

---

### 阶段 3：VLN 入门 + Habitat（W7–W8）

**目标**：理解任务与指标，在仿真里看到 agent 按指令走路。

#### W7：论文 + 概念

| 天 | 内容 |
|----|------|
| D1 | 精读 **R2R (2018)**：任务定义、动作空间、数据集划分 |
| D2 | 指标：SR、SPL、NE 公式手推一遍 |
| D3 | 读 **RCM (2019)** 或 Seq2Seq baseline 方法图 |
| D4 | 了解 seen/unseen split、teacher forcing 问题 |
| D5 | 浏览 VLN 综述（2022–2024），列出 5 个主流方向 |
| D6–D7 | 整理「VLN vs SLAM」对比表 → `notes/vln_metrics.md` |

**必读论文顺序**：

1. R2R (Anderson et al., 2018) — 任务定义
2. RCM (2019) — 经典 IL+RL
3. VLN-BERT (2020) — 预训练 + 历史编码
4. DUET (2022) — 拓扑/语义地图 + 语言

#### W8：Habitat 环境搭建

| 天 | 内容 |
|----|------|
| D1–D2 | 安装 Habitat-Sim + Habitat-Lab（conda，CUDA 12.x） |
| D3 | 下载 Matterport3D 场景子集，跑 random agent |
| D4 | 可视化：RGB 观测 + top-down map + 指令文本 |
| D5 | 跑官方 VLN baseline 的 **evaluation only**（预训练权重） |
| D6–D7 | 记录 val-unseen SR/SPL；截图 3 success + 3 failure case → `notes/week08_habitat.md` |

**验收**：本机跑通 eval，能口头解释一条 R2R 指令从输入到 STOP 的完整数据流。

---

### 阶段 4：核心项目——复现 DUET（W9–W12）

**目标**：简历可写——「复现 DUET，val-unseen SR 达到论文 ±2%」。

**为何选 DUET**：拓扑/语义地图与 SLAM 思维接近，转岗叙事好；比 NavGPT 不依赖 LLM API，适合 primeiro 复现。

#### W9：代码架构啃读

- 克隆 [DUET](https://github.com/chenjshn/DUET) 仓库，理清目录
- 画数据流：instruction → BERT → panorama features → graph map → action
- 对照论文 Figure 2，每个模块标注 tensor shape

#### W10–W11：训练与调试

- 先用作者预训练权重跑 eval，对齐论文数字
- 再跑 fine-tune（5090 可开较大 batch + `amp` 混合精度）
- 常见坑：MP3D 路径、Habitat 版本、feature 预提取文件

#### W12：实验报告

写 5–8 页报告 → `notes/duet_repro_report.md`：

1. 方法概述（自己的话）
2. 复现环境与配置
3. val-seen / val-unseen 结果对比表
4. 3 类 failure case 分析（语言歧义 / 视觉相似 / 长程漂移）
5. **SLAM 视角讨论**：DUET 的 graph 与拓扑 SLAM 的异同

**验收**：GitHub 有 README + 可复现命令 + 结果截图；报告可面试讲 15 分钟。

---

### 阶段 5：连续导航 + 前沿（W13–W14）

**目标**：从 discrete benchmark 过渡到「更像机器人」的设置。

#### W13：VLN-CE

- 读 VLN-CE 论文：离散 → 连续动作空间
- 了解 waypoint predictor + 低层 controller
- 跑 [VLN-CE](https://github.com/jacobkrantz/VLN-CE) baseline eval

#### W14：大模型 VLN（广度了解）

- 读 NavGPT / DiscussNav 之一（泛读）
- 理解分层架构：**LLM 规划 subgoal → 经典导航执行**
- 写笔记 → `notes/ros_demo_design.md`：「若在公司落地，我会选分层而非端到端」

---

### 阶段 6：ROS2 落地 Demo（W15–W17）

**目标**：把 VLN 接到熟悉的 ROS2 / 导航栈——SLAM 工程师的差异化作品。

> **定制方案（机器人工程 + ROS2-SLAM + 室内外激光 + Nav2 背景）**  
> 详见 **[`w15_w17_ros2_demo.md`](w15_w17_ros2_demo.md)**：基于 **2D 激光 SLAM 地图 + semantic_map.yaml + Nav2** 的分层语义导航，不用 Habitat 桥接。

#### 推荐架构

```text
文本指令
    ↓
[instruction_parser] 关键词 → open_clip（Phase 2）
    ↓
[semantic_map_server] 拓扑地标 YAML（衔接 DUET 思路）
    ↓
[goal_bridge] → Nav2 NavigateToPose
    ↓
map_server + AMCL/SLAM + Nav2（现有栈）
```

#### W15：仿真地图 + 离线「文本 → 坐标」

- Gazebo 室内场景 + `slam_toolbox` 建图 → `maps/office_sim/`
- 手写 `semantic_map.yaml`（≥3 个 landmark）
- `instruction_parser` Phase 1（关键词/同义词）+ 离线测试
- 笔记 → `notes/ros_demo_design.md`

#### W16：ROS2 集成（vln_ws）

- 包：`instruction_parser` / `semantic_map_server` / `goal_bridge`
- Nav2 + RViz MarkerArray；可选 open_clip 模糊匹配
- `launch/semantic_nav_demo.launch.py` 一键启动

#### W17：Demo 打磨

- 3 条固定验收指令 + 5 分钟视频
- GitHub `semantic-nav2-demo` + Sim2Real 待办（实机室内地图、室外 POI 写扩展）
- 与 DUET 复现报告对比「仿真拓扑 vs 实机 semantic_map」

**验收**：能打开视频向面试官讲清楚分层方案；详见 [`w15_w17_ros2_demo.md`](w15_w17_ros2_demo.md) 验收表。

---

### 阶段 7：求职准备（W18–W20）

#### W18：Portfolio 整理

GitHub 两个 repo：

1. **vln-duet-repro**：benchmark 复现（学术能力）
2. **vln-ros-demo**：分层导航 demo（工程落地）

简历 bullet 示例：

- 复现 DUET，R2R val-unseen SR xx%，分析 unseen 泛化失败模式
- 设计 language-to-Nav2 分层架构，ROS2 仿真 demo 验证 x 条指令

#### W19：面试知识库 → `notes/interview_qa.md`

准备能答的问题：

- R2R 指标含义？SPL 为什么比 SR 更严格？
- Teacher forcing 问题及常见解法（scheduled sampling、DAgger）
- DUET 的 map 表示 vs SLAM occupancy grid
- 端到端 VLN vs 分层导航的 trade-off
- 5090 上训练时如何调 batch size / mixed precision

#### W20：小改进（加分项）

选一个方向做 1 周小改动（不必 SOTA）：

- 把 DUET 的 map 模块换成 pose graph 表示
- 加入 depth 特征对比 RGB-only 的 ablation
- 用 CLIP 做 instruction-landmark 在线匹配

---

## 五、5090 使用建议

| 阶段 | 建议 |
|------|------|
| W1–W6 | 小模型练手，重点是写对代码 |
| W9–W12 | DUET fine-tune 可 `batch_size=32~64`，开 `amp` |
| W15+ | 可同时开 Habitat 渲染 + 小 VLM；CPU 瓶颈时加 `num_workers` |
| 环境 | CUDA 12.x + PyTorch 2.x，与 Habitat 版本对齐后再装 |

---

## 六、资源清单

| 阶段 | 资源 |
|------|------|
| PyTorch | [pytorch.org/tutorials](https://pytorch.org/tutorials/) |
| Transformer | 李宏毅视频 + [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) |
| CLIP | [open_clip](https://github.com/mlfoundations/open_clip) |
| VLN 论文 | R2R → RCM → VLN-BERT → DUET → VLN-CE |
| 仿真 | [habitat-lab](https://github.com/facebookresearch/habitat-lab) |
| 连续 VLN | [VLN-CE](https://github.com/jacobkrantz/VLN-CE) |
| 地图 VLN | [DUET](https://github.com/chenjshn/DUET) |
| ROS2 Demo | [`w15_w17_ros2_demo.md`](w15_w17_ros2_demo.md) · Nav2 · slam_toolbox |
| 落地 | Nav2 文档 + 现有 SLAM 栈经验 |

---

## 七、常见坑

1. **过分依赖 GPS/Compass**：benchmark 里常有，真实 VLN 要逐步去掉 oracle
2. **用 SLAM 指标评 VLN**：应看 SR/SPL，不是 ATE
3. **忽视 unseen split**：train/val seen 很高不代表会泛化
4. **环境版本不一致**：Habitat、MP3D、PyTorch 版本对结果影响大
5. **只做离散动作**：工业界更关心 continuous + 真实平台
6. **前 4 周觉得慢**：正常；基础不牢后面复现会反复卡住
7. **Habitat 安装失败**：预留 W8 整周；可先用 Docker 官方镜像

---

## 八、目录结构

```text
plan/                       ← 学习计划与跟敲清单
  learning_plan.md
  week01_pytorch.md
  w15_w17_ros2_demo.md      ← W15–W17 ROS2 语义导航 Demo（定制）
  week02_*.md               ← 后续每周清单（按需添加）

notes/                      ← 个人笔记（你自己写）
  week01_review.md
  week06_transformer_clip.md
  vln_metrics.md
  week08_habitat.md
  duet_repro_report.md
  ros_demo_design.md
  interview_qa.md

scripts/                    ← 练习与实验代码
  week01/
```

---

## 九、20 周后能力 checklist

- [ ] 独立写 PyTorch 训练与调试
- [ ] 讲清 Transformer 在 VLN 里的作用
- [ ] 跑通 Habitat + R2R 评测
- [ ] 1 个 DUET 级复现 + 实验报告
- [ ] 1 个 ROS2 分层 VLN demo
- [ ] 面试能聊 SR/SPL、地图式 VLN、落地架构

---

## 十、进度追踪

| 周次 | 主题 | 计划产出 | 完成日期 | 备注 |
|------|------|----------|----------|------|
| W1 | PyTorch 入门 | plan/week01_pytorch.md + notes/week01_review.md | | |
| W2 | CNN | 房间分类小项目 | | |
| W3 | RNN/Attention | Attention 手算笔记 | | |
| W4 | 工程化 | 跑通开源 CV 项目 | | |
| W5 | Transformer | VLN-BERT 论文笔记 | | |
| W6 | CLIP | notes/week06_transformer_clip.md | | |
| W7 | VLN 概念 | notes/vln_metrics.md | | |
| W8 | Habitat | notes/week08_habitat.md | | |
| W9 | DUET 读码 | 数据流图 | | |
| W10–W11 | DUET 训练 | eval 对齐论文 | | |
| W12 | 实验报告 | notes/duet_repro_report.md | | |
| W13 | VLN-CE | eval 结果 | | |
| W14 | 大模型 VLN | notes/ros_demo_design.md | | |
| W15–W17 | ROS2 demo | 视频 + README | | |
| W18–W20 | 求职准备 | notes/interview_qa.md + portfolio | | |
