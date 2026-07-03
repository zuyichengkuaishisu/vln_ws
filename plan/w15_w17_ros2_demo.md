# W15–W17：ROS2 语义导航 Demo（定制版）

> **适用背景**：机器人工程本科 · 嵌入式 → ROS2-SLAM · 室内外激光 SLAM · Nav2 部分经验 · ROS-SLAM 国一  
> **总计划**：[`learning_plan.md`](learning_plan.md) 阶段 6  
> **笔记输出**：`notes/ros_demo_design.md`（设计决策）、`notes/w15_w17_log.md`（每周记录）  
> **代码仓库**：`/home/zy/.wzy/vln_ws`（colcon 工作区）

---

## 一、Demo 定位（简历一句话）

**「基于已有激光 SLAM 地图的分层语言导航：文本指令 → 语义地标匹配 → Nav2 全局规划」**

和纯 Habitat VLN 的区别：

| 纯 VLN benchmark | 你的 Demo |
|------------------|-----------|
| 离散动作 FORWARD/LEFT/RIGHT | Nav2 连续路径 + 局部避障 |
| 仿真 RGB 全景 | **真实 2D 激光地图 / 拓扑图** |
| 端到端神经网络 | **分层：语言层 + SLAM 地图层 + 导航栈** |
| 难讲工程落地 | **直接对接现职技术栈** |

面试叙事：

> DUET 解决「语言如何在仿真里选 waypoint」；我的 Demo 解决「语言如何接到 ROS2 Nav2 和激光 SLAM 地图」—— 这是公司上线会用的架构。

---

## 二、总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│  输入层：文本指令（CLI / RViz Panel / 简单 Web，三选一）      │
│  例："去充电区" / "go to the corridor entrance"             │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  instruction_parser（Python 节点）                           │
│  · Phase 1：关键词 + 同义词表（先跑通）                       │
│  · Phase 2：open_clip 图文匹配（W16 接入）                     │
│  输出：landmark_id 或 semantic_label                          │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  semantic_map_server（C++/Python 均可，建议 Python 先跑通）   │
│  · 加载 semantic_map.yaml（节点 id、label、pose、同义词）      │
│  · 查询：label → map 坐标系下 PoseStamped                     │
│  · 可选：可视化 MarkerArray 在 RViz 显示语义节点               │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  goal_bridge → Nav2                                            │
│  · 发布 /goal_pose（NavigateToPose）或调用 BasicNavigator API  │
│  · frame_id 与 SLAM map 一致（通常 map）                       │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  已有栈：map_server + amcl（或 SLAM）+ Nav2                    │
│  · 仿真：Gazebo + 2D 激光 + 预建 map                          │
│  · 实机（可选 W17+）：复用公司室内地图 + bag 定位              │
└─────────────────────────────────────────────────────────────┘
```

**刻意不做（控制 scope）：**

- 不做端到端 VLN 神经网络控制电机  
- 不做 ASR 语音（文本足够；语音写进 Sim2Real 待办）  
- W15–W17 不做室外大场景（室外放「扩展方向」幻灯片即可）

---

## 三、语义地图格式（核心设计）

用 **YAML** 描述拓扑语义节点，和你熟悉的 pose graph / 拓扑 SLAM 思维一致：

```yaml
# semantic_map.yaml
frame_id: map
landmarks:
  - id: charge_station
    label: 充电区
    aliases: [充电站, charging station, charge]
    pose: { x: 2.5, y: 1.0, yaw: 0.0 }

  - id: corridor_entrance
    label: 走廊入口
    aliases: [走廊, corridor, hallway entrance]
    pose: { x: 5.0, y: 3.2, yaw: 1.57 }

  - id: office_door
    label: 办公室门口
    aliases: [办公室, office, office door]
    pose: { x: 8.1, y: 2.0, yaw: -1.57 }
```

**地图来源（按优先级）：**

1. **推荐**：Gazebo 仿真环境 + `slam_toolbox` 离线建图 → 导出 `map.pgm` + 手工标 3–5 个节点  
2. **加分**：脱敏后的 **公司室内地图**（注意保密，公开 repo 用仿真地图）  
3. **不推荐 W15 起步**：Habitat 导出（坐标系/传感器和 Nav2 不一致，转换成本高）

**和 DUET 的呼应（面试用）：**

- DUET：语言 → 拓扑图上的 node  
- 你的 Demo：语言 → `semantic_map.yaml` 里的 landmark → Nav2 goal  

---

## 四、ROS2 包结构（vln_ws）

```text
vln_ws/
├── src/
│   ├── semantic_nav_msgs/          # 可选：自定义 srv/msg
│   ├── instruction_parser/         # 文本 → landmark_id
│   ├── semantic_map_server/        # yaml → PoseStamped
│   └── goal_bridge/                # landmark → Nav2 goal
├── maps/
│   ├── office_sim/                 # map.yaml + map.pgm
│   └── semantic_map.yaml
├── launch/
│   └── semantic_nav_demo.launch.py # 一键启动
└── docs/
    └── demo_architecture.png
```

**话题 / 服务设计（最小集）：**

| 接口 | 类型 | 说明 |
|------|------|------|
| `/semantic_nav/instruction` | `std_msgs/String` | 输入自然语言 |
| `/semantic_nav/goal_pose` | `geometry_msgs/PoseStamped` | 输出给 Nav2 |
| `/semantic_nav/markers` | `visualization_msgs/MarkerArray` | RViz 语义点 |
| `/parse_instruction` | Service（可选） | 同步解析，方便调试 |

---

## 五、分周计划

### W15：设计 + 离线跑通 + 仿真地图（21h）

**目标**：不依赖 Nav2，完成「文本 → 地图坐标」闭环；准备好仿真地图。

#### Day 1–2（Hour 1 理论 / Hour 2 动手 / Hour 3 文档）

- [ ] 写 `notes/ros_demo_design.md`：分层 vs 端到端、为何选 Nav2（结合你 Nav2 实习/现职经验）
- [ ] 画架构图（paper/pencil 或 draw.io），放 `docs/demo_architecture.png`
- [ ] 选定仿真场景：**Gazebo Classic 或 Ignition + TurtleBot3 / 自研差分底盘（和你公司接近更好）**

#### Day 3–4

- [ ] Gazebo 建简单 **室内** 场景（走廊 + 2–3 房间即可）
- [ ] `slam_toolbox` 或 `cartographer` 离线建图 → `maps/office_sim/`
- [ ] 在 RViz 里点选位置，手写 `semantic_map.yaml`（**至少 3 个 landmark**）

#### Day 5–6

- [ ] **`instruction_parser` Phase 1**：纯 Python，同义词表匹配

```python
# 逻辑示意
def parse(instruction: str, landmarks) -> str:
    text = instruction.lower()
    for lm in landmarks:
        if any(alias in text for alias in lm.all_names):
            return lm.id
    raise ValueError("unknown landmark")
```

- [ ] **`semantic_map_server`**：读 yaml，查 id → `PoseStamped`
- [ ] 单元测试：3 条指令打印正确坐标（**不接 ROS 也能测**）

#### Day 7

- [ ] 端到端离线脚本：`python tools/run_offline.py "去充电区"` → 打印 pose
- [ ] 记录失败 case：歧义词、多 landmark 匹配 → 笔记

**W15 验收：**

- [ ] 有 `map.pgm` + `semantic_map.yaml`  
- [ ] 离线 3 条指令解析正确  
- [ ] `notes/ros_demo_design.md` 完成  

---

### W16：ROS2 集成 + Nav2 跑通（21h）

**目标**：colcon 编译通过，RViz 里输入指令，机器人导航到目标。

#### Day 1–2

- [ ] 创建 3 个 ROS2 包（ament_python 即可，你 Python 届时已熟）
- [ ] `semantic_map_server` 节点：启动时加载 yaml，提供查询
- [ ] RViz2 `MarkerArray` 显示语义点（**嵌入式/SLAM 背景的优势：TF 要对**）

#### Day 3–4

- [ ] 启动栈（与你熟悉的一致）：

```bash
# 典型组合（Humble 示例，按你环境调整）
ros2 launch nav2_bringup tb3_simulation_launch.py
# 或 map_server + amcl + nav2（已有 map 时）
```

- [ ] **`goal_bridge`**：订阅解析结果 → 调 `nav2_simple_commander` 或发 Action

```python
# nav2_simple_commander 示意
from nav2_simple_commander.robot_navigator import BasicNavigator
navigator = BasicNavigator()
navigator.goToPose(goal_pose)
```

- [ ] 打通：**publish 字符串到 `/semantic_nav/instruction` → 车动起来**

#### Day 5

- [ ] **`instruction_parser` Phase 2（可选但推荐）**：`open_clip` 对 landmark 名称 + 预存文本模板算相似度  
  - 5090 上推理无压力  
  - 面试可说：「关键词兜底 + CLIP 模糊匹配」

#### Day 6–7

- [ ] 写 `launch/semantic_nav_demo.launch.py` 一键启动  
- [ ] 调试 TF：`map` → `base_link` 必须稳定  
- [ ] 至少 **3 条不同指令** 仿真导航成功，截图/录屏初版  

**W16 验收：**

- [ ] `colcon build` 无 erro  
- [ ] Gazebo/RViz 中 3/3 指令 Nav2 到达（位置误差 < 0.5m 即可）  
- [ ] launch 一条命令可复现  

---

### W17：Demo 打磨 + 作品集（21h）

**目标**：可对外展示的 GitHub + 5 分钟视频 + 面试故事。

#### Day 1–2：可靠性

- [ ] 增加 **未知指令** 处理（返回错误提示，不发给 Nav2）  
- [ ] 增加 **最近 landmark 消歧**（可选：「去最近的充电区」）  
- [ ] 日志：instruction → landmark_id → goal → nav2 result  

#### Day 3：视频（5 分钟结构）

| 时间 | 内容 |
|------|------|
| 0:00–0:30 | 问题：VLN benchmark vs 真实机器人 Nav2 |
| 0:30–1:30 | 架构图 + 分层讲解 |
| 1:30–4:00 | 3 条指令 live demo（RViz 可见语义点 + 路径） |
| 4:00–5:00 | 和 DUET / 现职 SLAM 的关系 + Sim2Real 待办 |

#### Day 4：README（面试官第一眼）

```markdown
# semantic-nav2-demo

Language-conditioned navigation on 2D LiDAR SLAM maps via ROS2 Nav2.

## Stack
ROS2 Humble | Nav2 | slam_toolbox | open_clip (optional)

## Quick start
ros2 launch semantic_nav_demo semantic_nav_demo.launch.py
ros2 topic pub /semantic_nav/instruction std_msgs/String "data: '去充电区'"

## Architecture
[diagram]

## Related work
- DUET (VLN benchmark) — topological map + language
- This demo — same idea on real Nav2 stack
```

#### Day 5：Sim2Real 待办（写进 README，展示工程思维）

- [ ] 实机：复用公司 **室内** 地图（脱敏）  
- [ ] 定位：AMCL / 现职 SLAM 定位方案  
- [ ] 传感器：2D 激光（你已熟悉）→ 无需改 Demo 架构  
- [ ] 室外：POI 路点 + GNSS/激光 SLAM 切换（**写「扩展」不写进 W17 scope**）  
- [ ] 语音：Whisper ASR → `/semantic_nav/instruction`  

#### Day 6–7：和 DUET 项目联动

- [ ] 在 DUET 复现报告里加一节：「仿真拓扑图 vs 本 Demo 的 semantic_map.yaml」  
- [ ] 简历 bullet 定稿（见下文）  
- [ ] 更新 `notes/w15_w17_log.md` 复盘  

**W17 验收：**

- [ ] GitHub 公开 repo（仿真地图可公开；公司地图勿上传）  
- [ ] 5 分钟 demo 视频（B 站/私有链接均可）  
- [ ] 面试官 15 分钟能讲完整故事  

---

## 六、技术选型（贴合你背景）

| 组件 | 推荐 | 理由 |
|------|------|------|
| ROS2 发行版 | **Humble**（LTS）或公司现用版本 | 与现职/实习一致优先 |
| 仿真 | Gazebo + 差分 drive | 贴近 AMR；嵌入式熟悉驱动 |
| 建图 | **slam_toolbox** | ROS2 常用，你 SLAM 背景易上手 |
| 定位 | AMCL + 已知 map | 标准 indoor 方案 |
| 导航 | **Nav2** | 你已有部分经验，Demo 核心 |
| 语言层 Phase 1 | 关键词 + aliases | 先工程跑通 |
| 语言层 Phase 2 | **open_clip** | 5090 可用；和 W6 CLIP 学习衔接 |
| 不建议 W15–17 | Habitat ↔ ROS 桥接 | 工作量大、面试收益低 |

---

## 七、三条验收指令（固定脚本）

Demo 视频必须包含以下 3 条（可改成你的地图里的实际 landmark）：

1. **「去充电区」** — 单一明确 landmark  
2. **「到走廊入口」** — 测试 aliases / CLIP  
3. **「go to office door」** — 测试英文 / 跨语言匹配  

**加分第四条（消歧）：**

4. 「去办公室」— 若只有一个 office landmark 应成功；若有多个应报错或追问（写进 README）

---

## 八、简历 bullet（W17 完成后直接用）

```text
· 设计并实现 ROS2 分层语义导航 Demo：自然语言 → 语义地标 → Nav2 全局规划；
  基于 2D 激光 SLAM 地图与拓扑地标 YAML，Gazebo 仿真 3 类指令导航成功。

· 复现 DUET (R2R)，val-unseen SR xx%；分析语言歧义/长程漂移等 failure case；
  对比 DUET 仿真拓扑图与实机 semantic_map 的工程落地差异。

· 现职/实习：室内外 2D 激光 SLAM + Nav2 导航工程；ROS-SLAM 国一。
```

---

## 九、风险与 fallback

| 风险 | fallback |
|------|----------|
| Nav2 调不通 | W16 先用 `rviz2` 手动发 goal 验证地图；再接入 bridge |
| Gazebo 仿真耗时 | 用 **bag 回放 + 静态 map**，不做动态仿真（对 SLAM 工程师完全 OK） |
| CLIP 接入慢 | Phase 1 关键词足够交付；CLIP 写「optional enhancement」 |
| 公司地图保密 | 公开 repo 仅仿真地图；实机视频本地面试播放 |
| 时间不够 | **砍掉 CLIP、砍掉 Gazebo**，用 bag+静态 map+Nav2 仍算完整 Demo |

---

## 十、与 20 周计划其他阶段的衔接

```text
W6  CLIP          → W16 Phase 2 语言匹配
W8  Habitat       → 理解 VLN 任务与指标（Demo 不依赖 Habitat）
W9–12 DUET        → 拓扑图语言 grounding 理论；W17 写对比
W13 VLN-CE        → 面试解释「离散→连续」；Demo 就是连续 Nav2
W18 Portfolio     → 本 Demo 为 repo #2：semantic-nav2-demo
```

---

## 十一、进度追踪

| 周 | 关键产出 | 完成 | 备注 |
|----|----------|------|------|
| W15 | semantic_map.yaml + 离线解析 | | |
| W15 | ros_demo_design.md | | |
| W16 | 3 个 ROS2 包 + Nav2 跑通 | | |
| W16 | launch 一键启动 | | |
| W17 | 5min 视频 + GitHub README | | |
| W17 | 简历 bullet 定稿 | | |
