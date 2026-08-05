# 深度学习与目标检测基础概念

## 1. 训练流程中的基础概念
### Batch Size（批量大小）
模型在一次前向传播和反向传播中，同时处理的样本数量。

### Epoch（轮数）
把整个训练集完整遍历一遍，称为 1 个 epoch。

### Iteration（迭代次数）
模型完成 **一次参数更新**，通常称为 1 次 iteration。

在一个 epoch 中：

`iteration 数 = 训练集样本数 / batch size`

如果不能整除，通常应写为：

`iteration 数 = ceil(训练集样本数 / batch size)`

例如：
- 训练集有 1000 张图
- `batch size = 100`
- 那么 1 个 epoch 有 10 个 iteration

如果：
- 训练集有 1050 张图
- `batch size = 100`
- 那么 1 个 epoch 有 11 个 iteration

### 易错点
- `epoch` 不是一次参数更新
- `iteration` 不是“把整个数据集遍历一次”
- 正确理解应为：**一次 epoch 包含多次 iteration**

## 2. 数据集划分
### Training Set（训练集）
用于训练模型参数的数据集。

### Validation Set（验证集）
用于训练过程中评估模型效果、调超参数、选择最佳模型的数据集。

### Test Set（测试集）
用于训练完成后，最终评估模型泛化能力的数据集。

### 易错点
- 验证集用于“训练过程中的选择”
- 测试集用于“训练结束后的最终评估”
- 测试集不应参与调参，否则会造成评估结果失真

## 3. 目标检测中的常见评估指标
### IoU（Intersection over Union）
交并比，用于衡量预测框和真实框的重叠程度。

公式：

`IoU = 预测框与真实框的交集面积 / 预测框与真实框的并集面积`

IoU 越大，说明预测框与真实框越接近。

### TP / FP / FN
在目标检测中，通常先设定一个 IoU 阈值（例如 0.5），再判断预测结果属于哪一类：

- TP（True Positive）：正确检测到目标
- FP（False Positive）：错误检测到目标，或检测框与真实框不匹配
- FN（False Negative）：真实目标存在，但模型没有检测出来

### Precision（精确率）
表示“模型检测出来的目标中，有多少是真的”。

公式：

`Precision = TP / (TP + FP)`

### Recall（召回率）
表示“所有真实目标中，有多少被模型找出来了”。

公式：

`Recall = TP / (TP + FN)`

## 4. AP 与 mAP
### AP（Average Precision）
AP 是某一个类别在不同阈值下，综合 Precision-Recall 曲线得到的平均精度。

它不是 `Area Precision`，而是 `Average Precision`。

### mAP（mean Average Precision）
mAP 是对多个类别的 AP 再求平均。

它是目标检测中最常用的综合指标之一。

常见写法包括：
- `mAP@0.5`：IoU 阈值为 0.5 时的 mAP
- `mAP@0.5:0.95`：在多个 IoU 阈值下求平均，更严格，也更常见于 COCO 评估

## 5. 最终记忆版
- `batch size`：一次喂给模型多少样本
- `iteration`：一次参数更新
- `epoch`：整个训练集完整过一遍
- `Precision`：检出来的有多准
- `Recall`：该检出来的找到了多少
- `AP`：单类别检测效果
- `mAP`：多类别平均检测效果
