# DINOv3 + ViB 消融实验设置

## 1. 实验目的

本组消融实验旨在回答以下核心问题：

1. DINOv3-ViB 的性能提升是否仅来源于更大的分类头参数量？
2. 性能增益是否仅来源于更强的非线性映射能力，而非信息瓶颈机制本身？
3. 特征压缩是否对跨生成器泛化具有关键作用？
4. 随机瓶颈与显式 KL 正则是否分别带来额外贡献？

围绕上述问题，本文在统一的 DINOv3 冻结特征框架下，对不同检测头进行系统对照，以分离“参数规模”“非线性映射”“确定性压缩”“随机瓶颈”与“信息瓶颈正则”各自的作用。

---

## 2. 统一实验框架

### 2.1 Backbone 设置

所有实验均采用相同的 DINOv3 视觉基础模型作为特征提取骨干网络，并保持骨干参数完全冻结，仅训练检测头。  
输入图像首先送入冻结的 DINOv3 backbone 提取视觉特征，再经过统一的 token pooling 得到图像级表示，随后进行 L2 normalization，最后输入不同类型的分类头进行真假判别。

统一流程如下：

`Image -> Frozen DINOv3 -> Token Pooling -> L2 Normalize -> Detection Head -> Logit`

### 2.2 Pooling 策略

为避免额外变量干扰，本组主消融实验固定采用同一种 pooling 策略。  
推荐默认设置为：

- `pool_type = cls`

### 2.3 特征归一化

所有分类头统一使用 backbone 输出特征的 L2 normalization，以保证对比公平，避免由于特征范数差异带来额外偏置。  
即不同头部方法的唯一差异来自头部结构与损失设计，而非输入特征尺度不同。

---

## 3. 对比方法设计

为系统分析性能提升来源，本文设置以下五种检测头：

### 3.1 Linear

最基础的线性探测器：

`h -> Linear -> y`

作用：
- 作为最简单基线

### 3.2 MLP

两层感知机分类头：

`h -> Linear(in_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, 1)`

作用：
- 检验性能提升是否仅来源于更强的非线性映射能力

### 3.3 Bottleneck-MLP

带确定性压缩的 MLP 头：

`h -> Linear(in_dim, bottleneck_dim) -> GELU -> Dropout -> Linear(bottleneck_dim, 1)`

作用：
- 检验“仅靠维度压缩”是否已经足以带来明显收益
- 验证通用视觉特征中是否确实存在任务无关冗余

### 3.4 ViB-no-KL

保留随机瓶颈结构，但不引入 KL 正则项：

`h -> Encoder -> (mu, logvar) -> reparameterization -> z -> classifier`

训练损失仅包含分类损失：

`L = L_cls`

作用：
- 检验收益是否仅来自 stochastic bottleneck 或更复杂结构
- 排除“ViB 只是随机扰动正则化”的解释

### 3.5 ViB

完整的变分信息瓶颈头：

`h -> Encoder -> (mu, logvar) -> reparameterization -> z -> classifier`

训练目标为：

`L = L_cls + beta * KL(q(z|x) || N(0, I))`

作用：
- 验证显式信息压缩约束是否带来额外收益
- 检验 ViB 是否能更有效地过滤任务无关冗余特征，提升未见分布泛化能力

---

## 4. 参数量对齐原则

为避免审稿人将性能提升简单归因于“额外参数”，本文在设计各类头部时尽量控制参数规模接近，尤其重点对齐以下三组：

- `MLP` vs `ViB`
- `Bottleneck-MLP` vs `ViB-no-KL / ViB`
- `ViB-no-KL` vs `ViB`

### 4.1 参数量统计

所有实验均报告检测头的可训练参数量（trainable parameters），不计入冻结 backbone 参数。

参数统计方式为：

`#Params = sum(p.numel() for p in head.parameters() if p.requires_grad)`

### 4.2 推荐设置

若 DINOv3 输出特征维度为 `in_dim = 1024`，推荐初始配置如下：

- Linear: 无隐藏层
- MLP: `hidden_dim = 2048` 升维
- Bottleneck-MLP: `bottleneck_dim = 256` 降维
- ViB-no-KL: `hidden_dim = 256, latent_dim = 128`
- ViB: `hidden_dim = 256, latent_dim = 128`

最终以实际参数量打印结果为准，并可在不改变实验逻辑的前提下微调 hidden_dim / latent_dim 使其更加接近。

---

### 5.1 分类损失

所有方法均采用相同的二分类损失：

- `BCEWithLogitsLoss`

### 5.2 ViB 额外正则

仅在完整 `ViB` 中加入 KL 正则项：

`L_total = L_cls + beta * L_kl`

其中 `beta` 为信息压缩强度控制系数。

---


## 6. 预期结论与判别逻辑

本组消融实验的结论判别逻辑如下：

### 情况 A：若 `MLP > Linear`
说明更强非线性分类器确实有帮助，但不能说明信息瓶颈有效。

### 情况 B：若 `Bottleneck-MLP > MLP`
说明压缩特征比单纯增强容量更重要，支持“冗余特征”假设。

### 情况 C：若 `ViB-no-KL > Bottleneck-MLP`
说明随机瓶颈结构本身可能带来一定正则化效果。

### 情况 D：若 `ViB > ViB-no-KL`
说明性能增益不能仅归因于额外参数、随机采样或结构复杂度，而与显式 KL 信息约束密切相关。

### 最理想结果模式

若实验结果满足以下趋势：

`Linear < MLP ≈ Bottleneck-MLP < ViB-no-KL < ViB`

或至少满足：

`ViB > ViB-no-KL ≈ MLP`

则可以较有力地支持如下结论：

1. ViB 的收益并非仅来自额外参数量；
2. ViB 的收益并非仅来自更复杂的非线性映射；
3. 显式信息瓶颈约束在困难未见分布上具有关键作用；
4. ViB 更符合“从通用视觉表征中提纯任务相关鉴伪特征”的方法动机。

---
通过引入`Linear`、 `MLP`、`Bottleneck-MLP`、 与 `ViB` 四类结构对照，并控制参数量与训练协议一致，本文能够更严格地区分“容量提升”“确定性压缩”“随机瓶颈”与“显式信息瓶颈约束”的作用，从而使 ViB 的有效性论证更加完整和严谨。