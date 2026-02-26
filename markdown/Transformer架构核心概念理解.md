# Transformer 架构核心概念详解

> 对应代码文件：`1-3-LLMBase.js`  
> 本文档详细解释 Transformer 架构中的所有核心概念

---

## 📋 目录

1. [整体架构](#1-整体架构)
2. [位置编码 (Positional Encoding)](#2-位置编码-positional-encoding)
3. [注意力机制 (Attention Mechanism)](#3-注意力机制-attention-mechanism)
4. [多头注意力 (Multi-Head Attention)](#4-多头注意力-multi-head-attention)
5. [前馈网络 (Feed-Forward Network)](#5-前馈网络-feed-forward-network)
6. [编码器层 (Encoder Layer)](#6-编码器层-encoder-layer)
7. [解码器层 (Decoder Layer)](#7-解码器层-decoder-layer)
8. [残差连接与层归一化](#8-残差连接与层归一化)
9. [掩码机制 (Masking)](#9-掩码机制-masking)
10. [完整流程示例](#10-完整流程示例)

---

## 1. 整体架构

### 1.1 什么是 Transformer？

Transformer 是一种**基于注意力机制**的神经网络架构，由 Google 在 2017 年论文 "Attention is All You Need" 中提出。

**核心特点**：
- ✅ 完全基于注意力机制（不使用 RNN/LSTM）
- ✅ 可以并行处理序列（训练速度快）
- ✅ 能够捕捉长距离依赖关系
- ✅ 现代 LLM（GPT、BERT）的基础架构

### 1.2 基本结构

```
┌─────────────────────────────────────┐
│        输入序列（Source）            │
│    "我爱北京天安门"                  │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────┐
        │   词嵌入层    │  ← 将词转换为向量
        │ + 位置编码   │  ← 添加位置信息
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  编码器 (×N)  │  ← 理解输入
        │  - 自注意力   │
        │  - 前馈网络   │
        └──────┬───────┘
               ↓
        编码器输出
               ↓
        ┌──────────────┐
        │  解码器 (×N)  │  ← 生成输出
        │  - 掩码注意力 │
        │  - 交叉注意力 │  ← 连接编码器
        │  - 前馈网络   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │   输出层      │  ← 预测下一个词
        └──────┬───────┘
               ↓
┌─────────────────────────────────────┐
│        输出序列（Target）            │
│    "I love Beijing Tiananmen"       │
└─────────────────────────────────────┘
```

### 1.3 三种变体

| 类型 | 结构 | 代表模型 | 应用场景 |
|------|------|----------|----------|
| **仅编码器** | 只有编码器部分 | BERT, RoBERTa | 文本分类、情感分析、命名实体识别 |
| **仅解码器** | 只有解码器部分 | GPT, ChatGPT | 文本生成、对话系统、代码生成 |
| **编码器-解码器** | 完整结构 | T5, BART | 机器翻译、文本摘要、问答系统 |

**你的 Agent 使用的是 GPT（仅解码器）架构！**

---

## 2. 位置编码 (Positional Encoding)

### 2.1 为什么需要位置编码？

**问题**：Transformer 的注意力机制本身是**位置无关**的。

```javascript
// 这两个句子在没有位置信息时，注意力计算结果是一样的！
句子1: "我 爱 你"
句子2: "你 爱 我"
```

打乱顺序后，如果没有位置编码，模型无法区分它们。

### 2.2 位置编码公式

使用 **sin 和 cos 函数**生成位置编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    ← 偶数维度
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    ← 奇数维度
```

**参数说明**：
- `pos`：位置索引（0, 1, 2, ...）
- `i`：维度索引（0, 1, 2, ..., d_model/2）
- `d_model`：模型维度（通常 512）

### 2.3 为什么用 sin/cos？

#### ✅ 优点 1：周期性模式

```javascript
// 不同位置的编码有规律的周期性变化
位置 0: [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
位置 1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
位置 2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
```

#### ✅ 优点 2：相对位置关系

利用三角函数性质，模型可以学习相对位置：

```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
```

这意味着 `PE(pos+k)` 可以表示为 `PE(pos)` 的线性组合！

#### ✅ 优点 3：外推性

可以处理比训练时更长的序列。

### 2.4 代码实现映射

```javascript
// 对应代码：第 19-119 行
class PositionalEncoding {
    constructor(dModel = 512, dropout = 0.1, maxLen = 5000) {
        // maxLen: 支持的最大序列长度
        this.pe = this.createPositionalEncoding();
    }
    
    createPositionalEncoding() {
        // 预计算所有位置的编码矩阵 (maxLen × d_model)
        for (let pos = 0; pos < this.maxLen; pos++) {
            for (let i = 0; i < this.dModel; i++) {
                const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / this.dModel);
                
                if (i % 2 === 0) {
                    pe[pos][i] = Math.sin(angle);  // 偶数维度
                } else {
                    pe[pos][i] = Math.cos(angle);  // 奇数维度
                }
            }
        }
    }
    
    forward(x) {
        // 将位置编码加到词嵌入上
        return x.map((row, pos) => row.map((val, i) => val + this.pe[pos][i]));
    }
}
```

### 2.5 可视化示例

```
位置编码模式（前 5 个位置，前 8 个维度）:
pos\dim  0       1       2       3       4       5       6       7
0        0.000   1.000   0.000   1.000   0.000   1.000   0.000   1.000
1        0.841   0.540   0.010   1.000   0.001   1.000   0.000   1.000
2        0.909  -0.416   0.020   1.000   0.002   1.000   0.000   1.000
3        0.141  -0.990   0.030   1.000   0.003   1.000   0.000   1.000
4       -0.757  -0.654   0.040   0.999   0.004   1.000   0.000   1.000
```

**观察**：
- 低维度（0, 1）变化快 → 编码短距离信息
- 高维度（6, 7）变化慢 → 编码长距离信息

---

## 3. 注意力机制 (Attention Mechanism)

### 3.1 核心思想

注意力机制模拟人类阅读时的"选择性关注"：

```
例子：翻译 "The cat sat on the mat"

当处理 "sat" 时，人类会自然关注：
- 主语 "cat" (谁坐着？)
- 地点 "mat" (坐在哪里？)

而不是平等关注所有词。
```

### 3.2 Query-Key-Value (QKV) 机制

把注意力想象成**数据库查询**：

```
┌─────────────┬─────────────────────────────┐
│  组件       │  类比                        │
├─────────────┼─────────────────────────────┤
│ Query (Q)   │ 搜索词（"我想找什么"）       │
│ Key (K)     │ 数据库索引（"我是什么"）     │
│ Value (V)   │ 实际数据（"我的内容是什么"） │
└─────────────┴─────────────────────────────┘
```

### 3.3 注意力计算公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

#### 步骤详解：

**步骤 1：计算相似度（QK^T）**

```javascript
// Q: "sat" 的查询向量
// K: 所有词的键向量矩阵

scores = Q × K^T  // 矩阵乘法

// 结果（相似度分数）:
// [cat: 85, the: 20, sat: 100, on: 30, mat: 60]
```

**步骤 2：缩放（÷ √d_k）**

```javascript
// 为什么要缩放？
// 维度高时，点积值会很大 → softmax 梯度消失

scaled_scores = scores / Math.sqrt(d_k)

// 假设 d_k = 64
// [cat: 10.6, the: 2.5, sat: 12.5, on: 3.75, mat: 7.5]
```

**步骤 3：Softmax 归一化**

```javascript
// 将分数转换为概率分布（和为 1）

attention_weights = softmax(scaled_scores)

// 结果（注意力权重）:
// [cat: 0.35, the: 0.05, sat: 0.40, on: 0.05, mat: 0.15]
//  ↑ 关注猫     ↑ 自己     ↑ 关注地点
```

**步骤 4：加权求和（× V）**

```javascript
// 用注意力权重加权所有词的 Value

output = 0.35×V_cat + 0.05×V_the + 0.40×V_sat + 0.05×V_on + 0.15×V_mat

// 结果：融合了相关词信息的新表示
```

### 3.4 代码实现映射

```javascript
// 对应代码：第 268-293 行
scaledDotProductAttention(Q, K, V, mask = null) {
    // 步骤 1: QK^T
    const KT = this.transpose(K);
    let attnScores = this.matmul(Q, KT);
    
    // 步骤 2: 缩放
    const scale = Math.sqrt(this.dK);
    attnScores = attnScores.map(row => row.map(val => val / scale));
    
    // 步骤 3: 应用掩码（可选）
    if (mask !== null) {
        attnScores = attnScores.map((row, i) =>
            row.map((val, j) => mask[i][j] === 0 ? -1e9 : val)
        );
    }
    
    // 步骤 4: Softmax
    const attnProbs = this.softmax(attnScores);
    
    // 步骤 5: 加权求和
    const output = this.matmul(attnProbs, V);
    
    return output;
}
```

### 3.5 三种注意力类型

```
┌──────────────────┬──────────────┬──────────────┬──────────────┐
│      类型        │  Q 来源      │  K/V 来源    │  应用场景    │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 自注意力         │  自己        │  自己        │  编码器      │
│ (Self-Attention) │              │              │              │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 掩码自注意力     │  自己        │  自己        │  解码器      │
│ (Masked)         │              │  (只看过去)  │  (生成任务)  │
├──────────────────┼──────────────┼──────────────┼──────────────┤
│ 交叉注意力       │  解码器      │  编码器      │  解码器      │
│ (Cross)          │              │              │  (翻译任务)  │
└──────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 4. 多头注意力 (Multi-Head Attention)

### 4.1 为什么需要多头？

**问题**：单个注意力头只能关注一个方面。

```
例子：句子 "The cat sat on the mat"

单头注意力可能只关注：
- 语法关系（主谓宾）

但我们希望同时关注：
- 语法关系
- 语义关系
- 位置关系
- 上下文关系
```

### 4.2 多头机制

**核心思想**：将模型维度分成多个"头"，每个头独立学习不同的注意力模式。

```
┌─────────────────────────────────────────┐
│         输入 (seq_len × 512)            │
└──────────────┬──────────────────────────┘
               ↓
      ┌────────┴────────┐
      │  线性投影 Q,K,V  │
      └────────┬────────┘
               ↓
      ┌────────┴────────┐
      │  分割成 8 个头   │  ← 512 ÷ 8 = 64 维/头
      └────────┬────────┘
               ↓
    ┌──────────┴──────────┐
    │  并行计算注意力      │
    │  ┌────┬────┬────┐   │
    │  │头1 │头2 │... │   │  ← 每个头独立计算
    │  └────┴────┴────┘   │
    └──────────┬──────────┘
               ↓
      ┌────────┴────────┐
      │  拼接所有头      │  ← 8×64 = 512 维
      └────────┬────────┘
               ↓
      ┌────────┴────────┐
      │  输出线性层      │
      └────────┬────────┘
               ↓
┌─────────────────────────────────────────┐
│         输出 (seq_len × 512)            │
└─────────────────────────────────────────┘
```

### 4.3 数学公式

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head₈) × W^O

其中每个头：
head_i = Attention(Q×W^Q_i, K×W^K_i, V×W^V_i)
```

### 4.4 每个头学到什么？

实际研究发现，不同的头会自动学习不同的语言模式：

```
头 1: 关注【位置相邻】的词
      "The [cat] sat"
          ↑───↑

头 2: 关注【主谓关系】
      "[The cat] [sat]"
         ↑_______↑

头 3: 关注【介词短语】
      "sat [on the mat]"
           ↑________↑

头 4: 关注【长距离依赖】
      "[The] cat sat on the [mat]"
        ↑_________________↑

... (其他头关注其他模式)
```

### 4.5 代码实现映射

```javascript
// 对应代码：第 125-403 行
class MultiHeadAttention {
    constructor(dModel = 512, numHeads = 8) {
        this.numHeads = numHeads;
        this.dK = Math.floor(dModel / numHeads);  // 64 维/头
        
        // 可学习的权重矩阵
        this.WQ = this.initializeWeights(dModel, dModel);
        this.WK = this.initializeWeights(dModel, dModel);
        this.WV = this.initializeWeights(dModel, dModel);
        this.WO = this.initializeWeights(dModel, dModel);
    }
    
    splitHeads(x) {
        // 将 (seq_len, 512) 分割为 (8, seq_len, 64)
        // 512 维 → 8 个头 × 64 维
    }
    
    combineHeads(x) {
        // 将 (8, seq_len, 64) 合并为 (seq_len, 512)
        // 8 个头 × 64 维 → 512 维
    }
    
    forward(Q, K, V, mask = null) {
        // 1. 线性变换
        Q = this.linear(Q, this.WQ);
        K = this.linear(K, this.WK);
        V = this.linear(V, this.WV);
        
        // 2. 分割成多头
        const QHeads = this.splitHeads(Q);
        const KHeads = this.splitHeads(K);
        const VHeads = this.splitHeads(V);
        
        // 3. 每个头独立计算注意力
        const attnOutputs = [];
        for (let h = 0; h < this.numHeads; h++) {
            attnOutputs.push(this.scaledDotProductAttention(
                QHeads[h], KHeads[h], VHeads[h], mask
            ));
        }
        
        // 4. 合并所有头
        const combined = this.combineHeads(attnOutputs);
        
        // 5. 输出线性层
        return this.linear(combined, this.WO);
    }
}
```

### 4.6 参数量计算

```
以 d_model=512, num_heads=8 为例：

W^Q: 512×512 = 262,144 参数
W^K: 512×512 = 262,144 参数
W^V: 512×512 = 262,144 参数
W^O: 512×512 = 262,144 参数

总计：1,048,576 参数（约 1M）
```

---

## 5. 前馈网络 (Feed-Forward Network)

### 5.1 作用

注意力机制擅长**建模关系**，但缺乏**深度非线性变换**。前馈网络补充这一能力。

```
注意力：  "这个词和其他词的关系是什么？"
前馈网络："基于这些关系，我该如何变换这个词的表示？"
```

### 5.2 结构

```
输入 (seq_len × 512)
    ↓
第一层线性变换: 512 → 2048  (扩大 4 倍)
    ↓
ReLU 激活函数: max(0, x)    (非线性)
    ↓
Dropout (训练时)
    ↓
第二层线性变换: 2048 → 512  (压缩回原维度)
    ↓
输出 (seq_len × 512)
```

### 5.3 数学公式

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**特点**：
- **位置独立**：对序列中每个位置应用相同的变换
- **不共享信息**：不同位置之间不交互（这是注意力的工作）

### 5.4 为什么要扩大再压缩？

```
类比：图像处理中的"扩散-收缩"

512 维 → 2048 维:
- 更大的"画布"
- 更丰富的表示空间
- 模型可以学习更复杂的变换

2048 维 → 512 维:
- 压缩回原维度
- 提取最重要的特征
- 保持与其他层的维度一致
```

### 5.5 ReLU 激活函数

```javascript
ReLU(x) = max(0, x)

// 例子：
输入:  [-2, -1, 0, 1, 2]
输出:  [ 0,  0, 0, 1, 2]  ← 负值变为 0
```

**作用**：
- 引入非线性（否则多层线性变换等价于单层）
- 计算简单（只是比较和取最大值）
- 缓解梯度消失问题

### 5.6 代码实现映射

```javascript
// 对应代码：第 405-523 行
class PositionWiseFeedForward {
    constructor(dModel = 512, dFF = 2048, dropout = 0.1) {
        // 初始化两层的权重和偏置
        this.W1 = this.initializeWeights(dModel, dFF);    // 512→2048
        this.b1 = Array(dFF).fill(0);
        
        this.W2 = this.initializeWeights(dFF, dModel);    // 2048→512
        this.b2 = Array(dModel).fill(0);
        
        this.dropout = new Dropout(dropout);
    }
    
    linear(x, W, b) {
        // output = x × W^T + b
        // x: (seq_len, input_dim)
        // W: (output_dim, input_dim)
        // b: (output_dim,)
    }
    
    relu(x) {
        return x.map(row => row.map(val => Math.max(0, val)));
    }
    
    forward(x) {
        // 步骤 1: 第一层线性变换 (512 → 2048)
        let output = this.linear(x, this.W1, this.b1);
        
        // 步骤 2: ReLU 激活
        output = this.relu(output);
        
        // 步骤 3: Dropout
        output = this.dropout.forward(output);
        
        // 步骤 4: 第二层线性变换 (2048 → 512)
        output = this.linear(output, this.W2, this.b2);
        
        return output;
    }
}
```

### 5.7 参数量计算

```
以 d_model=512, d_ff=2048 为例：

W₁: 512×2048 = 1,048,576 参数
b₁: 2048 参数
W₂: 2048×512 = 1,048,576 参数
b₂: 512 参数

总计：2,099,712 参数（约 2M）

注意：这比多头注意力的参数量（1M）还要多！
```

---

## 6. 编码器层 (Encoder Layer)

### 6.1 整体结构

编码器的任务是**理解输入序列**（双向上下文）。

```
┌─────────────────────────────────────┐
│        输入 x (seq_len × 512)       │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────┐
        │ 多头自注意力  │  ← 理解词与词的关系
        └──────┬───────┘
               ↓
        [残差连接]  ← x + Attention(x)
               ↓
        [层归一化]  ← 稳定训练
               ↓
        ┌──────────────┐
        │  前馈网络    │  ← 非线性变换
        └──────┬───────┘
               ↓
        [残差连接]  ← x + FFN(x)
               ↓
        [层归一化]
               ↓
┌─────────────────────────────────────┐
│        输出 (seq_len × 512)         │
└─────────────────────────────────────┘
```

### 6.2 自注意力 (Self-Attention)

**"自"** 的含义：Q、K、V 都来自**同一个输入序列**。

```
输入句子: "The cat sat on the mat"

计算 "cat" 的表示时：
- Query: "cat" 的查询向量
- Key:   所有词（包括 "cat" 自己）的键向量
- Value: 所有词（包括 "cat" 自己）的值向量

结果：
"cat" 的新表示 = 
    0.05×"The" + 0.30×"cat"(自己) + 0.40×"sat" + 0.05×"on" + ...
```

### 6.3 残差连接 (Residual Connection)

```javascript
// 普通网络
output = F(x)

// 残差网络
output = x + F(x)  ← 保留原始输入
```

**作用**：
1. **梯度流通**：梯度可以直接流过（不经过变换层）
2. **恒等映射**：如果 F(x) 学不好，至少保留 x
3. **训练深层网络**：没有残差，几十层后训练会崩溃

**形象理解**：
```
残差 = "在原有基础上改进"
而不是 "完全重新学习"

就像修改文章：
- 没有残差：重写整篇文章
- 有残差：在原文基础上修改
```

### 6.4 层归一化 (Layer Normalization)

```javascript
// 对每个样本的所有特征维度归一化
normalized = (x - mean) / sqrt(variance + ε)
```

**作用**：
- 稳定训练（每层输入分布一致）
- 加速收敛
- 减少对学习率的敏感度

**与 Batch Normalization 的区别**：

```
Batch Norm:  对一个 batch 的同一特征归一化
             [样本1_特征i, 样本2_特征i, ...]

Layer Norm:  对一个样本的所有特征归一化
             [特征1, 特征2, 特征3, ...]
             
Transformer 用 Layer Norm 因为：
- 序列长度不固定（batch norm 难处理）
- RNN 类模型更适合 layer norm
```

### 6.5 代码实现映射

```javascript
// 对应代码：第 574-648 行
class EncoderLayer {
    constructor(dModel = 512, numHeads = 8, dFF = 2048, dropout = 0.1) {
        this.selfAttn = new MultiHeadAttention(dModel, numHeads);
        this.feedForward = new PositionWiseFeedForward(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.dropout = new Dropout(dropout);
    }
    
    forward(x, mask = null) {
        // 子层 1: 多头自注意力
        const attnOutput = this.selfAttn.forward(x, x, x, mask);
        x = this.norm1.forward(
            this.addResidual(x, this.dropout.forward(attnOutput))
        );
        // ↑ 等价于：x = LayerNorm(x + Dropout(Attention(x)))
        
        // 子层 2: 前馈网络
        const ffOutput = this.feedForward.forward(x);
        x = this.norm2.forward(
            this.addResidual(x, this.dropout.forward(ffOutput))
        );
        // ↑ 等价于：x = LayerNorm(x + Dropout(FFN(x)))
        
        return x;
    }
    
    addResidual(x, residual) {
        // 残差连接：逐元素相加
        return x.map((row, i) => row.map((val, j) => val + residual[i][j]));
    }
}
```

### 6.6 信息流动

```
输入: "The cat sat"

Step 1: 自注意力
- "The" 关注所有词 → 理解上下文
- "cat" 关注所有词 → 发现是主语
- "sat" 关注所有词 → 发现是谓语

Step 2: 残差 + 归一化
- 保留原始信息
- 稳定数值范围

Step 3: 前馈网络
- 对每个词独立进行深度变换
- 增强表达能力

Step 4: 残差 + 归一化
- 再次稳定

输出: 包含丰富上下文信息的词表示
```

---

## 7. 解码器层 (Decoder Layer)

### 7.1 与编码器的区别

```
┌────────────────┬──────────────┬──────────────┐
│    特性        │  编码器      │  解码器      │
├────────────────┼──────────────┼──────────────┤
│ 子层数量       │  2 层        │  3 层        │
├────────────────┼──────────────┼──────────────┤
│ 第1层          │ 自注意力     │ 掩码自注意力 │
├────────────────┼──────────────┼──────────────┤
│ 第2层          │ 前馈网络     │ 交叉注意力   │
├────────────────┼──────────────┼──────────────┤
│ 第3层          │ -            │ 前馈网络     │
├────────────────┼──────────────┼──────────────┤
│ 能否看未来     │ 可以         │ 不可以       │
├────────────────┼──────────────┼──────────────┤
│ 任务           │ 理解         │ 生成         │
└────────────────┴──────────────┴──────────────┘
```

### 7.2 完整结构

```
┌─────────────────────────────────────┐
│   解码器输入 (已生成的词)           │
│   "I love"                          │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────┐
        │掩码自注意力   │  ← 只能看过去的词
        └──────┬───────┘     (不能作弊看未来)
               ↓
        [残差 + 归一化]
               ↓
        ┌──────────────┐     ┌─────────────┐
        │ 交叉注意力    │ ←── │ 编码器输出  │
        │ Q:解码器      │     │ (源句子)    │
        │ K,V:编码器    │     └─────────────┘
        └──────┬───────┘
               ↓
        [残差 + 归一化]
               ↓
        ┌──────────────┐
        │  前馈网络    │
        └──────┬───────┘
               ↓
        [残差 + 归一化]
               ↓
┌─────────────────────────────────────┐
│   输出 → 预测下一个词 "Beijing"     │
└─────────────────────────────────────┘
```

### 7.3 掩码自注意力 (Masked Self-Attention)

**目的**：防止解码器"作弊"看到未来的词。

```
生成序列: "I love Beijing"

预测 "I" 时:
  可以看: []
  不能看: "love", "Beijing"

预测 "love" 时:
  可以看: "I"
  不能看: "Beijing"

预测 "Beijing" 时:
  可以看: "I", "love"
  不能看: (没有未来了)
```

**掩码矩阵**：

```javascript
// 下三角矩阵（1 表示可见，0 表示不可见）

       I  love Beijing
I    [ 1   0     0   ]  ← 预测 "I" 时只能看自己
love [ 1   1     0   ]  ← 预测 "love" 时可以看 I, love
Bei  [ 1   1     1   ]  ← 预测 "Beijing" 时可以看所有
```

实现：

```javascript
// 将不可见位置的注意力分数设为 -∞
if (mask[i][j] === 0) {
    attn_scores[i][j] = -1e9;  // softmax 后接近 0
}
```

### 7.4 交叉注意力 (Cross-Attention)

**目的**：让解码器关注源序列（编码器输出）。

```
翻译任务: "我爱北京" → "I love Beijing"

生成 "love" 时：
- Query: 来自解码器当前状态（"我想知道源句子哪个词重要"）
- Key/Value: 来自编码器输出（源句子 "我爱北京"）

交叉注意力计算：
"love" 关注源序列的哪些词？
  我:   0.10  ← 有点相关
  爱:   0.70  ← 最相关！"爱" → "love"
  北京: 0.20  ← 有点相关（上下文）

输出: 融合了源序列 "爱" 信息的 "love" 表示
```

**与自注意力的区别**：

```
自注意力 (Self-Attention):
  Q, K, V 都来自同一个序列
  
交叉注意力 (Cross-Attention):
  Q 来自解码器（目标序列）
  K, V 来自编码器（源序列）
```

### 7.5 代码实现映射

```javascript
// 对应代码：第 654-739 行
class DecoderLayer {
    constructor(dModel = 512, numHeads = 8, dFF = 2048, dropout = 0.1) {
        this.selfAttn = new MultiHeadAttention(dModel, numHeads);   // 掩码自注意力
        this.crossAttn = new MultiHeadAttention(dModel, numHeads);  // 交叉注意力
        this.feedForward = new PositionWiseFeedForward(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.norm3 = new LayerNorm(dModel);  // 多一个归一化层
        this.dropout = new Dropout(dropout);
    }
    
    forward(x, encoderOutput, srcMask = null, tgtMask = null) {
        // 子层 1: 掩码自注意力
        const attnOutput = this.selfAttn.forward(x, x, x, tgtMask);
        x = this.norm1.forward(this.addResidual(x, this.dropout.forward(attnOutput)));
        
        // 子层 2: 交叉注意力（关键！）
        const crossAttnOutput = this.crossAttn.forward(
            x,              // Query: 来自解码器
            encoderOutput,  // Key: 来自编码器
            encoderOutput,  // Value: 来自编码器
            srcMask
        );
        x = this.norm2.forward(this.addResidual(x, this.dropout.forward(crossAttnOutput)));
        
        // 子层 3: 前馈网络
        const ffOutput = this.feedForward.forward(x);
        x = this.norm3.forward(this.addResidual(x, this.dropout.forward(ffOutput)));
        
        return x;
    }
}
```

### 7.6 信息流动

```
翻译: "我爱北京" → "I love Beijing"

已生成: "I love"
目标:   预测下一个词

Step 1: 掩码自注意力
"I" 和 "love" 互相关注（但不能看未来）

Step 2: 交叉注意力
"love" 查询源句子:
  - 发现 "爱" 最相关
  - 获取 "爱" 的上下文信息

Step 3: 前馈网络
深度变换融合的信息

输出: 综合考虑了：
  ✓ 已生成的目标词（"I love"）
  ✓ 源句子的相关部分（"爱"）
  → 预测 "Beijing"
```

---

## 8. 残差连接与层归一化

### 8.1 为什么需要残差连接？

**问题**：深层网络的梯度消失。

```
没有残差的深层网络:

输入 → 层1 → 层2 → ... → 层96 → 输出
       ↓     ↓           ↓
    梯度×0.9 ×0.9    ...×0.9
                    
结果: 0.9^96 ≈ 0.00006  ← 梯度几乎消失！
```

**解决方案**：残差连接

```
output = input + F(input)

即使 F 的梯度很小，input 的梯度可以直接流回：
∂output/∂input = 1 + ∂F/∂input
                 ↑ 保证至少有 1
```

### 8.2 残差连接的数学

```
前向传播:
y = x + F(x)

反向传播:
∂Loss/∂x = ∂Loss/∂y × (1 + ∂F/∂x)
                       ↑ 恒等项
                       
好处:
- 梯度可以直接通过 "1" 这条路径流回
- 即使 F 很复杂，梯度也不会完全消失
```

### 8.3 层归一化公式

```
μ = (1/d) Σ x_i                    ← 计算均值
σ² = (1/d) Σ (x_i - μ)²           ← 计算方差
y_i = (x_i - μ) / √(σ² + ε)       ← 归一化
```

**参数说明**：
- `d`: 特征维度（如 512）
- `ε`: 小常数（如 1e-6），防止除零

### 8.4 代码实现

```javascript
// 对应代码：第 529-543 行
class LayerNorm {
    constructor(dModel, eps = 1e-6) {
        this.dModel = dModel;
        this.eps = eps;
    }
    
    forward(x) {
        // 计算均值
        const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
        
        // 计算方差
        const variance = x.reduce((sum, val) => 
            sum + Math.pow(val - mean, 2), 0
        ) / x.length;
        
        // 归一化
        return x.map(val => (val - mean) / Math.sqrt(variance + this.eps));
    }
}
```

### 8.5 Pre-LN vs Post-LN

**Post-LN（原始论文）**：
```
x → [Attention] → 残差 → [LayerNorm] → 输出
```

**Pre-LN（现代实现）**：
```
x → [LayerNorm] → [Attention] → 残差 → 输出
```

**Pre-LN 的优点**：
- 训练更稳定
- 不需要 warmup
- GPT-3、GPT-4 使用 Pre-LN

---

## 9. 掩码机制 (Masking)

### 9.1 两种掩码

```
┌──────────────────┬─────────────────┬─────────────────┐
│      类型        │  Padding Mask   │  Look-ahead Mask│
├──────────────────┼─────────────────┼─────────────────┤
│ 用途             │ 处理变长序列    │ 防止看到未来    │
├──────────────────┼─────────────────┼─────────────────┤
│ 应用位置         │ 编码器/解码器   │ 仅解码器        │
├──────────────────┼─────────────────┼─────────────────┤
│ 掩码形状         │ 任意形状        │ 下三角矩阵      │
└──────────────────┴─────────────────┴─────────────────┘
```

### 9.2 Padding Mask

**问题**：批处理时序列长度不同。

```
Batch:
  句子1: "I love you"          (长度 3)
  句子2: "Hello"               (长度 1)
  
填充后:
  句子1: "I love you"          [1, 1, 1]
  句子2: "Hello <PAD> <PAD>"   [1, 0, 0]
         ↑ 真实词  ↑ 填充
```

**掩码矩阵**：

```javascript
// 1 表示真实词，0 表示填充

句子1: [1, 1, 1]  → 注意力可以关注所有词
句子2: [1, 0, 0]  → 注意力只能关注第一个词
```

**实现**：

```javascript
if (mask[i][j] === 0) {
    attn_scores[i][j] = -1e9;  // 填充位置分数设为 -∞
}
// softmax(-1e9) ≈ 0  ← 填充位置注意力权重接近 0
```

### 9.3 Look-ahead Mask（前瞻掩码）

**问题**：生成任务时不能看到未来。

```
生成 "I love Beijing" 时：

预测位置 1 ("I"):
  可以看: []
  掩码:   [1, 0, 0]

预测位置 2 ("love"):
  可以看: "I"
  掩码:   [1, 1, 0]

预测位置 3 ("Beijing"):
  可以看: "I", "love"
  掩码:   [1, 1, 1]
```

**掩码矩阵（下三角）**：

```javascript
// 生成掩码矩阵
function createLookAheadMask(size) {
    const mask = [];
    for (let i = 0; i < size; i++) {
        const row = [];
        for (let j = 0; j < size; j++) {
            row.push(j <= i ? 1 : 0);  // 只能看当前及之前
        }
        mask.push(row);
    }
    return mask;
}

// 结果 (size=4):
// [1, 0, 0, 0]
// [1, 1, 0, 0]
// [1, 1, 1, 0]
// [1, 1, 1, 1]
```

### 9.4 组合掩码

有时需要同时应用两种掩码：

```javascript
// Padding Mask
padding_mask = [
    [1, 1, 1, 0],  // 第 4 个是填充
    [1, 1, 1, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 0]
]

// Look-ahead Mask
lookahead_mask = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

// 组合（逻辑与）
combined_mask = [
    [1, 0, 0, 0],  ← 既要下三角，又要避开填充
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 0]  ← 最后一位始终被掩盖（填充）
]
```

---

## 10. 完整流程示例

### 10.1 机器翻译任务

**任务**：将中文翻译成英文

```
源句子: "我 爱 北京 天安门"
目标句子: "I love Beijing Tiananmen"
```

### 10.2 详细流程

#### **阶段 1：编码器处理源句子**

```
输入: "我 爱 北京 天安门"
  ↓
【词嵌入】
[0.23, 0.45, ..., 0.12]  ← "我" 的向量
[0.67, 0.89, ..., 0.34]  ← "爱" 的向量
[0.11, 0.56, ..., 0.78]  ← "北京" 的向量
[0.90, 0.23, ..., 0.45]  ← "天安门" 的向量
  ↓
【位置编码】
加上 PE(0), PE(1), PE(2), PE(3)
  ↓
【编码器层 1】
  - 自注意力: 理解词之间关系
    "我" ←→ "爱"     (主谓关系)
    "爱" ←→ "北京"   (动宾关系)
    "北京" ←→ "天安门" (并列关系)
  - 前馈网络: 深度变换
  ↓
【编码器层 2】
  - 更深层次的语义理解
  - ...
  ↓
【编码器层 6】
  ↓
编码器输出: 源句子的深度表示
[向量1, 向量2, 向量3, 向量4]
```

#### **阶段 2：解码器生成目标句子**

**Step 1: 预测第一个词 "I"**

```
输入: <START> (开始标记)
  ↓
【掩码自注意力】
只能看 <START>
  ↓
【交叉注意力】
Query: <START> 的表示
Key/Value: 编码器输出（源句子）

<START> 查询源句子:
  我:     0.70  ← 最相关！第一个词
  爱:     0.15
  北京:   0.10
  天安门: 0.05
  ↓
【前馈网络】
  ↓
输出: 预测下一个词
  softmax([...]) = [I: 0.85, You: 0.05, ...]
  ↓
选择: "I"
```

**Step 2: 预测第二个词 "love"**

```
输入: <START> I
  ↓
【掩码自注意力】
"I" 可以关注 <START> 和自己
  ↓
【交叉注意力】
"I" 查询源句子:
  我:     0.20
  爱:     0.60  ← "I" 后面跟 "love"（对应 "爱"）
  北京:   0.15
  天安门: 0.05
  ↓
输出: 预测 "love"
```

**Step 3: 预测第三个词 "Beijing"**

```
输入: <START> I love
  ↓
【掩码自注意力】
"love" 可以关注 <START>, "I", "love"
  ↓
【交叉注意力】
"love" 查询源句子:
  我:     0.10
  爱:     0.15
  北京:   0.65  ← "love" 后面是 "Beijing"
  天安门: 0.10
  ↓
输出: 预测 "Beijing"
```

**Step 4: 预测第四个词 "Tiananmen"**

```
输入: <START> I love Beijing
  ↓
【掩码自注意力】
"Beijing" 可以看所有已生成的词
  ↓
【交叉注意力】
"Beijing" 查询源句子:
  我:     0.05
  爱:     0.10
  北京:   0.20
  天安门: 0.65  ← 对应 "Tiananmen"
  ↓
输出: 预测 "Tiananmen"
```

**Step 5: 结束**

```
输入: <START> I love Beijing Tiananmen
  ↓
输出: 预测 <END> (结束标记)
  ↓
完整翻译: "I love Beijing Tiananmen"
```

### 10.3 关键观察

1. **编码器是一次性的**：
   - 处理整个源句子
   - 输出固定表示

2. **解码器是自回归的**：
   - 逐词生成
   - 每次依赖之前生成的词

3. **交叉注意力连接两者**：
   - 解码器每一步都查询编码器输出
   - 决定关注源句子的哪些部分

### 10.4 代码对应

```javascript
// 对应代码：第 927-954 行
function exampleUsage() {
    // 初始化模型
    const model = new Transformer(
        2,      // 2 层（简化）
        512,    // 维度
        8,      // 8 个注意力头
        2048,   // 前馈网络维度
        0.1     // dropout
    );
    
    // 模拟输入
    const srcSequence = Array(4).fill(0).map(() => 
        Array(512).fill(0.5)
    );  // "我爱北京天安门" (4个词)
    
    const tgtSequence = Array(2).fill(0).map(() => 
        Array(512).fill(0.3)
    );  // "I love" (2个词，预测第3个)
    
    // 前向传播
    const output = model.forward(srcSequence, tgtSequence);
    
    // output: 预测下一个词 "Beijing" 的表示
}
```

---

## 11. 参数量计算

### 11.1 单层 Transformer 参数

以标准配置为例：`d_model=512, num_heads=8, d_ff=2048`

```
多头注意力:
  W^Q: 512×512 = 262,144
  W^K: 512×512 = 262,144
  W^V: 512×512 = 262,144
  W^O: 512×512 = 262,144
  小计: 1,048,576 参数 (1M)

前馈网络:
  W₁: 512×2048 = 1,048,576
  b₁: 2,048
  W₂: 2048×512 = 1,048,576
  b₂: 512
  小计: 2,099,712 参数 (2M)

层归一化:
  γ: 512 (可选)
  β: 512 (可选)
  小计: 1,024 参数

一层总计: 约 3.15M 参数
```

### 11.2 完整模型参数

```
标准 Transformer (6层编码器 + 6层解码器):
  编码器: 6 × 3.15M = 18.9M
  解码器: 6 × 3.15M = 18.9M (多一个交叉注意力层)
  词嵌入: vocab_size × 512 (如 30,000 × 512 = 15.36M)
  位置编码: 预计算，无参数
  输出层: 512 × vocab_size (如 512 × 30,000 = 15.36M)
  
总计: 约 70-80M 参数
```

### 11.3 现代 LLM 的参数量

```
GPT-2:        1.5B 参数 (48 层)
GPT-3:        175B 参数 (96 层)
GPT-4:        ~1.76T 参数 (估计，未公开)
```

---

## 12. 训练技巧

### 12.1 学习率预热 (Warmup)

```
学习率调度:

epoch 1-4000:  线性增长 (warmup)
epoch 4000+:   按 sqrt 衰减

lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

### 12.2 标签平滑 (Label Smoothing)

```
原始标签:  [0, 0, 1, 0, 0]  ← 硬目标
平滑标签:  [0.02, 0.02, 0.92, 0.02, 0.02]  ← 软目标

作用: 防止过拟合，提高泛化能力
```

### 12.3 Dropout 位置

```
Transformer 中应用 Dropout 的位置:
1. 位置编码后
2. 每个子层输出后（残差前）
3. 注意力权重上（可选）
```

---

## 13. 常见变体

### 13.1 GPT（仅解码器）

```
特点:
- 只有解码器
- 无交叉注意力
- 单向（从左到右）
- 预训练 + 微调

应用: 文本生成、对话系统
```

### 13.2 BERT（仅编码器）

```
特点:
- 只有编码器
- 双向注意力
- 掩码语言模型（MLM）
- 预训练 + 微调

应用: 文本分类、情感分析
```

### 13.3 T5（编码器-解码器）

```
特点:
- 完整 Transformer
- 所有任务统一为文本生成
- "Text-to-Text"

应用: 翻译、摘要、问答
```

---

## 14. 优化技巧

### 14.1 内存优化

```
1. 梯度检查点 (Gradient Checkpointing)
   - 训练时不保存中间激活
   - 反向传播时重新计算
   - 节省内存，略微增加计算

2. 混合精度训练 (Mixed Precision)
   - 使用 FP16 代替 FP32
   - 减少内存占用，加速训练
```

### 14.2 速度优化

```
1. Flash Attention
   - 优化注意力计算的内存访问模式
   - 速度提升 2-4 倍

2. 多查询注意力 (Multi-Query Attention)
   - K 和 V 只有一个头
   - 减少参数量和计算量

3. 分组查询注意力 (Grouped-Query Attention)
   - 介于 MHA 和 MQA 之间
   - 平衡性能和速度
```

---

## 15. 调试技巧

### 15.1 检查维度

```javascript
// 在每个关键步骤打印形状
console.log('输入形状:', x.length, '×', x[0].length);
console.log('注意力输出:', attn.length, '×', attn[0].length);
console.log('前馈输出:', ff.length, '×', ff[0].length);
```

### 15.2 可视化注意力

```javascript
// 打印注意力权重
function visualizeAttention(attnWeights) {
    console.log('注意力权重矩阵:');
    attnWeights.forEach((row, i) => {
        const values = row.map(v => v.toFixed(2)).join(' ');
        console.log(`词${i}: ${values}`);
    });
}
```

### 15.3 梯度检查

```javascript
// 检查梯度是否正常
if (isNaN(gradient) || !isFinite(gradient)) {
    console.error('梯度异常！');
}
```

---

## 16. 常见问题 FAQ

### Q1: 为什么 Transformer 比 RNN 快？

**A**: 并行化！
- RNN 必须顺序处理（t 依赖 t-1）
- Transformer 所有位置同时计算注意力

### Q2: 注意力的计算复杂度是多少？

**A**: O(n²d)
- n: 序列长度
- d: 维度
- 瓶颈：长序列时 n² 很大

### Q3: 如何处理超长文本？

**A**: 几种方法：
1. 截断（简单但损失信息）
2. 滑动窗口注意力
3. 稀疏注意力（Longformer）
4. 分层注意力

### Q4: 残差连接一定有用吗？

**A**: 对深层网络（>6 层）几乎必需
- 浅层网络可能不明显
- 深层网络没有残差训练会失败

### Q5: Layer Norm 能换成 Batch Norm 吗？

**A**: 不推荐
- Batch Norm 对序列长度敏感
- Layer Norm 更适合 NLP

---

## 17. 扩展阅读

### 17.1 必读论文

1. **Attention Is All You Need** (2017)
   - 原始 Transformer 论文
   - [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

2. **BERT** (2018)
   - 预训练语言模型
   - [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

3. **GPT-3** (2020)
   - 大规模语言模型
   - [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

### 17.2 可视化工具

- **Tensor2Tensor**: Google 的 Transformer 实现
- **The Illustrated Transformer**: 图解教程
- **BertViz**: 注意力可视化工具

### 17.3 代码资源

- **Hugging Face Transformers**: 最流行的库
- **fairseq**: Facebook 的实现
- **nanoGPT**: 极简 GPT 实现（教学用）

---

## 18. 总结

### 18.1 核心组件回顾

```
┌────────────────────┬─────────────────────────┐
│      组件          │       作用              │
├────────────────────┼─────────────────────────┤
│ 位置编码           │ 提供位置信息            │
├────────────────────┼─────────────────────────┤
│ 多头注意力         │ 建模词之间的关系        │
├────────────────────┼─────────────────────────┤
│ 前馈网络           │ 增加非线性表达能力      │
├────────────────────┼─────────────────────────┤
│ 残差连接           │ 稳定梯度流动            │
├────────────────────┼─────────────────────────┤
│ 层归一化           │ 稳定训练                │
├────────────────────┼─────────────────────────┤
│ 掩码机制           │ 控制信息流向            │
└────────────────────┴─────────────────────────┘
```

### 18.2 设计哲学

1. **并行化优先**：摒弃 RNN 的顺序依赖
2. **多头多角度**：不同头学习不同模式
3. **残差保信息**：深层网络的稳定剂
4. **归一化稳训练**：数值稳定性
5. **掩码控流向**：灵活控制信息传播

### 18.3 应用方向

```
编码器 → 理解任务
- 文本分类
- 情感分析
- 命名实体识别

解码器 → 生成任务
- 文本生成
- 对话系统
- 代码生成

编码器-解码器 → 转换任务
- 机器翻译
- 文本摘要
- 问答系统
```

---

## 附录：代码文件结构

```
1-3-LLMBase.js 文件结构:

第 1-10 行:     文件说明
第 19-119 行:   PositionalEncoding (位置编码)
第 125-403 行:  MultiHeadAttention (多头注意力)
第 405-523 行:  PositionWiseFeedForward (前馈网络)
第 529-543 行:  LayerNorm (层归一化)
第 549-568 行:  Dropout
第 574-648 行:  EncoderLayer (编码器层)
第 654-739 行:  DecoderLayer (解码器层)
第 745-826 行:  Transformer (完整模型)
第 835-978 行:  测试函数和示例
第 984-1002 行: 架构对比总结
第 1005-1024 行: 导出和执行
```

---

**最后更新**: 2026-02-25  
**作者**: AI Assistant  
**对应代码**: `1-3-LLMBase.js`
