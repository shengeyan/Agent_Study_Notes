/**
 * LLM 基础架构演进：从 N-gram 到 Transformer
 *
 * 本文件展示了自然语言处理模型的发展历程：
 * 1. N-gram: 基于统计的语言模型
 * 2. RNN: 循环神经网络
 * 3. LSTM: 长短期记忆网络
 * 4. Transformer: 基于注意力机制的模型（现代 LLM 基础）
 */

// ==============================================================================
// 第四部分：Transformer 架构（编码器-解码器）
// ==============================================================================

/**
 * 步骤 1：基础模块实现
 */

class PositionalEncoding {
    /**
     * 位置编码模块（Positional Encoding）
     *
     * 作用：为输入序列的词嵌入向量添加位置信息
     * 原因：Transformer 的 Attention 机制本身不包含位置信息（打乱顺序结果相同）
     *
     * 方法：使用不同频率的 sin 和 cos 函数
     *
     * 公式：
     *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    ← 偶数维度
     *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    ← 奇数维度
     *
     * 其中：
     *   - pos: 序列中的位置 (0, 1, 2, ...)
     *   - i: 维度索引 (0, 1, 2, ..., d_model/2)
     *   - d_model: 模型维度
     *
     * @param {number} dModel - 模型维度（默认 512）
     * @param {number} dropout - Dropout 比率（默认 0.1）
     * @param {number} maxLen - 最大序列长度（默认 5000）
     */
    constructor(dModel = 512, dropout = 0.1, maxLen = 5000) {
        this.dModel = dModel;
        this.maxLen = maxLen;
        this.dropout = new Dropout(dropout);

        // 预计算位置编码矩阵 (maxLen × d_model)
        this.pe = this.createPositionalEncoding();
    }

    /**
     * 创建位置编码矩阵
     * @returns {Array} 位置编码矩阵 (maxLen × d_model)
     */
    createPositionalEncoding() {
        const pe = Array(this.maxLen)
            .fill(0)
            .map(() => Array(this.dModel).fill(0));

        // 对每个位置
        for (let pos = 0; pos < this.maxLen; pos++) {
            // 对每个维度
            for (let i = 0; i < this.dModel; i++) {
                // 计算角度：pos / 10000^(2i/d_model)
                const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / this.dModel);

                // 偶数维度使用 sin，奇数维度使用 cos
                if (i % 2 === 0) {
                    pe[pos][i] = Math.sin(angle);
                } else {
                    pe[pos][i] = Math.cos(angle);
                }
            }
        }

        return pe;
    }

    /**
     * 前向传播
     *
     * 将位置编码加到输入的词嵌入上
     *
     * @param {Array} x - 输入词嵌入 (seq_len × d_model)
     * @returns {Array} 添加位置编码后的输出 (seq_len × d_model)
     */
    forward(x) {
        console.log('  → PositionalEncoding forward');

        const seqLen = x.length;

        // 将对应长度的位置编码加到输入上
        const output = x.map((row, pos) => {
            if (pos >= this.maxLen) {
                throw new Error(`序列长度 ${seqLen} 超过最大长度 ${this.maxLen}`);
            }
            return row.map((val, i) => val + this.pe[pos][i]);
        });

        // 应用 Dropout
        const result = this.dropout.forward(output);

        console.log(`    ✓ 添加位置编码 (seq_len=${seqLen})`);
        return result;
    }

    /**
     * 可视化位置编码（调试用）
     * 打印前 n 个位置的编码模式
     */
    visualize(numPositions = 10, numDims = 8) {
        console.log('\n位置编码可视化（前几个位置和维度）:');
        console.log('pos\\dim', '\t', Array.from({ length: numDims }, (_, i) => i).join('\t'));

        for (let pos = 0; pos < Math.min(numPositions, this.maxLen); pos++) {
            const values = this.pe[pos]
                .slice(0, numDims)
                .map(v => v.toFixed(3))
                .join('\t');
            console.log(`${pos}\t\t${values}`);
        }
    }
}

/**
 * 步骤 1（续）：占位符模块（将在后续实现）
 */

class MultiHeadAttention {
    /**
     * 多头注意力机制模块
     *
     * 作用：让模型从多个角度（多个头）关注输入序列
     *
     * 核心思想：
     * 1. 将 d_model 维度分成 num_heads 个头
     * 2. 每个头独立计算注意力
     * 3. 拼接所有头的输出
     * 4. 通过线性层整合
     *
     * 公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
     *
     * @param {number} dModel - 模型维度（必须能被 numHeads 整除）
     * @param {number} numHeads - 注意力头数
     */
    constructor(dModel = 512, numHeads = 8) {
        // 验证：dModel 必须能被 numHeads 整除
        if (dModel % numHeads !== 0) {
            throw new Error(`d_model (${dModel}) 必须能被 num_heads (${numHeads}) 整除`);
        }

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = Math.floor(dModel / numHeads); // 每个头的维度

        // 初始化 Q, K, V 和输出的线性变换矩阵（简化为恒等映射）
        // 实际应该是可学习的权重矩阵，这里用随机初始化模拟
        this.WQ = this.initializeWeights(dModel, dModel);
        this.WK = this.initializeWeights(dModel, dModel);
        this.WV = this.initializeWeights(dModel, dModel);
        this.WO = this.initializeWeights(dModel, dModel);
    }

    /**
     * 初始化权重矩阵（简化实现）
     * 实际应该使用 Xavier 或 Kaiming 初始化
     */
    initializeWeights(inputDim, outputDim) {
        const weights = [];
        const scale = Math.sqrt(2.0 / inputDim);
        for (let i = 0; i < outputDim; i++) {
            weights[i] = [];
            for (let j = 0; j < inputDim; j++) {
                weights[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return weights;
    }

    /**
     * 矩阵乘法：C = A × B
     * @param {Array} A - 矩阵 A (m × n)
     * @param {Array} B - 矩阵 B (n × p)
     * @returns {Array} C (m × p)
     */
    matmul(A, B) {
        if (!Array.isArray(A) || !Array.isArray(B)) {
            throw new Error('输入必须是数组');
        }

        // 处理一维数组情况
        if (!Array.isArray(A[0])) {
            A = [A];
        }
        if (!Array.isArray(B[0])) {
            B = B.map(x => [x]);
        }

        const m = A.length;
        const n = A[0].length;
        const p = B[0].length;

        if (B.length !== n) {
            throw new Error(`矩阵维度不匹配: (${m}×${n}) × (${B.length}×${p})`);
        }

        const C = Array(m)
            .fill(0)
            .map(() => Array(p).fill(0));

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                for (let k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    /**
     * 矩阵转置
     * @param {Array} matrix - 输入矩阵
     * @returns {Array} 转置后的矩阵
     */
    transpose(matrix) {
        if (!Array.isArray(matrix[0])) {
            return matrix.map(x => [x]);
        }
        const rows = matrix.length;
        const cols = matrix[0].length;
        const result = Array(cols)
            .fill(0)
            .map(() => Array(rows).fill(0));

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    /**
     * Softmax 函数（在最后一个维度上）
     * @param {Array} matrix - 输入矩阵
     * @returns {Array} Softmax 后的矩阵
     */
    softmax(matrix) {
        return matrix.map(row => {
            // 数值稳定性：减去最大值
            const maxVal = Math.max(...row);
            const expValues = row.map(x => Math.exp(x - maxVal));
            const sumExp = expValues.reduce((a, b) => a + b, 0);
            return expValues.map(x => x / sumExp);
        });
    }

    /**
     * 缩放点积注意力（Scaled Dot-Product Attention）
     *
     * 步骤：
     * 1. 计算 Q 和 K 的点积：QK^T
     * 2. 缩放：除以 sqrt(d_k)
     * 3. 应用掩码（可选）
     * 4. Softmax 归一化得到注意力权重
     * 5. 加权求和：权重 × V
     *
     * @param {Array} Q - Query 矩阵 (seq_len × d_k)
     * @param {Array} K - Key 矩阵 (seq_len × d_k)
     * @param {Array} V - Value 矩阵 (seq_len × d_k)
     * @param {Array} mask - 掩码矩阵（可选）
     * @returns {Array} 注意力输出
     */
    scaledDotProductAttention(Q, K, V, mask = null) {
        // 步骤 1: 计算注意力得分 QK^T
        const KT = this.transpose(K);
        let attnScores = this.matmul(Q, KT);

        // 步骤 2: 缩放（除以 sqrt(d_k)，防止点积过大）
        const scale = Math.sqrt(this.dK);
        attnScores = attnScores.map(row => row.map(val => val / scale));

        // 步骤 3: 应用掩码（如果提供）
        if (mask !== null) {
            attnScores = attnScores.map((row, i) => row.map((val, j) => (mask[i][j] === 0 ? -1e9 : val)));
        }

        // 步骤 4: Softmax 计算注意力权重
        const attnProbs = this.softmax(attnScores);

        // 步骤 5: 加权求和（注意力权重 × V）
        const output = this.matmul(attnProbs, V);

        return output;
    }

    /**
     * 分割多头
     * 将输入从 (seq_length, d_model) 分割为 (num_heads, seq_length, d_k)
     *
     * @param {Array} x - 输入矩阵 (seq_length × d_model)
     * @returns {Array} 分割后的矩阵 (num_heads × seq_length × d_k)
     */
    splitHeads(x) {
        const seqLength = x.length;
        const heads = [];

        // 将每个序列位置的 d_model 维度分成 num_heads 个 d_k 维度
        for (let h = 0; h < this.numHeads; h++) {
            const head = [];
            for (let i = 0; i < seqLength; i++) {
                const startIdx = h * this.dK;
                const endIdx = startIdx + this.dK;
                head.push(x[i].slice(startIdx, endIdx));
            }
            heads.push(head);
        }

        return heads;
    }

    /**
     * 合并多头
     * 将 (num_heads, seq_length, d_k) 合并为 (seq_length, d_model)
     *
     * @param {Array} x - 多头矩阵 (num_heads × seq_length × d_k)
     * @returns {Array} 合并后的矩阵 (seq_length × d_model)
     */
    combineHeads(x) {
        const seqLength = x[0].length;
        const combined = [];

        // 将所有头在最后一个维度上拼接
        for (let i = 0; i < seqLength; i++) {
            const row = [];
            for (let h = 0; h < this.numHeads; h++) {
                row.push(...x[h][i]);
            }
            combined.push(row);
        }

        return combined;
    }

    /**
     * 线性变换（简化实现）
     * @param {Array} x - 输入 (seq_len × d_model)
     * @param {Array} W - 权重矩阵 (d_model × d_model)
     * @returns {Array} 输出 (seq_len × d_model)
     */
    linear(x, W) {
        return this.matmul(x, this.transpose(W));
    }

    /**
     * 前向传播
     *
     * 完整流程：
     * 1. 对 Q, K, V 进行线性变换
     * 2. 分割成多个头
     * 3. 每个头独立计算缩放点积注意力
     * 4. 合并所有头的输出
     * 5. 通过输出线性层
     *
     * @param {Array} Q - Query (seq_len × d_model)
     * @param {Array} K - Key (seq_len × d_model)
     * @param {Array} V - Value (seq_len × d_model)
     * @param {Array} mask - 掩码（可选）
     * @returns {Array} 输出 (seq_len × d_model)
     */
    forward(Q, K, V, mask = null) {
        console.log('  → MultiHeadAttention forward');

        // 步骤 1: 线性变换 Q, K, V
        Q = this.linear(Q, this.WQ);
        K = this.linear(K, this.WK);
        V = this.linear(V, this.WV);

        // 步骤 2: 分割成多个头
        const QHeads = this.splitHeads(Q);
        const KHeads = this.splitHeads(K);
        const VHeads = this.splitHeads(V);

        // 步骤 3: 每个头独立计算注意力
        const attnOutputs = [];
        for (let h = 0; h < this.numHeads; h++) {
            const attnOutput = this.scaledDotProductAttention(QHeads[h], KHeads[h], VHeads[h], mask);
            attnOutputs.push(attnOutput);
        }

        // 步骤 4: 合并所有头
        const combinedOutput = this.combineHeads(attnOutputs);

        // 步骤 5: 通过输出线性层
        const output = this.linear(combinedOutput, this.WO);

        console.log(`    ✓ 计算了 ${this.numHeads} 个注意力头`);
        return output;
    }
}

class PositionWiseFeedForward {
    /**
     * 位置前馈网络模块（Position-wise Feed-Forward Network）
     *
     * 作用：对序列中的每个位置独立进行非线性变换
     * 结构：Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model)
     *
     * 特点：
     * 1. 对每个位置应用相同的变换（位置独立）
     * 2. 在不同位置之间不共享信息
     * 3. 增加模型的非线性表达能力
     *
     * 公式：FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
     *
     * @param {number} dModel - 输入/输出维度（默认 512）
     * @param {number} dFF - 隐藏层维度（默认 2048，通常是 dModel 的 4 倍）
     * @param {number} dropout - Dropout 比率（默认 0.1）
     */
    constructor(dModel = 512, dFF = 2048, dropout = 0.1) {
        this.dModel = dModel;
        this.dFF = dFF;

        // 初始化两个线性层的权重和偏置
        this.W1 = this.initializeWeights(dModel, dFF);
        this.b1 = Array(dFF)
            .fill(0)
            .map(() => (Math.random() - 0.5) * 0.1);

        this.W2 = this.initializeWeights(dFF, dModel);
        this.b2 = Array(dModel)
            .fill(0)
            .map(() => (Math.random() - 0.5) * 0.1);

        this.dropout = new Dropout(dropout);
    }

    /**
     * 初始化权重矩阵
     * 使用 Xavier 初始化
     */
    initializeWeights(inputDim, outputDim) {
        const weights = [];
        const scale = Math.sqrt(2.0 / inputDim);

        for (let i = 0; i < outputDim; i++) {
            weights[i] = [];
            for (let j = 0; j < inputDim; j++) {
                weights[i][j] = (Math.random() - 0.5) * 2 * scale;
            }
        }
        return weights;
    }

    /**
     * 矩阵乘法 + 偏置
     * output = x @ W^T + b
     */
    linear(x, W, b) {
        // x: (seq_len, input_dim)
        // W: (output_dim, input_dim)
        // b: (output_dim,)
        // output: (seq_len, output_dim)

        const seqLen = x.length;
        const outputDim = W.length;
        const result = [];

        for (let i = 0; i < seqLen; i++) {
            const row = [];
            for (let j = 0; j < outputDim; j++) {
                let sum = 0;
                for (let k = 0; k < x[i].length; k++) {
                    sum += x[i][k] * W[j][k];
                }
                row.push(sum + b[j]);
            }
            result.push(row);
        }

        return result;
    }

    /**
     * ReLU 激活函数
     * ReLU(x) = max(0, x)
     */
    relu(x) {
        return x.map(row => row.map(val => Math.max(0, val)));
    }

    /**
     * 前向传播
     *
     * 流程：
     * 1. 第一层线性变换：(seq_len, d_model) → (seq_len, d_ff)
     * 2. ReLU 激活
     * 3. Dropout（训练时）
     * 4. 第二层线性变换：(seq_len, d_ff) → (seq_len, d_model)
     *
     * @param {Array} x - 输入 (seq_len × d_model)
     * @returns {Array} 输出 (seq_len × d_model)
     */
    forward(x) {
        console.log('  → PositionWiseFeedForward forward');

        // 步骤 1: 第一层线性变换 (d_model → d_ff)
        let output = this.linear(x, this.W1, this.b1);

        // 步骤 2: ReLU 激活
        output = this.relu(output);

        // 步骤 3: Dropout
        output = this.dropout.forward(output);

        // 步骤 4: 第二层线性变换 (d_ff → d_model)
        output = this.linear(output, this.W2, this.b2);

        console.log(`    ✓ FFN: ${this.dModel} → ${this.dFF} → ${this.dModel}`);
        return output;
    }
}

/**
 * 工具函数：Layer Normalization
 * 作用：归一化每个样本的特征维度（均值=0，方差=1）
 */
class LayerNorm {
    constructor(dModel, eps = 1e-6) {
        this.dModel = dModel;
        this.eps = eps;
    }

    forward(x) {
        // 处理二维数组：对每一行（每个样本）独立归一化
        if (Array.isArray(x[0])) {
            return x.map(row => {
                // 计算该行的均值和方差
                const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
                const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;

                // 归一化该行
                return row.map(val => (val - mean) / Math.sqrt(variance + this.eps));
            });
        }

        // 处理一维数组
        const mean = x.reduce((sum, val) => sum + val, 0) / x.length;
        const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / x.length;
        return x.map(val => (val - mean) / Math.sqrt(variance + this.eps));
    }
}

/**
 * 工具函数：Dropout
 * 作用：训练时随机丢弃部分神经元，防止过拟合
 */
class Dropout {
    constructor(dropoutRate = 0.1) {
        this.dropoutRate = dropoutRate;
        this.isTraining = false; // 默认设为 false，简化实现
    }

    forward(x) {
        if (!this.isTraining || this.dropoutRate === 0) return x;

        // 处理二维数组
        if (Array.isArray(x[0])) {
            return x.map(row => row.map(val => (Math.random() > this.dropoutRate ? val / (1 - this.dropoutRate) : 0)));
        }

        // 处理一维数组
        return x.map(val => (Math.random() > this.dropoutRate ? val / (1 - this.dropoutRate) : 0));
    }
}

// ==============================================================================
// 步骤 2：编码器层（EncoderLayer）
// ==============================================================================

class EncoderLayer {
    /**
     * Transformer 编码器的单层结构
     *
     * 结构：
     *   输入 x
     *     ↓
     *   [多头自注意力] ← 让序列中每个词关注所有词
     *     ↓ (残差连接 + 层归一化)
     *   [前馈网络] ← 独立处理每个位置
     *     ↓ (残差连接 + 层归一化)
     *   输出
     *
     * @param {number} dModel - 模型维度（默认 512）
     * @param {number} numHeads - 注意力头数（默认 8）
     * @param {number} dFF - 前馈网络隐藏层维度（默认 2048）
     * @param {number} dropout - Dropout 比率（默认 0.1）
     */
    constructor(dModel = 512, numHeads = 8, dFF = 2048, dropout = 0.1) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;

        // 子模块初始化
        this.selfAttn = new MultiHeadAttention(dModel, numHeads);
        this.feedForward = new PositionWiseFeedForward(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.dropout = new Dropout(dropout);
    }

    /**
     * 前向传播
     *
     * @param {Array} x - 输入序列（形状：[seq_len, d_model]）
     * @param {Array} mask - 掩码矩阵（可选，用于处理 padding）
     * @returns {Array} 输出序列
     */
    forward(x, mask = null) {
        console.log('--- EncoderLayer Forward Pass ---');

        // 步骤 1: 多头自注意力 + 残差连接 + 层归一化
        console.log('Step 1: Multi-Head Self-Attention');
        const attnOutput = this.selfAttn.forward(x, x, x, mask);
        // 残差连接：x + dropout(attn_output)
        const attnWithResidual = this.addResidual(x, this.dropout.forward(attnOutput));
        // 层归一化
        x = this.norm1.forward(attnWithResidual);

        // 步骤 2: 前馈网络 + 残差连接 + 层归一化
        console.log('Step 2: Feed-Forward Network');
        const ffOutput = this.feedForward.forward(x);
        // 残差连接：x + dropout(ff_output)
        const ffWithResidual = this.addResidual(x, this.dropout.forward(ffOutput));
        // 层归一化
        x = this.norm2.forward(ffWithResidual);

        console.log('EncoderLayer output shape:', x.length);
        return x;
    }

    /**
     * 辅助函数：残差连接
     * 作用：保留原始信息，防止梯度消失
     */
    addResidual(x, residual) {
        if (Array.isArray(x[0])) {
            // 二维数组
            return x.map((row, i) => row.map((val, j) => val + residual[i][j]));
        } else {
            // 一维数组
            return x.map((val, i) => val + residual[i]);
        }
    }
}

// ==============================================================================
// 步骤 3：解码器层（DecoderLayer）
// ==============================================================================

class DecoderLayer {
    /**
     * Transformer 解码器的单层结构
     *
     * 结构：
     *   输入 x                    编码器输出
     *     ↓                           ↓
     *   [掩码自注意力] ← 防止看到未来    |
     *     ↓ (残差 + 归一化)            |
     *   [交叉注意力] ←─────────────────┘ (关注源序列)
     *     ↓ (残差 + 归一化)
     *   [前馈网络]
     *     ↓ (残差 + 归一化)
     *   输出
     *
     * @param {number} dModel - 模型维度
     * @param {number} numHeads - 注意力头数
     * @param {number} dFF - 前馈网络隐藏层维度
     * @param {number} dropout - Dropout 比率
     */
    constructor(dModel = 512, numHeads = 8, dFF = 2048, dropout = 0.1) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;

        // 子模块初始化
        this.selfAttn = new MultiHeadAttention(dModel, numHeads); // 掩码自注意力
        this.crossAttn = new MultiHeadAttention(dModel, numHeads); // 交叉注意力（新增）
        this.feedForward = new PositionWiseFeedForward(dModel, dFF);
        this.norm1 = new LayerNorm(dModel);
        this.norm2 = new LayerNorm(dModel);
        this.norm3 = new LayerNorm(dModel); // 多一个归一化层
        this.dropout = new Dropout(dropout);
    }

    /**
     * 前向传播
     *
     * @param {Array} x - 解码器输入（目标序列）
     * @param {Array} encoderOutput - 编码器输出（源序列）
     * @param {Array} srcMask - 源序列掩码（处理 padding）
     * @param {Array} tgtMask - 目标序列掩码（防止看到未来）
     * @returns {Array} 输出序列
     */
    forward(x, encoderOutput, srcMask = null, tgtMask = null) {
        console.log('--- DecoderLayer Forward Pass ---');

        // 步骤 1: 掩码自注意力 + 残差 + 归一化
        console.log('Step 1: Masked Multi-Head Self-Attention');
        const attnOutput = this.selfAttn.forward(x, x, x, tgtMask);
        const attnWithResidual = this.addResidual(x, this.dropout.forward(attnOutput));
        x = this.norm1.forward(attnWithResidual);

        // 步骤 2: 交叉注意力 + 残差 + 归一化（关键！）
        console.log('Step 2: Cross-Attention (Encoder-Decoder Attention)');
        // Query 来自解码器，Key/Value 来自编码器
        const crossAttnOutput = this.crossAttn.forward(
            x, // query: 解码器当前状态
            encoderOutput, // key: 编码器输出
            encoderOutput, // value: 编码器输出
            srcMask
        );
        const crossWithResidual = this.addResidual(x, this.dropout.forward(crossAttnOutput));
        x = this.norm2.forward(crossWithResidual);

        // 步骤 3: 前馈网络 + 残差 + 归一化
        console.log('Step 3: Feed-Forward Network');
        const ffOutput = this.feedForward.forward(x);
        const ffWithResidual = this.addResidual(x, this.dropout.forward(ffOutput));
        x = this.norm3.forward(ffWithResidual);

        console.log('DecoderLayer output shape:', x.length);
        return x;
    }

    /**
     * 辅助函数：残差连接
     */
    addResidual(x, residual) {
        if (Array.isArray(x[0])) {
            return x.map((row, i) => row.map((val, j) => val + residual[i][j]));
        } else {
            return x.map((val, i) => val + residual[i]);
        }
    }
}

// ==============================================================================
// 步骤 4：完整的 Transformer 模型
// ==============================================================================

class Transformer {
    /**
     * 完整的 Transformer 模型（编码器-解码器架构）
     *
     * 用途：序列到序列任务（机器翻译、文本摘要等）
     *
     * @param {number} numLayers - 编码器/解码器的层数（默认 6）
     * @param {number} dModel - 模型维度（默认 512）
     * @param {number} numHeads - 注意力头数（默认 8）
     * @param {number} dFF - 前馈网络维度（默认 2048）
     * @param {number} dropout - Dropout 比率（默认 0.1）
     */
    constructor(numLayers = 6, dModel = 512, numHeads = 8, dFF = 2048, dropout = 0.1) {
        // 初始化 N 层编码器
        this.encoderLayers = Array.from({ length: numLayers }, () => new EncoderLayer(dModel, numHeads, dFF, dropout));

        // 初始化 N 层解码器
        this.decoderLayers = Array.from({ length: numLayers }, () => new DecoderLayer(dModel, numHeads, dFF, dropout));

        this.numLayers = numLayers;
    }

    /**
     * 编码器前向传播
     * @param {Array} src - 源序列
     * @param {Array} srcMask - 源序列掩码
     */
    encode(src, srcMask = null) {
        console.log(`\n=== Encoder: Processing ${this.numLayers} layers ===`);
        let output = src;

        for (let i = 0; i < this.encoderLayers.length; i++) {
            console.log(`\nEncoder Layer ${i + 1}/${this.numLayers}`);
            output = this.encoderLayers[i].forward(output, srcMask);
        }

        return output;
    }

    /**
     * 解码器前向传播
     * @param {Array} tgt - 目标序列
     * @param {Array} encoderOutput - 编码器输出
     * @param {Array} srcMask - 源序列掩码
     * @param {Array} tgtMask - 目标序列掩码
     */
    decode(tgt, encoderOutput, srcMask = null, tgtMask = null) {
        console.log(`\n=== Decoder: Processing ${this.numLayers} layers ===`);
        let output = tgt;

        for (let i = 0; i < this.decoderLayers.length; i++) {
            console.log(`\nDecoder Layer ${i + 1}/${this.numLayers}`);
            output = this.decoderLayers[i].forward(output, encoderOutput, srcMask, tgtMask);
        }

        return output;
    }

    /**
     * 完整的前向传播
     * @param {Array} src - 源序列（如 "我爱北京"）
     * @param {Array} tgt - 目标序列（如 "I love"）
     */
    forward(src, tgt, srcMask = null, tgtMask = null) {
        console.log('\n========== Transformer Forward Pass ==========');

        // 编码阶段
        const encoderOutput = this.encode(src, srcMask);

        // 解码阶段
        const decoderOutput = this.decode(tgt, encoderOutput, srcMask, tgtMask);

        console.log('\n========== Forward Pass Complete ==========\n');
        return decoderOutput;
    }
}

// ==============================================================================
// 步骤 5：使用示例
// ==============================================================================

/**
 * 示例 1：位置编码测试
 */
function testPositionalEncoding() {
    console.log('\n' + '='.repeat(60));
    console.log('示例 1：位置编码（Positional Encoding）');
    console.log('='.repeat(60));

    const dModel = 8; // 简化维度便于观察
    const seqLen = 5;

    // 创建位置编码器
    const posEncoder = new PositionalEncoding(dModel, 0, 100);

    // 可视化位置编码模式
    posEncoder.visualize(10, 8);

    // 创建模拟输入（词嵌入）
    const input = Array(seqLen)
        .fill(0)
        .map(() => Array(dModel).fill(1.0));
    console.log('\n输入词嵌入（全 1）:');
    console.log(input.slice(0, 2).map(row => row.map(v => v.toFixed(2))));

    // 添加位置编码
    const output = posEncoder.forward(input);
    console.log('\n添加位置编码后:');
    console.log(output.slice(0, 2).map(row => row.map(v => v.toFixed(2))));
}

/**
 * 示例 2：前馈网络测试
 */
function testFeedForward() {
    console.log('\n' + '='.repeat(60));
    console.log('示例 2：位置前馈网络（Feed-Forward Network）');
    console.log('='.repeat(60));

    const dModel = 512;
    const dFF = 2048;
    const seqLen = 3;

    // 创建前馈网络
    const ffn = new PositionWiseFeedForward(dModel, dFF, 0);

    // 模拟输入
    const input = Array(seqLen)
        .fill(0)
        .map(() =>
            Array(dModel)
                .fill(0)
                .map(() => Math.random() * 0.1)
        );

    console.log(`\n输入形状: (${seqLen}, ${dModel})`);
    console.log(
        '输入样例（前 3 个维度）:',
        input[0].slice(0, 3).map(v => v.toFixed(4))
    );

    // 前向传播
    const output = ffn.forward(input);

    console.log(`\n输出形状: (${output.length}, ${output[0].length})`);
    console.log(
        '输出样例（前 3 个维度）:',
        output[0].slice(0, 3).map(v => v.toFixed(4))
    );
    console.log(`\n变换路径: ${dModel} → ${dFF} → ${dModel}`);
}

/**
 * 示例 3：多头注意力测试
 */
function testMultiHeadAttention() {
    console.log('\n' + '='.repeat(60));
    console.log('示例 3：多头注意力（Multi-Head Attention）');
    console.log('='.repeat(60));

    const dModel = 64; // 简化维度
    const numHeads = 4;
    const seqLen = 3;

    // 创建多头注意力
    const mha = new MultiHeadAttention(dModel, numHeads);

    // 模拟输入序列
    const Q = Array(seqLen)
        .fill(0)
        .map(() =>
            Array(dModel)
                .fill(0)
                .map(() => Math.random() * 0.1)
        );
    const K = Q; // 自注意力：K = Q
    const V = Q; // 自注意力：V = Q

    console.log(`\n配置: d_model=${dModel}, num_heads=${numHeads}, d_k=${dModel / numHeads}`);
    console.log(`输入序列长度: ${seqLen}`);
    console.log('每个头的维度:', dModel / numHeads);

    // 前向传播
    const output = mha.forward(Q, K, V, null);

    console.log(`\n输出形状: (${output.length}, ${output[0].length})`);
    console.log(`✓ ${numHeads} 个注意力头并行计算完成`);
}

/**
 * 示例 4：完整 Transformer 测试（机器翻译）
 */
function exampleUsage() {
    console.log('\n' + '='.repeat(60));
    console.log('示例 4：完整 Transformer 模型（机器翻译）');
    console.log('='.repeat(60));

    // 初始化模型
    const model = new Transformer(
        2, // 简化为 2 层（实际 GPT-3 有 96 层）
        512, // 每个 token 的向量维度
        8, // 8 个注意力头
        2048, // 前馈网络维度
        0.1 // dropout
    );

    // 模拟输入（实际应该是词嵌入向量）
    const srcSequence = Array(4)
        .fill(0)
        .map(() => Array(512).fill(0.5)); // "我爱北京天安门"
    const tgtSequence = Array(2)
        .fill(0)
        .map(() => Array(512).fill(0.3)); // "I love"

    console.log('\n源序列长度:', srcSequence.length);
    console.log('目标序列长度:', tgtSequence.length);
    console.log(`模型配置: ${model.numLayers} 层编码器 + ${model.numLayers} 层解码器`);

    // 前向传播
    const output = model.forward(srcSequence, tgtSequence);

    console.log('\n输出序列长度:', output.length);
    console.log('输出维度:', output[0] ? output[0].length : 'N/A');
}

/**
 * 运行所有示例
 */
function runAllExamples() {
    console.log('\n');
    console.log('█'.repeat(60));
    console.log('  Transformer 架构完整测试');
    console.log('█'.repeat(60));

    // 测试各个模块
    testPositionalEncoding();
    testFeedForward();
    testMultiHeadAttention();

    // 注意：完整模型计算量大，如需测试请取消下面注释
    exampleUsage();

    console.log('\n' + '█'.repeat(60));
    console.log('  模块测试完成！');
    console.log('  提示：完整 Transformer 模型计算量较大');
    console.log('  如需测试，请调用 exampleUsage() 函数');
    console.log('█'.repeat(60) + '\n');
}

// ==============================================================================
// 步骤 6：架构对比总结
// ==============================================================================

/**
 * 编码器 vs 解码器对比
 *
 * ┌─────────────────┬──────────────┬──────────────┐
 * │      特性       │  EncoderLayer │ DecoderLayer │
 * ├─────────────────┼──────────────┼──────────────┤
 * │ 子层数量        │      2       │      3       │
 * │ 自注意力        │     ✓        │     ✓        │
 * │ 交叉注意力      │     ✗        │     ✓        │
 * │ 前馈网络        │     ✓        │     ✓        │
 * │ 掩码            │  可选(padding)│  必须(未来)  │
 * │ 用途            │  理解源序列   │  生成目标序列│
 * └─────────────────┴──────────────┴──────────────┘
 *
 * 典型应用：
 * - 仅编码器（BERT）: 文本分类、命名实体识别
 * - 仅解码器（GPT）: 文本生成、对话系统 ← 你的 Agent 使用的
 * - 编码器-解码器（T5）: 机器翻译、文本摘要
 */

runAllExamples();
