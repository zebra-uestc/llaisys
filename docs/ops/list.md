# 算子列表
目前仅支持 `Float32`、`Float16` 和 `BFloat16` 数据类型。

## Add（加法）

```cpp
void add(tensor_t c, tensor_t a, tensor_t b);
```

对张量 `a` 和 `b` 执行逐元素加法，结果存入张量 `c`。所有输入与输出张量必须**具有相同形状且为连续内存布局**。

## Argmax（取最大值索引）

```cpp
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
```

在 1D 输入张量 `vals` 中查找最大值及其对应索引。最大值写入 `max_val`，其索引写入 `max_idx`。`max_val` 与 `max_idx` 均为包含单个元素的 1D 张量。输入 `vals` 必须是 1D 且连续的张量。

## Embedding（嵌入查找）

```cpp
void embedding(tensor_t out, tensor_t index, tensor_t weight);
```

根据 1D 的 `index` 向量（数据类型为 `int64`），从 2D 的 `weight` 矩阵中按行索引取出对应行，结果写入 2D 输出张量 `out`。`weight` 必须是 2D 连续张量，`index` 必须是 1D 的 `int64` 张量。

## Linear（线性变换）

```cpp
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
```

$$ Y = XW^T + b $$

计算线性变换 $Y = XW^T + b$，其中：

- `in` 即 $X$，为 2D 连续输入张量；
- `weight` 即 $W$，为 2D 连续权重张量（无需预先转置）；
- `bias` 即 $b$，为可选的 1D 偏置张量（若未提供，则不加偏置）；
- `out` 即 $Y$，为 2D 连续输出张量。

## RMS Normalization（均方根归一化）

```cpp
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
```

对输入张量 `in` 的每一行沿最后一个维度进行 RMS 归一化，公式为：

$$
Y_i = \frac{W \odot  X_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d X_{i,j}^2 + \varepsilon}}
$$

其中：

- `in` 即 $X$，为 2D 连续输入张量；
- `weight` 即 $W$，为长度等于输入行宽 $d$ 的 1D 权重张量；
- `eps` 即 $\varepsilon$，为防止除零的小常数；
- `out` 即 $Y$，为 2D 连续输出张量。

## RoPE（旋转位置编码）

```cpp
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
```
为输入张量`in`的每个向量（这些向量与 pos_ids 中的位置 id 相对应）计算以下内容：

设 $\mathbf{x}_i = [\mathbf{a}_i, \mathbf{b}_i] \in \mathbb{R}^d$ 为输入向量， $\mathbf{y}_i = [\mathbf{a}'_i, \mathbf{b}'_i] \in \mathbb{R}^d$ 为索引 $i$ 处的输出向量，其中 $\mathbf{a}_i, \mathbf{b}_i,\mathbf{a}'_i, \mathbf{b}'_i \in \mathbb{R}^{d/2}$ 。

设 $\theta$ 为固定基数（例如 $\theta = 10000$）， $j = 0, 1, \ldots, d/2 - 1$。

设 $p_i \in \mathbb{N}$ 是输入索引 $i$ 处token的位置id。

那么RoPE的角度为 $\phi_{i,j} = \frac{p_i}{\theta^{2j/d}}$

输出向量 $\mathbf{y}_i = [\mathbf{a}'_i, \mathbf{b}'_i]$ 计算如下：

$$a_{i,j}' = a_{i,j} \cos(\phi_{i,j}) - b_{i,j} \sin(\phi_{i,j})$$

$$b_{i,j}' = b_{i,j} \cos(\phi_{i,j}) + a_{i,j} \sin(\phi_{i,j})$$

- `out`：编码后的查询（Q）或键（K）张量。形状应该是 `[seqlen, nhead, d]` 或 `[seqlen, nkvhead, d]`。暂时可以假设张量是连续的。
- `in`：原始查询（Q）或键（K）张量。形状应该是 `[seqlen, nhead, d]` 或 `[seqlen, nkvhead, d]`。暂时可以假设张量是连续的。
- `pos_ids`：输入序列中每个token的位置id（整个上下文中的索引）。形状应该是 `[seqlen,]`，dtype应该是`int64`。
- `theta`：频率向量的基值（如 10000）。
## Self-Attention（自注意力）

```cpp
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
```
为查询张量`q`、键张量`k`和值张量`v`计算带因果掩码的自注意力。

$$
A = Q K^\top * scale \\
$$

$$
Y = \mathrm{causalsoftmax}(A) \cdot V \\
$$

- `attn_val`：结果注意力值张量。形状应该是`[seqlen, nhead, dv]`。暂时可以假设张量是连续的。
- `q`：查询张量。形状应该是 `[seqlen, nhead, d]`。暂时可以假设张量是连续的。
- `k`：键张量。形状应该是 `[total_len, nkvhead, d]`。暂时可以假设张量是连续的。
- `v`：值张量。形状应该是 `[total_len, nkvhead, dv]`。暂时可以假设张量是连续的。
- `scale`：缩放因子。在大多数情况下取值为 $\frac{1}{\sqrt{d}}$ 。

> `total_len` = `past_len` + `seq_len`，计算当前批次`seq_len`个token的attention时，需要注意到前面的`past_len`个token以及当前批次的`1..seq_len`个token的键（K）,此处需要注意 kvcache 的拼接。

## SwiGLU（Swish-Gated Linear Unit）

```cpp
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
```

计算 SwiGLU 激活函数：

$$
out_{i} = up_{i} \odot \frac { gate_{i}}{1 + e^{-gate_{i}}}
$$

$e^{−gate_i}$ 表示对 $gate_i$ 向量中每个元素单独进行指数运算。

其中 `gate`、`up` 和 `out` 均为形状相同的 2D 连续张量，形状为 `[seqlen, intermediate_size]`。

## Rearrange（重排）

```cpp
void rearrange(tensor_t out, tensor_t in);
```

将数据从输入张量 `in` 复制到输出张量 `out`，两者具有相同逻辑形状但可能具有不同的内存步长（strides）。该算子可用于实现张量的 `contiguous()` 功能，确保输出为连续内存布局。