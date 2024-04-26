# 基于改进的Informer模型的时间序列回归预测分析

## 算法思路

采取Informer模型进行ETTh数据集中指标的预测，并且在Informer模型的基础上加入了滑动时间窗口改进。Informer模型的实质是注意力机制+Transformer模型，Informer模型的核心思想是将输入序列进行自注意力机制的处理，以捕捉序列中的长期依赖关系，并利用Transformer的编码器-解码器结构进行预测。

### Informer模型

#### Informer模型的特点

Informer是一种用于长序列时间序列预测的Transformer模型,但是它与传统的Transformer模型又有些不同点，与传统的Transformer模型相比，Informer具有以下几个独特的特点：

1. ProbSparse自注意力机制：Informer引入了ProbSparse自注意力机制，该机制在时间复杂度和内存使用方面达到了O(Llog L)的水平，能够有效地捕捉序列之间的长期依赖关系。
2. 自注意力蒸馏：通过减少级联层的输入，自注意力蒸馏技术可以有效处理极长的输入序列，提高了模型处理长序列的能力。
3. 生成式解码器：Informer采用生成式解码器，可以一次性预测整个长时间序列，而不是逐步进行预测。这种方式大大提高了长序列预测的推理速度。



#### Informer的机制原理

 Informer模型概述见下图，左侧：编码器接收大规模的长序列输入（绿色序列），使用提出的ProbSparse自注意力替代传统的自注意力。蓝色梯形表示自注意力蒸馏操作，用于提取主导的注意力，大幅减小网络大小。层堆叠的副本增加了模型的稳健性。右侧：解码器接收长序列输入，将目标元素填充为零，测量特征图的加权注意力组合，并以生成式风格即时预测输出元素（橙色序列）。

![image-20240420000312286](C:\Users\HUAWEI\AppData\Roaming\Typora\typora-user-images\image-20240420000312286.png)

Informer模型编码器结构的视觉表示如下图，以下是对其内容的解释：

- 编码器堆栈：图像中的水平堆栈代表Informer编码器结构中的一个编码器副本。每个堆栈都是一个独立单元，处理部分或全部输入序列。

- 主堆栈：图中显示的主堆栈处理整个输入序列。主堆栈之后，第二个堆栈处理输入序列的一半，以此类推，每个后续的堆栈都处理上一个堆栈输入的一半。

- 点积矩阵：堆栈内的红色层是点积矩阵，它们是自注意力机制的一部分。通过在每一层应用自注意力蒸馏，这些矩阵的大小逐层递减，可能降低了计算复杂度，并集中于序列中最相关的信息。

- 输出的拼接：通过自注意力机制处理后，所有堆栈的特征图被拼接起来，形成编码器的最终输出。然后，模型的后续部分（如解码器）通常使用这个输出，基于输入序列中学习到的特征和关系生成预测。

  

总体来说，Informer模型，成功提高了在LSTF问题中的预测能力，验证了类似Transformer的模型在捕捉长序列时间序列输出和输入之间的个体长期依赖关系方面的潜在价值。

提出了ProbSparse自注意力机制，以高效地替代传统的自注意力机制,
提出了自注意力蒸馏操作，可优化J个堆叠层中主导的注意力得分，并将总空间复杂度大幅降低。
提出了生成式风格的解码器，只需要一步前向传播即可获得长序列输出，同时避免在推理阶段累积误差的传播。



## 代码结构

#### 嵌入（embedding）

在给出的数据集中一共有8列，第一列日期(date)会处理成4维向量(freq='h'的情况下)作为时间戳输入到Encoder和Decoder中进行embedding。如果进行多变量预测任务，则预测为后7列变量的值，如果进行的是单变量预测任务，则预测最后一列变量的值。Encoder和Decoder输入的embedding包括三个部分：数据的embedding(下图中Scalar部分)、位置编码(下图中Local Time Stamp部分)以及时间戳编码(下图中Global Time Stamp部分)。

![image-20240420081613837](C:\Users\HUAWEI\AppData\Roaming\Typora\typora-user-images\image-20240420081613837.png)

对输入的原始数据进行一个1维卷积得到，将输入数据从$C_{in}(=7)$维映射为$d_{model}(=512)$维。

```python
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
```

#### 位置编码

输入序列中的元素是一起处理的，而不是像RNN一样是一个一个处理的，这样做加快了速度但是忽略了序列中元素的先后关系，这个时候就需要位置编码。

位置编码的公式为：
$$
PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}})\\ PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}})
$$

``` python
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
```

#### 时间戳编码

对时间戳的编码主要分为TemporalEmbedding和TimeFeatureEmbedding这两种方式，前者使用month_embed、day_embed、weekday_embed、hour_embed和minute_embed(可选)多个embedding层处理输入的时间戳，将结果相加；后者直接使用一个全连接层将输入的时间戳映射到512维的embedding。

TemporalEmbedding中的embedding层可以使用Pytorch自带的embedding层，再训练参数，也可以使用定义的FixedEmbedding，它使用位置编码作为embedding的参数，不需要训练参数。

```python
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)
```

最后将这三部分的embedding加起来，就得到了最终的embedding。

``` python
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
```

#### 注意力机制

使用了两种attention，一种是普通的多头自注意力层(FullAttention)，一种是Informer新提出来的ProbSparse self-attention层(ProbAttention)。

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
```

Informer模型中提出了一种新的注意力层——ProbSparse Self-Attention，Q，K，V为输入的embedding分别乘上一个权重矩阵得到的query、key、value。ProbSparse Self-Attention首先对K进行采样，得到K_sample，对每个$q_i\in Q$关于K_sample求M值：
$$
M(q_i,K)=\max_j\{\frac{q_ik_k^\top}{\sqrt{d}}\}-\frac{1}{L_K}\sum_{j=1}^{L_K}\frac{q_ik_k^\top}{\sqrt{d}}
$$
找到M值最大的u个 $q_i$，对这Top-u个 $q_i$ 关于K求score值：
$$
\mathcal{A}(Q,K,V)=Softmax(\frac{\bar{Q}K^\top}{\sqrt{d}})V
$$
其中$\bar{Q}$是Top-u的$q_i$组成的矩阵，这样就得到了$S_0$ ，对于没有被选中的那些 $q_i$的score值取$mean(V)$。其实现过程如下：

```python
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn
```

AttentionLayer是定义的attention层，会先将输入的embedding分别通过线性映射得到query、key、value。还将输入维度$d_{model}$划分为多头，接着就执行前面定义的attention操作，最后经过一个线性映射得到输出。

```python
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
```

#### 编码器（Encoder）

ConvLayer类实现的是Informer中的Distilling操作，本质上就是一个1维卷积+ELU激活函数+最大池化。公式如下：
$$
X_{j+1}^t=MaxPool(ELU(Conv1d[X_j^t]_{AB}))
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
```

EncoderLayer类实现的是一个Encoder层，整体架构和Transformer是大致相同的，主要包含两个子层：多头注意力层(Informer中改为提出的ProbSparse Self-Attention层)和两个线性映射组成的前馈层(Feed Forward)，两个子层后都带有一个批量归一化层，子层之间有跳跃连接。

```python
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn
```

Encoder类是将前面定义的Encoder层和Distilling操作组织起来，形成一个Encoder模块。其中distilling层总比EncoderLayer少一层，即最后一层EncoderLayer后不再做distilling操作。

```python
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
```

为了增强distilling操作的鲁棒性，文章中提到可以采用多个replicas并行执行，不同replicas采用不同长度的embedding(L、L/2、L/4、...)，embedding长度减半对应的attention层也减少一层，distilling层也会随之减少一层，最终得到的结果拼接起来作为输出。

```python
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
```

#### 解码器（Decoder）

Decoder部分结构可以参考Transformer中的Decoder结构，包括两层attention层和一个两层线性映射的Feed Forward部分。

需要注意的是，第一个attention层中的query、key、value都是根据Decoder输入的embedding乘上权重矩阵得到的，而第二个attention层中的query是根据前面attention层的输出乘上权重矩阵得到的，key和value是根据Encoder的输出乘上权重矩阵得到的。

```python
import torch

import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
```

#### Mask机制

mask机制用在Decoder的第一个attention层中，目的是为了保证t时刻解码的输出只依赖于t时刻之前的输出。

生成的mask矩阵右上角部分为1（不包括对角线），将mask矩阵作用到score矩阵上会使得mask矩阵中为1的位置在score矩阵中为 −∞ ，这样softmax后就为0。TriangularCausalMask是用在Fullattention层上的，ProbMask是用在ProbSparseAttention层上的。

```python
import torch

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
```



## 结果展示

更详细的结果放在了压缩包中。总体预测结果如下所示，其中每一列都代表了一个独立的指标。

![](总体预测.png)

各个指标的预测结果如下所示，其中对于每一个预测结果，我们以2000作为一个数据的批次，将17500个数据点分为了9张图展示。

#### HULL的预测

![](\new\HULL_Chunk_9.png)

#### OT的预测

![](\new\OT_Chunk_9.png)

#### MULL的预测

![](\new\MULL_Chunk_9.png)

#### MUFL的预测

![](\new\MUFL_Chunk_9.png)



#### LULL的预测

![](\new\LULL_Chunk_9.png)

#### LUFL的预测

![](\new\LUFL_Chunk_9.png)

#### HUFL的预测

![](\new\HUFL_Chunk_9.png)



## 模型的评估

Mean Squared Error (MSE) 和 Mean Absolute Error (MAE) 是用来评估估计器、预测或模型质量的统计度量。MSE 是错误的平方的平均值，即预估值和实际值之间平方差的平均值。而 MAE 则是预测值和实际观察值之间绝对差异的平均值，其计算公式如下：
$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2,\\
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|.
$$
下图展示了部分滚动时间内的MAE和MSE数值：

![img](data:image/png;base64,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RERERETKx8BJZmzjRERERESkfAycFIIlTkREREREysXASWasqkdEREREpHwMnGTGqnpERERERMrHwEkhWOJERERERKRcDJxkxqp6RERERETKx8BJZgyciIiIiIiUj4GTzNjGiYiIiIhI+Rg4KQRLnIiIiIiIlIuBk8xYVY+IiIiISPkYOMmMgRMRERERkfIxcJIZ2zgRERERESkfAyeFYIkTEREREZFyMXCSGavqEREREREpHwMnmbGqHhERERGR8jFwUgiWOBERERERKRcDJ5mxqh4RERERkfIxcJIZAyciIiIiIuVj4CQztnEiIiIiIlI+Bk4KwRInIiIiIiLlYuAkM1bVIyIiIiJSPgZOMmPgRERERESkfAycZMY2TkREREREysfASSFY4kREREREpFwMnGTGqnpERERERMrHwElmrKpHRERERKR8DJwUgiVORERERETKxcBJZqyqR0RERESkfAycZMbAiYiIiIhI+Rg4yYxtnIiIiIiIlI+Bk0KwxImIiIiISLkYOMmMVfWIiIiIiJSPgZPMGDgRERERESkfAyeZsY0TEREREZHyMXBSCJY4EREREREpFwMnmbGqHhERERGR8jFwkhmr6hERERERKR8DJ4VgiRMRERERkXIxcJIZq+oRERERESkfAyeZMXAiIiIiIlI+Bk4yYxsnIiIiIiLlY+CkECxxIiIiIiJSLgZOMmNVPSIiIiIi5WPgJDMGTkREREREysfASWZs40REREREpHwMnBSCJU5ERERERMrFwElmrKpHRERERKR8DJxkxqp6RERERETKx8BJIVjiRERERESkXAycZMaqekREREREysfASWYMnIiIiIiIlE9xgdNXX30FlUql81e8eHHN/Ldv32LAgAEICgpC5syZERERgejoaBlT7Bi2cSIiIiIiUj7FBU4A8N577+HRo0eav0OHDmnmDR06FJs2bcLatWuxf/9+PHz4EG3atJExtc7BEiciIiIiIuXykjsBxnh5eSEkJMRgemxsLBYtWoRVq1ahXr16AIDIyEiUKFECx44dQ7Vq1dI6qQ5jVT0iIiIiIuVTZInT9evXkTt3bhQsWBCdO3fG3bt3AQCnTp1CUlISGjRooFm2ePHiCA0NxdGjR01uLyEhAXFxcTp/SsHAiYiIiIhI+RQXOFWtWhVLlizB9u3bMW/ePNy6dQu1atXCy5cvERUVhQwZMiAwMFBnneDgYERFRZnc5vTp0xEQEKD5y5cvn4s/hfXYxomIiIiISPkUV1WvadOmmtdlypRB1apVkT9/fvz+++/ImDGjXdscM2YMhg0bpnkfFxenqOAJYIkTEREREZGSKa7ESV9gYCCKFi2Kf//9FyEhIUhMTERMTIzOMtHR0UbbRKn5+PjA399f508pWFWPiIiIiEj5FB84vXr1Cjdu3ECuXLlQsWJFeHt7Y/fu3Zr5V69exd27dxEeHi5jKu3HqnpERERERMqnuKp6I0aMQIsWLZA/f348fPgQEyZMgKenJzp27IiAgAD07t0bw4YNQ7Zs2eDv749BgwYhPDzcLXvUk2KJExERERGRcikucLp//z46duyIZ8+eIUeOHKhZsyaOHTuGHDlyAABmzpwJDw8PREREICEhAY0bN8bcuXNlTrX9WFWPiIiIiEj5VILw7j2yx8XFISAgALGxsbK3d9q/H6hTByheHLh8WdakEBERERG9c6yNDRTfxim9YxsnIiIiIiLlY+CkEO9euR8RERERkftg4CQztnEiIiIiIlI+Bk4yY+BERERERKR8DJxkxjZORERERETKx8BJIVjiRERERESkXAycZMaqekREREREysfASWasqkdEREREpHwMnBSCJU5ERERERMrFwElmrKpHRERERKR8DJxkxsCJiIiIiEj5GDjJjG2ciIiIiIiUj4GTzLy9xf9v38qbDiIiIiIiMo2Bk8yyZRP/v3jB6npERERERErFwElm6sApIQF480betBARERERkXEMnGSWOTPg5SW+fv5c3rQQEREREZFxDJxkplIB/v7i65cv5U0LEREREREZx8BJATw9xf8pKfKmg4iIiIiIjGPgpAAMnIiIiIiIlI2BkwIwcCIiIiIiUjYGTgqgDpySk+VNBxERERERGcfASQFY4kREREREpGwMnBSAgRMRERERkbIxcFIABk5ERERERMrGwEkBGDgRERERESkbAycFYOBERERERKRsDJwUgIETEREREZGyMXBSAHZHTkRERESkbAycFMDjv19BEORNBxERERERGcfASQFUKvE/AyciIiIiImVi4KQA6sCJiIiIiIiUiYGTgrDEiYiIiIhImRg4KQCr6hERERERKRsDJwVgVT0iIiIiImVj4KQgLHEiIiIiIlImBk4KYKmq3qNHQHx82qWHiIiI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hmzZssHf3x+DBg1CeHi42/aoZ4m7dQ5BRERERJQeKS5wevz4Mbp164ZHjx4hICAAZcqUwY4dO9CwYUMAwMyZM+Hh4YGIiAgkJCSgcePGmDt3rsypdkx6KnEiIiIiIkqPFBc4LVq0yOx8X19fzJkzB3PmzEmjFCmHUkucGPgRERERUXrnFm2c0jtzgUdqatqlw15KDeiIiIiIiJyFgZPCuUPgRERERESU3jFwUgBzJU7SwXBZskNEREREJA8GTgonDZyUim2ciIiIiCi9Y+CkANa2cVJqiRO7TCciIiKi9I6Bk8K5Q4kTEREREVF6x8BJAdy9Vz1W1SMiIiKi9I6BkwKkp84h3CGNRERERES2YuCkcKyqR0REREQkPwZOCuDunUMQEREREaV3DJwUzh3aOBERERERpXcMnBTA2jZOREREREQkDwZOCseqekRERERE8mPgpADpqcSJwR0RERERpUcMnBTO3QInIiIiIqL0iIGTAljbq17evK5PCxERERERGWLgpHDSEqf69eVLh7VYVY+IiIiI0iMGTgpgbYkTERERERHJg4GTwrGNExERERGR/Bg4KYC1veoptRqcufQTEREREaUHDJwUwN2r6ik1oCMiIiIichYGTgrHAXCJiIiIiOTHwEkBWFWPiIiIiEjZGDgpHDuHICIiIiKSHwMnBXD3EidpupSaRiIiIiIiRzBwUjgGIkRERERE8mPgpADWthFSahDFNk5ERERElN4xcFI4pQZLprhbeomIiIiIrMHASQHMldiw/RARERERkfwYOCmcfrAkCMCdOwyiiIiIiIjSEgMnBTBX4qQ/AO533wEFCgCff+7yZBERERER0X8YOCmcfsnSqFHi/xkz0j4tRERERETvKgZOCuDuveoREREREaV3DJwUzt2CJXdLLxERERGRNRg4KQB71SMiIiIiUjYGTgpgbecQSsUBcImIiIgovWPgpHDuUOKk1HQRERERETkLAycF0C+xKVVK+5pBCRERERGR/Bg4KVDu3Manu0MQ5Q5pJCIiIiKyFQMnBbC2cwilYhsnIiIiIkrvGDgpkDRYknYOodQgSqnpIiIiIiJyFgZOCuDuJU5EREREROkdAyeFc4de9VhVj4iIiIjSOwZOCqAfeLhDsERERERE9C5h4KRAZ89qX7tbEOUOaSQiIiIishUDJwXQL3F68kT72t0CEXdL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gK+//hqJiYn/fceJiOiZwsSJiIjatAcTQCxatEjtEr6cnBxs2LChXvtRo0bBxsYGy5cvx/79++vV19bWIjU1VXVfoVBgwoQJuH37NtatWwdra2ts2rQJHTp0wLvvvoubN282waiIiKi58TdORETUpvn5+eG3335DfHw8nJ2dMXz4cJSWliIiIgKvvvoqdu3apdZeJpNh69atGD58OIYOHYqXX34ZDg4OkEgkuHjxIg4cOAATExOcPn0aALB06VKkpKRg1qxZGDFiBADAzs4O4eHhmDhxIiZPnowdO3Y0+7iJiOjvxTNORETUpmloaCAmJgbz589HaWkpVq5ciUOHDmHFihWYO3dug4/p378/srKyEBAQgKKiInz//ff45ZdfcPr0aYwePRqrV68GoJyBb8mSJejTpw+WLVumto0JEybA19cXO3fuRHh4eJOPk4iImpZECCFauhNERERERETPMp5xIiIiIiIiagQTJyIiIiIiokYwcSIiIiIiImoEEyciIiIiIqJGMHEiIiIiIiJqBBMnIiIiIiKiRjBxIiIiIiIiagQTJyIiIiIiokYwcSIiIiIiImoEEyciIiIiIqJGMHEiIiIiIiJqBBMnIiIiIiKiRvwfOTypoMzpMKIAAAAASUVORK5CYII=)
