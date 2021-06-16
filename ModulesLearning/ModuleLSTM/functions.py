import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# def a_norm(Q, K):
#     m = torch.matmul(Q, K.transpose(2, 1).float())
#     # m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
#
#     # return torch.softmax(m, -1)
#     return m
#
#
# def attention(Q, K, V):
#     # Attention(Q, K, V) = norm(QK)V
#     a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)
#
#
#     mask = np.triu(np.ones(a.size()), k=1).astype('bool')
#     mask = torch.from_numpy(mask)
#     if torch.cuda.is_available():
#         mask = mask.cuda()
#     # do masked_fill_ on data rather than Variable because PyTorch doesn't
#     # support masked_fill_ w/-inf directly on Variables for some reason.
#     a.data.masked_fill_(mask, float('-inf'))
#     a = F.softmax(a, dim=1) / torch.sqrt(torch.tensor(Q.shape[-1]).float())
#
#     return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)
#
#
#
#
# class AttentionBlock(torch.nn.Module):
#     def __init__(self, dim_val, dim_attn):
#         super(AttentionBlock, self).__init__()
#         self.value = Value(dim_val, dim_val)
#         self.key = Key(dim_val, dim_attn)
#         self.query = Query(dim_val, dim_attn)
#
#     def forward(self, x, kv=None):
#         if (kv is None):
#             # Attention with x connected to Q,K and V (For encoder)
#             return attention(self.query(x), self.key(x), self.value(x))
#
#         # Attention with x as Q, external vector kv as K an V (For decoder)
#         return attention(self.query(x), self.key(kv), self.value(kv))
#
#
# class MultiHeadAttentionBlock(torch.nn.Module):
#     def __init__(self, dim_val, dim_attn, n_heads):
#         super(MultiHeadAttentionBlock, self).__init__()
#         self.heads = []
#         for i in range(n_heads):
#             self.heads.append(AttentionBlock(dim_val, dim_attn))
#
#         self.heads = nn.ModuleList(self.heads)
#
#         self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)
#
#     def forward(self, x, kv=None):
#         a = []
#         for h in self.heads:
#             a.append(h(x, kv=kv))
#
#         a = torch.stack(a, dim=-1)  # combine heads
#         a = a.flatten(start_dim=2)  # flatten all head outputs
#
#         x = self.fc(a)
#
#         return x
#
#
# class Value(torch.nn.Module):
#     def __init__(self, dim_input, dim_val):
#         super(Value, self).__init__()
#         self.dim_val = dim_val
#
#         self.fc1 = nn.Linear(dim_input, dim_val, bias=False)
#         # self.fc2 = nn.Linear(5, dim_val)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         # x = self.fc2(x)
#
#         return x
#
#
# class Key(torch.nn.Module):
#     def __init__(self, dim_input, dim_attn):
#         super(Key, self).__init__()
#         self.dim_attn = dim_attn
#
#         self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
#         # self.fc2 = nn.Linear(5, dim_attn)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         # x = self.fc2(x)
#
#         return x
#
#
# class Query(torch.nn.Module):
#     def __init__(self, dim_input, dim_attn):
#         super(Query, self).__init__()
#         self.dim_attn = dim_attn
#
#         self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
#         # self.fc2 = nn.Linear(5, dim_attn)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         # print(x.shape)
#         # x = self.fc2(x)
#
#         return x
#
#
# # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(1), :].squeeze(1)
#         return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None
        self.sqrt_k = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features]
        """
        if isinstance(self.layer, nn.MultiheadAttention):
            src = x.transpose(0, 1)     # [seq_len, N, features]

            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)

            output, self.attn_weights = self.layer(src, src, src, attn_mask=mask)

            output = output.transpose(0, 1)     # [N, seq_len, features]

        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = F.softmax(mask,dim=0) / self.sqrt_k
        return mask



class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size * 2, hidden_size, 1)
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.transpose(1, 2)
        tensor = self.conv(tensor)
        tensor = tensor.transpose(1, 2)

        return tensor


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()

        self.attention = ResidualBlock(
            nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout_rate
        )
        self.ffn = ResidualBlock(PositionWiseFeedForward(embed_dim), embed_dim, p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ffn(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2) -> None:
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x

class DenseInterpolation(nn.Module):
    def __init__(self, seq_len: int, factor: int) -> None:
        """
        :param seq_len: sequence length
        :param factor: factor M
        """
        super(DenseInterpolation, self).__init__()

        W = np.zeros((factor, seq_len), dtype=np.float32)

        for t in range(seq_len):
            s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
            for m in range(factor):
                tmp = np.array(1 - (np.abs(s - (1+m)) / factor), dtype=np.float32)
                w = np.power(tmp, 2, dtype=np.float32)
                W[m, t] = w

        W = torch.tensor(W).float().unsqueeze(0)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.repeat(x.shape[0], 1, 1).requires_grad_(False)
        u = torch.bmm(w, x)
        return u.transpose_(1, 2)
