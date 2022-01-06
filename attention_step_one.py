import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clone(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# query, key, value = Variable(torch.randn(35, 512)), Variable(torch.randn(35, 512)), Variable(torch.randn(35, 512))
n_batches = 20
time_size = 35
d_k = 64
h = 8
d_model = 512
query = Variable(torch.randn(n_batches, time_size, d_model))
key = Variable(torch.randn(n_batches, time_size, d_model))
value = Variable(torch.randn(n_batches, time_size, d_model))

linears = clone(torch.nn.Linear(d_model, d_model), 4)

query_p, key_p, value_p = [l(x).view(n_batches, -1, h, d_k).transpose(1, 2) for l, x in
                           zip(linears, (query, key, value))]
print(f"query_p.shape : {query_p.shape}")

x, attn = attention(query_p, key_p, value_p)
print(f"x.shape {x.shape}, attn.shape : {attn.shape}")
