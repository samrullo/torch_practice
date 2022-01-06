import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable

x = torch.rand(1, 5, 5)
print(f"x.shape : {x.shape}")


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


query = Variable(torch.randn(35, 64))
key = Variable(torch.randn(35, 64))
val = Variable(torch.randn(35, 64))

att_res, p_attn = attention(query, key, val)
