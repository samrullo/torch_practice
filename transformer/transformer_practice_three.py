import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from transformer.transformer_modules import Embeddings, Encoder, EncoderLayer, MultiHeadedAttention, \
    PositionwiseFeedForward, make_model, subsequent_mask, LabelSmoothing, NoamOpt, run_epoch, data_gen, \
    SimpleLossCompute, greedy_decode

V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)

folder = "/Users/samrullo/Documents/learning/data_science/nlp_related/pytorch_transformer"
file = "pytorch_transforfer_params"
model.load_state_dict(torch.load(os.path.join(folder, file)))
data_iter = data_gen(V, 1, 1)
batch = next(data_iter)
ys = greedy_decode(model, batch.src, batch.src_mask, 10, start_symbol=1)
print(f"batch.src : {batch.src}")
print(f"ys : {ys}")