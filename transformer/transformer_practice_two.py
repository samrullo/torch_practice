import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from transformer.transformer_modules import Embeddings, Encoder, EncoderLayer, MultiHeadedAttention, \
    PositionwiseFeedForward, make_model, subsequent_mask, LabelSmoothing, NoamOpt, run_epoch, data_gen, \
    SimpleLossCompute

V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(100):
    model.train()
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))

folder = "/Users/samrullo/Documents/learning/data_science/nlp_related/pytorch_transformer"
file = "pytorch_transformer_params"
torch.save(model.state_dict(), os.path.join(folder, file))
