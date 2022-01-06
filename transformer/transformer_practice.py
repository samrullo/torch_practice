import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from transformer.transformer_modules import Embeddings, Encoder, EncoderLayer, MultiHeadedAttention, \
    PositionwiseFeedForward, make_model, subsequent_mask

vocab_size = 1000
batch_size = 20
time_size = 35
d_model = 512
h = 8
ff_size = 2048

corpus = np.concatenate([np.arange(vocab_size), np.arange(vocab_size), np.arange(vocab_size)])
corpus = corpus.astype(np.int32)
np.random.shuffle(corpus)


def make_batch(time_seq_idx=0):
    jump = len(corpus) // batch_size
    x_batch = np.zeros((batch_size, time_size),dtype=np.int32)
    for b in range(batch_size):
        for t in range(time_size):
            pos_in_corpus = b * jump + time_seq_idx * time_size + t
            if pos_in_corpus >= len(corpus):
                break
            x_batch[b, t] = corpus[pos_in_corpus]
    return x_batch


# corpus_t = Variable(corpus)
src_batch_np = make_batch(0)
src_batch = torch.from_numpy(src_batch_np)
print(f"src_batch shape : {src_batch.shape}")

tgt_batch_np = make_batch(1)
tgt_batch = torch.from_numpy(tgt_batch_np)
tgt_mask = subsequent_mask(time_size)
print(f"tgt_batch shape : {tgt_batch.shape}")
# x_embed = Embeddings(d_model, vocab_size)(x)
# print(f"x_embed shape : {x_embed.shape}")
# self_attn = MultiHeadedAttention(h, d_model, dropout=0.1)
# ff = PositionwiseFeedForward(d_model, ff_size, dropout=0.1)
# enc_layer = EncoderLayer(d_model, self_attn, ff, dropout=0.1)
# x_encoded = Encoder(enc_layer, 6)(x_embed, mask=None)
# print(f"x_encoded shape {x_encoded.shape}")

model = make_model(src_vocab=vocab_size, tgt_vocab=vocab_size, N=6, d_model=d_model, d_ff=ff_size, h=h, dropout=0.1)
y_out = model(src_batch, tgt_batch, src_mask=None, tgt_mask=tgt_mask)
