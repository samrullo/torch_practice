import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformer.transformer_modules import Embeddings
from dataset.addition_dataset import AdditionSeqDataset
from torch.utils.data import DataLoader
from dataset import sequence
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s %(lineno)s]')

batch_size = 100
max_epoch = 25
train_set = AdditionSeqDataset(reverse=True)
train_loader = DataLoader(train_set, batch_size=batch_size)
test_set = AdditionSeqDataset(train=False, reverse=True)
test_loader = DataLoader(test_set, batch_size=1)
char_to_id, id_to_char = sequence.get_vocab()

V = len(char_to_id)
d_model = 16
n_head = 8
n_layers = 6


class Transformer(nn.Module):
    def __init__(self, d_model, vocab, n_head, n_layers):
        super(Transformer, self).__init__()
        self.src_embed = Embeddings(d_model, vocab)
        self.tgt_embed = Embeddings(d_model, vocab)
        self.fc = nn.Linear(d_model, vocab)
        self.transformer = nn.Transformer(d_model, n_head, n_layers,
                                          n_layers, dim_feedforward=d_model * 4,
                                          batch_first=True)

    def forward(self, src, tgt):
        src_embedding = self.src_embed(src)
        tgt_embedding = self.tgt_embed(tgt)
        out = self.transformer(src_embedding, tgt_embedding)
        score = self.fc(out)
        return score


model = Transformer(d_model, V, n_head, n_layers)
folder = "/Users/samrullo/Documents/learning/data_science/nlp_related/pytorch_transformer"
file = "pytorch_nn_transformer_addition_seq_params"
model.load_state_dict(torch.load(os.path.join(folder, file)))
model.eval()

src_text = "512+121"
src = torch.tensor([[char_to_id[char] for char in src_text]])
src_embedding = model.src_embed(src)
encoded = model.transformer.encoder(src_embedding, None)
decoded = model.transformer.decoder(encoded)
print(decoded)
