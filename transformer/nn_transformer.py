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
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(max_epoch):
    start = time.time()
    for i, (src, tgt) in enumerate(train_loader):
        score = model(src, tgt)

        optimizer.zero_grad()
        loss = 0
        for out_, tgt_ in zip(score, tgt):
            loss = loss + criterion(out_, tgt_)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            logging.info(f"epoch {epoch}/{max_epoch} | iter {i + 1}/{len(train_set)/batch_size} : {loss}")
    logging.info(f"epoch {epoch + 1}/{max_epoch} took {time.time() - start} seconds")


def eval_model(model,id_to_char):
    correct = 0
    total_count = 0
    model.eval()
    for i, (src, tgt) in enumerate(test_loader):
        score = model(src, tgt)
        for out_, tgt_ in zip(score, tgt):
            total_count += 1
            _,predicted=torch.max(out_,-1)
            predicted_seq="".join([id_to_char[token.item()]for token in predicted.data])
            tgt_seq = "".join([id_to_char[token.item()] for token in tgt_.data])
            if predicted_seq == tgt_seq:
                correct += 1
                logging.info(f"{predicted_seq} vs {tgt_seq}")
    accuracy = correct / total_count
    logging.info(f"accuracy : {accuracy}")
    return accuracy


# save model parameters
folder = "/Users/samrullo/Documents/learning/data_science/nlp_related/pytorch_transformer"
file = "pytorch_nn_transformer_addition_seq_params"
torch.save(model.state_dict(), os.path.join(folder, file))

accuracy = eval_model(model,id_to_char)
