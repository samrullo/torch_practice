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
from transformer.transformer_modules import Batch
from transformer.transformer_modules import greedy_decode
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
test_set = AdditionSeqDataset(train=False,reverse=True)
test_loader = DataLoader(test_set, batch_size=1)
char_to_id, id_to_char = sequence.get_vocab()

V = len(char_to_id)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=6, d_model=128, d_ff=128*4, h=8, dropout=0.1)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def seq_to_text(src, id_to_char):
    return "".join([id_to_char[int(_id)] for _id in src.data[0]])


def seq_pick_predict(test_generator, verbose=False):
    b = next(test_generator)
    src, tgt = b.src, b.trg
    ys = greedy_decode(model, b.src, b.src_mask, 7, start_symbol=tgt.data[0][0])
    ys = remove_blank_char(ys)
    ys = ys.unsqueeze(-2)
    if verbose:
        print(f"{seq_to_text(src, id_to_char)}={seq_to_text(tgt, id_to_char)} pred : {seq_to_text(ys, id_to_char)}")
    return src, tgt, ys


def remove_blank_char(ys):
    return ys[(ys != 5) & (ys!=8)]


def accuracy(test_generator, verbose=False):
    start = time.time()
    correct_count = 0
    total_count = 0
    for b in test_generator:
        total_count += 1
        src, tgt = b.src, b.trg
        tgt=remove_blank_char(tgt)
        tgt=tgt.unsqueeze(-2)
        ys = greedy_decode(model, b.src, b.src_mask, 7, start_symbol=tgt.data[0][0])
        ys = remove_blank_char(ys)
        ys = ys.unsqueeze(-2)
        if verbose:
            print(
                f"{total_count}/{len(test_set)} {seq_to_text(src, id_to_char)}={seq_to_text(tgt, id_to_char)} pred : {seq_to_text(ys, id_to_char)}")
        if seq_to_text(tgt, id_to_char) == seq_to_text(ys, id_to_char):
            correct_count += 1
    logging.info(f"it took {time.time() - start} seconds to calculate accuracy")
    return correct_count / total_count + 1e-10


def data_gen_addition(my_dataloader):
    for src, tgt in my_dataloader:
        batch_obj = Batch(src, tgt)
        yield batch_obj


best_test_accuracy = 0
for epoch in range(max_epoch):
    start = time.time()
    logging.info(f"--- epoch {epoch + 1}/{max_epoch} -----")

    model.train()
    run_epoch(data_gen_addition(train_loader), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    # model.eval()
    # print(run_epoch(data_gen_addition(test_loader), model,
    #                 SimpleLossCompute(model.generator, criterion, None)))
    if epoch % 10 == 0:
        model.eval()
        test_gen = data_gen_addition(test_loader)
        with torch.no_grad():
            test_acc = accuracy(test_gen)
        logging.info(f"test accuracy : {test_acc}")
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            logging.info(f"best_accuracy so far : {best_test_accuracy}")
            folder = "/Users/samrullo/Documents/learning/data_science/nlp_related/pytorch_transformer"
            file = "pytorch_transformer_addition_seq_params"
            torch.save(model.state_dict(), os.path.join(folder, file))
    end = time.time()
    logging.info(f"---- epoch {epoch + 1}/{max_epoch} took {end - start} seconds ----")

# folder = "/Users/samrullo/Documents/learning/data_science/nlp_related/pytorch_transformer"
# file = "pytorch_transformer_addition_seq_params"
# torch.save(model.state_dict(), os.path.join(folder, file))
test_gen = data_gen_addition(test_loader)
test_accuracy = accuracy(test_gen, False)
print(f"test accuracy : {test_accuracy}")
