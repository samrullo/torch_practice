import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy
from dataset import sequence


class AdditionSeqDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
        if train:
            self.src_seq = x_train
            self.tgt_seq = t_train
        else:
            self.src_seq = x_test
            self.tgt_seq = t_test

    def __len__(self):
        return len(self.src_seq)

    def __getitem__(self, idx):
        return self.src_seq[idx], self.tgt_seq[idx]


if __name__ == "__main__":
    train_set = AdditionSeqDataset(train=True)
    test_set = AdditionSeqDataset(train=False)
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
    x, t = train_set[0]
    print(f"train_set length : {len(train_set)}, test_set length : {len(test_set)}")
    print(f"x : {x}")
    print(f"t : {t}")

    print("------batch from train_loader ------")
    count = 0
    for x, t in train_loader:
        print(f"{count + 1}/{len(train_set) / 20} | x.shape : {x.shape}, t.shape : {t.shape}")
        count+=1
