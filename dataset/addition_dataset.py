from torch.utils.data import Dataset
from dataset import sequence


class AdditionSeqDataset(Dataset):
    def __init__(self, train=True, reverse=True):
        super().__init__()
        (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
        if reverse:
            x_train = x_train[:, ::-1].copy()
            x_test = x_test[:, ::-1].copy()
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
