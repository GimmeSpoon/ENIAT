import torch
from torch import nn, rand, rand_like
from torch.utils.data import Dataset, DataLoader


class RandDataset(Dataset):
    def __init__(self, total: int, _hidden: int, _gt: int):
        self.data = rand((total, _hidden))
        self.target = rand((total, _gt))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


class SimpleLinear(nn.Module):
    def __init__(self, layers: int, _in: int, _hidden: int, _out: int):
        super().__init__()

        self.in_layer = nn.Sequential(nn.Linear(_in, _hidden), nn.ReLU())
        self.out_layer = nn.Linear(_hidden, _out)
        self.mid_layer = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(_hidden, _hidden), nn.ReLU())
                for l in range(layers - 2)
            ]
        )

    def forward(self, x):
        return self.out_layer(self.mid_layer(self.in_layer(x)))
