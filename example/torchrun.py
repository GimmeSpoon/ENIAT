import eniat
import eniat.eniat
from eniat.pytorch.trainer import TorchDistributedTrainer
from eniat.pytorch.learner import TorchLearner
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group

cfg = eniat.eniat.init_conf()
print(OmegaConf.to_yaml(cfg))

class FishNet(nn.Module):
    def __init__(self) -> None:
        super(FishNet, self).__init__()
        self.net = nn.SoftMax()

    def forward(self, x):
        x = x + x
        return self.net(x)

class Trainset(Dataset):
    def __init__(self, size:int = 10000000) -> None:
        super(Trainset, self).__init__()
        self.x = torch.rand((size, 100))
        self.y = torch.rand((size, 1))
    
    def __len__(self) -> int:
        return len(self.y)

    def __getitem__ (self, idx:int):
        return self.x[idx], self.y[idx]

class Evalset(Dataset):
    def __init__(self, size:int = 10000) -> None:
        super(Evalset, self).__init__()
        self.x = torch.rand((size, 100))
        self.y = torch.rand((size, 1))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx:int):
        return self.x[idx], self.y[idx]

#optimizer = torch.optim.

if __name__ == "__main__":
    init_process_group()