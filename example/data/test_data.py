import torch
from torch.utils.data import Dataset
import pickle as pkl
import numpy as np

class Testset (Dataset):
    def __init__(self, batch_size:int=10000, dim:int=256, target:bool=True) -> None:
        super(Testset, self).__init__()
        self.data = torch.rand((batch_size, dim))
        self.target = torch.rand((batch_size))
        self.train = target

    def __len__ (self) -> int:
        return self.target.shape[0]

    def __getitem__ (self, idx:int):
        return self.data[idx], self.target[idx] if self.train else self.data[idx]

if __name__ == "__main__":
    trainset = Testset()
    testset = Testset(target=False)

    with open('./data/pickle.pkl', 'wb') as f:
        pkl.dump(trainset, f)
    with open('./data/numpy.npy', 'wb') as f:
        np.save(f, testset.data)
