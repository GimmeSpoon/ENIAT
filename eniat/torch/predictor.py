from typing import Union, Sequence
from torch.utils.data import DataLoader

class TorchPredictor():
    def predict(self, device:Union[int, str, Sequence[int]], silent:bool=False, log=None):
        