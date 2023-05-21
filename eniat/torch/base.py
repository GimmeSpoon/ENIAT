from typing import TypeVar, Union, Sequence, Callable
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import os

C = TypeVar('C', bound=Course)

class TorchPredictor():

    def distributed (fn:Callable) -> Callable:
        def wrapper(self, *args):
            if not dist.is_initialized():

                if self.conf.distributed.debug:
                    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
                    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

                warnings.showwarning = Warning(self.log)

                if dist.is_torchelastic_launched():
                    #torchrun
                    self.hc = HydraConfig.get()
                    self._torchrun_init(self)
                    return fn(int(os.environ['LOCAL_RANK']), int(os.environ['RANK']))
                else:
                    #DDP
                    spawn(self._ddp_init, (fn.__name__, HydraConfig.get()), nprocs=self.conf.distributed.local_size, join=True)
            else:
                if dist.get_rank() == 0:
                    warnings.showwarning = Warning(self.log)
                    return fn(self, *args)
                else:
                    warnings.filterwarnings("ignore")
                    with self.log.silent():
                        return fn(self, *args)
        return wrapper

    @staticmethod
    def to_tensor(batch):
        if isinstance(batch, list):
            for data in batch:
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(data)
        return batch

    def get_loader(self, dataset:Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)
    
    def predict(self, device:Union[int, str, Sequence[int]], loader:DataLoader, silent:bool=False, log=None):
        log.info("Inference Started...")

        ret = []
        with logging_redirect_tqdm():
            for batch in (step_bar:=tqdm(loader, desc='Steps', unit='step', position=2, leave=False, disable=silent)):
                batch = self.to_tensor(batch)
                self.learner.to(device)
                self.learner.eval()
                pred = self.learner.predict(batch).detach().cpu().numpy()
                ret.append(pred)

        ret = np.concatenate(ret)
        return ret
