from typing import TypeVar, Literal, Callable
from ..base import Trainer
from ..data.course import Course, FullCourse
from .learner import TorchLearner
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.multiprocessing import spawn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import PostLocalSGDOptimizer, ZeroRedundancyOptimizer
import torch.distributed.algorithms.model_averaging.averagers as averagers
import os
import random
import numpy as np

T = TypeVar('T', bound=TorchLearner)
C = TypeVar('C', Course, FullCourse)

class TorchTrainer(Trainer):

    def __init__(self, course:C=None, learner:T=None, conf=None, evaluate_fn:Callable=None, logger=None) -> None:
        super(TorchTrainer, self).__init__(course, learner, conf, evaluate_fn, logger)

        self.course = course

        self.learner = learner
        self.conf = conf
        self.eval_fn = evaluate_fn
        self.log = logger

        if not self.conf.distributed or self.conf.distributed.type == 'none':
            self._dist = False
        else: # distributed learning set
            self._dist = True

    def rand_all(self, seed):
        if self._dist:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def save_checkpoint(self, dir:str):
        self.learner.save_all(os.path.join(self.io.output_dir, dir + '/model.pt'), os.path.join(self.io.output_dir, dir + '/optimizer.pt'))

    def dist(self, fn, *args) -> None:
        self.learner.set_model(DDP(self.learner.model), self.learner.model_name + '(DDP)')
        self.learner.set_optimizer(self.get_dist_opt(self.task.dist_optimizer, self.learner.get_params()))
        if self.conf.distributed.debug:
            os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if self.distributed.type == "torchrun":
            local_size = self.conf.distributed.local_size = int(os.environ['LOCAL_WORLD_SIZE'])
            self.conf.distributed.local_rank = int(os.environ['LOCAL_RANK'])
            self.conf.distributed.global_rank = int(os.environ['RANK'])
            self.conf.distributed.world_size = int(os.environ['WORLD_SIZE'])
            if __name__ == "__main__":
                init_process_group(backend=self.conf.backend)
                fn(self.conf.distributed.local_rank)
        elif self.distributed.type == "DDP":
            os.environ["MASTER_ADDR"] = self.conf.master_address
            os.environ["MASTER_PORT"] = self.conf.master_port
            if __name__ == "__main__":
                spawn(fn, args, local_size, join=True)
        else:
            raise ValueError("The type of distributed config must be one of the following literals: ['torchrun', 'DDP', 'none']")

    def distributed (fn:Callable):
        def wrapper(*args):
            if (self:=args[0])._dist:
                return self.dist(fn, *args)
            else:
                self.loader = DataLoader()
                return fn(*args)
        return wrapper

    @distributed
    def fit(self, device:int=0, silent:bool=False, seed=None):
        if seed:
            self.rand_all(seed)

        current_step = 0
        for epoch in (epoch_bar:=tqdm(range(self.conf.hyperparameters.epoch), desc='Training', unit='epoch', position=0, leave=False, disable=silent)):
            for batch in (step_bar:=tqdm(self.course, desc='Batch', unit='step', position=1, leave=False, disable=silent)):
                tr_loss = self.learner.fit(batch)
                current_step += 1
                step_postfix = {'training_loss' : tr_loss.item()}
                step_bar.set_postfix(step_postfix)
                # Step Log
                if self.conf.log_strategy == 'step' and current_step % self.conf.log_interval == 0:
                    self.logger.log_scalar('training loss', tr_loss.item(), current_step)
                # Step Eval
                if self.conf.eval_strategy == 'step' and current_step % self.conf.eval_interval == 0:
                    self.eval(silent=silent)
                # Step Save
                if self.conf.save_strategy == 'step' and current_step % self.conf.save_interval == 0:
                    self.save_checkpoint('step' + current_step)
            self.scheduler.step()
            # Epoch Log
            if self.conf.log_strategy == 'epoch' and epoch % self.conf.log_interval == 0:
                    self.logger.log_scalar('training loss', tr_loss.item(), epoch)
            # Epoch Save
            if self.conf.save_strategy == 'epoch' and epoch % self.conf.save_interval == 0:
                self.save_checkpoint('epoch' + epoch)
            # Epoch Eval
            if self.conf.eval_strategy == 'epoch' and epoch % self.conf.eval_interval == 0:
                self.eval(silent=silent, final=False)

        self.logger.log(epoch, current_step, tr_loss) # Log after done
        self.save_checkpoint('final')

        if self.distributed:
            destroy_process_group()

    @distributed
    def eval(self, local_rank:int, final:bool=False, silent:bool=False):
        # Evaluation
        if not self.course:
            raise AttributeError("No evaluation dataset.")
            
        whole_batch = None
        for batch in tqdm(self.course, desc='Validation', unit='step', leave=False, disable=silent):
            output = self.learner.infer(batch, local_rank)
            if not whole_batch:
                whole_batch = torch.empty((0, *output.shape[1:]))
            whole_batch = torch.cat((whole_batch, output), dim=0)

        if final and self.distributed:
            destroy_process_group()

        if self.eval_fn:
            return self.eval_fn(whole_batch)
        else:
            return whole_batch

    @distributed
    def predict(self, local_rank:int, final:bool=False, silent:bool=False):
        # Batch Inference
        outputs = None
        for batch in tqdm(self.loader, unit='Steps', position=1, disable=silent):
            output = self.learner.predict(batch)
            if not outputs:
                outputs = torch.empty((0, *output.shape[1:]))
            outputs = torch.cat((outputs, output), dim=0)

        if final and self.distributed:
            destroy_process_group()

        return outputs
    
    def accelerate(self):
        self.learner.set_model(torch.compile(self.learner.model))

    def get_dist_opt(self, method:Literal['zero', 'postlocal'], params=None, **kwargs):
        if method == 'zero':
            if not params:
                raise ValueError("model parameters require for zero redundancy optimizer.")
            return ZeroRedundancyOptimizer(params, optimizer_class=getattr(torch.optim, type(self.learner.opt).__name__), parameters_as_bucket_view=False, overlap_with_ddp=False, **kwargs)
        if method == 'postlocal':
            return PostLocalSGDOptimizer(self.learner.opt, averagers.PeriodicModelAverager(**kwargs))
        
    def get_loader(self, dataset:Literal['train', 'eval', 'predict']) -> DataLoader:
        return DataLoader(getattr(self, dataset + 'set'), batch_size=self.task.batch_size, num_workers=self.task.num_workers) if not self.distributed else \
        DataLoader(getattr(self, dataset + 'set'), batch_size=self.task.batch_size,  num_workers=self.task.num_workers, sampler=DistributedSampler(self.dataset, self.world_size, self.global_rank))