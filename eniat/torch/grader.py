from typing import Callable, TypeVar, Union, Literal, Sequence, Any
from ..base import Grader, Learner
from ..data import Course
from .predictor import TorchPredictor
import torch
import torch.distributed as dist
from omegaconf import DictConfig
import os

C = TypeVar('C', bound=Course)
L = TypeVar('L', bound=Learner)

class TorchGrader (Grader, TorchPredictor):

    def __init__(self, conf: DictConfig, methods: Union[str, Callable[..., Any], Sequence[Union[str, Callable[..., Any]]]] = None, logger=None, course:C = None, options: list[dict] = None) -> None:
        super().__init__(conf, methods, logger, course, options)

    def eval(
            self,
            learner:L,
            data:C=None,
            timestep:int=None,
            unit:Literal['epoch', 'step']=None,
            step_check:bool=False,
            final:bool=True,
            position:int = 0,
            env_conf:DictConfig = None,
            skip_prepare:bool = False
            ):
            
        if not skip_prepare:
            self.prepare()

        if step_check:
            if self.conf.unit is None or self.conf.unit == 'none' or self.conf.interval is None or self.conf.interval == 0:
                raise ValueError("You're trying to evaluate at preconfigured steps but configurations of Grader have no settings for eval strategy.")
            if self.conf.unit != unit or timestep % self.conf.interval != 0:
                return

        self.log.info("Evaluation started...")

        if data is None:
            data = self.course

        if self.conf.env.type == 'keep':
            self.conf.env = env_conf
        
        self.loader = self.get_loader('eval', data)
        
        self.learner = learner

        self.log.info("Test dataset prepared.")

        if self.conf.env.type == 'remote':
            raise NotImplementedError("sorry, remote grader is still in development.")
        elif self.conf.env.type == 'single':
            res, gt = self.predict(device=self.conf.env.dev_id, global_rank=0, position=position, silent=False, final=final, data_label='eval', skip_prepare=True)
        elif dist.is_initialized():
            res, gt = self.predict(device=int(os.environ['LOCAL_RANK']), global_rank=dist.get_rank(), position=position, silent=False, final=final, data_label='eval', skip_prepare=True)
        else:
            res, gt = self.predict(position=position, silent=False, final=final, data_label='eval', skip_prepare=True)

        if gt is not None:
            if res.shape == gt.shape:
                eval_result = self.compute(res, gt)
                eval_result = {k:float(v) for k, v in eval_result.items()}
                eval_result[unit] = timestep
                self.log.log_state(eval_result)
                self.log.info("Evaluation completed.")
                return eval_result
            else:
                raise ValueError(f"Shapes of Predictions are different from ground truth. {res.shape}, {gt.shape}")