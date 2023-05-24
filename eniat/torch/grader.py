from typing import Callable, TypeVar, Union, Literal, Sequence, Any
from ..base import Grader, Learner
from ..data import Course
from .base import TorchPredictor
import torch
import torch.distributed as dist
from omegaconf import DictConfig

C = TypeVar('C', bound=Course)
L = TypeVar('L', bound=Learner)

class TorchGrader (Grader, TorchPredictor):

    def __init__(self, conf: DictConfig, methods: Union[str, Callable[..., Any], Sequence[Union[str, Callable[..., Any]]]] = None, logger=None, course:C = None, options: list[dict] = None) -> None:
        super().__init__(conf, methods, logger, course, options)

    def eval(
            self,
            learner:L,
            data:C=None,
            env_conf:DictConfig=None,
            timestep:int=None,
            unit:Literal['epoch', 'step']=None,
            step_check:bool=False,
            final:bool=True,
            position:int = 0
            ):
            
        if step_check:
            if self.conf.unit is None or self.conf.unit == 'none' or self.conf.eval_interval is None or self.conf.eval_interval == 0:
                raise ValueError("You're trying to evaluate at preconfigured steps but configurations of Grader have no settings for eval strategy.")
            if self.conf.unit != unit or timestep % self.conf.eval_interval != 0:
                    return

        self.log.info("Evaluation started...")

        if data is None:
            data = self.course
        
        self.loader = self.get_loader('eval', data)
        
        self.learner = learner

        self.log.info("Test dataset prepared.")

        if self.conf.env.type == 'remote':
            raise NotImplementedError("sorry, remote grader is still in development.")
        
        res, gt = self.predict(position=position, silent=False, final=final, data_label='eval', skip_prepare=True)
        eval_result = self.compute(res, gt)

        self.log.log_state(eval_result.dict)
        self.log.info("Evaluation completed. The result is as below.\n"+eval_result.__repr__())

        return eval_result