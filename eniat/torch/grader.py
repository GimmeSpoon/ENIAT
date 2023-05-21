from typing import TypeVar, Union, Literal, Sequence
from ..base import Grader, Learner
from ..data import Course
from . import TorchPredictor

C = TypeVar('C', bound=Course)
L = TypeVar('L', bound=Learner)

class TorchGrader (Grader, TorchPredictor):

    def eval(
            self,
            learner:L,
            device:Union[int, str, Sequence[int]]=None,
            data:C=None,
            timestep:int=None,
            unit:Literal['epoch', 'step']=None,
            step_check:bool=False
            ):
            
        if step_check:
            if self.conf.unit is None or self.conf.unit == 'none' or self.conf.eval_interval is None or self.conf.eval_interval == 0:
                raise ValueError("You're trying to evaluate at preconfigured steps but configurations of Grader have no settings for eval strategy.")
            if self.conf.unit != unit or timestep % self.conf.eval_interval != 0:
                    return

        self.log.info("Evaluation started...")

        if isinstance(device, str):
            if device == 'cpu':
                self.log.info("Evaluation environment set on CPU.")
            elif device == 'gpu' or device == 'cuda':
                device = 0
                self.log.warning("Evaluation environment is not explicitly set. Device set to 0 as default.")
            else:
                raise ValueError(f"Evaluation environment configuration is not valid: {device}")
        elif isinstance(device, int):
            self.log.info(f"Evaluation environment set on device {device}.")
        else:
            if not self.conf.distributed.enable:
                raise ValueError("Tried to initiate evaluation processes on parallel, but the configuration has been set to disable it.")
            self.log.info(f"Evaluation environment set on devices {','.join(device)}")
            
        if not data and self.course:
            data = self.course

        if data is None:
            raise ValueError("No data is given to grader. Evaluation aborted.")

        self.log.info("Test dataset prepared.")

        if self.conf.distributed.enable:
            # Distributed Evaluation
            
        else:
            learner.to(device)

        self.log.info("Evaluation ended. The result is as below.\n" + )