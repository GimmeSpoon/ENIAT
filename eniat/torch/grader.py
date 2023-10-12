from typing import Callable, TypeVar, Union, Literal, Sequence, Any
from ..core import Grader, Learner, Course
from .predictor import TorchPredictor
from ..utils import Logger
import torch
import torch.distributed as dist
from omegaconf import DictConfig
import os

C = TypeVar("C", bound=Course)
M = TypeVar("M", bound=Learner)
L = TypeVar("L", bound=Logger)


def load_grader(grader_conf: DictConfig, log: L, course=None, learner=None):
    grader = TorchGrader(grader_conf, course=course, learner=learner)
    log.info("Grader instance created.")
    return grader


class TorchGrader(Grader, TorchPredictor):
    def __init__(
        self,
        conf: DictConfig,
        methods: Union[
            str, Callable[..., Any], Sequence[Union[str, Callable[..., Any]]]
        ] = None,
        options: list[dict] = None,
        logger=None,
        course: C = None,
        learner: M = None,
    ) -> None:
        super().__init__(conf, methods, options, logger, course, learner)

    def eval(
        self,
        learner: M = None,
        course: C = None,
        timestep: int = None,
        unit: Literal["epoch", "step"] = None,
        final: bool = True,
        position: int = 0,
        env_conf: DictConfig = None,
    ):

        if timestep is not None and unit is not None:
            if (
                self.conf.scheme.unit != unit
                or timestep % self.conf.scheme.interval != 0
            ):
                return

        if self.conf.env.type == "keep":
            self.conf.env = env_conf

        if course is None:
            self.course = course

        if learner is None:
            self.learner = learner

        self.log.info("Evaluation started...")

        if self.conf.env.type == "remote":
            raise NotImplementedError("sorry, remote grader is still in development.")
        elif self.conf.env.type == "single":
            res, gt = self.predict(
                device=self.conf.env.dev_id,
                global_rank=0,
                position=position,
                silent=False,
                final=final,
                data_label="eval",
                skip_prepare=True,
            )
        elif dist.is_initialized():
            res, gt = self.predict(
                device=int(os.environ["LOCAL_RANK"]),
                global_rank=dist.get_rank(),
                position=position,
                silent=False,
                final=final,
                data_label="eval",
                skip_prepare=True,
            )
        else:
            res, gt = self.predict(
                position=position,
                silent=False,
                final=final,
                data_label="eval",
                skip_prepare=True,
            )

        if gt is not None:
            if res.shape == gt.shape:
                eval_result = self.compute(res, gt)
                eval_result = {k: float(v) for k, v in eval_result.items()}
                eval_result[unit] = timestep
                self.log.log_state(eval_result)
                self.log.info("Evaluation completed.")
                return eval_result
            else:
                raise ValueError(
                    f"A shape of output is different from that of ground truth. {res.shape}, {gt.shape}"
                )
