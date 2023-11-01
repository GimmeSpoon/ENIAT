from typing import Callable, TypeVar, Union, Literal, Sequence, Any
from ..core import Grader, Learner, Course
from .predictor import TorchPredictor, distributed
from ..utils import Logger, end
import torch.distributed as dist
from copy import deepcopy
from omegaconf import DictConfig
import os

C = TypeVar("C", bound=Course)
M = TypeVar("M", bound=Learner)
L = TypeVar("L", bound=Logger)


def load_grader(grader_conf: DictConfig, log: L, course=None, learner=None):
    grader = TorchGrader(grader_conf, logger=log, course=course, learner=learner)
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
        self.unloaded = (methods, deepcopy(conf.scheme.metrics))
        conf.scheme.metrics = None

        super().__init__(conf, None, options, logger, course, learner)

    def load_metrics(self) -> None:
        if self.unloaded is not None:
            self.append_metric(self.unloaded[0])
            self.append_metric(self.unloaded[1])
            self.unloaded = None

    @distributed
    def eval(
        self,
        device,
        global_rank,
        silent: bool = False,
        final: bool = True,
        learner: M = None,
        course: C = None,
        timestep: int = None,
        unit: Literal["epoch", "step"] = "epoch",
        env_conf: DictConfig = None,
    ):
        if timestep is not None and unit is not None:
            if (
                self.conf.scheme.unit != unit
                or timestep % self.conf.scheme.eval_interval != 0
            ):
                return

        if not dist.is_initialized() or dist.get_rank() == 0:
            self.load_metrics()
            master_prog = True
        else:
            master_prog = False

        if self.conf.env.type == "keep":
            self.conf.env = env_conf

        if course is not None:
            self.course = course

        if learner is not None:
            self.learner = learner

        if not final:
            self.loader = self.get_loader(self.conf.data_label)
        else:
            self.prepare(device, self.log)

        if self.conf.env.type == "remote":
            raise NotImplementedError("sorry, remote grader is still in development.")
        elif self.conf.env.type == "single":
            res, gt = self.predict(
                device=self.conf.env.dev_id,
                global_rank=0,
                silent=True,
                final=final,
                msg="Evaluating...",
            )
        elif dist.is_initialized():
            res, gt = self.predict(
                device=int(os.environ["LOCAL_RANK"]),
                global_rank=dist.get_rank(),
                silent=True,
                final=final,
                msg="Evaluating...",
            )
        else:
            res, gt = self.predict(
                silent=True,
                final=final,
                msg="Evaluating...",
            )

        if master_prog:

            if gt is None:
                self.log.warning("No ground truth for evaluation.")
            elif res.shape != gt.shape:
                self.log.warning("Predictions and ground truth have different shapes.")

            eval_result = self.compute(res, gt)
            eval_result = {k: float(v) for k, v in eval_result.items()}
            if final:
                self.log.log_state(
                    eval_result, epoch=1, unit="epoch", training_state=False
                )
            else:
                self.log.log_state(
                    eval_result, epoch=timestep, unit=unit, training_state=False
                ) if unit == "epoch" else self.log.log_state(
                    eval_result, step=timestep, unit=unit, training_state=False
                )
            return eval_result
