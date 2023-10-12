from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import torch.nn as nn
from ..utils.statelogger import StateLogger


def load_learner_components(conf: DictConfig, log: StateLogger):

    if hasattr(conf.model, "_target_"):
        inst_conf = OmegaConf.create(
            {"_target_": conf.model._target_, **conf.model.options}
        )
        model = instantiate(inst_conf)
        log.info("Model loaded...")

        if hasattr(conf.optimizer, "_target_"):
            inst_conf = OmegaConf.create(
                {"_target_": conf.optimizer._target_, **conf.optimizer.options}
                if conf.optimizer.options
                else {"_target_": conf.optimizer._target_}
            )
            opt = instantiate(inst_conf, params=model.parameters())
            log.info("Optimizer loaded...")

            if hasattr(conf.scheduler, "_target_"):
                inst_conf = OmegaConf.create(
                    {"_target_": conf.scheduler._target_, **conf.scheduler.options}
                    if conf.scheduler.options
                    else {"_target": conf.scheduler._target_}
                )
                sch = instantiate(inst_conf, optimizer=opt)
                log.info("Scheduler loaded...")

                return model, opt, sch
            else:
                log.warning(
                    "Scheduler is not provided. If this is not your intention, please add '_target_' attribute to scheduelr config."
                )
                return model, opt, None
        else:
            log.warning(
                "Optimizer is not provided. If this is not your intention, please add '_target_' attribute to optimizer config."
            )
            return model, None, None
    else:
        raise ModuleNotFoundError(
            f"Cannot load {conf.model.name} from {conf.model._target_}"
        )
