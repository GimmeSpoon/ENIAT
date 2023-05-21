from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import pkg_resources
import os
from importlib import import_module
from shutil import copytree, ignore_patterns
from .torch import torchload
from .utils.cli import introduce
from .utils.conf import _dynamic_import, _merge_conf_by_path, conf_instantiate
from .data.course import FullCourse, Course, get_course_instance

version_base = None
eniat_path = os.path.abspath(pkg_resources.resource_filename(__name__, 'config'))

class Manager():
    def __init__(self, cfg: DictConfig, silent:bool = False) -> None:
        self.cfg = cfg
        self.type = cfg.type
        self.task = cfg.task

        self.log = getattr(import_module('.utils.statelogger', 'eniat'), cfg.logger.type)(__name__, cfg.logger.level, conf=cfg.logger)

        if not silent:
            introduce()

        self.log.info("Experiment Config:\n"+OmegaConf.to_yaml(cfg))

    def run(self) -> None:
        try:
            if self.cfg.type == "torch":
                self.log.info(f"Initiating an experiment based on PyTorch.")

                if self.cfg.trainer.distributed.type == "none" or self.cfg.trainer.distributed.type == "DP":
                    _courses = get_course_instance(self.cfg.data, self.log)
                    self.log.info('Loaded dataset.\n' + _courses.__repr__())
                    learner, trainer = torchload(self.cfg, self.log)
                else:
                    trainer = getattr(import_module('.torch', 'eniat'), 'TorchDistributedTrainer')(conf=self.cfg.trainer, data_conf=self.cfg.data, learner_conf=self.cfg.learner, logger_conf=self.cfg.logger)
                    self.log.info("Distributed Learning (Torch) is configured.")

            if self.cfg.type == "scikit":

                self.log.info(f"Initiating an experiment based on Scikit-learn.")
                model = instantiate(self.cfg.learner.model)

                learner = scikit_load(self.cfg.learner, self.cfg.trainer)

            if self.cfg.trainer.task == "fit" or self.cfg.trainer.task == "train":
                trainer.fit()
            if self.cfg.trainer.task == "fit_n_eval" or self.cfg.trainer.task == "train_n_test":
                trainer.fit(eval=True)
            elif self.cfg.trainer.task == "eval" or self.cfg.trainer.task == "test":
                trainer.eval()
            elif self.cfg.trainer.task == "predict" or self.cfg.trainer.task == "infer":
                trainer.predict()
            else:
                raise ValueError("<task> of trainer config must be one of the following literals : ['fit', 'train', 'eval', 'test', 'predict', 'infer', 'fit_n_eval', 'train_n_test']")

        except BaseException as err:
            self.log.exception(err)
            print("If you need some help, type command : 'eniat do=help'")



@hydra.main(version_base=version_base, config_path=eniat_path, config_name='eniat')
def eniat(cfg: DictConfig) -> None:

    cfg = _merge_conf_by_path(cfg, cfg.config)

    if cfg.init:
        copytree(eniat_path, './config', ignore=ignore_patterns('*.yaml'))
        print("Config files copied to current directory.")
    
    manager = Manager(cfg, cfg.silent)

    log = getattr(import_module('.utils.statelogger', 'eniat'), cfg.logger.type)(__name__, cfg.logger.level, conf=cfg.logger)

    introduce()
    log.info("Experiment Config:\n"+OmegaConf.to_yaml(cfg))

    try:
        # Torch
        if cfg.trainer.type == "torch":

            log.info(f"Initiating an experiment based on PyTorch.")

            if cfg.trainer.distributed.type == "none" or cfg.trainer.distributed.type == "DP":
                _courses = get_course_instance(cfg.data, log)
                log.info('Loaded dataset.\n' + _courses.__repr__())
                learner, trainer = torchload(cfg, log)
            else:
                trainer = getattr(import_module('.torch', 'eniat'), 'TorchDistributedTrainer')(conf=cfg.trainer, data_conf=cfg.data, learner_conf=cfg.learner, logger_conf=cfg.logger)
                log.info("Distributed Learning (Torch) is configured.")

            if cfg.trainer.task == "fit" or cfg.trainer.task == "train":
                trainer.fit()
            elif cfg.trainer.task == "eval" or cfg.trainer.task == "test":
                trainer.eval()
            elif cfg.trainer.task == "predict" or cfg.trainer.task == "infer":
                trainer.predict()
            else:
                raise ValueError("<task> of trainer config must be one of the following literals : ['fit', 'train', 'eval', 'test', 'predict', 'infer']")

        # Scikit-learn
        if cfg.trainer.type == "scikit":

            log.info(f"Initiating an experiment based on Scikit-learn.")
            model = instantiate(cfg.learner.model)


    except BaseException as err:
        log.exception(err)
        print("If you need some help, type command : 'eniat do=help'")

if __name__ == "__main__":
    eniat(version='0.1.0')