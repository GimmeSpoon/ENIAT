from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.utils import instantiate
import pkg_resources
import os
from importlib import import_module
from shutil import copytree, ignore_patterns
from .utils.cli import introduce
from ._dyn import _dynamic_load, _dynamic_import, _merge_conf_by_path, conf_instantiate
from .data.course import FullCourse, Course, get_course_instance

version_base = None
eniat_path = os.path.abspath(pkg_resources.resource_filename(__name__, 'config'))

@hydra.main(version_base=version_base, config_path=eniat_path, config_name='eniat')
def eniat(cfg: DictConfig) -> None:

    cfg = _merge_conf_by_path(cfg, cfg.config)

    if cfg.do == 'init':
        copytree(eniat_path, './config', ignore=ignore_patterns('*.yaml'))
        print("Config files copied to current directory.")

    log = getattr(import_module('.utils.statelogger', 'eniat'), cfg.logger.type)(__name__, cfg.logger.level, conf=cfg.logger)

    introduce()
    print("===============CONFIG================")
    log.info("Experiment Config:\n"+OmegaConf.to_yaml(cfg))
    print("=====================================")

    try:
        # Torch
        if cfg.trainer.type == "torch":

            log.info(f"Initiating an experiment based on PyTorch.")

            if cfg.trainer.distributed.type == "none" or cfg.trainer.distributed.type == "DP":
            
            # DATA LOAD
                _courses = get_course_instance(cfg.data, log)
                log.info('Loaded dataset.\n' + _courses.__repr__())
                
                # instantiate learner components

                # Model Load
                model = conf_instantiate(cfg.learner.model)
                log.info("Model loaded...")

                if not cfg.learner.resume:
                    log.warning("'resume' is set to False. The model will be initialized without loading a checkpoint.")
                # loss
                loss = instantiate(cfg.learner.loss) if cfg.learner.loss and cfg.learner.loss._target_ else None
                if loss:
                    log.info("Loss function loaded...")
                else:
                    log.warning("Loss function is not defined. Are you sure you wanted this?")
                # optimizer
                optim = instantiate(cfg.learner.optimizer, params=model.parameters()) if cfg.learner.optimizer and cfg.learner.optimizer._target_ else None
                if optim:
                    log.info("Optimizer loaded...")
                else:
                    log.warning("Optimizer is not defined. Are you sure you wanted this?")
                # scheduler
                schlr = None# instantiate(cfg.learner.scheduler, lr_lambda=lambda x: x**cfg.learner.scheduler.lr_lambda, optimizer=optim) if cfg.learner.scheduler and cfg.learner.scheduler._target_ else None
                if schlr:
                    log.info("Scheduler loaded...")
                else:
                    log.warning("Scheduler is not defined. Edit the configuration if this is not what you wanted.")
                
                # instantiate learner
                if 'path' in cfg.learner and cfg.learner.path:
                    _mod, _bn = _dynamic_import(cfg.learner.path)
                    learner = getattr(_mod, cfg.learner.cls)(model, loss, optim, schlr, cfg.learner.resume, cfg.learner.resume_path)
                else:
                    learner = getattr(import_module('.pytorch.learner', 'eniat'), cfg.learner.cls)(model, loss, optim, schlr, cfg.learner.resume, cfg.learner.resume_path)
                if learner:
                    log.info("Learner instance created.")

                # instantiate trainer
                trainer = getattr(import_module('.pytorch', 'eniat'), 'TorchTrainer')(course=_courses, learner=learner, conf=cfg.trainer, logger=log)
            else:
                trainer = getattr(import_module('.pytorch', 'eniat'), 'TorchDistributedTrainer')(conf=cfg.trainer, data_conf=cfg.data, learner_conf=cfg.learner, logger_conf=cfg.logger)
                log.info("Distributed Learning (Torch) is configured.")
            
            if trainer:
                log.info("Trainer instance created.")

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

            trainer = _dynamic_load('ScikitTrainer', '.scikit')

    except BaseException as err:
        log.exception(err)
        print("If you need some help, type command : 'eniat do=help'")

if __name__ == "__main__":
    eniat(version='0.1.0')