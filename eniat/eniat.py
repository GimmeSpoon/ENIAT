from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
import pkg_resources
import os
import sys
from importlib import import_module
from importlib.util import spec_from_file_location, module_from_spec
from typing import Callable, Sequence, Union
from shutil import copytree, ignore_patterns
from .data.course import Course, FullCourse, batch_load
from .utils.cli import introduce, help

from torch.distributed.elastic.multiprocessing.errors import record

version_base = None

global_config_path = os.path.abspath(pkg_resources.resource_filename(__name__, 'config'))

def dynamic_import(path:str, name:str = None):
    _spec = spec_from_file_location(_bn:=name if name else os.path.basename(path), os.path.abspath(path))
    if not _spec:
        raise FileNotFoundError(f"Can not read from the location : {path}")
    _mod = module_from_spec(_spec)
    sys.modules[_bn] = _mod
    _spec.loader.exec_module(_mod)
    return _mod, _bn

def dynamic_load(name:str, path:str):
    _mod, _bn = dynamic_import(path, name)
    return getattr(_mod, name), _bn

def _conf_instantiate(options:DictConfig):
    print("_conf_instantiate()")
    if not options:
        return None
    if '_target_' in options:
        return instantiate(options)
    elif 'path' in options and 'cls' in options:
        _cls, _bn = dynamic_load(options.cls, options.path)
        options = OmegaConf.to_container(options)
        options.pop('_target_', None)
        options.pop('path', None)
        options.pop('cls', None)
        print(options)
        return _cls(**options)
    else:
        raise ValueError(f"Instantiation failed. Current config is not valid: {options}")

def init_conf(job_name:str="eniat"):
    # Global Config
    with initialize(version_base=version_base, config_path=global_config_path, job_name=job_name):
        cfg = compose(config_name="eniat")
        print(cfg)
    return cfg

def _merge_conf_by_path(conf:DictConfig, paths:Union[str,Sequence[str]]) -> DictConfig:

    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        new_cfg = OmegaConf.load(path)
        #conf = OmegaConf.merge(new_cfg, conf)
        conf = OmegaConf.merge(conf, new_cfg)

    return conf

@record
@hydra.main(version_base=version_base, config_path=global_config_path, config_name='eniat')
def eniat(cfg: DictConfig) -> None:

    cfg = _merge_conf_by_path(cfg, cfg.config)

    if cfg.do == 'init':
        copytree(global_config_path, './config', ignore=ignore_patterns('*.yaml'))
        print("Config files copied to current directory.")
        return
    elif cfg.do == 'help':
        help()
        return

    log = getattr(import_module('.utils.statelogger', 'eniat'), cfg.log.type)(__name__, cfg.log.level, conf=cfg.log)

    introduce()
    print("===============CONFIG================")
    log.info("Experiment Config:\n"+OmegaConf.to_yaml(cfg))
    print("=====================================")

    try:
        # DATA LOAD
        _courses = FullCourse()
        log.info("Loading data...")
        for label in cfg.data:
            if 'cls' in cfg.data[label]:
                _courses.append(Course(label ,_conf_instantiate(cfg.data[label])))
                log.info(f"'{label}' data is loaded.")
            elif 'path' in cfg.data[label]:
                _courses.append(course=Course(label, data=batch_load(cfg.data[label]['path'], cfg.data[label].type)))
                log.info(f"'{label}' data is loaded.")
            else:
                log.warning(f"Data(:{label}) is not loaded because the path of data is not specified.")
        if not len(_courses):
            log.warning("No data is given! Terminating the task...")
            return
        log.info('Loaded dataset info\n' + _courses.__repr__())

        # Torch
        if cfg.trainer.type == "torch":

            log.info(f"Task based on PyTorch.")
            # instantiate learner components

            # Model Load
            model = _conf_instantiate(cfg.learner.model)
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
            schlr = instantiate(cfg.learner.scheduler, lr_lambda=lambda x: x**cfg.learner.scheduler.lr_lambda, optimizer=optim) if cfg.learner.scheduler and cfg.learner.scheduler._target_ else None
            if schlr:
                log.info("Scheduler loaded...")
            else:
                log.warning("Scheduler is not defined. Edit the configuration if this is not what you wanted.")
            
            # instantiate learner
            if 'path' in cfg.learner and cfg.learner.path:
                _mod, _bn = dynamic_import(cfg.learner.path)
                learner = getattr(_mod, cfg.learner.cls)(model, loss, optim, schlr, cfg.learner.resume, cfg.learner.resume_path)
            else:
                learner = getattr(import_module('.pytorch.learner', 'eniat'), cfg.learner.cls)(model, loss, optim, schlr, cfg.learner.resume, cfg.learner.resume_path)
            if learner:
                log.info("Learner instance created.")
            # instantiate trainer
            trainer = getattr(import_module('.pytorch', 'eniat'), 'TorchTrainer')(_courses, learner, conf=cfg.trainer, logger=log)
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

            log.info(f"Task based on Scikit-learn.")
            model = instantiate(cfg.learner.model)

            trainer = dynamic_load('ScikitTrainer', '.scikit')

    except BaseException as err:
        log.exception(err)
        print("If you need some help, type command : 'eniat do=help'")

if __name__ == "__main__":
    eniat(version='0.1.0')