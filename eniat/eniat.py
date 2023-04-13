from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import pkg_resources
import os
from importlib import import_module
from shutil import copytree, ignore_patterns
from .utils.cli import introduce, help
from ._dyn import dynamic_load, _merge_conf_by_path

version_base = None
global_config_path = os.path.abspath(pkg_resources.resource_filename(__name__, 'config'))

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
        # _courses = FullCourse()
        # log.info("Loading data...")
        # for label in cfg.data:
        #     if 'cls' in cfg.data[label]:
        #         _courses.append(Course(label ,_conf_instantiate(cfg.data[label])))
        #         log.info(f"'{label}' data is loaded.")
        #     elif 'path' in cfg.data[label]:
        #         _courses.append(course=Course(label, data=batch_load(cfg.data[label]['path'], cfg.data[label].type)))
        #         log.info(f"'{label}' data is loaded.")
        #     else:
        #         log.warning(f"Data(:{label}) is not loaded because the path of data is not specified.")
        # if not len(_courses):
        #     log.warning("No data is given! Terminating the task...")
        #     return
        # log.info('Loaded dataset info\n' + _courses.__repr__())

        # # Torch
        if cfg.trainer.type == "torch":

        #     log.info(f"Task based on PyTorch.")
        #     # instantiate learner components

        #     # Model Load
        #     model = _conf_instantiate(cfg.learner.model)
        #     log.info("Model loaded...")

        #     if not cfg.learner.resume:
        #         log.warning("'resume' is set to False. The model will be initialized without loading a checkpoint.")
        #     # loss
        #     loss = instantiate(cfg.learner.loss) if cfg.learner.loss and cfg.learner.loss._target_ else None
        #     if loss:
        #         log.info("Loss function loaded...")
        #     else:
        #         log.warning("Loss function is not defined. Are you sure you wanted this?")
        #     # optimizer
        #     optim = instantiate(cfg.learner.optimizer, params=model.parameters()) if cfg.learner.optimizer and cfg.learner.optimizer._target_ else None
        #     if optim:
        #         log.info("Optimizer loaded...")
        #     else:
        #         log.warning("Optimizer is not defined. Are you sure you wanted this?")
        #     # scheduler
        #     schlr = None# instantiate(cfg.learner.scheduler, lr_lambda=lambda x: x**cfg.learner.scheduler.lr_lambda, optimizer=optim) if cfg.learner.scheduler and cfg.learner.scheduler._target_ else None
        #     if schlr:
        #         log.info("Scheduler loaded...")
        #     else:
        #         log.warning("Scheduler is not defined. Edit the configuration if this is not what you wanted.")
            
        #     # instantiate learner
        #     if 'path' in cfg.learner and cfg.learner.path:
        #         _mod, _bn = dynamic_import(cfg.learner.path)
        #         learner = getattr(_mod, cfg.learner.cls)(model, loss, optim, schlr, cfg.learner.resume, cfg.learner.resume_path)
        #     else:
        #         learner = getattr(import_module('.pytorch.learner', 'eniat'), cfg.learner.cls)(model, loss, optim, schlr, cfg.learner.resume, cfg.learner.resume_path)
        #     if learner:
        #         log.info("Learner instance created.")
            # instantiate trainer
            trainer = getattr(import_module('.pytorch', 'eniat'), 'TorchTrainer')(conf=cfg.trainer, data_conf=cfg.data, learner_conf=cfg.learner, logger=log)
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