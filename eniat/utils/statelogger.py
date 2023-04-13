import logging
from logging import Logger
from typing import Callable, Literal
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json

class DummyLogger():
    r"""This class is for debugging. You can ignore this."""
    def __init__(self, name, level, conf) -> None:
        self.conf = conf

    def info(self, msg, *args, **kwargs):
        print(msg)
    
    def debug(self, msg, *args, **kwargs):
        print(msg)

    def warning(self, msg, *args, **kwargs):
        print(msg)

    def error(self, msg, *args, **kwargs):
        print(msg)

    def critical(self, msg, *args, **kwargs):
        print(msg)

    def log(self, level, msg, *args, **kwargs):
        print(msg)

    def exception(self, msg, *args, **kwargs):
        print(msg)

    def log_state(self, data:dict):
        return
class StateLogger():

    __silent:bool =False

    class silent:
        def __enter__(self):
            self.prev_silent = StateLogger.__silent
            StateLogger.__silent = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            StateLogger.__silent = self.prev_silent

    def __init__(self, name: str, level = 0, conf:DictConfig = None) -> None:
        self.conf = conf
        self.console = logging.getLogger(name)
        self.console.setLevel(level)
        self.unit = conf.unit
        self.interval = conf.interval
        self.silent = False

    def _stepfilter(fn:Callable) -> Callable:
        def wrapper(self, data:dict, force:bool=False):
            if 'epoch' in data:
                timestep = data['epoch']
                unit = 'epoch'
            elif 'step' in data:
                timestep = data['step']
                unit = 'step'
            else:
                raise ValueError("Dict 'data' must contain timestep variable when logging states.")
            if not force and (self.unit != unit or timestep % self.interval):
                return
            return fn(data)
        return wrapper

    def setLevel(self, level):
        return self.console.setLevel(level)

    def isEnabledFor(self, level):
        return self.console.isEnabledFor(level)

    def getEffectiveLevel(self, level):
        return self.console.getEffectiveLevel(level)

    def addFilter(self, filter):
        return self.console.addFilter(filter)

    def removeFilter(self, filter):
        return self.console.removeFilter(filter)

    def filter(self, record):
        return self.console.filter(record)

    def addHandler(self, hdlr):
        return self.console.addHandler(hdlr)

    def removeHandler(self, hdlr):
        return self.console.removeHandler(hdlr)
    
    def handle(self, record):
        return self.console.handle(record)
    
    def hasHandlers(self):
        return self.hasHandlers()

    def findCaller(self, stack_info=False, stacklevel=1):
        return self.console.findCaller(stack_info, stacklevel)

    def info(self, msg, *args, **kwargs):
        return self.console.info(msg, *args, **kwargs) if not self.__silent else None
    
    def debug(self, msg, *args, **kwargs):
        return self.console.debug(msg, *args, **kwargs) if not self.__silent else None

    def warning(self, msg, *args, **kwargs):
        return self.console.warning(msg, *args, **kwargs) if not self.__silent else None

    def error(self, msg, *args, **kwargs):
        return self.console.error(msg, *args, **kwargs) if not self.__silent else None

    def critical(self, msg, *args, **kwargs):
        return self.console.critical(msg, *args, **kwargs) if not self.__silent else None

    def log(self, level, msg, *args, **kwargs):
        return self.console.log(level, msg, *args, **kwargs) if not self.__silent else None

    def exception(self, msg, *args, **kwargs):
        return self.console.exception(msg, *args, **kwargs) if not self.__silent else None 

    @_stepfilter
    def log_state(self, data:dict):
        log = json.dumps(data, ensure_ascii=False, indent=2)
        if self.conf.log.file_log:
            with open(Path(self.conf.log.dir).joinpath('state', "a")) as f:
                f.write(log)
        self.console.info(log)

class TBoardLogger(StateLogger):

    def __init__(self, name: str, conf: DictConfig, file_path: str = None) -> None:
        super().__init__(name, conf, file_path)
        self.tb = __import__("torch.utils.tensorboard", fromlist=["torch.utils"])
        self.tb_logger = self.tb.SummaryWriter("tensorboard", conf.logger.logging_dir)

    def __getattr__(self, __name: str):
        if __name.startswith('add_'):
            return getattr(self.tb_logger, __name)
        return getattr(self.console, __name)

class MFlowLogger(StateLogger):

    def __init__(self, name: str, conf: DictConfig, file_path: str = None, uri:str='localhost') -> None:
        super().__init__(name, conf, file_path)

        self.mf_logger = __import__("mlflow")
        self.mf_logger.set_tracking_uri(uri)
        if autolog:
            self.mf_logger.autolog(**autolog_options)
        else:
            
            if experiment_id:
                self.mf_logger.set_experiment(experiment_id=experiment_id)
            else:
                self.mf_logger.set_experiment(experiment_name=experiment_name)

            self.mf_logger.start_run(run_id)

class TotalLogger(StateLogger):

    console = None
    mf_logger = None
    tb_logger = None

    def __init__(self, *args, **kwargs) -> None:
        self.init_logger(*args, **kwargs)

    def __del__(self) -> None:
        if not self.autolog:
            self.mf_logger.end_run()
        self.tb_logger.close()

    def get_mlflow_logger(self):
        return self.mf_logger

    def get_tensorboard_logger(self):
        return self.tb_logger

    def init_logger(
            self,
            name:str,
            file:bool=True,
            mlflow:bool=True,
            remote_model_save:bool=False,
            tensorboard:bool=True,
            autolog:bool=False,
            autolog_options:dict=False,
            uri:str="",
            experiment_id:str=None,
            run_id:str=None,
            experiment_name:str=None
            ):
        
        self.console = logging.getLogger(name)
        
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.file = file
        self.mlflow = mlflow
        self.remote_model = remote_model_save
        self.tensorboard = tensorboard
        self.autolog = autolog
        self.uri = uri

        if mlflow:
            self.mf_logger = __import__("mlflow")
            self.mf_logger.set_tracking_uri(uri)
            if autolog:
                self.mf_logger.autolog(**autolog_options)
            else:
                
                if experiment_id:
                    self.mf_logger.set_experiment(experiment_id=experiment_id)
                else:
                    self.mf_logger.set_experiment(experiment_name=experiment_name)

                self.mf_logger.start_run(run_id)
                    
        if tensorboard:
            self.tb = __import__("torch.utils.tensorboard", fromlist=["torch.utils"])
            self.tb_logger = self.tb.SummaryWriter("tensorboard")

    def log_scalar(self, key:str, value, step:int):
        if self.mlflow and not self.autolog:
            self.mf_logger.log_metric(key, value, step)
        if self.tensorboard:
            self.tb_logger.add_scalar(key, value, step)
        self.console.info(key + ": " + value)

    def log_scalars(self, key_value:dict, step:int, tag:str=None):
        if self.mlflow and not self.autolog:
            self.mf_logger.log_metrics(key_value, step)
        if self.tensorboard:
            self.tb_logger.add_scalars(tag, key_value, step)
        for k, v in key_value.items():
            self.console.info(k + ": " + v)

    def log_image(self, image, tag:str=None, step:int=None, format:str=None, filepath:str=None):
        if self.mlflow and not self.autolog:
            self.mf_logger.log_image(image, filepath)
            self.console.info(f"Logged an image to {self.mf_logger.get_tracking_uri()}")
        if self.tensorboard:
            self.tb_logger.add_image(tag, image, step, format)
            self.console.info(f"Logged an image to tensorboard")
    
    def log_figure(self, figure, tag:str=None, step:int=None, close:bool=None, filepath:str=None):
        if self.mlflow and not self.autolog:
            self.mf_logger.log_figure(figure, filepath)
            self.console.info(f"Logged a figure to {self.mf_logger.get_tracking_uri()}")
        if self.tensorboard:
            self.tb_logger.add_figure(tag, figure, step, close)
            self.console.info(f"Logged a figure to tensorboard")