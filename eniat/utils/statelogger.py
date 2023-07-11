import logging
from logging import Logger
from typing import Callable, Literal
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
import csv
import openpyxl
from openpyxl import Workbook

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

    _silent:bool =False

    class silent:
        def __enter__(self):
            self.prev_silent = StateLogger._silent
            StateLogger._silent = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            StateLogger._silent = self.prev_silent

    def __init__(self, name: str, level = 0, conf:DictConfig = None, resume_path:str=None) -> None:
        self.conf = conf
        self.console = logging.getLogger(name)
        self.console.setLevel(conf.level)
        self.hc = HydraConfig.get()
        if resume_path is not None:
            _ext=resume_path.split('.')[-1]

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
        return self.console.info(msg, *args, **kwargs) if not self._silent else None
    
    def debug(self, msg, *args, **kwargs):
        return self.console.debug(msg, *args, **kwargs) if not self._silent else None

    def warning(self, msg, *args, **kwargs):
        return self.console.warning(msg, *args, **kwargs) if not self._silent else None

    def error(self, msg, *args, **kwargs):
        return self.console.error(msg, *args, **kwargs) if not self._silent else None

    def critical(self, msg, *args, **kwargs):
        return self.console.critical(msg, *args, **kwargs) if not self._silent else None

    def log(self, level, msg, *args, **kwargs):
        return self.console.log(level, msg, *args, **kwargs) if not self._silent else None

    def exception(self, msg, *args, **kwargs):
        return self.console.exception(msg, *args, **kwargs) if not self._silent else None 

    def log_state(self, data:dict, timestep:int, unit:Literal['epoch', 'step'], tag:str=None, to_json:bool=None, to_xls:bool=None, silent:bool=False):
        if not self.check_loss_policy(data, unit):
            return
        if (to_json if to_json is not None else self.conf.json):
            json.dump({'epoch':self.e_state, 'step':self.s_state}, open(Path.joinpath(Path(self.hc.runtime.output_dir), './chekcpoints/state.json')))
        if (to_xls if to_xls is not None else self.conf.xls):
            raise NotImplementedError("sorry. xls is not supported yet.")
        if not silent:
            _str = '\n'.join([f"{item[0]} : {item[1]}" for item in data.items()])
            with logging_redirect_tqdm():
                self.console.info("Current training state" + _str)

class TensorboardLogger(StateLogger):

    def __init__(self, name: str, conf: DictConfig = None, resume_path: str = None) -> None:
        super().__init__(name, conf, resume_path)
    
    def prepare(self) -> None:
        tb = __import__("torch.utils.tensorboard", fromlist=["SummaryWriter"])
        self.tb = tb.SummaryWriter(self.conf.logging_dir or Path.joinpath(Path(self.hc.runtime.output_dir), 'tensorboard'))

    def log_state(self, data:dict, timestep:int, unit:Literal['epoch', 'step'], tag:str=None, to_json:bool=None, to_xls:bool=None, silent:bool=False):
        if not self.check_loss_policy(data, unit):
            return
        if len(data) == 1:
            data = list(data.items())[0]
            self.log_scalar(data[0], data[1], timestep, silent=silent)
        elif len(data) > 1:
            self.log_scalars(data, timestep, tag, silent=silent)

        super().log_state(data, timestep, unit, tag, to_json, to_xls, silent=True)
        
    def log_scalar(self, key:str, value, step:int, silent:bool=False):
        self.tb.add_scalar(key, value, step)
        if not silent:
            with logging_redirect_tqdm():
                self.console.info("Current state logged\n" + key + ": " + str(value))

    def log_scalars(self, key_value:dict, step:int, tag:str=None, silent:bool=False):
        self.tb.add_scalars(tag, key_value, step)
        if not silent:
            with logging_redirect_tqdm():
                self.console.info("Current state logged\n"+(f"{key} : {value}\n" for key, value in key_value.items()))

    def log_image(self, image, tag:str=None, step:int=None, format:str=None, filepath:str=None, silent:bool=False):
        self.tb.add_image(tag, image, step, format)
        if not silent:
            self.console.info(f"Logged an image to tensorboard")
    
    def log_figure(self, figure, tag:str=None, step:int=None, close:bool=None, filepath:str=None, silent:bool=False):
        self.tb.add_figure(tag, figure, step, close)
        if not silent:
            self.console.info(f"Logged a figure to tensorboard")

class MLFlowLogger(StateLogger):

    def __init__(
            self,
            name: str,
            conf: DictConfig = None,
            resume_path = None,
            ) -> None:
        super().__init__(name, conf, resume_path)

    def prepare(self, ) -> None:
        
        self.uri = self.conf.mlflow.uri
        self.mf = __import__('mlflow')
        self.mf.set_tracking_uri(self.uri)

        if self.conf.mlflow.autolog:
            # Refer to autolog reference and fix here
            self.mf.autolog()    
        elif self.conf.mlflow.run_id is not None:
            self.current_run = self.mf.start_run(self.conf.mlflow.run_id, ru_name=self.conf.mlflow.run_name)
        else:
            if self.conf.mlflow.exp_id is not None:
                self.experiment = self.mf.set_experiment(experiment_id=self.conf.mlflow.exp_id)
            else:
                self.experiment = self.mf.set_experiment(experiment_name=self.conf.mlflow.exp_name)
            self.current_run = self.mf.start_run(experiment_id=self.conf.mlflow.exp_id)

    def __del__ (self) -> None:

        if not 'mf' in self.__dict__:
            return
        
        if self.mf is not None and self.mf.active_run() is not None:
            self.mf.end_run()

    def log_scalar(self, key:str, value, step:int, silent:bool=False):
        self.mf.log_metric(key, value, step)
        if not silent:
            with logging_redirect_tqdm():
                self.console.info("Current state logged\n" + key + ": " + str(value))

    def log_scalars(self, key_value:dict, step:int, tag:str=None, silent:bool=False):
        self.mf.log_metrics(key_value, step)
        if not silent:
            with logging_redirect_tqdm():
                self.console.info("Current state logged\n"+(f"{key} : {value}\n" for key, value in key_value.items()))

    def log_image(self, image, tag:str=None, step:int=None, format:str=None, filepath:str=None, silent:bool=False):
        self.mf.log_image(image, filepath)
        if not silent:
            self.console.info(f"Logged an image to {self.mf.get_tracking_uri()}")
    
    def log_figure(self, figure, tag:str=None, step:int=None, close:bool=None, filepath:str=None, silent:bool=False):
        self.mf.log_figure(figure, filepath)
        if not silent:
            self.console.info(f"Logged a figure to {self.mf.get_tracking_uri()}")

    def log_state(self, data:dict, timestep:int, unit:Literal['epoch', 'step'], tag:str=None, to_json:bool=None, to_xls:bool=None, silent:bool=False):

        if not self.check_loss_policy(data, unit):
            return
        if len(data) == 1:
            data = list(data.items())[0]
            self.log_scalar(data[0], data[1], timestep, silent=silent)
        elif len(data) > 1:
            self.log_scalars(data, timestep, tag, silent=silent)

        super().log_state(data, timestep, unit, tag, to_json, to_xls, True)

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