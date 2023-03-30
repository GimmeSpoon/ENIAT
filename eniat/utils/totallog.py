import logging
from typing import Callable
import json

class TotalLogger():

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