"""statelogger module includes loggers to record any information about
experiments you conduct. The basic feature is to log outputs to stdout
just like builtin package 'logging', but also it can log to files or
other tools such as Tensorboard or MLFlow."""

import logging
from logging import Formatter, LogRecord, StreamHandler
from datetime import datetime
from typing import Callable, Literal, TypeVar, Union, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
import csv
import openpyxl
from openpyxl import Workbook
from abc import abstractmethod
import os
import sys
from rich.traceback import install
from rich.console import Console
from .style import init_display, LogHandler, PreLogHandler, LogFileHandler
from datetime import datetime

install(show_locals=True)

FILE_HND_MAX_BYTES = 10_485_760
FILE_HND_BCK_COUONT = 1

T_co = TypeVar("T_co", covariant=True)

g_logger = None

class Logger:
    @abstractmethod
    def info(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def debug(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def warning(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def error(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def critical(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def log(self, level, msg, *args, **kwargs):
        pass

    @abstractmethod
    def exception(self, msg, *args, **kwargs):
        pass

    @abstractmethod
    def log_state(self, data: dict) -> None:
        pass


class DummyLogger(Logger):
    r"""DummyLogger just prints to stdout every log messages. It doesn't
    keep any training state or output any kind of results as files. If
    you need to record any information about experiments, You should
    consider using other Loggers such as StateLogger."""

    def __init__(
        self, name, level, conf, silent: bool = False, inactive: bool = False
    ) -> None:
        self.name = name
        self.level = level
        self.conf = conf
        self.silent = conf.silent
        if silent:
            self.silent = True
        self.inactive = inactive
        if inactive:
            self.silent = True

        self.init_console_logger(name, level, conf.logging_dir)

    def init_console_logger(
        self, name: str, level: int = 0, logging_dir: Union[str, Path] = None
    ) -> None:

        if isinstance(logging_dir, str):
            logging_dir = Path(logging_dir)

        self.logging_dir = logging_dir

        logging_dir.mkdir(parents=True, exist_ok=True)

        rich_handler = PreLogHandler(
            rich_tracebacks=True, show_path=False, tracebacks_suppress=[]
        )

        file_handler = LogFileHandler(
            logging_dir.joinpath("eniat.log"),
            maxBytes=FILE_HND_MAX_BYTES,
            backupCount=1,
        )

        logging.basicConfig(level=level, handlers=[rich_handler, file_handler])
        self.console = logging.getLogger(name)
        self.rich_console = None

    def load_rich_console(self) -> None:
        self.rich_console = init_display(silent=self.silent)
        file_handler = LogFileHandler(
            self.logging_dir.joinpath("eniat.log"),
            maxBytes=FILE_HND_MAX_BYTES,
            backupCount=1,
        )
        logging.basicConfig(
            level=self.level,
            handlers=[LogHandler(self.level), file_handler],
            force=True,
        )

    def be_silent(self) -> None:
        self.silent = True

    def be_inactive(self) -> None:
        self.inactive = True
        self.silent = True

    def reload(self, name=None, level=None, logging_dir=None):
        self.init_console_logger(
            name if name is not None else self.name,
            level if level is not None else self.level,
            logging_dir if logging_dir is not None else self.conf.logging_dir,
        )

    def info(self, msg, *args, **kwargs):
        return self.console.info(msg, *args, **kwargs) if not self.silent else None

    def debug(self, msg, *args, **kwargs):
        return self.console.debug(msg, *args, **kwargs) if not self.silent else None

    def warning(self, msg, *args, **kwargs):
        return self.console.warning(msg, *args, **kwargs) if not self.silent else None

    def error(self, msg, *args, **kwargs):
        return self.console.error(msg, *args, **kwargs) if not self.silent else None

    def critical(self, msg, *args, **kwargs):
        return self.console.critical(msg, *args, **kwargs) if not self.silent else None

    def log(self, level, msg, *args, **kwargs):
        return (
            self.console.log(level, msg, *args, **kwargs) if not self.silent else None
        )

    def exception(self, msg, *args, **kwargs):
        return self.console.exception(msg, *args, **kwargs) if not self.silent else None

    def log_state(
        self,
        data: dict,
        epoch: int = None,
        step: int = None,
        unit: Literal["epoch", "step"] = "epoch",
        training_state: bool = True,
        to_json: bool = None,
        to_xls: bool = None,
        to_csv: bool = None,
        silent: bool = False,) -> None:
        return None


class StateLogger(DummyLogger):
    """StateLogger keeps tracks of training states and can output the
    results as files. Specifically, data transferred by 'log_state'
    will be recorded into State instance."""

    class StateTable:
        def __init__(self) -> None:
            self.__state = {}
            self.__e = self.__state["epoch"] = {}
            self.__s = self.__state["step"] = {}
            self.__n = self.__state["global"] = []

        def add(self, data: dict, epoch: int = None, step: int = None) -> None:
            if epoch is None:
                if step is None:
                    self.__n.append(data)
                else:
                    if step in self.__s:
                        self.__s[step].append(data)
                    else:
                        self.__s[step] = [data]
            else:
                if not epoch in self.__e:
                    self.__e[epoch] = {"data": []}

                if step is None:
                    self.__e[epoch]["data"].append(data)
                else:
                    if step in self.__e[epoch]:
                        self.__e[epoch][step].append(data)
                    else:
                        self.__e[epoch][step] = [data]

        def get(self) -> T_co:
            pass

        def to_json(self, path: Union[Path, str]) -> None:
            with open(path, "w", encoding="utf8") as f:
                json.dump(self.__state, f, ensure_ascii=False, indent=4)

        def to_csv(self, path: Union[Path, str]) -> None:
            pass

        def to_xls(self, path: Union[Path, str]) -> None:
            pass

    def __init__(
        self, name: str, level: int = 0, conf: DictConfig = None, silent: bool = False
    ) -> None:
        super().__init__(name, level, conf, silent)
        self.__state = self.StateTable()

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

    def check_loss_policy(self, _t: int, _u: str) -> bool:
        if _t == None or _u == None or self.conf.log_interval == 0:
            return True
        return (_u == self.conf.unit) and (_t % self.conf.log_interval) == 0

    @property
    def state(self):
        return self.__state

    def log_state(
        self,
        data: dict,
        epoch: int = None,
        step: int = None,
        unit: Literal["epoch", "step"] = "epoch",
        training_state: bool = True,
        to_json: bool = None,
        to_xls: bool = None,
        to_csv: bool = None,
        silent: bool = False,
    ):

        if (
            not self.check_loss_policy(epoch if unit == "epoch" else step, unit)
            or self.inactive
        ):
            return

        self.__state.add(data, epoch, step)

        if to_json if to_json is not None else self.conf.json:
            self.__state.to_json(
                Path(self.conf.logging_dir).joinpath("training_state.json").absolute()
            )
        if to_xls if to_xls is not None else self.conf.xls:
            raise NotImplementedError("sorry. xls is not supported yet.")
        if to_csv if to_csv is not None else self.conf.csv:
            raise NotImplementedError("Sorry, csv is not supported yet.")

        if not silent:
            _str = (
                f"{'Training State' if training_state else 'Evaluation Result'} ({unit.capitalize()} {epoch if unit=='epoch' else step})\n"
                + "\n".join(
                    [
                        f"                       {item[0]} : {item[1]}"
                        for item in data.items()
                    ]
                )
            )
            self.info(_str, extra={"markup": True})


class TensorboardLogger(StateLogger):
    """TensorboardLogger literally features Tensorboard from torch.utils.
    This inherits StateLogger, thus you can record Tensorboard and files at
    the same time if you wish."""

    def __init__(
        self, name: str, conf: DictConfig = None, resume_path: str = None
    ) -> None:
        super().__init__(name, conf, resume_path)

    def prepare(self) -> None:
        tb = __import__("torch.utils.tensorboard", fromlist=["SummaryWriter"])
        self.tb = tb.SummaryWriter(
            Path(self.conf.logging_dir).joinpath("tensorboard")
        )

    def reload(self, *args, **kwargs) -> None:
        super().reload(*args, **kwargs)
        self.prepare()

    def log_state(
        self,
        data: dict,
        epoch: int = None,
        step: int = None,
        unit: Literal["epoch", "step"] = "epoch",
        training_state: bool = True,
        to_json: bool = False,
        to_xls: bool = False,
        to_csv: bool = False,
        silent: bool = False,
    ):
        if not self.check_loss_policy(epoch if unit == "epoch" else step, unit) or self.inactive:
            return
        if len(data) == 1:
            data = list(data.items())[0]
            self.log_scalar(data[0], data[1], epoch if unit == "epoch" else step, silent=silent)
        elif len(data) > 1:
            tag = "Trainig state" if training_state else "Evaluation Result"
            self.log_scalars(data, epoch if unit == "epoch" else step, tag, silent=silent)

        super().log_state(data, epoch, step, unit, training_state, to_json, to_xls, to_csv, silent=True)

    def log_scalar(self, key: str, value, step: int, silent: bool = False):
        if self.inactive:
            return
        self.tb.add_scalar(key, value, step)
        if not silent:
            self.console.info("Current state logged\n" + key + ": " + str(value))

    def log_scalars(
        self, key_value: dict, step: int, tag: str = None, silent: bool = False
    ):
        if self.inactive:
            return
        self.tb.add_scalars(tag, key_value, step)
        if not silent:
            self.console.info(
                "Current state logged\n"
                + str(f"{key} : {value}\n" for key, value in key_value.items())
            )

    def log_image(
        self,
        image,
        tag: str = None,
        step: int = None,
        format: str = None,
        filepath: str = None,
        silent: bool = False,
    ):
        if self.inactive:
            return
        self.tb.add_image(tag, image, step, format)
        if not silent:
            self.console.info(f"Logged an image to tensorboard")

    def log_figure(
        self,
        figure,
        tag: str = None,
        step: int = None,
        close: bool = None,
        filepath: str = None,
        silent: bool = False,
    ):
        if self.inactive:
            return
        self.tb.add_figure(tag, figure, step, close)
        if not silent:
            self.console.info(f"Logged a figure to tensorboard")


class MLFlowLogger(StateLogger):
    """TensorboardLogger literally features Tensorboard from torch.utils.
    This inherits StateLogger, thus you can record Tensorboard and files at
    the same time if you wish."""

    def __init__(
        self,
        name: str,
        conf: DictConfig = None,
        resume_path=None,
    ) -> None:
        super().__init__(name, conf, resume_path)

    def prepare(
        self,
    ) -> None:

        self.uri = self.conf.mlflow.uri
        self.mf = __import__("mlflow")
        self.mf.set_tracking_uri(self.uri)

        if self.conf.mlflow.autolog:
            # Refer to autolog reference and fix here
            self.mf.autolog()
        elif self.conf.mlflow.run_id is not None:
            self.current_run = self.mf.start_run(
                self.conf.mlflow.run_id, ru_name=self.conf.mlflow.run_name
            )
        else:
            if self.conf.mlflow.exp_id is not None:
                self.experiment = self.mf.set_experiment(
                    experiment_id=self.conf.mlflow.exp_id
                )
            else:
                self.experiment = self.mf.set_experiment(
                    experiment_name=self.conf.mlflow.exp_name
                )
            self.current_run = self.mf.start_run(experiment_id=self.conf.mlflow.exp_id)

    def reload(self, *args, **kwargs) -> None:
        super().reload(*args, **kwargs)
        self.prepare()

    def __del__(self) -> None:

        if not "mf" in self.__dict__:
            return

        if self.mf is not None and self.mf.active_run() is not None:
            self.mf.end_run()

    def log_scalar(self, key: str, value, step: int, silent: bool = False):
        if self.inactive:
            return
        self.mf.log_metric(key, value, step)
        if not silent:
            self.console.info("Current state logged\n" + key + ": " + str(value))

    def log_scalars(
        self, key_value: dict, step: int, tag: str = None, silent: bool = False
    ):
        if self.inactive:
            return
        self.mf.log_metrics(key_value, step)
        if not silent:
            self.console.info(
                "Current state logged\n"
                + (f"{key} : {value}\n" for key, value in key_value.items())
            )

    def log_image(
        self,
        image,
        tag: str = None,
        step: int = None,
        format: str = None,
        filepath: str = None,
        silent: bool = False,
    ):
        if self.inactive:
            return
        self.mf.log_image(image, filepath)
        if not silent:
            self.console.info(f"Logged an image to {self.mf.get_tracking_uri()}")

    def log_figure(
        self,
        figure,
        tag: str = None,
        step: int = None,
        close: bool = None,
        filepath: str = None,
        silent: bool = False,
    ):
        if self.inactive:
            return
        self.mf.log_figure(figure, filepath)
        if not silent:
            self.console.info(f"Logged a figure to {self.mf.get_tracking_uri()}")

    def log_state(
        self,
        data: dict,
        epoch: int = None,
        step: int = None,
        unit: Literal["epoch", "step"] = "epoch",
        training_state: bool = True,
        to_json: bool = False,
        to_xls: bool = False,
        to_csv: bool = False,
        silent: bool = False,
    ):

        if not self.check_loss_policy(data, unit) or self.inactive:
            return
        if len(data) == 1:
            data = list(data.items())[0]
            self.log_scalar(data[0], data[1], epoch if unit == "epoch" else step, silent=silent)
        elif len(data) > 1:
            self.log_scalars(data, epoch if unit == "epoch" else step, "Training State" if training_state else "Evaluation Result", silent=silent)

        super().log_state(data, epoch, step, unit, training_state, to_json, to_xls, to_csv, True)


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
        name: str,
        file: bool = True,
        mlflow: bool = True,
        remote_model_save: bool = False,
        tensorboard: bool = True,
        autolog: bool = False,
        autolog_options: dict = False,
        uri: str = "",
        experiment_id: str = None,
        run_id: str = None,
        experiment_name: str = None,
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

    def log_scalar(self, key: str, value, step: int):
        if self.inactive:
            return
        if self.mlflow and not self.autolog:
            self.mf_logger.log_metric(key, value, step)
        if self.tensorboard:
            self.tb_logger.add_scalar(key, value, step)
        self.console.info(key + ": " + value)

    def log_scalars(self, key_value: dict, step: int, tag: str = None):
        if self.inactive:
            return
        if self.mlflow and not self.autolog:
            self.mf_logger.log_metrics(key_value, step)
        if self.tensorboard:
            self.tb_logger.add_scalars(tag, key_value, step)
        for k, v in key_value.items():
            self.console.info(k + ": " + v)

    def log_image(
        self,
        image,
        tag: str = None,
        step: int = None,
        format: str = None,
        filepath: str = None,
    ):
        if self.inactive:
            return
        if self.mlflow and not self.autolog:
            self.mf_logger.log_image(image, filepath)
            self.console.info(f"Logged an image to {self.mf_logger.get_tracking_uri()}")
        if self.tensorboard:
            self.tb_logger.add_image(tag, image, step, format)
            self.console.info(f"Logged an image to tensorboard")

    def log_figure(
        self,
        figure,
        tag: str = None,
        step: int = None,
        close: bool = None,
        filepath: str = None,
    ):
        if self.inactive:
            return
        if self.mlflow and not self.autolog:
            self.mf_logger.log_figure(figure, filepath)
            self.console.info(f"Logged a figure to {self.mf_logger.get_tracking_uri()}")
        if self.tensorboard:
            self.tb_logger.add_figure(tag, figure, step, close)
            self.console.info(f"Logged a figure to tensorboard")


def load_logger(logger_conf: DictConfig) -> Logger:
    logger = getattr(sys.modules[__name__], logger_conf.type)(
        logger_conf.name, logger_conf.level, logger_conf
    )
    return logger

def init_logger(
        log_cls = StateLogger,
        conf:DictConfig = None,
        **kwargs
    ) -> Logger:
    global g_logger
    if g_logger is not None:
        if isinstance(g_logger, log_cls):
            return g_logger
        else:
            g_logger = None
    if conf is not None:
        g_logger = load_logger(conf)
    else:
        g_logger = log_cls(**kwargs)
    return g_logger
    
def get_logger() -> Logger:
    return g_logger or init_logger()