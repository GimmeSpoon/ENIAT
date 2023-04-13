from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.utils import instantiate
import os
import sys
from importlib.util import spec_from_file_location, module_from_spec
from typing import Callable, Sequence, Union

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
        return _cls(**options)
    else:
        raise ValueError(f"Instantiation failed. Current config is not valid: {options}")

def init_conf(global_config_path:str, job_name:str="eniat", version_base=None):
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