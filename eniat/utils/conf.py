from omegaconf import DictConfig, OmegaConf
from importlib.util import spec_from_file_location, module_from_spec
from hydra import compose, initialize
from hydra.utils import instantiate
import pkg_resources
import sys
import os
from typing import Union, Sequence

def init_conf(global_config_path:str, job_name:str="eniat", version_base=None):
    # Global Config
    with initialize(version_base=version_base, config_path=global_config_path, job_name=job_name):
        cfg = compose(config_name="eniat")
        print(cfg)
    return cfg

def _dynamic_import(path:str, name:str = None):
    _spec = spec_from_file_location(_bn:=name if name else os.path.basename(path), os.path.abspath(path))
    if not _spec:
        raise FileNotFoundError(f"Can not read from the location : {path}")
    _mod = module_from_spec(_spec)
    sys.modules[_bn] = _mod
    _spec.loader.exec_module(_mod)
    return _mod, _bn

def _dynamic_load(name:str, path:str):
    _mod, _bn = _dynamic_import(path, name)
    return getattr(_mod, name)

def conf_instantiate(conf:DictConfig):
    if not conf:
        return None
    if '_target_' in conf and conf['_target_']: #Hydra instance
        return instantiate(conf)
    elif 'path' in conf and conf['path']:
        if 'cls' in conf and conf['cls']: #dynamic load
            _cls = _dynamic_load(conf.cls, conf.path)
            options = OmegaConf.to_container(conf)
            options.pop('_target_', None)
            options.pop('path', None)
            options.pop('cls', None)
            return _cls(**options)
    raise ValueError(f"Instantiation failed. Current config is not valid: {options}")

def _merge_conf_by_path(conf:DictConfig, paths:Union[str,Sequence[str]]) -> DictConfig:

    if isinstance(paths, str):
        paths = [paths]
    
    for path in paths:
        new_cfg = OmegaConf.load(path)
        #conf = OmegaConf.merge(new_cfg, conf)
        conf = OmegaConf.merge(conf, new_cfg)

    return conf
