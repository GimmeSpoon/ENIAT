from omegaconf import DictConfig, OmegaConf
from importlib.resources import files
from importlib.util import spec_from_file_location, module_from_spec
from importlib import import_module
import sys
import os
from pathlib import Path
from typing import Union, Sequence
from omegaconf import OmegaConf
from copy import deepcopy
from functools import partial

RESKEYS = (
    RESKEY_LAMBDA := "ld",
    RESKEY_EXP := "exp",
    RESKEY_INST := "ins",
)

RESOLVERS = {
    RESKEY_LAMBDA : lambda x: eval(x),
    RESKEY_EXP : lambda e: lambda x: x ** e,
    RESKEY_INST : lambda p, c, o: load_class(p, c)(**o)
}

for reskey in RESKEYS:
    OmegaConf.register_new_resolver(reskey, RESOLVERS[reskey], replace=True)

def recursive_merge(*configs) -> DictConfig:
    merged = OmegaConf.create({})

    if len(configs) == 1:
        for config in configs[0]:
            merged = OmegaConf.unsafe_merge(merged, config)
    else:
        for config in configs:
            merged = OmegaConf.unsafe_merge(merged, config)

    return merged

def recursive_merge_by_path(*paths) -> DictConfig:
    if len(paths) == 1:
        return OmegaConf.load(paths[0])
    else:
        configs = [OmegaConf.load(path) for path in paths]
        return recursive_merge(*configs)

def load_default_conf(path:Union[str, Path]=None) -> DictConfig:
    if path is not None:
        return OmegaConf.load(path)
    else:
        return OmegaConf.load(files('eniat').joinpath('config/default.yaml'))

def import_by_file(path:Union[str,Path], name:str = None):
    _spec = spec_from_file_location(_bn:=name if name else os.path.basename(path), os.path.abspath(path))
    if not _spec:
        raise FileNotFoundError(f"Can not read from the location : {path}")
    _mod = module_from_spec(_spec)
    sys.modules[_bn] = _mod
    _spec.loader.exec_module(_mod)
    return _mod, _bn

def load_class(path:Union[str, Path]=None, _class:str=None):
    if path is None:
        if _class is None:
            raise ValueError("No valid arguments")
        _cls = _class.split('.')
        try:
            return getattr(import_module('.'.join(_cls[:-1])), _cls[-1])
        except:
            return getattr(import_module('eniat.' + '.'.join(_cls[:-1]), 'eniat'), _cls[-1])
    else:
        return getattr(import_by_file(path)[0], _class)

def instantiate(conf:DictConfig, *args, _partial:bool=False, **kwargs):
    conf = deepcopy(conf)
    if not conf:
        return None
    try:
        _cls = load_class(conf.path, conf.cls)
    except:
        raise ValueError(f"Instantiation failed. Current config is not valid: {conf}")
    
    args = list(args)
    options = {}
    if 'options' in conf and conf.options:
        options = OmegaConf.to_container(conf.options)
        args += options.pop('__args__', [])
    return partial(_cls, *args, **options, **kwargs) if _partial else _cls(*args, **options, **kwargs)