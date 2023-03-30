from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import logging
import hydra
from hydra import compose, initialize
import pkg_resources
import os

def 

def init_conf(config_name, overrides=None, config_path:str="config", job_name:str="eniat"):
    # Global Config
    global_config_path = os.path.abspath(pkg_resources.resource_filename(__name__, 'config'))
    with initialize(version_base=None, config_path=os.path.relpath(global_config_path), job_name=job_name):
        global_cfg = compose(config_name="eniat")
        print(global_cfg)
    # User Config
    with initialize(version_base=None, config_path=config_path, job_name=job_name):
        user_cfg = compose(config_name=config_name, overrides=overrides)
    return OmegaConf.merge(global_cfg, user_cfg)

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def eniat(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    eniat()