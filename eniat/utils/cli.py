# from typing import
import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .conf import load_conf, recursive_merge, recursive_merge_by_path


def datetime_to_str() -> str:
    return datetime.now().strftime("%y%m%d_%H:%M:%S")


MLTASK = ("fit", "train", "eval", "test", "infer", "predict")

parser = argparse.ArgumentParser(
    "ENIAT",
    epilog="",
    description="ENIAT supports and boosts your machine learning experiments!",
    exit_on_error=False,
)

parser.add_argument("-v", "--version", action="store_true")

parser.add_argument(
    "-p",
    "--package",
    type=str,
    choices=["torch", "sklearn"],
    help="Based package",
)
parser.add_argument("-t", "--task", choices=MLTASK)
parser.add_argument(
    "-c", "--config", nargs="*", type=Path, help="yaml config files to parse."
)
parser.add_argument(
    "-s",
    "--silent",
    action="store_true",
    help="if True, no console output. (not affect File output)",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=Path,
    default=Path("./checkpoints"),
    help="Output directory",
)
parser.add_argument(
    "-l",
    "--logging_dir",
    type=Path,
    help="Logging directory",
)

def init() -> tuple:
    main_args, cli_args = parser.parse_known_args()

    if main_args.version:
        print("eniat", version("eniat"))
        return

    default_conf = load_conf()
    user_conf = None
    if main_args.config:
        user_conf = recursive_merge_by_path(*main_args.config)
    cli_conf = OmegaConf.from_dotlist(cli_args)

    conf = recursive_merge(default_conf, user_conf, cli_conf or OmegaConf.create({}))

    (
        conf.package,
        conf.task,
        conf.output_dir,
        conf.logger.silent,
        conf.logger.logging_dir,
    ) = (
        main_args.package or conf.package,
        main_args.task or conf.task,
        main_args.output_dir,
        main_args.silent,
        main_args.logging_dir or main_args.output_dir,
    )

    OmegaConf.resolve(conf)

    return conf
