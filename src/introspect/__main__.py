"""Entry point for running the introspect package as a module.

Usage: python -m introspect [experiment] [action] --config path/to/config.yaml
"""
# takes in config file and any parameters that override the config
# config is a yaml file


import yaml
import argparse
from omegaconf import OmegaConf
from pathlib import Path

from .utils import set_seed


def _set_nested(cfg, k, v):
    keys = k.split(".")
    for key in keys[:-1]:
        cfg = cfg[key]
    cfg[keys[-1]] = v


def _load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["feature", "attrib", "activation"])
    parser.add_argument("action", type=str)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", type=str)
    args = parser.parse_args()
    # build yaml config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # override with any arguments
    if args.override is not None:
        for override in args.override:
            key, value = override.split("=")
            _set_nested(config, key, value)
    return args.experiment, args.action, OmegaConf.create(config)


def main():
    experiment, action, cfg = _load_config()
    set_seed(cfg.seed)
    if experiment == "feature":
        from .feature import run
    elif experiment == "attrib":
        from .attrib import run
    elif experiment == "activation":
        from .activation import run
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    run(action, cfg)


if __name__ == "__main__":
    main()
