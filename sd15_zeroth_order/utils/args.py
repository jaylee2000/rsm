import argparse
import os
import yaml
from types import SimpleNamespace

def parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    tmp_args = parser.parse_args()
    with open(tmp_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    if not config_dict.get("pretrained_model"):
        raise ValueError("pretrained_model is required in config file")

    args = SimpleNamespace(**config_dict)
    # assert args.algorithm in ["ddpo", "logrho", "vw", "uw-sigma", "uw-lambda"], \
    #     f"Algorithm {args.algorithm} not supported."

    return args
