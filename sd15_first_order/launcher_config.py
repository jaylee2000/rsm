import argparse


def parse_config_path(default_config_path: str | None, description: str) -> str:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        required=default_config_path is None,
        help="Path to config YAML.",
    )
    args = parser.parse_args()
    return args.config
