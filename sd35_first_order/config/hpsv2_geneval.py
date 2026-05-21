from config.default_config import get_default_configs


def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "from_jsonl"
    config.experiment.reward_fn = "hpscore"
    config.experiment.prompt_fn_kwargs = {
        "path": "geneval/train_metadata.jsonl",
        "prompt_key": "prompt",
    }
    config.logging.save_freq = 50

    return config
