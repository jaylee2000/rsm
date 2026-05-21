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

    config.training.use_saved_rgrad_x1 = True
    config.model.eta_mode = 'linear'
    config.model.reward_scale = 1.5 * 1e7

    return config
