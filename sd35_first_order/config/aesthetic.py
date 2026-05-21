from config.default_config import get_default_configs

def get_config():
    config = get_default_configs()
    config.experiment.prompt_fn = "simple_animals"
    config.experiment.reward_fn = "aesthetic_score"
    config.experiment.prompt_fn_kwargs = {}

    return config