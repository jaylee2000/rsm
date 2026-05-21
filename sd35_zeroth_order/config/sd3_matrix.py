import importlib.util
import os


def _load_base_module():
    base_path = os.path.join(os.path.dirname(__file__), "base.py")
    spec = importlib.util.spec_from_file_location("base_config", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base config module from {base_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


def _load_reward_presets_module():
    preset_path = os.path.join(os.path.dirname(__file__), "sd3_reward_presets.py")
    spec = importlib.util.spec_from_file_location("sd3_reward_presets", preset_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reward preset module from {preset_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward_presets = _load_reward_presets_module()


SUPPORTED_PROFILES = ("base", "highsnr", "highsnr2", "base2", "lowsnr", "lowsnr2")
SUPPORTED_SAMPLERS = ("default", "branch")
SUPPORTED_REWARDS = (
    "geneval",
    "ocr",
    "pickscore",
    "deqa",
    "imagereward",
    "qwenvl",
    "aesthetic",
    "jpeg_compressibility",
    "unifiedreward",
)
SUPPORTED_LOSSES = ("ppo", "matching")
SUPPORTED_REWEIGHTS = ("base", "tempflow", "pcpo", "guard", "fairclip", "fairclip2")

_REWEIGHT_TO_CONFIG_VALUE = {
    "base": "base",
    "tempflow": "tempflow_reweight",
    "pcpo": "pcpo_reweight",
    "guard": "grpo_guard",
    "fairclip": "fair_clip",
    "fairclip2": "fair_clip2"
}

_PROFILE_TO_BRANCH_EXPLORATION_K = {
    "base": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "highsnr": [3, 3, 4, 5, 6, 7, 7, 8, 11, 6],
    "highsnr2": [4, 5, 5, 6, 7, 8, 9, 10, 0, 6],
    "base2": [7, 7, 7, 7, 7, 7, 6, 6, 0, 6],
    "lowsnr": [6, 6, 8, 10, 11, 13, 0, 0, 0, 6],
    "lowsnr2": [6, 6, 8, 10, 0, 0, 0, 0, 0, 6],
}


def _parse_config_name(name):
    if not isinstance(name, str):
        raise ValueError(f"Config name must be a string, got {type(name).__name__}.")

    parts = name.strip().split(".")
    if len(parts) != 6 or parts[0] != "sd3":
        raise ValueError(
            "Config name must follow `sd3.<profile>.<sampler>.<reward>.<loss>.<reweight>`, "
            f"got: `{name}`."
        )

    _, profile_raw, sampler_raw, reward_raw, loss_raw, reweight_raw = parts

    profile = str(profile_raw)
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported profile: `{profile_raw}`. "
            f"Supported profiles: {', '.join(SUPPORTED_PROFILES)}"
        )

    sampler = str(sampler_raw)
    if sampler not in SUPPORTED_SAMPLERS:
        raise ValueError(
            f"Unsupported sampler: `{sampler_raw}`. "
            f"Supported samplers: {', '.join(SUPPORTED_SAMPLERS)}"
        )

    reward = str(reward_raw)
    if reward not in SUPPORTED_REWARDS:
        raise ValueError(
            f"Unsupported reward: `{reward_raw}`. "
            f"Supported rewards: {', '.join(SUPPORTED_REWARDS)}"
        )

    loss = str(loss_raw)
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported loss: `{loss_raw}`. "
            f"Supported losses: {', '.join(SUPPORTED_LOSSES)}"
        )

    reweight = str(reweight_raw)
    if reweight not in SUPPORTED_REWEIGHTS:
        raise ValueError(
            f"Unsupported reweight: `{reweight_raw}`. "
            f"Supported reweights: {', '.join(SUPPORTED_REWEIGHTS)}"
        )

    return profile, sampler, reward, loss, reweight


def _build_sd3_base_config():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.use_lora = True

    # Base defaults; concrete sampler builders overwrite where needed.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # TempFlow grouping controls for branching mode.
    config.sample.group_strategy = "prompt"
    config.train.heuristic_kldenom_trick = True

    config.activation_checkpointing = True
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.resolution = 512
    config.save_freq = 50
    config.eval_freq = 50
    config.train.ema = True
    config.num_epochs = 301

    config.prompt_fn = "general_ocr"
    config.reward_fn = {"pickscore": 1.0}
    config.per_prompt_stat_tracking = True

    return config


def _apply_sampler_settings(config, profile, sampler):
    if sampler == "default":
        config.sample.sampling_mode = "default"
        config.sample.train_batch_size = 32
        config.sample.test_batch_size = 16
        config.train.batch_size = config.sample.train_batch_size
        config.train.num_inner_epochs = 1
        config.train.timestep_fraction = 0.99
        config.sample.global_std = True
        config.sample.same_latent = False
        config.train.ema = True
        return config

    if sampler == "branch":
        config.sample.sampling_mode = "sde_branching"
        if "train_batch_size" in config.sample:
            del config.sample.train_batch_size

        config.sample.exploration_k = list(_PROFILE_TO_BRANCH_EXPLORATION_K[profile])
        config.sample.collection_batch_size = 4
        config.sample.test_batch_size = 16
        config.latent_chunk_size = 12
        config.sample.branch_batch_size = config.latent_chunk_size
        config.train.batch_size = config.latent_chunk_size
        config.train.updates_per_epoch = 2
        config.train.num_inner_epochs = 1
        config.train.timestep_fraction = 0.99
        config.sample.global_std = False
        return config

    raise ValueError(f"Unsupported sampler `{sampler}`.")


def _apply_reward_settings(config, reward):
    return reward_presets._apply_reward_settings(config, reward)


def _apply_objective_settings(config, sampler, loss, reweight):
    config.train.loss_type = loss
    config.train.reweight_type = _REWEIGHT_TO_CONFIG_VALUE[reweight]
    if sampler == "branch":
        config.sample.group_strategy = "seed"
    return config


def _apply_save_dir(config, profile, sampler, reward, loss, reweight):
    config.save_dir = f"../logs/sd3/{profile}/{sampler}/{reward}/{loss}_{reweight}"
    return config


def build_config(profile, sampler, reward, loss, reweight):
    config = _build_sd3_base_config()
    config = _apply_sampler_settings(config, profile, sampler)
    config = _apply_reward_settings(config, reward)
    config = _apply_objective_settings(config, sampler, loss, reweight)
    config = _apply_save_dir(config, profile, sampler, reward, loss, reweight)
    return config


def get_config(name):
    profile, sampler, reward, loss, reweight = _parse_config_name(name)
    return build_config(profile, sampler, reward, loss, reweight)


def list_config_ids():
    ids = []
    for profile in SUPPORTED_PROFILES:
        for sampler in SUPPORTED_SAMPLERS:
            for reward in SUPPORTED_REWARDS:
                for loss in SUPPORTED_LOSSES:
                    for reweight in SUPPORTED_REWEIGHTS:
                        ids.append(
                            f"sd3.{profile}.{sampler}.{reward}.{loss}.{reweight}"
                        )
    return ids
