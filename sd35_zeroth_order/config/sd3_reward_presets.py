import os


# Fallback defaults used when a reward does not define an override.
DEFAULT_REWARD_PRESET = {
    "gpu_count": 2,
    "save_freq": 20,
    "eval_freq": 20,
    "num_epochs": 301,
    "beta": 0.01,
    "sampler": {
        "default": {
            "num_prompt_per_epoch": 16,
            "num_image_per_prompt": 24,
        },
        "branch": {
            "num_prompt_per_epoch": 16,
            "num_image_per_prompt": 4,
        },
    },
}

# Reward-specific overrides. Keep each reward entry tweakable.
# Top-level keys can override save_freq/eval_freq/num_epochs/beta.
# sampler.{default|branch} can override num_prompt_per_epoch/num_image_per_prompt.
REWARD_PRESETS = {
    "geneval": {
        "gpu_count": 4,
        "save_freq": 60,
        "eval_freq": 60,
        "num_epochs": 1081,
        "beta": 0.004,
        "sampler": {
            "default": {
                "num_prompt_per_epoch": 48,
                "num_image_per_prompt": 24,
            },
            "branch": {
                "num_prompt_per_epoch": 48,
                "num_image_per_prompt": 4,
            },
        },
    },
    "ocr": {},
    "pickscore": {},
    "deqa": {},
    "imagereward": {},
    "qwenvl": {},
    "aesthetic": {
        "save_freq": 30,
        "eval_freq": 30,
        "num_epochs": 151,
    },
    "jpeg_compressibility": {},
    "unifiedreward": {},
}


def _resolve_prompt_dataset_and_fn(reward_name):
    if reward_name == "geneval":
        return "geneval", "geneval"
    if reward_name == "ocr":
        return "ocr", "general_ocr"
    return "pickscore", "general_ocr"


def _apply_preprocessed_prompt_paths(config, prompt_dataset_name):
    dataset_dir = os.path.join(os.getcwd(), f"dataset/{prompt_dataset_name}")
    embedding_dir = os.path.join(os.getcwd(), f"../cache/sd35_embeddings/{prompt_dataset_name}")
    prompt_ext = "jsonl" if prompt_dataset_name == "geneval" else "txt"
    train_prompt_filename = f"train_metadata.{prompt_ext}" if prompt_ext == "jsonl" else f"train.{prompt_ext}"
    test_prompt_filename = f"test_metadata.{prompt_ext}" if prompt_ext == "jsonl" else f"test.{prompt_ext}"

    config.dataset = dataset_dir
    config.train_hdf5_path = os.path.join(embedding_dir, "train_embeddings.h5")
    config.train_prompt_file_path = os.path.join(dataset_dir, train_prompt_filename)
    config.test_hdf5_path = os.path.join(embedding_dir, "test_embeddings.h5")
    config.test_prompt_file_path = os.path.join(dataset_dir, test_prompt_filename)
    return config


def _sampler_preset_key(sampling_mode):
    normalized = str(sampling_mode).lower()
    if normalized == "default":
        return "default"
    if normalized == "sde_branching":
        return "branch"
    raise ValueError(
        "Unsupported sample.sampling_mode for reward preset application: "
        f"{sampling_mode!r}."
    )


def _resolve_reward_preset(reward, sampling_mode):
    sampler_key = _sampler_preset_key(sampling_mode)
    reward_override = REWARD_PRESETS.get(reward, {})

    sampler_defaults = DEFAULT_REWARD_PRESET["sampler"][sampler_key]
    sampler_override = reward_override.get("sampler", {}).get(sampler_key, {})

    return {
        "gpu_count": reward_override.get("gpu_count", DEFAULT_REWARD_PRESET["gpu_count"]),
        "save_freq": reward_override.get("save_freq", DEFAULT_REWARD_PRESET["save_freq"]),
        "eval_freq": reward_override.get("eval_freq", DEFAULT_REWARD_PRESET["eval_freq"]),
        "num_epochs": reward_override.get("num_epochs", DEFAULT_REWARD_PRESET["num_epochs"]),
        "beta": reward_override.get("beta", DEFAULT_REWARD_PRESET["beta"]),
        "num_prompt_per_epoch": sampler_override.get(
            "num_prompt_per_epoch", sampler_defaults["num_prompt_per_epoch"]
        ),
        "num_image_per_prompt": sampler_override.get(
            "num_image_per_prompt", sampler_defaults["num_image_per_prompt"]
        ),
    }


def _derive_num_batches_per_epoch(config, gpu_count):
    sampling_mode = str(config.sample.sampling_mode).lower()
    num_prompt_per_epoch = int(config.sample.num_prompt_per_epoch)
    num_image_per_prompt = int(config.sample.num_image_per_prompt)
    total_samples = num_prompt_per_epoch * num_image_per_prompt

    if sampling_mode == "default":
        denominator = int(config.sample.train_batch_size) * int(gpu_count)
        denominator_desc = "sample.train_batch_size * gpu_count"
    elif sampling_mode == "sde_branching":
        denominator = int(config.sample.collection_batch_size) * int(gpu_count)
        denominator_desc = "sample.collection_batch_size * gpu_count"
    else:
        raise ValueError(
            f"Unsupported sample.sampling_mode for num_batches_per_epoch derivation: {sampling_mode!r}."
        )

    if denominator <= 0:
        raise ValueError(
            "Denominator for num_batches_per_epoch must be positive, got "
            f"{denominator} from {denominator_desc}."
        )

    assert total_samples % denominator == 0, (
        "Reward/sampler settings must satisfy exact batch divisibility: "
        f"(num_prompt_per_epoch * num_image_per_prompt) % ({denominator_desc}) == 0, got "
        f"{total_samples} % {denominator}."
    )

    config.sample.num_batches_per_epoch = total_samples // denominator
    assert config.sample.num_batches_per_epoch % 2 == 0, (
        "sample.num_batches_per_epoch must be even, got "
        f"{config.sample.num_batches_per_epoch} "
        f"(derived from {total_samples} // {denominator})."
    )

    if sampling_mode == "default":
        config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2

    return config


def _apply_reward_settings(config, reward):
    prompt_dataset_name, prompt_fn = _resolve_prompt_dataset_and_fn(reward)
    config = _apply_preprocessed_prompt_paths(config, prompt_dataset_name)
    config.prompt_fn = prompt_fn
    if reward == "geneval":
        # Keep geneval as the optimized objective while logging pickscore as an auxiliary metric.
        config.reward_fn = {"geneval": 1.0, "pickscore": 0.0}
        config.train.heuristic_kldenom_trick = False
        config.train.use_ode_kl_anchor = True
    else:
        config.reward_fn = {reward: 1.0}
        config.train.use_ode_kl_anchor = False

    reward_preset = _resolve_reward_preset(reward, config.sample.sampling_mode)
    config.save_freq = int(reward_preset["save_freq"])
    config.eval_freq = int(reward_preset["eval_freq"])
    config.num_epochs = int(reward_preset["num_epochs"])
    config.train.beta = float(reward_preset["beta"])
    config.sample.num_prompt_per_epoch = int(reward_preset["num_prompt_per_epoch"])
    config.sample.num_image_per_prompt = int(reward_preset["num_image_per_prompt"])
    config = _derive_num_batches_per_epoch(config, gpu_count=int(reward_preset["gpu_count"]))
    return config
