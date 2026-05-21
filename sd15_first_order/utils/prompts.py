from importlib import resources
import os
import functools
import random
import json
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("utils.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `utils/assets` directory for a file named `path`.
    """
    if os.path.exists(path):
        resolved_path = path
    else:
        resolved_path = ASSETS_PATH.joinpath(path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Could not find {path} or utils.assets/{path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def load_all_prompts_from_file(path: str = "simple_animals.txt") -> list[str]:
    return _load_lines(path)


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def hpd_prompts():
    return from_file("hpd_prompts.txt")


def geneval(split="train"):
    if split not in {"train", "test"}:
        raise ValueError(
            f"Invalid Geneval split '{split}'. Supported values are 'train' and 'test'."
        )
    return from_file(f"geneval_{split}_prompts.txt")


@functools.cache
def read_hpd(style=None):
    if style is None:
        # 800 prompts for each of the 4 styles
        styles = ["anime", "concept-art", "paintings", "photo"]
    else:
        styles = [style]

    prompts_ls = []
    for current_style in styles:
        benchmark_path = ASSETS_PATH.joinpath(f"HPDv2/benchmark_{current_style}.json")
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(
                f"Could not find HPDv2 benchmark file for style '{current_style}': {benchmark_path}"
            )
        with open(benchmark_path, "r", encoding="utf-8") as f:
            # Keep training split behavior aligned with nabla-gfn.
            prompts_ls.extend(json.load(f)[10:])
    return prompts_ls


def hpd_photo_painting():
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("paintings"))  # not "painting"
    return random.choice(prompts_ls), {}


def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}


def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata
