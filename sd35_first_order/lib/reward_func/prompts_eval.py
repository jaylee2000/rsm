"""
Prompt functions for evaluation.

These functions return all prompts from a dataset (not random samples)
for systematic evaluation.
"""

import os
import functools
import inflect
import json

IE = inflect.engine()

import sys
if sys.version_info < (3, 9):
    from importlib_resources import files
else:
    from importlib.resources import files
ASSETS_PATH = files("lib.assets")


def _resolve_asset_path(path):
    if os.path.exists(path):
        return path

    asset_path = ASSETS_PATH.joinpath(path)
    if os.path.exists(asset_path):
        return asset_path

    raise FileNotFoundError(f"Could not find {path} or lib.assets/{path}")


@functools.lru_cache()
def _load_lines(path):
    """
    Load lines from a file with caching for performance.

    First tries to load from `path` directly. If that doesn't exist, searches the
    `lib.assets` directory for a file named `path`.

    Args:
        path: Path to the file (relative or absolute)

    Returns:
        List of stripped lines from the file
    """
    path = _resolve_asset_path(path)
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


@functools.lru_cache()
def _load_jsonl_prompts(path, prompt_key="prompt"):
    path = _resolve_asset_path(path)
    prompts = []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if prompt_key not in row:
                raise KeyError(
                    f"Missing prompt key '{prompt_key}' in {path} at line {line_num}"
                )

            metadata = dict(row)
            prompt = metadata.pop(prompt_key)
            prompts.append((prompt, {}))

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    return prompts


def from_file(path, low=None, high=None):
    """Load prompts from file for evaluation."""
    prompts = _load_lines(path)[low:high]
    return prompts, {}


def from_jsonl(path, prompt_key="prompt"):
    """Load JSONL prompts for evaluation in file order."""
    return _load_jsonl_prompts(path, prompt_key=prompt_key)


short_names = {
    "imagenet_all": "inall",
    "imagenet_animals": "inanm",
    "imagenet_dogs": "indog",
    "simple_animals": "simanm",
    "drawbench": "drawb",
    "hpd": "hpd",
    "hpd_photo": "hppho",
    "hpd_photo_painting": "hpphopa",
    "hpd_photo_anime": "hpphoan",
    "hpd_photo_concept": "hpphoct",
    "nouns_activities": "nounact",
    "counting": "count",
    "pick_a_pic": "pap",
}


def pick_a_pic():
    """Load Pick-a-Pic prompts for evaluation."""
    return from_file("pick_a_pic.txt")


def imagenet_all():
    """Load all ImageNet class names for evaluation."""
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    """Load ImageNet animal class names (first 398 classes)."""
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    """Load ImageNet dog class names (classes 151-269)."""
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    """Load simple animal prompts for evaluation."""
    return from_file("simple_animals.txt")


import csv
import collections


@functools.lru_cache()
def read_csv(path):
    """
    Read DrawBench prompts from CSV file.

    Filters out 'Misspellings' and 'Rare Words' categories.
    Reserves first 2 prompts from each category for training (returns the rest).

    Returns:
        List of prompts (165 prompts total after filtering)
    """
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        reader = [row for row in reader]

    info = collections.defaultdict(list)
    for row in reader:
        info[row["Category"]].append(row["Prompts"])

    # Filter out 'Misspellings' and 'Rare Words' categories
    # Use only prompts [2:] from each category (first 2 reserved for training)
    filtered_info = {}
    for k, v in info.items():
        if k in ["Misspellings", "Rare Words"]:
            continue
        filtered_info[k] = v[2:]
    drawbench_prompt_ls = sum(filtered_info.values(), [])
    return drawbench_prompt_ls


def drawbench():
    """Load DrawBench prompts for evaluation."""
    drawbench_prompt_ls = read_csv(ASSETS_PATH.joinpath("DrawBench Prompts.csv"))
    return drawbench_prompt_ls, {}


import json


@functools.lru_cache()
def read_hpd(style=None):
    """
    Load HPDv2 prompts with caching.

    Args:
        style: Specific style to load ('anime', 'concept-art', 'paintings', 'photo')
               If None, loads all styles (~800 prompts each)

    Returns:
        List of prompt strings (first 10 from each style reserved for training)
    """
    if style is None:
        styles = ["anime", "concept-art", "paintings", "photo"]
    else:
        styles = [style]

    prompts_ls = []
    for style in styles:
        with open(ASSETS_PATH.joinpath(f"HPDv2/benchmark_{style}.json"), "r") as f:
            # Skip first 10 prompts (reserved for training)
            prompts_ls.extend(json.load(f)[10:])

    return prompts_ls


def hpd():
    """Load all HPDv2 prompts (all styles) for evaluation."""
    prompts_ls = read_hpd()
    return prompts_ls, {}


def hpd_photo():
    """Load HPDv2 photo style prompts for evaluation."""
    prompts_ls = read_hpd("photo")
    return prompts_ls, {}


def hpd_photo_painting():
    """Load HPDv2 photo + painting style prompts for evaluation."""
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("paintings"))  # Note: style is "paintings", not "painting"
    return prompts_ls, {}


def hpd_photo_anime():
    """Load HPDv2 photo + anime style prompts for evaluation."""
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("anime"))
    return prompts_ls, {}


def hpd_photo_concept():
    """Load HPDv2 photo + concept-art style prompts for evaluation."""
    prompts_ls = read_hpd("photo")
    prompts_ls.extend(read_hpd("concept-art"))
    return prompts_ls, {}


def nouns_activities(nouns_file, activities_file):
    """
    Generate evaluation prompts by combining nouns and activities.

    Returns all combinations as a list of prompts.
    """
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)

    prompts = []
    for noun in nouns:
        for activity in activities:
            prompt = f"{IE.a(noun)} {activity}"
            prompts.append(prompt)

    return prompts, {}


def counting(nouns_file, low, high):
    """
    Generate counting evaluation prompts.

    Returns prompts for all nouns and numbers in the specified range.
    """
    nouns = _load_lines(nouns_file)

    prompts_and_metadata = []
    for noun in nouns:
        for count in range(low, high + 1):
            number_word = IE.number_to_words(count)
            plural_noun = IE.plural(noun)
            prompt = f"{number_word} {plural_noun}"
            metadata = {
                "questions": [
                    f"How many {plural_noun} are there in this image?",
                    f"What animal is in this image?",
                ],
                "answers": [
                    number_word,
                    noun,
                ],
            }
            prompts_and_metadata.append((prompt, metadata))

    return prompts_and_metadata, {}
