import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import DiffusionPipeline

pipes = {}


def get_pipe(name):
    if name not in pipes:
        pipes[name] = DiffusionPipeline.from_pretrained(
            name,
            torch_dtype=torch.float16,
        )
        pipes[name].to("cuda")
    return pipes[name]


def inference_sd_finetune():
    results_dir = Path("results/chessmen-finetune-1500")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    with open("../../data/chessmen-finetune/info.json") as f:
        samples = json.load(f)

    prompts = set([sample["prompt"] for sample in samples])

    pipe = get_pipe("./weights/chessman-sd-finetune-1500")

    for i, prompt in enumerate(prompts):
        prompt = prompt.replace("chess piece", "sks chess piece", 1)
        print(prompt)

        for j in range(5):
            images = pipe(
                [prompt] * 8,
                negative_prompt=["a photo of many sks chess pieces"] * 8,
            ).images

            for k, image in enumerate(images):
                image.save(results_dir / f"{i:04d}_{j * 8 + k:05d}.jpg")


def inference_openjourney_finetune():
    results_dir = Path("results/chessmen-openjourney-finetune")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    with open("../../data/chessmen-finetune/info.json") as f:
        samples = json.load(f)

    prompts = set([sample["prompt"] for sample in samples])

    # pipe = get_pipe("prompthero/openjourney-v2")
    pipe = get_pipe("./weights/chessman-openjourney-finetune")

    for i, prompt in enumerate(prompts):
        prompt = prompt.replace("chess piece", "sks chess piece", 1)
        print(prompt)

        for j in range(5):
            images = pipe(
                [prompt] * 8,
                negative_prompt=["a photo of many sks chess pieces"] * 8,
            ).images

            for k, image in enumerate(images):
                image.save(results_dir / f"{i:04d}_{j * 8 + k:05d}.jpg")


def inference_dreambooth():
    results_dir = Path("results/dreambooth-02")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    with open("../../data/chessmen/Chessmen 20/prompt.json") as f:
        meta = json.load(f)

    prompt = "This is a photo of sks chess piece."
    if meta["culture"]:
        prompt += f" {meta['culture']} culture."

    if meta["date"]:
        prompt += f" {meta['date']}."

    if meta["intro"]:
        prompt += f" {meta['intro']}"

    pipe = get_pipe("./weights/chessman-02")

    for i in range(20):
        images = pipe(
            [prompt] * 8,
            negative_prompt=["a photo of many sks chess pieces"] * 8,
        ).images

        for j, image in enumerate(images):
            image.save(results_dir / f"{i * 8 + j:05d}.jpg")


if __name__ == "__main__":
    inference_dreambooth()
    # inference_openjourney_finetune()
