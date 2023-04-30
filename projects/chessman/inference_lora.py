import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipes = {}


def get_pipe(base_model_name, model_path):
    if model_path not in pipes:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet.load_attn_procs(model_path)
        pipe.enable_xformers_memory_efficient_attention()

        def dummy_checker(images, **kwargs): return images, False
        pipe.safety_checker = dummy_checker

        pipe.to("cuda")
        pipes[model_path] = pipe
    return pipes[model_path]


def inference_sd_finetune():
    results_dir = Path("results/chessman-sd2.1-lora-02")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    with open("../../data/chessmen-finetune-all-v2/metadata.jsonl") as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in lines]

    prompts = set([sample["text"] for sample in samples])

    pipe = get_pipe(
        "stabilityai/stable-diffusion-2-1",
        "./weights/chessman-sd2.1-lora-02",
    )

    num_iterations = 8
    batch_size = 16
    for i, prompt in enumerate(sorted(prompts)):
        for j in range(num_iterations):
            images = pipe(
                [prompt] * batch_size,
                negative_prompt=[
                    "a photo of many chess pieces, lowres, bad anatomy, bad hands, "
                    "text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, "
                    "low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                ] * batch_size,
                height=512,
                width=512,
                num_inference_steps=50,
                generator=[
                    torch.Generator(device="cpu").manual_seed(j * batch_size + k + 233)
                    for k in range(batch_size)
                ],
            ).images

            for k, image in enumerate(images):
                image.save(results_dir / f"{i:04d}_{j * batch_size + k:05d}.jpg")


if __name__ == "__main__":
    inference_sd_finetune()
