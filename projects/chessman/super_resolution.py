from pathlib import Path

import torch
from diffusers import LDMSuperResolutionPipeline
from PIL import Image
from tqdm import tqdm


def main():
    src_path = Path("./results/chessman-sd2.1-lora-00")
    tgt_path = Path("./results/chessman-sd2.1-lora-00-x4")

    if not tgt_path.exists():
        tgt_path.mkdir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"

    pipe = LDMSuperResolutionPipeline.from_pretrained(model_id)

    pipe.enable_xformers_memory_efficient_attention()

    def dummy_checker(images, **kwargs): return images, False
    pipe.safety_checker = dummy_checker

    pipe = pipe.to(device)

    for image_path in tqdm(sorted(src_path.glob("*.jpg"))):
        upscaled_image = pipe(
            Image.open(image_path).convert("RGB"), num_inference_steps=100, eta=1
        ).images[0]

        upscaled_image.save(tgt_path / image_path.name)


if __name__ == "__main__":
    main()
