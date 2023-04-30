import torch
from PIL import Image
from diffusers import DiffusionPipeline


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def main():
    pipe = DiffusionPipeline.from_pretrained(
        # "runwayml/stable-diffusion-v1-5",
        "./weights/chessman-sd-dreambooth-2",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    images = pipe(
        ["a photo of sks bishop chess piece. the bishop is lying on the chess piece"] * 8,
        negative_prompt=["a photo of sks bishop chess piece. the bishop is standing on the chess piece"] * 8,
    ).images

    image_grid(
        images, rows=2, cols=4,
    ).save("results/chessman.png")


if __name__ == "__main__":
    main()
