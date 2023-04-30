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
        "./weights/chessman-sd-dreambooth-3",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    for i in range(20):
        images = pipe(
            ["a photo of one sks chess piece, not many chess pieces, just one chess piece. a big white chess pawn piece, chess pawn, chess pawn, chess pawn, chess pawn, chess pawn, battlefield background, bright art masterpiece artstation. 8 k, sharp high quality artwork in style of jose daniel cabrera pena and greg rutkowski, concept art by tooth wu, blizzard warcraft artwork, hearthstone card game artwork, chess pawn"] * 8,
            negative_prompt=["a photo of many sks chess pieces"] * 8,
        ).images

        for j, image in enumerate(images):
            image.save(f"results/chessman-{i * 8 + j}.png")
        # image_grid(
        #     images, rows=2, cols=4,
        # ).save("results/chessman.png")


if __name__ == "__main__":
    main()
