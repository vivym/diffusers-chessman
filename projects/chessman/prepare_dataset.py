import shutil
import json
from pathlib import Path

import pandas as pd


def main():
    data_root = Path("../../data/chessmen_v2")

    finetune_data_root = Path("../../data/chessmen-finetune-all-v2")
    if not finetune_data_root.exists():
        finetune_data_root.mkdir()

    df = pd.read_excel(data_root / "chessmen_v2.xlsx", header=1)

    info = []

    for _, row in df.iterrows():
        title = row["Title"]
        image_dir = data_root / f"{title}"
        if not image_dir.exists():
            print(f"{image_dir} does not exist")
            continue

        culture, date, intro = row["Culture"], row["Date"], row["introduce"]

        prompt = "This is a photo of one chess piece."

        if isinstance(culture, str):
            prompt += f" This item is from {culture} culture."

        if isinstance(date, str):
            prompt += f" This item probably dates black to the {date}."

        if isinstance(intro, str):
            prompt += (" " + intro)

        for image_path in image_dir.glob("*.*"):
            if image_path.suffix not in [".png", ".jpeg", ".jpg"]:
                continue

            file_name = f"{title}_{image_path.stem}{image_path.suffix}"

            shutil.copyfile(
                image_path,
                finetune_data_root / file_name,
            )
            info.append({
                "file_name": file_name,
                "text": prompt,
            })

    for image_path in (data_root / "new chess").glob("*.png"):
        file_name = f"new chess_{image_path.stem}{image_path.suffix}"

        shutil.copyfile(
            image_path,
            finetune_data_root / file_name,
        )
        info.append({
            "file_name": file_name,
            "text": "This is a photo of one chess piece.",
        })

    print("num_images", len(info))

    with open(finetune_data_root / "metadata.jsonl", "w") as f:
        for meta in info:
            f.write(json.dumps(meta) + "\n")


if __name__ == "__main__":
    main()
