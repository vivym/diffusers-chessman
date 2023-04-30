import shutil
import json
from pathlib import Path

import pandas as pd


def main():
    data_root = Path("../../data/chessmen")

    df = pd.read_excel(data_root / "chessmen.xlsx", header=1)

    for _, row in df.iterrows():
        title = row["Title"]
        image_dir = data_root / f"{title}"
        if not image_dir.exists():
            print(f"{image_dir} does not exist")
            continue

        culture, date, intro = row["Culture"], row["Date"], row["introduce"]

        if not isinstance(culture, str):
            culture = ""

        if not isinstance(date, str):
            date = ""

        if not isinstance(intro, str):
            intro = ""

        with open(image_dir / "prompt.json", "w") as f:
            json.dump({
                "culture": culture,
                "date": date,
                "intro": intro,
            }, f)


if __name__ == "__main__":
    main()
