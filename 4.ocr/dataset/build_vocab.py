import json
from pathlib import Path
from utils.logger import logger


def build_vocabulary(
        labels_file,
        output_file
):
    labels_file = Path(labels_file)
    characters = set()

    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            _, text = line.split(maxsplit=1)
            characters.update(text)

    characters = sorted(characters)

    vocabulary = {
        "blank": 0,
        "characters": characters,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            vocabulary,
            f,
            ensure_ascii=False,
            indent=4,
        )

    logger.info(f"Vocabulary size : {len(characters)}")
