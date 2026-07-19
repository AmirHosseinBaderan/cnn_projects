from pathlib import Path
import random


def create_train_valid_split(
        labels_file,
        output_dir,
        train_rate=0.9,
        seed=42
):
    random.seed(seed)
    labels_file = Path(labels_file)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(labels_file, "r", encoding="utf-8") as f:
        samples = [
            line.strip()
            for line in f
            if line.strip()
        ]

    random.shuffle(samples)

    split_index = int(len(samples) * train_rate)
    train_samples = samples[:split_index]
    valid_samples = samples[split_index:]

    with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_samples))

    with open(output_dir / "valid.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(valid_samples))

    print(f"Train : {len(train_samples)}")
    print(f"Valid : {len(valid_samples)}")
