import os
import shutil
import random

random.seed(42)

for split in ["train", "val", "test"]:
    for label in ["genuine", "tampered"]:
        os.makedirs(f"data/{split}/{label}", exist_ok=True)

files = [
    f for f in os.listdir("data/genuine")
    if f.endswith(".png")
]

random.shuffle(files)

n = len(files)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

splits = {
    "train": files[:train_end],
    "val": files[train_end:val_end],
    "test": files[val_end:]
}

for split, selected in splits.items():
    for file in selected:
        shutil.copy2(
            f"data/genuine/{file}",
            f"data/{split}/genuine/{file}"
        )

        tampered_file = file.replace("genuine", "tampered")

        shutil.copy2(
            f"data/tampered/{tampered_file}",
            f"data/{split}/tampered/{tampered_file}"
        )

print("Paired dataset split completed")

for split in splits:
    print(
        split,
        len(os.listdir(f"data/{split}/genuine")),
        len(os.listdir(f"data/{split}/tampered"))
    )
