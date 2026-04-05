import os
from PIL import Image


def check_dataset(directory):
    print(f"\nChecking dataset: {directory}")

    total = 0
    corrupted = 0

    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            total += 1

            try:
                img = Image.open(path)
                img.verify()
            except:
                corrupted += 1
                print(f"Corrupted: {path}")

    print(f"\nTotal files: {total}")
    print(f"Corrupted files: {corrupted}")


if __name__ == "__main__":
    check_dataset("data/raw/train")
    check_dataset("data/raw/val")