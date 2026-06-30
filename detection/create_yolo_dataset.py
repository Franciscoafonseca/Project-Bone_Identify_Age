from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
LABELS_DIR = ROOT / "data" / "auto_labels"
OUT_DIR = ROOT / "data" / "yolo_dataset"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

random.seed(123)


def clear_folder(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {RAW_DIR}")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {LABELS_DIR}")

    images = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]

    pairs = []

    for img in images:
        label = LABELS_DIR / f"{img.stem}.txt"

        if label.exists():
            pairs.append((img, label))
        else:
            print(f"[AVISO] Sem label correspondente: {img.name}")

    if not pairs:
        raise RuntimeError("Não encontrei pares imagem + label.")

    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split in ["train", "val", "test"]:
        clear_folder(OUT_DIR / "images" / split)
        clear_folder(OUT_DIR / "labels" / split)

    for split, split_pairs in splits.items():
        images_out = OUT_DIR / "images" / split
        labels_out = OUT_DIR / "labels" / split

        for img, label in split_pairs:
            shutil.copy2(img, images_out / img.name)
            shutil.copy2(label, labels_out / label.name)

        print(f"{split}: {len(split_pairs)} imagens")

    yaml_text = """path: data/yolo_dataset

train: images/train
val: images/val
test: images/test

names:
  0: hand
"""

    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("\nDataset YOLO criado com sucesso em:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
