from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "yolo_dataset"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
SPLITS = ["train", "val", "test"]


def check_label_file(label_path: Path):
    errors = []

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    if not lines:
        errors.append("label vazia")
        return errors

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            errors.append(f"linha {line_number}: deve ter 5 valores")
            continue

        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"linha {line_number}: valores inválidos")
            continue

        if class_id != 0:
            errors.append(f"linha {line_number}: classe diferente de 0")

        for v in values:
            if v < 0 or v > 1:
                errors.append(f"linha {line_number}: coordenada fora de [0, 1]")

    return errors


def main():
    total_images = 0
    total_labels = 0
    total_errors = 0

    for split in SPLITS:
        images_dir = DATASET / "images" / split
        labels_dir = DATASET / "labels" / split

        images = [
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        labels = list(labels_dir.glob("*.txt"))

        print(f"\n[{split}]")
        print(f"Imagens: {len(images)}")
        print(f"Labels: {len(labels)}")

        total_images += len(images)
        total_labels += len(labels)

        for img in images:
            label = labels_dir / f"{img.stem}.txt"

            if not label.exists():
                print(f"[ERRO] Falta label para: {img.name}")
                total_errors += 1
                continue

            label_errors = check_label_file(label)

            if label_errors:
                print(f"[ERRO] {label.name}:")
                for err in label_errors:
                    print("  -", err)
                total_errors += len(label_errors)

    print("\nResumo:")
    print(f"Total imagens: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Total erros: {total_errors}")

    if total_errors == 0:
        print("\nDataset parece estar correto.")


if __name__ == "__main__":
    main()
