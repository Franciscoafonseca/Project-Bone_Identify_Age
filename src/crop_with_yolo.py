from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def clip(value, min_value, max_value):
    return max(min_value, min(int(value), max_value))


def crop_image_with_yolo(
    model,
    image_path: Path,
    output_dir: Path,
    preview_dir: Path | None = None,
    conf: float = 0.30,
    padding_x: float = 0.04,
    padding_y: float = 0.04,
):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERRO] Não consegui abrir: {image_path}")
        return False

    img_h, img_w = image.shape[:2]

    results = model.predict(
        source=str(image_path), conf=conf, iou=0.45, max_det=1, imgsz=640, verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print(f"[AVISO] Nenhuma mão detetada em: {image_path.name}")
        return False

    # Como usamos max_det=1, só há uma caixa
    box = result.boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    # Pequena margem extra para não cortar dedos/punho
    x1 -= box_w * padding_x
    x2 += box_w * padding_x
    y1 -= box_h * padding_y
    y2 += box_h * padding_y

    x1 = clip(x1, 0, img_w - 1)
    y1 = clip(y1, 0, img_h - 1)
    x2 = clip(x2, 1, img_w)
    y2 = clip(y2, 1, img_h)

    crop = image[y1:y2, x1:x2]

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), crop)

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 3)

        conf_value = float(result.boxes.conf[0].cpu().numpy())
        text = f"hand {conf_value:.2f}"

        cv2.putText(
            preview,
            text,
            (max(5, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        preview_path = preview_dir / image_path.name
        cv2.imwrite(str(preview_path), preview)

    print(f"[OK] {image_path.name} -> {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/yolo_hand/best.pt",
        help="Caminho para o modelo YOLO treinado",
    )

    parser.add_argument("--input", required=True, help="Imagem ou pasta com imagens RX")

    parser.add_argument(
        "--output", default="data/cropped_yolo", help="Pasta onde guardar os recortes"
    )

    parser.add_argument(
        "--preview",
        default="data/yolo_crop_preview",
        help="Pasta onde guardar previews com a caixa",
    )

    parser.add_argument(
        "--conf", type=float, default=0.30, help="Confiança mínima da deteção"
    )

    parser.add_argument(
        "--padding-x",
        type=float,
        default=0.04,
        help="Margem lateral extra aplicada ao recorte",
    )

    parser.add_argument(
        "--padding-y",
        type=float,
        default=0.04,
        help="Margem vertical extra aplicada ao recorte",
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Não guardar imagem preview com a caixa desenhada",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    preview_dir = None if args.no_preview else Path(args.preview)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input não encontrado: {input_path}")

    model = YOLO(str(model_path))

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = [
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    image_paths = sorted(
        image_paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    print(f"Imagens encontradas: {len(image_paths)}")

    ok_count = 0

    for image_path in image_paths:
        success = crop_image_with_yolo(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            preview_dir=preview_dir,
            conf=args.conf,
            padding_x=args.padding_x,
            padding_y=args.padding_y,
        )

        if success:
            ok_count += 1

    print("\nProcesso concluído.")
    print(f"Recortes criados: {ok_count}/{len(image_paths)}")
    print(f"Recortes guardados em: {output_dir}")

    if preview_dir is not None:
        print(f"Previews guardados em: {preview_dir}")


if __name__ == "__main__":
    main()
