from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def crop_image_with_yolo(model, image_path, output_path, conf=0.25, margin_ratio=0.12):
    image_path = Path(image_path)
    output_path = Path(output_path)

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[ERRO] Não foi possível abrir: {image_path}")
        return False

    h, w = img.shape[:2]

    results = model.predict(source=img, conf=conf, verbose=False)

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print(
            f"[AVISO] Nenhuma mão/punho detetado em {image_path.name}. A guardar imagem original."
        )
        crop = img
    else:
        boxes = results[0].boxes

        # Escolhe a bounding box com maior confiança
        best_idx = boxes.conf.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

        box_w = x2 - x1
        box_h = y2 - y1

        margin_x = int(box_w * margin_ratio)
        margin_y = int(box_h * margin_ratio)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        crop = img[y1:y2, x1:x2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)

    return True


def crop_folder(input_dir, output_dir, weights_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Modelo YOLO não encontrado: {weights_path}")

    model = YOLO(str(weights_path))

    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        image_paths.extend(input_dir.glob(ext))

    print(f"Encontradas {len(image_paths)} imagens em {input_dir}")

    for image_path in image_paths:
        output_path = output_dir / image_path.name
        crop_image_with_yolo(model, image_path, output_path)

    print("Recorte concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pasta com imagens originais")
    parser.add_argument(
        "--output", required=True, help="Pasta onde guardar imagens recortadas"
    )
    parser.add_argument(
        "--weights",
        default="models/yolo_hand/best.pt",
        help="Pesos treinados do YOLO11",
    )

    args = parser.parse_args()

    crop_folder(input_dir=args.input, output_dir=args.output, weights_path=args.weights)
