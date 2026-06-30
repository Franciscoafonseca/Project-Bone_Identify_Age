from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT / "data" / "yolo_dataset" / "data.yaml"

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project=str(ROOT / "runs" / "detect"),
    name="yolo_hand",
)
