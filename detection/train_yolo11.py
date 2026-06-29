from ultralytics import YOLO


def main():
    # YOLO11n é a versão mais leve. Boa para começar.
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="detection/hand_xray.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device="cpu",  # usa 0 se tiveres GPU NVIDIA com CUDA
        workers=2,
        project="runs/yolo_hand",
        name="yolo11n_hand_wrist",
    )

    print(results)


if __name__ == "__main__":
    main()
