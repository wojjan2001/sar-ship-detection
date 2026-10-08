"""Train a YOLO ship detector on the SSDD dataset."""

from ultralytics import YOLO


if __name__ == "__main__":
    # Start from COCO-pretrained weights (transfer learning).
    # The file is downloaded automatically on the first run.
    model = YOLO("yolov8n.pt")

    # Smoke test: 1 epoch on 10% of the training data, just to verify
    # that the dataset config and the whole pipeline work end to end.
    model.train(
        data="configs/ssdd.yaml",
        epochs=1,
        batch=8,
        imgsz=640,
        fraction=0.1,
        device="cpu",
        workers=0,
        # project="runs",
        name="smoke_test",
        exist_ok=True,
    )