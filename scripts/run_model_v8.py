from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics import YOLOWorld


# Load a pretrained YOLOv8 model
# model = YOLO('yolov8m.pt')

model = YOLO('yolov8s-world.pt')

model.set_classes(["cup","bottle"])

# Realtime Detection Transformer
# model = RTDETR('rtdetr-l.pt')

while True:
    results = model.predict(
        source=0,
        vid_stride=1,
        stream=True,
        show=True
    )

    for r in results:
        print(f"Box: {r.boxes}")
