from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')
yaml = "data/Annotations/data.yaml"

results = model.train(
    data = yaml,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    plots=True,
    project="models",
    name="plate_detector"
)

# 
# 


