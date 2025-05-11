# Suspicious Object Detection – YOLOv8

This branch contains the trained **YOLOv8m** model for suspicious object detection.

## 📦 Files
- `best.pt` – The best trained model (mAP50-95 ≈ 75.4%, Precision ≈ 92.2%, Recall ≈ 90.3%)
- `data.yaml` – Dataset configuration (7 classes: Grenade, Knife, Pistol, RPG, Masked Face, Machine Guns, Truck)

## 🧠 Training Info
- Model: YOLOv8m
- Dataset: 23,536 images – 171,847 annotations
- Classes: 7 suspicious objects
- Epochs: 100
- Image size: 640x640

## 📤 How to use
```bash
yolo task=detect mode=predict model=best.pt source=yourimage.jpg

to start the entrainement : 
yolo detect train data=data.yaml model=yolov8m.pt epochs=100 imgsz=640 batch=16 device=0 patience=20 save_period=5 optimizer=AdamW   