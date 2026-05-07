#!/bin/bash
# Train YOLOv8n on animal dataset
cd /mnt/d/PycharmProject/picturetest/new_animal
source .venv/bin/activate

# Train YOLOv8n from scratch
# --data: dataset config
# --epochs: training epochs
# --batch: batch size (auto-adjusts based on GPU memory)
# --imgsz: input size (224 for the small kmodel, but 640 is better for training)
# --device: GPU
yolo detect train \
  data=datasets/animal/animal.yaml \
  model=yolov8n.yaml \
  epochs=100 \
  batch=-1 \
  imgsz=640 \
  device=0 \
  project=runs/train \
  name=animal_yolov8n \
  exist_ok=True

echo "Training complete!"
