# YOLOv5 Animal Dataset — YOLOv5 动物检测

基于 YOLOv5 的动物检测项目，包含训练、推理、树莓派 GPIO 控制。

- `final_test.py` — 最终测试
- `camera_obtain.py` — 摄像头采集
- `Gpio_control.py` — 树莓派 GPIO 控制
- `animal_map.py` — mAP 计算
- `image_augmented.py` — 数据增强
- `train.py` / `detect.py` — 使用上级 `yolov5/` 框架
- `animal.yaml` — 数据集配置
- `best.onnx` / `best.tflite` — 导出模型
