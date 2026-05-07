# New Animal — K230 边缘端动物检测（YOLOv8）

基于 K230 AI 芯片的 YOLOv8n 动物检测项目。

## 项目结构

```
new_animal/
├── README.md              # 本文档
├── deploy_config.json     # 部署配置
├── calib/                 # 校准图片（量化用）
├── data/                  # 数据集
│   ├── Annotations/       #   3,930 个 YOLO 格式标注文件
│   ├── JPEGImages/        #   3,932 张 800×480 图片
│   ├── labels.txt         #   类别映射：tiger/wolf/monkey/peacock/elephant
│   └── data.zip           #   数据集压缩包
├── models/                # 编译好的模型文件
│   ├── best.kmodel        #   主力模型
│   ├── best.onnx          #   ONNX 格式
│   ├── best_nano_320.kmodel   # 轻量版（320 输入）
│   ├── best_nano_320.onnx     # 轻量版 ONNX
│   ├── yolov8n_224.kmodel     # 更小的版本
│   └── best_AnchorBaseDet_... # Anchor-Based 模型
└── scripts/               # 推理 & 转换脚本
    ├── animal_detect_best.py
    ├── animal_detect_anchorbasedet.py
    ├── k230_animal_detect.py
    ├── object_detect_yolov8n.py
    └── onnx_to_kmodel.py
```
