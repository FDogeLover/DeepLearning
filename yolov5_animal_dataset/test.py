import time
import torch
import cv2
import numpy as np
from torchvision.ops import nms
import yaml
from collections import defaultdict

# ---------------------- 参数 ----------------------
weights_path = 'best.torchscript'
img_size = 320
conf_thres = 0.3
iou_thres = 0.45
log_interval = 0
crop_ratio = 0.6  # 中心裁剪比例（例如 0.5 表示仅检测中心50%的区域）

# ---------------------- 模型加载 ----------------------
model = torch.jit.load(weights_path, map_location='cpu').eval()

with open('animal.yaml') as f:
    class_names = yaml.safe_load(f)['names']

# ---------------------- 预处理 ----------------------
def preprocess(image, img_size=640, crop_ratio=0.5):
    h, w = image.shape[:2]
    ch = int(h * crop_ratio)
    cw = int(w * crop_ratio)
    y1 = h // 2 - ch // 2
    x1 = w // 2 - cw // 2
    cropped = image[y1:y1+ch, x1:x1+cw]

    r = img_size / max(ch, cw)
    new_unpad = (int(cw * r), int(ch * r))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(cropped, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img = padded[:, :, ::-1]  # BGR to RGB
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img, r, dw, dh, x1, y1, (x1, y1, cw, ch)

# ---------------------- 后处理 ----------------------
def postprocess(pred, img0, ratio, dw, dh, dx, dy):
    pred = pred.squeeze(0)
    boxes, scores, class_ids = [], [], []

    for det in pred:
        x, y, w, h = det[:4]
        objectness = det[4].item()
        class_scores = det[5:]
        class_id = torch.argmax(class_scores).item()
        class_conf = class_scores[class_id].item()
        conf = objectness * class_conf
        if conf > conf_thres:
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
            scores.append(conf)
            class_ids.append(class_id)

    if not boxes:
        return img0, {}

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    indices = nms(boxes, scores, iou_thres)

    counts = defaultdict(int)
    for i in indices:
        i = i.item()
        x1, y1, x2, y2 = boxes[i]
        x1 = int((x1 - dw) / ratio) + dx
        y1 = int((y1 - dh) / ratio) + dy
        x2 = int((x2 - dw) / ratio) + dx
        y2 = int((y2 - dh) / ratio) + dy
        cls_id = class_ids[i]
        conf_score = scores[i].item()
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        counts[cls_name] += 1
        label = f"{cls_name} {conf_score:.2f}"
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img0, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img0, counts

# ---------------------- 主循环 ----------------------
cap = cv2.VideoCapture(1)
last_log_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, ratio, dw, dh, dx, dy, crop_box = preprocess(frame, crop_ratio=crop_ratio)

    with torch.no_grad():
        pred = model(input_tensor)[0]

    output_frame, counts = postprocess(pred, frame.copy(), ratio, dw, dh, dx, dy)

    # 画出裁剪区域（蓝色）
    x, y, cw, ch = crop_box
    cv2.rectangle(output_frame, (x, y), (x+cw, y+ch), (255, 0, 0), 1)

    # 打印信息（限频）
    now = time.time()
    if now - last_log_time >= log_interval and counts:
        print("检测到：", dict(counts))
        last_log_time = now

    cv2.imshow('YOLOv5n 中心裁剪推理', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
