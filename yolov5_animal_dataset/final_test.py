import cv2
import numpy as np
import tensorflow as tf
import time
from collections import defaultdict
import torch
from torchvision.ops import nms

# 配置参数
MODEL_PATH = "best1.tflite"
IMG_SIZE = 320
CONF_THRESH = 0.3
IOU_THRESH = 0.45

class_names = ['monkey', 'peacock', 'elephant', 'tiger', 'wolf']

def letterbox(img, new_shape=IMG_SIZE, color=(114,114,114)):
    shape = img.shape[:2]  # h,w
    r = min(new_shape/shape[0], new_shape/shape[1])
    new_unpad = (int(shape[1]*r), int(shape[0]*r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw //= 2
    dh //= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, dw, dh

def preprocess(image):
    img, r, dw, dh = letterbox(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # 这里不转置，保持 HWC 格式
    img = np.expand_dims(img, axis=0)  # 变成 [1, H, W, C]
    return img, r, dw, dh


def scale_coords(boxes, r, dw, dh, original_shape):
    # boxes 是Tensor格式 (x1,y1,x2,y2)
    boxes[:, [0,2]] -= dw
    boxes[:, [1,3]] -= dh
    boxes /= r
    boxes[:, 0].clamp_(0, original_shape[1])
    boxes[:, 1].clamp_(0, original_shape[0])
    boxes[:, 2].clamp_(0, original_shape[1])
    boxes[:, 3].clamp_(0, original_shape[0])
    return boxes

def postprocess(pred, img, r, dw, dh):
    """
    pred: numpy.ndarray, shape=(N, 5+num_classes)
    """
    boxes = []
    scores = []
    class_ids = []

    for det in pred:
        if len(det) < 6:
            continue
        x_c, y_c, w, h = det[:4]
        obj_conf = det[4]
        class_probs = det[5:]
        cls_id = np.argmax(class_probs)
        cls_conf = class_probs[cls_id]
        conf = obj_conf * cls_conf
        if conf > CONF_THRESH:
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(cls_id)

    if len(boxes) == 0:
        return {}, img

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    class_ids = torch.tensor(class_ids)

    keep = nms(boxes, scores, IOU_THRESH)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes = scale_coords(boxes, r, dw, dh, img.shape).int()

    counts = defaultdict(int)

    for i in range(len(boxes)):
        cls_id = class_ids[i].item()
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'id:{cls_id}'
        counts[cls_name] += 1

        x1, y1, x2, y2 = boxes[i].tolist()
        label = f"{cls_name} {scores[i]:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return counts, img


def main():
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(1)  # 修改为你的摄像头索引

    print("开始检测...")

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        t1 = time.time()

        if not ret:
            print("读取帧失败")
            break

        img_input, r, dw, dh = preprocess(frame)
        t2 = time.time()

        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        pred = np.squeeze(pred, axis=0)
        t3 = time.time()

        counts, output_img = postprocess(pred, frame, r, dw, dh)
        t4 = time.time()

        # 每帧耗时分析
        print(f"[帧耗时统计]")
        print(f" 读取帧       : {(t1 - t0) * 1000:.1f} ms")
        print(f" 预处理       : {(t2 - t1) * 1000:.1f} ms")
        print(f" 模型推理     : {(t3 - t2) * 1000:.1f} ms")
        print(f" 后处理+画图  : {(t4 - t3) * 1000:.1f} ms")
        print(f" 总耗时       : {(t4 - t0) * 1000:.1f} ms")
        print(f" FPS          : {1 / (t4 - t0):.2f}")
        print("-" * 40)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
