import onnxruntime as ort
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from collections import defaultdict
import serial
import time
# from Gpio_control import GpioLaserController
# ==== 配置选项 ====
ENABLE_BT = False  # 是否开启蓝牙串口通讯
SHOW_IMAGE = True  # 是否显示图像窗口
DRAW_BOX = True  # 是否在图像上绘制框和标签
SLEEP_IF_DETECTED = 3.0  # 检测到动物后暂停时间（秒）

BT_PORT = '/dev/ttyUSB0'  # 蓝牙串口（修改为你的）
BT_BAUDRATE = 9600

weights_path = "best.onnx"
img_size = 640
conf_thres = 0.3
iou_thres = 0.45

class_names = ['monkey', 'peacock', 'elephant', 'tiger', 'wolf']
animal_id_map = {'monkey': 6, 'peacock': 8, 'elephant': 4, 'tiger': 0, 'wolf': 2}


# ==== 工具函数 ====
def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32, auto=True):
    shape = img.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad):
    gain = ratio_pad[0][0]
    pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, [0, 2]] /= gain
    coords[:, [1, 3]] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def preprocess(image):
    img0 = image.copy()
    img, ratio, (dw, dh) = letterbox(img0, new_shape=img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, 0)
    return img, img0, ratio, dw, dh


def postprocess(outputs, img0, ratio, dw, dh):
    pred = outputs[0]
    if pred.ndim > 2:
        pred = np.squeeze(pred)
    boxes, scores, class_ids = [], [], []

    for det in pred:
        if len(det) < 6:
            continue
        x, y, w, h = det[:4]
        obj_conf = det[4]
        class_scores = det[5:]
        cls = np.argmax(class_scores)
        cls_conf = class_scores[cls]
        conf = obj_conf * cls_conf
        if conf > conf_thres:
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(int(cls))

    if not boxes:
        return {}, img0

    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    indices = nms(boxes, scores, iou_thres).cpu().numpy().tolist()
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = [class_ids[i] for i in indices]
    boxes = scale_coords((img_size, img_size), boxes, img0.shape[:2], ((ratio[0], ratio[1]), (dw, dh))).round()

    counts = defaultdict(int)
    for i, box in enumerate(boxes):
        cls_id = class_ids[i]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f'id:{cls_id}'
        counts[cls_name] += 1
        if DRAW_BOX:
            # 设定中心区域范围（例如图像中间 50% 区域）
            img_h, img_w = img0.shape[:2]
            center_x1 = int(img_w * 0.25)
            center_y1 = int(img_h * 0.25)
            center_x2 = int(img_w * 0.75)
            center_y2 = int(img_h * 0.75)

            counts = defaultdict(int)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.int().tolist()
                box_cx = (x1 + x2) // 2
                box_cy = (y1 + y2) // 2

                # 判断框中心点是否在中心区域内
                if center_x1 <= box_cx <= center_x2 and center_y1 <= box_cy <= center_y2:
                    cls_id = class_ids[i]
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f'id:{cls_id}'
                    counts[cls_name] += 1

                if DRAW_BOX:
                    label = f"{class_names[class_ids[i]]} {scores[i]:.2f}"
                    color = (0, 255, 0) if center_x1 <= box_cx <= center_x2 and center_y1 <= box_cy <= center_y2 else (
                    128, 128, 128)
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img0, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 可选：可视化中间区域
            if DRAW_BOX:
                cv2.rectangle(img0, (center_x1, center_y1), (center_x2, center_y2), (255, 0, 0), 1)

    return counts, img0


# ==== 主函数 ====
def main():
    # gp = GpioLaserController()
    bt = None
    if ENABLE_BT:
        try:
            bt = serial.Serial(BT_PORT, BT_BAUDRATE, timeout=1)
            print(f"[INFO] 蓝牙串口已打开: {BT_PORT}")
        except Exception as e:
            print(f"[WARN] 蓝牙串口打开失败: {e}")
            bt = None

    session = ort.InferenceSession(weights_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print('开始检测')

    from collections import deque, defaultdict

    detect_history = deque()
    HISTORY_WINDOW = 3.0
    last_detect_name = None
    last_detect_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        input_tensor, img0, ratio, dw, dh = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})
        counts, result_img = postprocess(outputs, img0, ratio, dw, dh)

        now = time.time()

        if counts:
            # 选出现次数最多的物种
            primary_animal = max(counts, key=counts.get)

            # ---------- 记录历史检测 ----------
            detect_history.append((primary_animal, now))

            # 移除窗口外记录
            while detect_history and now - detect_history[0][1] > HISTORY_WINDOW:
                detect_history.popleft()

            # 统计窗口内该物种出现次数
            freq_count = sum(1 for name, t in detect_history if name == primary_animal)

            # ---------- 检测持续性 ----------
            sustained = False
            if primary_animal == last_detect_name:
                duration = now - last_detect_time
                if duration >= HISTORY_WINDOW:
                    sustained = True
            else:
                last_detect_name = primary_animal
                last_detect_time = now
            #last_detect_name = primary_animal
            #last_detect_time = now
            # ---------- 满足任一条件则触发 ----------
            if freq_count >= 5:
                print("[检测结果]", dict(counts))
                print(f"[触发] 物种：{primary_animal}，频次：{freq_count}，持续：{now - last_detect_time:.2f}s")

                send_str = ",".join(f"{animal_id_map[k]}:{v}" for k, v in counts.items() if k in animal_id_map)
                if ENABLE_BT and bt and bt.is_open:
                    try:
                        for _ in range(5):
                            # gp.set_laser(1)
                            bt.write((send_str + "/p").encode('utf-8'))
                        print(f"[蓝牙发送] {send_str}")
                    except Exception as e:
                        print(f"[蓝牙发送失败] {e}")

                # 暂停 5 秒
                time.sleep(5)

                # 重置状态
                detect_history.clear()
                last_detect_name = None
                last_detect_time = 0
                continue

        else:
            # 无检测结果则清空状态
            detect_history.clear()
            last_detect_name = None
            last_detect_time = 0
            if ENABLE_BT and bt and bt.is_open:
                bt.write(b"none\n")

        if SHOW_IMAGE:
            cv2.imshow("Result", result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if SHOW_IMAGE:
        cv2.destroyAllWindows()
    if ENABLE_BT and bt and bt.is_open:
        bt.close()


if __name__ == '__main__':
    main()
