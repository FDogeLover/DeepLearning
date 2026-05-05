import cv2
import numpy as np

# 加载模型和类别标签
net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
classes = ["bird", "mammal", "reptile"]  # 替换为实际类别

# 摄像头初始化
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # 宽度
cap.set(4, 240)  # 高度

while True:
    _, frame = cap.read()
    height, width = frame.shape[:2]

    # 预处理
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)

    # 推理（记录耗时）
    import time

    start = time.time()
    outs = net.forward(net.getUnconnectedOutLayersNames())
    print(f"Inference time: {(time.time() - start) * 1000:.2f}ms")

    # 解析检测结果
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 置信度阈值
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 绘制框和标签
                cv2.rectangle(frame, (center_x - w // 2, center_y - h // 2),
                              (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(frame, f"{classes[class_id]}: {confidence:.2f}",
                            (center_x - w // 2, center_y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示结果（低分辨率节省资源）
    cv2.imshow("Wildlife Detection", cv2.resize(frame, (320, 240)))
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()