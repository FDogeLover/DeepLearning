import cv2
import numpy as np
import time

# 模型 & 类别配置
MODEL_PATH = "animal_model.onnx"
CLASS_NAMES = ["elephant", "monkey", "peacock", "tiger", "wolf"]
CONFIDENCE_THRESHOLD = 3

# 加载模型
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# 图像预处理
def preprocess_frame(frame):
    image_resized = cv2.resize(frame, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std

    blob = np.transpose(image_normalized, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    return blob

# 分类预测
def predict(frame):
    blob = preprocess_frame(frame)
    net.setInput(blob)
    output = net.forward()

    class_id = np.argmax(output)
    confidence = output[0][class_id]
    if confidence < CONFIDENCE_THRESHOLD:
        return "No animal", confidence
    else:
        return CLASS_NAMES[class_id], confidence

# 视频流处理函数
def classify_video(source=0):  # 0 表示摄像头；也可换成视频路径
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ 无法打开视频或摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict(frame)
        text = f"{label} ({conf:.2f})"
        print(text)

        # 显示结果
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Animal Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 执行
if __name__ == "__main__":
    # classify_video(0)  # 打开摄像头
    classify_video(1)  # 替换为你的视频路径
