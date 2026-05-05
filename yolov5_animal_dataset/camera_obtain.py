import cv2
import os

# 参数配置
camera_index =1
save_dir = "captured_center_integral"
capture_interval = 60     # 每隔多少帧保存一次
resize_scale = 1.0        # 缩放比例（如 0.5 表示缩小后保存）
crop_ratio = 0.5          # 中心区域比例（0.5 表示截取画面中心的 50%）

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("❌ 无法打开摄像头！")
    exit()

frame_count = 0
saved_count = 0

print("📷 按 'q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 无法读取帧")
        break

    # 获取原始尺寸
    h, w, _ = frame.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_y = (h - ch) // 2
    start_x = (w - cw) // 2
    center_crop = frame[start_y:start_y+ch, start_x:start_x+cw]

    # 显示截取画面（调试用）
    cv2.imshow("Center Crop", center_crop)

    # 保存每隔 capture_interval 帧
    if frame_count % capture_interval == 0:
        output = cv2.resize(center_crop, (0, 0), fx=resize_scale, fy=resize_scale)
        filename = os.path.join(save_dir, f"center_{saved_count:03d}.jpg")
        cv2.imwrite(filename, output)
        print(f"✅ 已保存：{filename}")
        saved_count += 1

    frame_count += 1

    # 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 已退出")
