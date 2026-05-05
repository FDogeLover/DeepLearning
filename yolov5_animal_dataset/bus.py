import cv2

# 使用 DirectShow 避免 MSMF 问题
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

resolutions = [
    (3840, 2160),  # 4K
    (2560, 1440),  # 2K
    (1920, 1080),  # FHD
    (1280, 720),   # HD
    (1024, 768),
    (800, 600),
    (640, 480),
    (320, 240),
]

for width, height in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ret, frame = cap.read()

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Requested: {width}x{height}, Got: {int(actual_width)}x{int(actual_height)}")

cap.release()
