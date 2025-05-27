import cv2
import numpy as np
from ultralytics import YOLO


# Відкриваємо файли
cap = cv2.VideoCapture("data/Match.mp4")
template = cv2.imread("data/template.png", 0)
h, w = template.shape[:2]


# Output data
output_size = (1080, 1920)
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter("data/output_yolo.mp4", fourcc, 30.0, output_size)


# Source points from video frame
pts_src = np.array([[210, 60], [945, 60], [1113, 1240], [114, 1240]], dtype=np.float32)

# Corresponding points on the "straightened" image
pts_dst = np.array([[0, 0], [1080, 0], [1080, 1920], [0, 1920]], dtype=np.float32)

# Homography matrix for perspective correction
H, _ = cv2.findHomography(pts_src, pts_dst)


last_valid_positions = []  # Store last few valid positions to detect jumps
max_positions = 15  # Number of positions to keep in history


model = YOLO("weights/best.pt")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply perspective transformation
    warped = cv2.warpPerspective(frame, H, (1080, 1920))

    # Отримуємо результати трекінгу з YOLO
    results = model.track(warped, persist=True, conf=0.5)[0]

    # Перевірка, чи є знайдені об'єкти
    if results and results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Малюємо прямокутник
            cv2.rectangle(warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                warped,
                f"Ball {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # Центр об'єкта
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            last_valid_positions.append(center)
            if len(last_valid_positions) > max_positions:
                last_valid_positions.pop(0)

            # Малюємо траєкторію
            for i in range(1, len(last_valid_positions)):
                cv2.line(
                    warped,
                    last_valid_positions[i - 1],
                    last_valid_positions[i],
                    (0, 0, 255),
                    2,
                )

            break  # Тільки перший знайдений об'єкт

    # Запис результату
    out.write(warped)

    # # Показ
    # scaled_warped = cv2.resize(warped, (0, 0), fx=0.4, fy=0.4)
    # cv2.imshow("Warped View", scaled_warped)

    # if cv2.waitKey(30) & 0xFF == 27:
    #     break
