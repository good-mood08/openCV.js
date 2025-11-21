import cv2
import numpy as np

# === Настройки ===
ARUCO_DICT_TYPE = cv2.aruco.DICT_APRILTAG_16h5

# Камера (пример для 640x480)
camera_matrix = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

dist_coeffs = np.zeros(5, dtype=np.float32)

# === Инициализация детектора ===
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Видео
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("AprilTag 16h5 | q = выход")

def draw_marker_vectors(frame, corners):
    """
    corners — массив 4 углов:
      0—TL, 1—TR, 2—BR, 3—BL
    Строим вектора сторон маркера.
    """
    c = corners.reshape((4, 2)).astype(int)

    # Вектора сторон
    # v_top    = c[1] - c[0]   # верхняя сторона
    # v_right  = c[2] - c[1]   # правая
    # v_bottom = c[3] - c[2]   # нижняя
    # v_left   = c[0] - c[3]   # левая

    # Рисуем стрелки
    # cv2.arrowedLine(frame, tuple(c[0]), tuple(c[1]), (0,0,255), 2) # сверху
    # cv2.arrowedLine(frame, tuple(c[1]), tuple(c[2]), (0,255,0), 2) # справа
    # cv2.arrowedLine(frame, tuple(c[2]), tuple(c[3]), (255,0,0), 2) # снизу
    # cv2.arrowedLine(frame, tuple(c[3]), tuple(c[0]), (255,255,0), 2) # слева

def draw_zone(frame, marker_positions):
    """
    marker_positions = { id: center_xy }
    ids = 0,1,2,3
    Рисуем зону.
    """
    pts = np.array([
        marker_positions[0],
        marker_positions[1],
        marker_positions[2],
        marker_positions[3]
    ], dtype=np.int32)

    cv2.polylines(frame, [pts], True, (0,255,255), 3)
    # cv2.fillPoly(frame, [pts], (0,255,255,50))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Для зоны 0-3
        detected = {}

        for i, marker_id in enumerate(ids):
            c = corners[i][0]

            # Подписываем ID
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(frame, f"ID {marker_id}",
                        (cx-10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

            # Рисуем вектора границ
            draw_marker_vectors(frame, corners[i])

            # Сохраняем центр маркера
            if marker_id in [0,1,2,3]:
                detected[marker_id] = (cx, cy)

        # Если нашли все 4
        if len(detected) == 4:
            draw_zone(frame, detected)

    cv2.imshow("AprilTag 16h5 Zone Builder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
