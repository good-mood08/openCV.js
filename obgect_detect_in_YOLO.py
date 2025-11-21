import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

# ================= Настройки =================
YOLO_MODEL = "yolov8n-seg.pt"
ARUCO_DICT_TYPE = cv2.aruco.DICT_APRILTAG_16h5
YOLO_RUN_INTERVAL = 0.1      # запуск YOLO каждые X секунд
MIN_OBJECT_AREA = 500
SMOOTHING_HISTORY = 5        # количество последних положений маркеров для сглаживания

# ================= Инициализация =================
model = YOLO(YOLO_MODEL)

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_polygon = None
last_yolo_time = 0

# Для сглаживания зоны
marker_history = {0: deque(maxlen=SMOOTHING_HISTORY),
                  1: deque(maxlen=SMOOTHING_HISTORY),
                  2: deque(maxlen=SMOOTHING_HISTORY),
                  3: deque(maxlen=SMOOTHING_HISTORY)}

# ================= Функции =================
def draw_zone(frame, marker_dict):
    pts = np.array([marker_dict[i] for i in sorted(marker_dict)], dtype=np.int32)
    cv2.polylines(frame, [pts], True, (255, 255, 0), 3)
    return pts

def point_inside_polygon(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def smooth_contour(poly, epsilon_ratio=0.01):
    poly = poly.reshape(-1, 1, 2)
    per = cv2.arcLength(poly, True)
    eps = max(2.0, per * epsilon_ratio)
    approx = cv2.approxPolyDP(poly, eps, True)
    return approx.reshape(-1, 2)

def smooth_marker_position(marker_id, new_pos):
    marker_history[marker_id].append(new_pos)
    pts = np.array(marker_history[marker_id], dtype=np.float32)
    return tuple(np.mean(pts, axis=0).astype(int))

# ================= Главный цикл =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ============ Обработка маркеров ============
    corners, ids, _ = detector.detectMarkers(frame)
    zone_pts = None
    detected = {}

    if ids is not None:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, m_id in enumerate(ids):
            cx, cy = corners[i][0].mean(axis=0).astype(int)
            cv2.putText(frame, f"{m_id}", (cx-10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if m_id in (0, 1, 2, 3):
                smoothed_pos = smooth_marker_position(m_id, (cx, cy))
                detected[m_id] = smoothed_pos

        if len(detected) == 4:
            zone_pts = draw_zone(frame, detected)

    # Если зона потеряна, сбросить latest_polygon
    if zone_pts is None:
        latest_polygon = None

    # ============ YOLO по таймеру ============
    now = time.time()
    if zone_pts is not None and (now - last_yolo_time) >= YOLO_RUN_INTERVAL:
        last_yolo_time = now

        xs, ys = zone_pts[:, 0], zone_pts[:, 1]
        x_min, x_max = max(xs.min(), 0), min(xs.max(), frame.shape[1])
        y_min, y_max = max(ys.min(), 0), min(ys.max(), frame.shape[0])

        if x_max - x_min > 20 and y_max - y_min > 20:
            crop = frame[y_min:y_max, x_min:x_max]

            results = model(crop, verbose=False)
            polygons = []

            for r in results:
                if r.masks is None:
                    continue

                for poly in r.masks.xy:
                    poly = np.array(poly, dtype=np.int32)

                    # возврат в координаты полного кадра
                    poly[:, 0] += x_min
                    poly[:, 1] += y_min

                    # фильтр "объект внутри зоны"
                    inside = sum(
                        point_inside_polygon((int(px), int(py)), zone_pts)
                        for px, py in poly
                    )
                    if inside < len(poly) * 0.6:
                        continue

                    poly = smooth_contour(poly)

                    if cv2.contourArea(poly) >= MIN_OBJECT_AREA:
                        polygons.append(poly)

            if polygons:
                latest_polygon = max(polygons, key=cv2.contourArea)

    # ============ Отрисовка найденного объекта ============
    if latest_polygon is not None:
        cv2.polylines(frame, [latest_polygon], True, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Segmentation in Zone", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
