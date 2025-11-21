import cv2
import numpy as np
from ultralytics import YOLO

# ================= Настройки =================
YOLO_MODEL = "yolov8n-seg.pt"  # YOLOv8 с сегментацией
ARUCO_DICT_TYPE = cv2.aruco.DICT_APRILTAG_16h5
YOLO_RUN_EVERY_N_FRAMES = 3  # запуск YOLO раз в N кадров
MIN_OBJECT_AREA = 500  # минимальная площадь полигона, чтобы не было шума

# ================= Инициализация =================
model = YOLO(YOLO_MODEL)
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
latest_polygon = None

# ================= Функции =================
def draw_zone(frame, marker_positions):
    pts = np.array([
        marker_positions[0],
        marker_positions[1],
        marker_positions[2],
        marker_positions[3]
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], True, (255, 255, 0), 3)
    return pts

def point_inside_polygon(pt, poly):
    x, y = pt
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

def smooth_contour(poly, epsilon_ratio=0.01):
    """Сглаживание контура с динамическим epsilon"""
    perimeter = cv2.arcLength(poly, True)
    epsilon = max(2.0, perimeter * epsilon_ratio)
    approx = cv2.approxPolyDP(poly, epsilon, True)
    if approx.ndim == 3:
        approx = approx.reshape(-1, 2)
    return approx

# ================= Главный цикл =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)
    zone_pts = None
    detected = {}

    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            c = corners[i][0]
            cx, cy = c.mean(axis=0).astype(int)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, f"ID {marker_id}", (cx-10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if marker_id in [0,1,2,3]:
                detected[marker_id] = (cx, cy)

        if len(detected) == 4:
            zone_pts = draw_zone(frame, detected)

    frame_count += 1

    # ===== YOLO раз в N кадров и только если зона есть =====
    if zone_pts is not None and frame_count % YOLO_RUN_EVERY_N_FRAMES == 0:
        xs, ys = zone_pts[:,0], zone_pts[:,1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        crop = frame[y_min:y_max, x_min:x_max].copy()
        results = model(crop, stream=False)

        polygons = []

        for r in results:
            if r.masks is None:
                continue
            for poly in r.masks.xy:
                poly = np.array(poly, dtype=np.int32)

                # переводим координаты из кропа в исходный кадр
                poly[:,0] += x_min
                poly[:,1] += y_min

                # проверка, что большая часть точек внутри зоны
                inside_count = sum(point_inside_polygon((int(px), int(py)), zone_pts) for px, py in poly)
                if inside_count < len(poly) * 0.6:
                    continue

                # Сглаживаем контур
                poly = smooth_contour(poly)

                # фильтруем по площади
                if cv2.contourArea(poly) < MIN_OBJECT_AREA:
                    continue

                polygons.append(poly)

        # берем самый крупный объект
        if polygons:
            largest_poly = max(polygons, key=cv2.contourArea)
            latest_polygon = largest_poly

    # ===== Рисуем контуры =====
    if latest_polygon is not None:
        cv2.polylines(frame, [latest_polygon], True, (0,0,255), 2)
        tx, ty = latest_polygon[0]
        cv2.putText(frame, "Object", (tx, ty-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # ===== Показываем зону =====
    if zone_pts is not None:
        cv2.polylines(frame, [zone_pts], True, (255, 255, 0), 3)

    cv2.imshow("YOLOv8 Segmentation in Zone", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
