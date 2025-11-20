# aruco_axes_demo.py
import cv2
import numpy as np

# === Настройки ===
ARUCO_DICT_TYPE = cv2.aruco.DICT_APRILTAG_16h5
MARKER_LENGTH = 0.05  # 5 см — физический размер стороны маркера в метрах

# === Калибровочные параметры камеры (пример для веб-камеры 640x480) ===
# Эти значения приблизительны — для MVP достаточно
camera_matrix = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

dist_coeffs = np.zeros(5, dtype=np.float32)  # Игнорируем дисторсию

# === Инициализация детектора ===
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# === Захват видео ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 Обнаружение ArUco-маркеров с отображением осей (X=красная, Y=зелёная, Z=синяя)")
print("Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение маркеров
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Отображаем маркеры с ID (зелёные рамки)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Для каждого маркера — оценка позы и отрисовка осей
        for i in range(len(ids)):
            # 3D-точки маркера в его системе координат (центр — (0,0,0), Z — вверх)
            obj_points = np.array([
                [-MARKER_LENGTH / 2,  MARKER_LENGTH / 2, 0],
                [ MARKER_LENGTH / 2,  MARKER_LENGTH / 2, 0],
                [ MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
                [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
            ], dtype=np.float32)

            # Решаем PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                corners[i],
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # Рисуем оси координат (длина = 1.5 * размер маркера)
                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    length=MARKER_LENGTH * 1.5,
                    thickness=2
                )

    cv2.imshow("ArUco Markers + Axes (X=Red, Y=Green, Z=Blue)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()