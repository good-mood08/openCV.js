import cv2
import numpy as np

print("🔍 Проверка установки...")
print("✅ OpenCV:", cv2.__version__)
print("✅ NumPy:", np.__version__)

# Проверка ArUco
try:
    _ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    print("✅ ArUco модуль доступен")
except Exception as e:
    print("❌ ArUco ошибка:", e)

# Генерация простого маркера (тест ArUco)
print("\n🖨️ Пробуем сгенерировать маркер ID=0...")
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
for i in range(4):
    marker = cv2.aruco.generateImageMarker(dictionary, i, 200)
    cv2.imwrite(f'test_marker{i}.png', marker)
    
print("✅ Маркер сохранён как 'test_marker.png'")