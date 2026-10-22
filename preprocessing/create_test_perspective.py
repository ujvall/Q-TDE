import cv2
import numpy as np

image = cv2.imread("data/raw/test_qr.png")

h, w = image.shape[:2]

src = np.float32([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
])

dst = np.float32([
    [40, 20],
    [w - 30, 0],
    [w - 1, h - 30],
    [0, h - 1]
])

matrix = cv2.getPerspectiveTransform(src, dst)

warped = cv2.warpPerspective(image, matrix, (w, h))

cv2.imwrite("data/raw/tilted_qr.png", warped)

print("Tilted QR created successfully")
