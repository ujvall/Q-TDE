import cv2
import numpy as np
import os
import random

input_dir = "data/genuine"
output_dir = "data/tampered"

os.makedirs(output_dir, exist_ok=True)

random.seed(42)
np.random.seed(42)

for file in sorted(os.listdir(input_dir)):
    if not file.endswith(".png"):
        continue

    image = cv2.imread(
        os.path.join(input_dir, file),
        cv2.IMREAD_GRAYSCALE
    )

    h, w = image.shape
    tampered = image.copy()

    mode = random.randint(0, 4)

    x1 = random.randint(int(w * 0.30), int(w * 0.60))
    y1 = random.randint(int(h * 0.30), int(h * 0.60))

    size = random.randint(
        int(w * 0.04),
        int(w * 0.09)
    )

    x2 = min(x1 + size, int(w * 0.72))
    y2 = min(y1 + size, int(h * 0.72))

    if mode == 0:
        cur_x1, cur_y1, cur_x2, cur_y2 = x1, y1, x2, y2
        patch = tampered[cur_y1:cur_y2, cur_x1:cur_x2].copy()
        if len(np.unique(patch)) == 1:
            for dx in range(-15, 16, 5):
                for dy in range(-15, 16, 5):
                    nx1 = max(0, min(w - size, cur_x1 + dx))
                    ny1 = max(0, min(h - size, cur_y1 + dy))
                    nx2 = min(w, nx1 + size)
                    ny2 = min(h, ny1 + size)
                    candidate = tampered[ny1:ny2, nx1:nx2]
                    if len(np.unique(candidate)) > 1:
                        cur_x1, cur_y1, cur_x2, cur_y2 = nx1, ny1, nx2, ny2
                        break
                if len(np.unique(tampered[cur_y1:cur_y2, cur_x1:cur_x2])) > 1:
                    break
        patch = cv2.GaussianBlur(tampered[cur_y1:cur_y2, cur_x1:cur_x2].copy(), (5, 5), 0)
        tampered[cur_y1:cur_y2, cur_x1:cur_x2] = patch

    elif mode == 1:
        factor = random.uniform(0.75, 1.25)
        patch = tampered[y1:y2, x1:x2].astype(np.float32)
        if factor < 1.0:
            patch = patch * factor
        else:
            fade = (factor - 1.0) * 160 + 20
            patch = np.where(patch < 128, fade, 255.0)
        tampered[y1:y2, x1:x2] = np.clip(
            patch, 0, 255
        ).astype(np.uint8)

    elif mode == 2:
        noise = np.random.normal(
            0,
            8,
            (y2 - y1, x2 - x1)
        )
        patch = tampered[y1:y2, x1:x2].astype(np.float32)
        patch += noise
        tampered[y1:y2, x1:x2] = np.clip(
            patch, 0, 255
        ).astype(np.uint8)

    elif mode == 3:
        patch = tampered[y1:y2, x1:x2].copy()

        shadow = np.linspace(
            0.75,
            1.0,
            x2 - x1
        )

        patch = (
            patch.astype(np.float32)
            * shadow[np.newaxis, :]
        )

        tampered[y1:y2, x1:x2] = np.clip(
            patch,
            0,
            255
        ).astype(np.uint8)

    else:
        for _ in range(random.randint(2, 4)):
            mx = random.randint(x1, x2 - 1)
            my = random.randint(y1, y2 - 1)

            cell = random.randint(2, 5)

            yy2 = min(my + cell, h)
            xx2 = min(mx + cell, w)

            tampered[my:yy2, mx:xx2] = (
                255 - tampered[my:yy2, mx:xx2]
            )

    cv2.imwrite(
        os.path.join(
            output_dir,
            file.replace("genuine", "tampered")
        ),
        tampered
    )

print("Subtle realistic tampered dataset generated successfully")
