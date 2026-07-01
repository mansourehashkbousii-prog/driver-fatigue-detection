import cv2
import numpy as np
import os
import random

def draw_face(img, eye_open=True):
    """یه چهره ساده با چشم باز یا بسته رسم کن"""
    # صورت (دایره)
    cv2.circle(img, (100, 120), 60, (200, 200, 200), -1)
    
    # چشم‌ها
    if eye_open:
        cv2.circle(img, (75, 110), 15, (255, 255, 255), -1)
        cv2.circle(img, (75, 110), 5, (0, 0, 0), -1)
        cv2.circle(img, (125, 110), 15, (255, 255, 255), -1)
        cv2.circle(img, (125, 110), 5, (0, 0, 0), -1)
    else:
        cv2.line(img, (60, 110), (90, 110), (0, 0, 0), 3)
        cv2.line(img, (110, 110), (140, 110), (0, 0, 0), 3)
    
    # دهان
    cv2.ellipse(img, (100, 160), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    return img

# پوشه‌ها رو بساز
os.makedirs("full_dataset/open", exist_ok=True)
os.makedirs("full_dataset/closed", exist_ok=True)

# ۵۰ تا تصویر با چشم باز
for i in range(50):
    img = np.zeros((240, 200, 3), dtype=np.uint8)
    img = draw_face(img, eye_open=True)
    cv2.imwrite(f"full_dataset/open/open_{i}.png", img)

# ۵۰ تا تصویر با چشم بسته
for i in range(50):
    img = np.zeros((240, 200, 3), dtype=np.uint8)
    img = draw_face(img, eye_open=False)
    cv2.imwrite(f"full_dataset/closed/closed_{i}.png", img)

print("✅ دیتاست کامل با ۱۰۰ تصویر ساخته شد!")
print(f"📁 پوشه: full_dataset/")
print(f"🖼️ ۵۰ تصویر چشم باز + ۵۰ تصویر چشم بسته")