# ================================================================
# تست روی دیتاست تصاویر JPG
# ================================================================

import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================================================================
# کلاس‌های اصلی
# ================================================================

class LowPassFilter:
    def __init__(self, alpha=0.12):
        self.alpha = alpha
        self.initialized = False
        self.value = 0.0
    
    def filter(self, x):
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

class EARDetector:
    import mediapipe as mp
    import math
    
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    def __init__(self):
        import mediapipe as mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def distance(self, p1, p2):
        import math
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def ear(self, landmarks, eye):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
        v1 = self.distance(p2, p6)
        v2 = self.distance(p3, p5)
        h = self.distance(p1, p4)
        return 0 if h == 0 else (v1 + v2) / (2 * h)
    
    def get_ear(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        face = results.multi_face_landmarks[0]
        left = self.ear(face.landmark, self.LEFT_EYE)
        right = self.ear(face.landmark, self.RIGHT_EYE)
        return (left + right) / 2

# ================================================================
# تست روی دیتاست
# ================================================================

def test_on_dataset(dataset_path):
    print("="*60)
    print("🧪 تست روی دیتاست تصاویر JPG")
    print("="*60)
    
    detector = EARDetector()
    lowpass = LowPassFilter(alpha=0.12)
    
    results = []
    all_ears = []
    y_true = []
    y_pred = []
    
    # پیدا کردن همه تصاویر JPG
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"🖼️ {len(image_files)} تصویر پیدا شد")
    
    for img_path in image_files:
        ear = detector.get_ear(img_path)
        
        if ear is not None:
            ear_filtered = lowpass.filter(ear)
            all_ears.append(ear_filtered)
            
            # تشخیص وضعیت (آستانه ۰.۲ برای خواب)
            is_drowsy = 1 if ear_filtered < 0.2 else 0
            
            # برچسب واقعی رو از فایل TXT بخون (اگه وجود داشته باشه)
            txt_path = img_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    label = f.read().strip()
                    try:
                        true_label = int(label)
                        y_true.append(true_label)
                        y_pred.append(is_drowsy)
                    except:
                        pass
            
            results.append({
                'image': os.path.basename(img_path),
                'ear': ear_filtered,
                'is_drowsy': is_drowsy
            })
    
    # نمایش نتایج
    print("\n" + "="*60)
    print("📊 نتایج")
    print("="*60)
    
    if results:
        print(f"✅ تعداد تصاویر پردازش شده: {len(results)}")
        print(f"📈 میانگین EAR: {np.mean(all_ears):.3f}")
        print(f"📉 کمترین EAR: {np.min(all_ears):.3f}")
        print(f"📈 بیشترین EAR: {np.max(all_ears):.3f}")
        
        # شمارش وضعیت‌ها
        drowsy_count = sum(1 for r in results if r['is_drowsy'] == 1)
        awake_count = len(results) - drowsy_count
        print(f"\n📋 وضعیت‌ها:")
        print(f"   😊 بیدار: {awake_count} تصویر")
        print(f"   😴 خواب‌آلود: {drowsy_count} تصویر")
        
        # محاسبه معیارها (اگه برچسب داشتیم)
        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print("\n📊 معیارهای ارزیابی (با برچسب‌ها):")
            print(f"   ✅ Accuracy: {accuracy*100:.1f}%")
            print(f"   ✅ Precision: {precision*100:.1f}%")
            print(f"   ✅ Recall: {recall*100:.1f}%")
            print(f"   ✅ F1-Score: {f1*100:.1f}%")
        
        # ذخیره نتایج
        df = pd.DataFrame(results)
        df.to_csv('jpg_dataset_results.csv', index=False)
        print("\n✅ نتایج ذخیره شد: jpg_dataset_results.csv")
    else:
        print("❌ هیچ تصویری پردازش نشد!")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🧪 تست روی دیتاست تصاویر JPG                              ║
║  ═════════════════════════════════════════════════════════   ║
║  پردازش تصاویر و استخراج EAR برای تشخیص خستگی              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("📁 مسیر پوشه دیتاست را وارد کنید: ")
    
    if not os.path.exists(dataset_path):
        print(f"❌ مسیر پیدا نشد: {dataset_path}")
        sys.exit(1)
    
    test_on_dataset(dataset_path)