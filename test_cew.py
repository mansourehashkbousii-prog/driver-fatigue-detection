import cv2
import mediapipe as mp
import math
import numpy as np
import os
import pandas as pd

# ================================================================
# کلاس تشخیص EAR
# ================================================================
class EARDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def distance(self, p1, p2):
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
def test_cew_dataset(dataset_path):
    print(f"📁 Scanning: {dataset_path}")
    
    detector = EARDetector()
    results = []
    
    # پیدا کردن همه تصاویر
    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"🖼️ Found {len(image_files)} images")
    
    for i, img_path in enumerate(image_files):
        ear = detector.get_ear(img_path)
        if ear is not None:
            results.append({
                'image': os.path.basename(img_path),
                'ear': ear
            })
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(image_files)}")
    
    # نمایش نتایج
    if results:
        ears = [r['ear'] for r in results]
        print("\n" + "="*50)
        print("📊 RESULTS")
        print("="*50)
        print(f"✅ Processed: {len(results)} images")
        print(f"📊 Mean EAR: {np.mean(ears):.3f}")
        print(f"📊 Min EAR: {np.min(ears):.3f}")
        print(f"📊 Max EAR: {np.max(ears):.3f}")
        print(f"📊 Std EAR: {np.std(ears):.3f}")
        
        # ذخیره نتایج
        df = pd.DataFrame(results)
        df.to_csv('cew_results.csv', index=False)
        print("\n✅ Results saved to: cew_results.csv")
    else:
        print("❌ No faces detected!")

# ================================================================
# MAIN - با مسیر مستقیم
# ================================================================
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════╗
║  🧪 CEW DATASET TESTER                   ║
║  ═══════════════════════════════════════  ║
║  تست روی دیتاست CEW (تصاویر)             ║
╚═══════════════════════════════════════════╝
    """)
    
    # ===== مسیر رو اینجا تنظیم کن =====
    dataset_path = "full_dataset"  # یا "test_faces"
    # =================================
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path not found: {dataset_path}")
        print("💡 Run: python make_full_dataset.py first!")
        exit()
    
    test_cew_dataset(dataset_path)