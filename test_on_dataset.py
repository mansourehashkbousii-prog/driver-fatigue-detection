import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
from collections import deque
from datetime import datetime

# ================================================================
# کلاس‌های مشابه برنامه اصلی (فقط بخش‌های ضروری)
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
    
    def get_ear_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

class DatasetTester:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.results = []
        self.detector = EARDetector()
        self.filter = LowPassFilter()
        self.fps = 30
        self.ear_history = deque(maxlen=self.fps * 3)  # 3 ثانیه
        
    def process_video(self, video_path, label_file=None):
        """
        پردازش یک ویدیو از دیتاست
        label_file: فایل CSV که برچسب‌های واقعی رو داره
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        video_results = {
            'video': os.path.basename(video_path),
            'frames': 0,
            'ear_values': [],
            'drowsy_frames': 0,
            'total_frames': 0
        }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            ear = self.detector.get_ear_from_frame(frame)
            if ear is not None:
                ear_filtered = self.filter.filter(ear)
                video_results['ear_values'].append(ear_filtered)
                
                # تشخیص خواب (EAR < 0.2)
                if ear_filtered < 0.2:
                    video_results['drowsy_frames'] += 1
                
                video_results['total_frames'] += 1
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        
        if video_results['total_frames'] > 0:
            video_results['drowsy_percentage'] = (video_results['drowsy_frames'] / video_results['total_frames']) * 100
        else:
            video_results['drowsy_percentage'] = 0
        
        self.results.append(video_results)
        return video_results
    
    def process_dataset(self):
        """پردازش کل دیتاست"""
        print(f"📁 Scanning dataset: {self.dataset_path}")
        
        # پیدا کردن همه فایل‌های ویدیو
        video_files = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.avi', '.mp4', '.mov')):
                    video_files.append(os.path.join(root, file))
        
        print(f"🎬 Found {len(video_files)} videos")
        
        for i, video_file in enumerate(video_files):
            print(f"\n📹 Processing video {i+1}/{len(video_files)}: {os.path.basename(video_file)}")
            self.process_video(video_file)
        
        return self.results
    
    def generate_report(self):
        """گزارش نهایی"""
        if len(self.results) == 0:
            print("❌ No results to report")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("📊 DATASET TEST RESULTS")
        print("="*60)
        
        print(f"\n📹 Total videos processed: {len(df)}")
        print(f"📊 Average drowsy percentage: {df['drowsy_percentage'].mean():.2f}%")
        print(f"📊 Min drowsy percentage: {df['drowsy_percentage'].min():.2f}%")
        print(f"📊 Max drowsy percentage: {df['drowsy_percentage'].max():.2f}%")
        
        # ذخیره نتایج
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"dataset_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Results saved to: {csv_file}")
        
        return df

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import sys
    
    print("""
╔═══════════════════════════════════════════╗
║  🧪 DATASET TESTER FOR NTHU-DDD          ║
║  ═══════════════════════════════════════  ║
║  برای تست روی دیتاست استاندارد           ║
╚═══════════════════════════════════════════╝
    """)
    
    # مسیر دیتاست رو از کاربر بگیر
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("📁 Enter path to NTHU-DDD dataset: ")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Path not found: {dataset_path}")
        sys.exit(1)
    
    tester = DatasetTester(dataset_path)
    tester.process_dataset()
    tester.generate_report()