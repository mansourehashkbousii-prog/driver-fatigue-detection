# ================================================================
# 🚗 DRIVER FATIGUE DETECTION SYSTEM v5.0 - CLEAN VERSION
# ================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime
import json
import os
import warnings
import time
import platform
import threading
import ctypes

warnings.filterwarnings('ignore')

# ================================================================
# BEEP FUNCTION
# ================================================================
def beep():
    try:
        ctypes.windll.kernel32.Beep(800, 400)
    except:
        try:
            import winsound
            winsound.Beep(800, 400)
        except:
            print('\a')

# ================================================================
# LOW PASS FILTER
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

# ================================================================
# KALMAN FILTER
# ================================================================
class KalmanFilter:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.q = process_noise
        self.r = measurement_noise
        self.p = 1.0
        self.k = 0.0
        self.x = 0.0
        self.initialized = False
    
    def filter(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return measurement
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return self.x

# ================================================================
# PERCLOS DETECTOR
# ================================================================
class PERCLOSDetector:
    def __init__(self, fps=30, window_seconds=60):
        self.fps = fps
        self.window_size = fps * window_seconds
        self.eye_states = deque(maxlen=self.window_size)
    
    def update(self, ear, threshold=0.2):
        is_closed = ear < threshold
        self.eye_states.append(1 if is_closed else 0)
        if len(self.eye_states) == self.window_size:
            return sum(self.eye_states) / len(self.eye_states)
        return None

# ================================================================
# MAR DETECTOR
# ================================================================
class MARDetector:
    MOUTH_UPPER = [13, 14, 78, 308]
    MOUTH_LOWER = [87, 88, 95, 96]
    
    def __init__(self, yawn_threshold=0.6):
        self.mar_value = 0.0
        self.yawn_threshold = yawn_threshold
        self.is_yawning = False
    
    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def mar(self, landmarks):
        v1 = self.distance(landmarks[self.MOUTH_UPPER[0]], landmarks[self.MOUTH_LOWER[0]])
        v2 = self.distance(landmarks[self.MOUTH_UPPER[1]], landmarks[self.MOUTH_LOWER[1]])
        v3 = self.distance(landmarks[self.MOUTH_UPPER[2]], landmarks[self.MOUTH_LOWER[2]])
        v4 = self.distance(landmarks[self.MOUTH_UPPER[3]], landmarks[self.MOUTH_LOWER[3]])
        h = self.distance(landmarks[self.MOUTH_UPPER[0]], landmarks[self.MOUTH_LOWER[0]])
        if h == 0:
            return 0
        return (v1 + v2 + v3 + v4) / (4 * h)
    
    def update(self, landmarks):
        self.mar_value = self.mar(landmarks)
        self.is_yawning = self.mar_value > self.yawn_threshold
        return self.mar_value, self.is_yawning

# ================================================================
# ADAPTIVE THRESHOLD
# ================================================================
class AdaptiveThreshold:
    def __init__(self, initial_samples=150):
        self.base_ear_values = []
        self.initial_samples = initial_samples
        self.calibrated = False
        self.threshold_drowsy = 0.26
        self.threshold_sleep = 0.20
    
    def calibrate(self, ear):
        if not self.calibrated:
            self.base_ear_values.append(ear)
            if len(self.base_ear_values) >= self.initial_samples:
                mean_ear = np.mean(self.base_ear_values)
                self.threshold_drowsy = max(0.15, mean_ear * 0.7)
                self.threshold_sleep = max(0.10, mean_ear * 0.55)
                self.calibrated = True
        return self.threshold_drowsy, self.threshold_sleep

# ================================================================
# DATA LOGGER
# ================================================================
class DataLogger:
    def __init__(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.save_dir = f"fatigue_data_{today}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data = {'timestamp': [], 'ear': [], 'perclos': [], 'mar': [], 'state': [], 'drowsy_score': []}
    
    def log(self, ear, perclos, mar, state, drowsy_score):
        self.data['timestamp'].append(datetime.now().isoformat())
        self.data['ear'].append(ear)
        self.data['perclos'].append(perclos)
        self.data['mar'].append(mar)
        self.data['state'].append(state)
        self.data['drowsy_score'].append(drowsy_score)
    
    def save(self):
        if len(self.data['ear']) == 0:
            return None
        df = pd.DataFrame(self.data)
        csv_file = f"{self.save_dir}/session_{self.session_id}.csv"
        df.to_csv(csv_file, index=False)
        return {'session_id': self.session_id, 'duration': len(self.data['ear']) / 30, 'mean_ear': np.mean(self.data['ear'])}

# ================================================================
# VIDEO RECORDER
# ================================================================
class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.start_time = None
    
    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.frames = []
            self.start_time = time.time()
    
    def add_frame(self, frame):
        if self.recording:
            self.frames.append(frame.copy())
            if time.time() - self.start_time > 10:
                self.stop_recording()
    
    def stop_recording(self):
        if self.recording and len(self.frames) > 10:
            self.recording = False
            self._save_video()
            self.frames = []
            return True
        self.recording = False
        self.frames = []
        return False
    
    def _save_video(self):
        if len(self.frames) == 0:
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sleep_alert_{timestamp}.avi"
            h, w = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 15.0, (w, h))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"✅ Video saved: {filename}")
        except Exception as e:
            print(f"Video save error: {e}")

# ================================================================
# WEB DASHBOARD
# ================================================================
class WebDashboard:
    def __init__(self):
        self.server_thread = None
        self.running = False
        self.data = {'ear': 0, 'state': 'AWAKE', 'score': 0, 'samples': 0}
    
    def start(self):
        if self.running:
            return
        try:
            from flask import Flask, jsonify
            from flask_cors import CORS
            self.running = True
            self.server_thread = threading.Thread(target=self._run, daemon=True)
            self.server_thread.start()
            print("🌐 Web: http://localhost:5000")
        except:
            print("❌ Web server failed")
    
    def _run(self):
        try:
            from flask import Flask, jsonify
            from flask_cors import CORS
            app = Flask(__name__)
            CORS(app)
            
            @app.route('/')
            def index():
                return '''
                <html><head><title>Fatigue Dashboard</title>
                <style>body{background:#1a1a2e;color:white;font-family:Arial;text-align:center;padding:20px}
                .card{background:#16213e;padding:30px;border-radius:15px;display:inline-block;margin:10px;min-width:150px}
                .value{font-size:48px;font-weight:bold}
                .label{color:#a8d8ea}
                .awake{color:#00b894}.drowsy{color:#fdcb6e}.sleep{color:#e74c3c}
                </style></head>
                <body>
                <h1>🚗 Fatigue Dashboard</h1>
                <div>
                <div class="card"><div class="label">EAR</div><div class="value" id="ear">0.000</div></div>
                <div class="card"><div class="label">Status</div><div class="value" id="state">AWAKE</div></div>
                <div class="card"><div class="label">Score</div><div class="value" id="score">0%</div></div>
                </div>
                <script>
                function update(){fetch('/api/data').then(r=>r.json()).then(d=>{
                document.getElementById('ear').textContent=d.ear.toFixed(3);
                document.getElementById('state').textContent=d.state;
                document.getElementById('state').className='value '+d.state.toLowerCase();
                document.getElementById('score').textContent=d.score+'%';})}
                setInterval(update,1000);update();
                </script>
                </body></html>
                '''
            
            @app.route('/api/data')
            def get_data():
                return jsonify(self.data)
            
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except:
            self.running = False
    
    def update(self, ear, state, score):
        self.data['ear'] = ear
        self.data['state'] = state
        self.data['score'] = score
        self.data['samples'] += 1

# ================================================================
# EAR DETECTOR
# ================================================================
class EARDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def ear(self, landmarks, eye):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
        v1 = self.distance(p2, p6)
        v2 = self.distance(p3, p5)
        h = self.distance(p1, p4)
        return 0 if h == 0 else (v1 + v2) / (2 * h)
    
    def get_ear_and_landmarks(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return frame, None, None
        face = results.multi_face_landmarks[0]
        left = self.ear(face.landmark, self.LEFT_EYE)
        right = self.ear(face.landmark, self.RIGHT_EYE)
        return frame, (left + right) / 2, face.landmark
    
    def release(self):
        self.cap.release()

# ================================================================
# MAIN DASHBOARD
# ================================================================
class FatigueDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Driver Fatigue Detection v5.0")
        self.root.geometry("1300x800")
        self.root.configure(bg='#1a1a2e')
        
        self.fps = 30
        self.running = False
        self.paused = False
        self.frame_index = 0
        
        self.detector = EARDetector()
        self.lowpass = LowPassFilter()
        self.kalman = KalmanFilter()
        self.perclos = PERCLOSDetector()
        self.mar = MARDetector()
        self.threshold = AdaptiveThreshold()
        self.logger = DataLogger()
        self.recorder = VideoRecorder()
        self.web = WebDashboard()
        
        self.times = deque(maxlen=self.fps * 15)
        self.values = deque(maxlen=self.fps * 15)
        self.raw_values = deque(maxlen=self.fps * 15)
        
        self.current_state = "AWAKE"
        self.drowsy_score = 0.0
        self.perclos_value = None
        self.mar_value = 0.0
        self.is_yawning = False
        self.calibration_counter = 0
        
        self.web.start()
        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
    
    def _build_gui(self):
        self.main = tk.Frame(self.root, bg='#1a1a2e')
        self.main.pack(fill='both', expand=True)
        
        # Header
        header = tk.Frame(self.main, bg='#0f3460', height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="🚗 Driver Fatigue Detection v5.0", bg='#0f3460', fg='white', font=('Segoe UI', 18, 'bold')).pack(pady=(10,0))
        tk.Label(header, text="EAR | PERCLOS | MAR | Yawn Detection", bg='#0f3460', fg='#a8d8ea', font=('Segoe UI', 10)).pack()
        
        # Toolbar
        toolbar = tk.Frame(self.main, bg='#16213e', height=60)
        toolbar.pack(fill='x')
        toolbar.pack_propagate(False)
        
        btn_style = {'width': 11, 'height': 1, 'font': ('Segoe UI', 9, 'bold'), 'relief': 'flat', 'cursor': 'hand2'}
        
        self.start_btn = tk.Button(toolbar, text="▶ Start", command=self.start, bg='#00b894', fg='white', **btn_style)
        self.start_btn.pack(side='left', padx=3, pady=12)
        
        self.stop_btn = tk.Button(toolbar, text="⏹ Stop", command=self.stop, bg='#e74c3c', fg='white', state='disabled', **btn_style)
        self.stop_btn.pack(side='left', padx=3, pady=12)
        
        self.save_btn = tk.Button(toolbar, text="💾 Save", command=self.save_data, bg='#6c5ce7', fg='white', state='disabled', **btn_style)
        self.save_btn.pack(side='left', padx=3, pady=12)
        
        self.web_btn = tk.Button(toolbar, text="🌐 Web", command=lambda: __import__('webbrowser').open('http://localhost:5000'), bg='#00cec9', fg='white', **btn_style)
        self.web_btn.pack(side='left', padx=3, pady=12)
        
        # Status
        status_frame = tk.Frame(toolbar, bg='#16213e')
        status_frame.pack(side='right', padx=15)
        
        self.led = tk.Canvas(status_frame, width=16, height=16, bg='#16213e', highlightthickness=0)
        self.led.pack(side='left', padx=5)
        self.led_circle = self.led.create_oval(2, 2, 14, 14, fill='#95a5a6')
        
        tk.Label(status_frame, text="Status:", bg='#16213e', fg='#a8d8ea', font=('Segoe UI', 11, 'bold')).pack(side='left', padx=5)
        self.status_label = tk.Label(status_frame, text="⚪ Stopped", bg='#16213e', fg='#95a5a6', font=('Segoe UI', 13, 'bold'))
        self.status_label.pack(side='left', padx=5)
        
        # Main content
        main_frame = tk.Frame(self.main, bg='#1a1a2e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Video
        video_frame = tk.Frame(main_frame, bg='#1a1a2e')
        video_frame.pack(side='left', fill='both', expand=True)
        self.video_label = tk.Label(video_frame, bg='#16213e')
        self.video_label.pack(padx=5, pady=5, fill='both', expand=True)
        
        # Info
        info_frame = tk.Frame(video_frame, bg='#1a1a2e')
        info_frame.pack(fill='x', pady=8)
        
        cards = [('EAR', 'ear_val', '#00b894'), ('PERCLOS', 'perclos_val', '#00cec9'), ('Score', 'score_val', '#fdcb6e'), ('Cal', 'cal_val', '#6c5ce7')]
        self.info_labels = {}
        for label, key, color in cards:
            card = tk.Frame(info_frame, bg='#16213e')
            card.pack(side='left', padx=4, expand=True, fill='both')
            tk.Label(card, text=label, bg='#16213e', fg='#a8d8ea', font=('Segoe UI', 8, 'bold')).pack(pady=(4,0))
            self.info_labels[key] = tk.Label(card, text='---', bg='#16213e', fg=color, font=('Segoe UI', 14, 'bold'))
            self.info_labels[key].pack(pady=(0,4))
        
        # Plot
        plot_frame = tk.Frame(main_frame, bg='#1a1a2e')
        plot_frame.pack(side='right', fill='both', expand=True)
        
        self.figure = Figure(figsize=(8, 5.5), dpi=100, facecolor='#1a1a2e')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#16213e')
        self.ax.set_title('EAR Monitoring', color='#a8d8ea')
        self.ax.set_xlabel('Time (s)', color='#a8d8ea')
        self.ax.set_ylabel('EAR', color='#a8d8ea')
        self.ax.tick_params(colors='#a8d8ea')
        self.ax.set_ylim(0, 0.5)
        self.ax.grid(True, alpha=0.15)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Progress
        progress_frame = tk.Frame(self.main, bg='#1a1a2e', height=35)
        progress_frame.pack(fill='x', padx=20, pady=5)
        progress_frame.pack_propagate(False)
        
        tk.Label(progress_frame, text="Drowsiness:", bg='#1a1a2e', fg='#a8d8ea', font=('Segoe UI', 10)).pack(side='left', padx=5)
        self.progress = ttk.Progressbar(progress_frame, length=450, mode='determinate')
        self.progress.pack(side='left', padx=10, fill='x', expand=True)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('green.Horizontal.TProgressbar', background='#00b894', troughcolor='#2d3436')
        self.progress.configure(style='green.Horizontal.TProgressbar')
        
        # Stats
        stats_frame = tk.Frame(self.main, bg='#0f3460', height=35)
        stats_frame.pack(fill='x', side='bottom')
        stats_frame.pack_propagate(False)
        
        self.stats_text = tk.Label(stats_frame, text="📊 Press Start", bg='#0f3460', fg='#a8d8ea', font=('Segoe UI', 10))
        self.stats_text.pack(side='left', padx=20, pady=5)
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.paused = False
        self.frame_index = 0
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_label.config(text="🟢 Running", fg='#00b894')
        self.led.itemconfig(self.led_circle, fill='#00b894')
        self.stats_text.config(text="📊 Calibrating...")
        self.update_waveform()
    
    def stop(self):
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.status_label.config(text="⚪ Stopped", fg='#95a5a6')
        self.led.itemconfig(self.led_circle, fill='#95a5a6')
        self.root.configure(bg='#1a1a2e')
        self.main.configure(bg='#1a1a2e')
        self.save_data()
    
    def save_data(self):
        report = self.logger.save()
        if report:
            messagebox.showinfo("✅ Data Saved", f"Session: {report['session_id']}\nDuration: {report['duration']:.0f}s\nMean EAR: {report['mean_ear']:.3f}")
    
    def update_waveform(self):
        if not self.running:
            return
        
        frame, ear_raw, landmarks = self.detector.get_ear_and_landmarks()
        
        if ear_raw is not None and landmarks is not None:
            ear_filtered = self.lowpass.filter(ear_raw)
            ear_kalman = self.kalman.filter(ear_filtered)
            
            if not self.threshold.calibrated:
                self.calibration_counter += 1
                self.threshold.calibrate(ear_kalman)
                if self.threshold.calibrated:
                    self.status_label.config(text="✅ Calibrated", fg='#00b894')
                    self.stats_text.config(text="📊 Monitoring")
            
            drowsy_thresh, sleep_thresh = self.threshold.threshold_drowsy, self.threshold.threshold_sleep
            self.perclos_value = self.perclos.update(ear_kalman, threshold=sleep_thresh)
            self.mar_value, self.is_yawning = self.mar.update(landmarks)
            
            prev_state = self.current_state
            
            if ear_kalman < sleep_thresh:
                self.drowsy_score = min(100, self.drowsy_score + 2)
                self.current_state = "SLEEPING"
            elif ear_kalman < drowsy_thresh:
                self.drowsy_score = min(100, self.drowsy_score + 0.5)
                self.current_state = "DROWSY"
            else:
                self.drowsy_score = max(0, self.drowsy_score - 0.3)
                self.current_state = "AWAKE"
            
            # ===== YAWN + BEEP =====
            if self.is_yawning and self.mar_value > 0.6:
                print(f"😮 Yawn! MAR: {self.mar_value:.2f}")
                beep()  # ← صدا
                if self.current_state == "AWAKE":
                    self.current_state = "DROWSY"
                    self.drowsy_score = min(100, self.drowsy_score + 10)
            
            # ===== SLEEP + BEEP =====
            if self.current_state == "SLEEPING" and prev_state != "SLEEPING":
                beep()  # ← صدا
                self.recorder.start_recording()
            elif self.current_state != "SLEEPING" and prev_state == "SLEEPING":
                self.recorder.stop_recording()
            
            if self.current_state == "SLEEPING":
                self.recorder.add_frame(frame)
            
            self.web.update(ear_kalman, self.current_state, self.drowsy_score)
            
            t = self.frame_index / self.fps
            self.times.append(t)
            self.values.append(ear_kalman)
            self.raw_values.append(ear_raw)
            self.logger.log(ear_kalman, self.perclos_value, self.mar_value, self.current_state, self.drowsy_score)
            
            self._update_ui(ear_kalman)
            self._update_plot(drowsy_thresh, sleep_thresh)
            self.frame_index += 1
        
        if frame is not None:
            self._display_frame(frame)
        
        self.root.after(int(1000 / self.fps), self.update_waveform)
    
    def _update_ui(self, ear):
        self.info_labels['ear_val'].config(text=f"{ear:.3f}")
        if self.perclos_value is not None:
            self.info_labels['perclos_val'].config(text=f"{self.perclos_value:.3f}")
        self.info_labels['score_val'].config(text=f"{self.drowsy_score:.0f}%")
        self.info_labels['cal_val'].config(text="✓" if self.threshold.calibrated else f"{self.calibration_counter}/150", fg='#00b894' if self.threshold.calibrated else '#fdcb6e')
        
        if self.current_state == "SLEEPING":
            self.root.configure(bg='#8B0000')
            self.main.configure(bg='#8B0000')
            self.status_label.config(text="😴 SLEEPING", fg='#ff0000')
            self.led.itemconfig(self.led_circle, fill='#ff0000')
        elif self.current_state == "DROWSY":
            self.root.configure(bg='#8B6508')
            self.main.configure(bg='#8B6508')
            self.status_label.config(text="😫 DROWSY", fg='#ff8c00')
            self.led.itemconfig(self.led_circle, fill='#ff8c00')
        else:
            self.root.configure(bg='#1a1a2e')
            self.main.configure(bg='#1a1a2e')
            self.status_label.config(text="😊 AWAKE", fg='#00b894')
            self.led.itemconfig(self.led_circle, fill='#00b894')
        
        self.progress['value'] = self.drowsy_score
        style = ttk.Style()
        if self.drowsy_score > 70:
            style.configure('green.Horizontal.TProgressbar', background='#e74c3c')
        elif self.drowsy_score > 40:
            style.configure('green.Horizontal.TProgressbar', background='#fdcb6e')
        else:
            style.configure('green.Horizontal.TProgressbar', background='#00b894')
        
        if len(self.logger.data['ear']) > 0:
            df = pd.DataFrame(self.logger.data)
            drowsy = df['state'].value_counts().get('DROWSY', 0) / len(df) * 100
            sleep = df['state'].value_counts().get('SLEEPING', 0) / len(df) * 100
            self.stats_text.config(text=f"📊 {len(df)} samples | Drowsy: {drowsy:.1f}% | Sleep: {sleep:.1f}%")
    
    def _update_plot(self, drowsy_thresh, sleep_thresh):
        if len(self.times) > 0:
            xmin = max(0, self.times[-1] - 15)
            self.ax.clear()
            self.ax.set_facecolor('#16213e')
            self.ax.set_xlim(xmin, xmin + 15)
            self.ax.set_ylim(0, 0.5)
            self.ax.axhline(drowsy_thresh, color='#fdcb6e', linestyle='--')
            self.ax.axhline(sleep_thresh, color='#e74c3c', linestyle='--')
            self.ax.plot(list(self.times), list(self.raw_values), color='#e74c3c', linewidth=1, alpha=0.3)
            self.ax.plot(list(self.times), list(self.values), color='#00b894', linewidth=2)
            self.ax.set_title('EAR Monitoring', color='#a8d8ea')
            self.ax.set_xlabel('Time (s)', color='#a8d8ea')
            self.ax.set_ylabel('EAR', color='#a8d8ea')
            self.ax.tick_params(colors='#a8d8ea')
            self.ax.grid(True, alpha=0.15)
            self.canvas.draw()
    
    def _display_frame(self, frame):
        display_w, display_h = 500, 380
        frame_resized = cv2.resize(frame, (display_w, display_h))
        
        color = (0, 255, 0) if self.current_state == "AWAKE" else (0, 165, 255) if self.current_state == "DROWSY" else (0, 0, 255)
        cv2.rectangle(frame_resized, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.putText(frame_resized, f"Status: {self.current_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame_resized, f"Score: {self.drowsy_score:.0f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.is_yawning and self.mar_value > 0.6:
            cv2.rectangle(frame_resized, (10, display_h - 50), (300, display_h - 10), (0, 0, 0), -1)
            cv2.putText(frame_resized, "😮 YAWN!", (20, display_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if self.recorder.recording:
            cv2.putText(frame_resized, "🔴 REC", (display_w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.ppm', frame_rgb)[1].tobytes())
        self.video_label.config(image=img, width=display_w, height=display_h)
        self.video_label.image = img
    
    def close(self):
        self.running = False
        self.detector.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════╗
║  🚗 DRIVER FATIGUE DETECTION v5.0        ║
║  ═══════════════════════════════════════  ║
║  ✅ EAR Detection                        ║
║  ✅ PERCLOS                              ║
║  ✅ MAR (Yawn Detection)                 ║
║  ✅ Sound Alert                          ║
║  ✅ Web Dashboard                        ║
╚═══════════════════════════════════════════╝
    """)
    
    root = tk.Tk()
    app = FatigueDashboard(root)
    root.mainloop()