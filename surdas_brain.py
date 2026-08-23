import os
import sys
import warnings
import argparse

# Suppress background PyTorch MPS & OpenCV warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_SETTINGS_ANALYTICS"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")

# ── CLI argument parsing (must happen before heavy imports) ──────────────────
_parser = argparse.ArgumentParser(description="SURDAS Assistive Vision Brain")
_parser.add_argument(
    "--model", "-m",
    default="",
    metavar="MODEL",
    help="Ollama model name to use for LLM. Defaults to gemma3:1b.",
)
_parser.add_argument(
    "--mic",
    default=None,
    metavar="DEVICE",
    help="Audio input device name keyword or index (e.g. 'airpods', 'boat', 'buds', '0'). Auto-detects TWS by default.",
)
_parser.add_argument(
    "--mic-gain",
    type=float,
    default=1.0,
    metavar="GAIN",
    help="Software gain multiplier for microphone (e.g. 1.5, 2.0, 2.5 for quiet TWS). Defaults to 1.8x for TWS.",
)
_args, _ = _parser.parse_known_args()
LLM_MODEL = _args.model  # empty string = use LocalLLM default (gemma3:1b)
MIC_DEVICE = _args.mic
MIC_GAIN = _args.mic_gain

# Guarantee working directory is the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import cv2
import torch
import numpy as np
import threading
import queue
import time
import urllib.request
from ultralytics import YOLO
import easyocr

from voice.tts import VoiceEngine
from voice.assistant import VoiceAssistant
from voice.llm import LocalLLM
from voice.app_launcher import open_app, close_app
from telemetry import start_telemetry, broadcast_event

# =====================================================
# CONFIGURATION
# =====================================================
ESP32_IP = "192.168.4.1"
STREAM_URL = f"http://{ESP32_IP}/stream"
LED_URL = f"http://{ESP32_IP}/led"

# Best available compute accelerator (CUDA / Apple Silicon MPS / CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[SYSTEM] Hardware Acceleration: {DEVICE}")
if LLM_MODEL:
    print(f"[SYSTEM] LLM Model (CLI): {LLM_MODEL}")


# =====================================================
# 1. ROBUST VIDEO STREAM RECEIVER
# =====================================================
class CameraStream:
    def __init__(self, src):
        self.src = src
        print(f"[STREAM] Attempting to connect to {src}...")
        self.cap = cv2.VideoCapture(src)
        
        # Fallback to local webcam (0) if stream is completely unreachable
        if not self.cap.isOpened():
            print(f"[STREAM] Warning: {src} unreachable. Falling back to local webcam (0).")
            self.cap = cv2.VideoCapture(0)
            
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.connected = False
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.connected = True
                with self.lock:
                    self.frame = frame
            else:
                self.connected = False
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# =====================================================
# 2. SURDAS PERCEPTION & VOICE BRAIN (YOLO + MIDAS WALLS + VOICE)
# =====================================================
class SurdasBrain:
    def __init__(self):
        # Start Dashboard Telemetry
        start_telemetry()
        
        # 1. TTS Voice Output Engine
        self.voice = VoiceEngine()
        self.voice.speak("SURDAS Vision and Voice System starting.")

        # Connect to ESP32 stream in background
        print(f"[STREAM] Connecting to: {STREAM_URL}")
        self.stream = CameraStream(STREAM_URL)

        # 2. Parallel AI Model Loading (YOLOv8, MiDaS, EasyOCR, Voice/Whisper)
        print("[SYSTEM] ⚡ Loading AI models in parallel (YOLOv8, MiDaS, EasyOCR, Whisper/Voice)...")
        t0_load = time.time()

        def _load_yolo():
            yolo_path = os.path.join(SCRIPT_DIR, "yolov8n.pt")
            self.yolo = YOLO(yolo_path)
            print("[AI] ✅ YOLOv8 loaded.")

        def _load_midas():
            midas_repo = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
            if os.path.isdir(midas_repo):
                self.midas = torch.hub.load(midas_repo, "MiDaS_small", source="local", pretrained=True).to(DEVICE).eval()
                midas_transforms = torch.hub.load(midas_repo, "transforms", source="local")
                self.transform = midas_transforms.small_transform
            else:
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(DEVICE).eval()
                self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
            print("[AI] ✅ MiDaS Depth Estimator loaded.")

        def _load_ocr():
            self.ocr = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"), download_enabled=False)
            print("[AI] ✅ EasyOCR loaded.")

        def _load_voice():
            self.llm = LocalLLM(model_name=LLM_MODEL)
            if self.llm.is_available():
                print(f"[AI] ✅ Ollama ready — model: {self.llm.model_name}")
            else:
                print("[AI] Ollama not detected. Start with: ollama serve")
            try:
                self.voice_assistant = VoiceAssistant(self, mic_device=MIC_DEVICE, mic_gain=MIC_GAIN)
                self.voice_assistant.start()
                print("[AI] ✅ Voice Assistant active.")
            except Exception as e:
                print(f"[VOICE] Voice Assistant warning: {e}. Vision continues normally.")
                self.voice_assistant = None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_yolo = executor.submit(_load_yolo)
            f_midas = executor.submit(_load_midas)
            f_ocr = executor.submit(_load_ocr)
            f_voice = executor.submit(_load_voice)

            f_yolo.result()
            f_midas.result()
            f_ocr.result()
            f_voice.result()

        print(f"[SYSTEM] 🚀 All AI models loaded in parallel in {time.time() - t0_load:.2f}s!")

        self.mode = "NAV"  # 'NAV' or 'OCR'
        self.last_speech_time = 0
        self.last_spoken = ""
        self.led_on = False
        self.conf_threshold = 0.25

        # Live Vision Context State
        self.latest_detected_objects = []
        self.latest_closest_obstacle = None
        self.wall_detected = False

        self.voice.speak("Ready. Navigation mode active. Say Hey Surdas.")
        
        # Dashboard integration state
        self.log_history = []
        self.latest_display_frame = None
        self.headless = False

    # ── LLM helper ─────────────────────────────────────────────────
    def query_llm(self, prompt: str, speak: bool = True) -> str:
        """Ask the LLM a question. Streams response sentence-by-sentence.
        Returns the full response as a string."""
        full = []
        for sentence in self.llm.query(prompt, vision_context=self.get_vision_context()):
            if sentence:
                full.append(sentence)
                if speak:
                    self.voice.speak(sentence)
        return " ".join(full)

    # ── App launcher helpers ─────────────────────────────────────────
    def launch_app(self, app_name: str, speak: bool = True) -> bool:
        """Open a macOS app by name. Returns True on success."""
        success, msg = open_app(app_name)
        if speak:
            self.voice.speak(msg)
        return success

    def quit_app(self, app_name: str, speak: bool = True) -> bool:
        """Close a macOS app by name. Returns True on success."""
        success, msg = close_app(app_name)
        if speak:
            self.voice.speak(msg)
        return success
        
    def add_log(self, msg: str):
        print(msg)
        timestamp = time.strftime("%H:%M:%S")
        self.log_history.append(f"[{timestamp}] {msg}")
        if len(self.log_history) > 100:
            self.log_history.pop(0)

    def get_vision_context(self) -> dict:
        """Provides current live scene state for LLM and Command Router."""
        return {
            "mode": self.mode,
            "torch_on": self.led_on,
            "detected_objects": list(self.latest_detected_objects),
            "closest_obstacle": self.latest_closest_obstacle,
            "wall_ahead": self.wall_detected
        }

    def get_status_summary(self) -> str:
        """Returns verbal status report."""
        stream_stat = "connected" if self.stream.connected else "disconnected"
        torch_stat = "on" if self.led_on else "off"
        obj_count = len(self.latest_detected_objects)
        wall_stat = "Wall detected ahead." if self.wall_detected else "No wall."
        return f"System status: Camera is {stream_stat}. Flashlight is {torch_stat}. Mode is {self.mode}. Detecting {obj_count} objects. {wall_stat}"

    def toggle_esp32_led(self, state: bool):
        """Toggles the ESP32-CAM flash torch."""
        try:
            val = "on" if state else "off"
            urllib.request.urlopen(f"{LED_URL}?state={val}", timeout=0.8)
            self.led_on = state
        except Exception:
            pass

    def compute_depth(self, frame):
        """Computes dense relative depth map."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb).to(DEVICE)
        
        with torch.no_grad():
            depth = self.midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_np = depth.cpu().numpy()
        depth_vis = cv2.normalize(depth_np, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return depth_np, depth_vis

    def process_navigation(self, frame):
        """
        Processes both Discrete Objects (YOLO) and Dense Continuous Barriers/Walls (MiDaS).
        """
        h, w, _ = frame.shape
        raw_depth, depth_vis = self.compute_depth(frame)
        
        # -------------------------------------------------------------
        # 1. DENSE MIDAS SPATIAL ANALYSIS (WALL & BARRIER DETECTION)
        # -------------------------------------------------------------
        third_w = w // 3
        center_depth_crop = raw_depth[:, third_w : 2 * third_w]
        left_depth_crop = raw_depth[:, :third_w]
        right_depth_crop = raw_depth[:, 2 * third_w:]

        center_med = np.median(center_depth_crop) if center_depth_crop.size > 0 else 0
        left_med = np.median(left_depth_crop) if left_depth_crop.size > 0 else 0
        right_med = np.median(right_depth_crop) if right_depth_crop.size > 0 else 0

        # Close proximity pixel density in central walking corridor
        center_close_ratio = np.mean(center_depth_crop > 950) if center_depth_crop.size > 0 else 0

        self.wall_detected = False
        immediate_danger = None
        guidance_text = None

        # -------------------------------------------------------------
        # 2. RUN YOLOV8 OBJECT DETECTION
        # -------------------------------------------------------------
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)[0]

        detected_obstacles = []
        current_objects_in_view = []
        center_object_found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.yolo.names[cls_id]
            current_objects_in_view.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2
            
            # Spatial position
            if cx < third_w:
                pos = "on your left"
                color = (255, 200, 0)
            elif cx > 2 * third_w:
                pos = "on your right"
                color = (0, 200, 255)
            else:
                pos = "ahead"
                color = (0, 255, 0)
                center_object_found = True

            # Sample depth inside object bounding box
            crop_depth = raw_depth[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            med_depth = np.median(crop_depth) if crop_depth.size > 0 else 0

            # Proximity estimation
            if med_depth > 1200:
                proximity = "very close"
                color = (0, 0, 255)  # Red for close obstacle
            elif med_depth > 650:
                proximity = "nearby"
            else:
                proximity = "distance"

            if proximity == "very close" and pos == "ahead":
                immediate_danger = f"Caution! {label} directly ahead."

            detected_obstacles.append((label, pos, proximity, med_depth, conf))

            # Draw standard YOLO Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {int(conf*100)}% | {proximity}"
            
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(frame, tag, (x1 + 3, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. WALL / CONTINUOUS SURFACE DETECTION (MIDAS)
        # -------------------------------------------------------------
        # If central area is blocked across a broad field with high depth and no single object explains it
        if center_close_ratio > 0.38 and not center_object_found:
            self.wall_detected = True
            if center_med > 1150:
                immediate_danger = "Stop! Wall directly in front of you."
            else:
                immediate_danger = "Caution! Wall ahead."

            # Calculate best clearance bypass
            if left_med < center_med - 150 and left_med < right_med:
                guidance_text = "Wall ahead. Path is clear on your left."
            elif right_med < center_med - 150:
                guidance_text = "Wall ahead. Path is clear on your right."

        # Update Live Vision State
        self.latest_detected_objects = current_objects_in_view

        # -------------------------------------------------------------
        # 4. VOICE FEEDBACK DECISION ENGINE
        # -------------------------------------------------------------
        now = time.time()

        # Skip low-priority nav chatter while voice assistant is processing/speaking
        assistant_busy = (
            self.voice_assistant is not None
            and (self.voice_assistant._processing or not self.voice._queue.empty())
        )

        # Priority 1: Immediate Collision Hazard (always fires, interrupts everything)
        if immediate_danger:
            self.latest_closest_obstacle = immediate_danger
            speech_to_say = guidance_text if guidance_text else immediate_danger
            if speech_to_say != self.last_spoken or now - self.last_speech_time > 3.0:
                self.voice.speak(speech_to_say, force=True)
                self.last_speech_time = now
                self.last_spoken = speech_to_say

        # Priority 2: Routine navigation (only when assistant is idle, cooldown 4s)
        elif not assistant_busy and now - self.last_speech_time > 4.0:
            if detected_obstacles:
                detected_obstacles.sort(key=lambda x: x[3], reverse=True)
                top_lbl, top_pos, top_prox, _, _ = detected_obstacles[0]
                speech_text = f"{top_lbl} {top_pos}."
                self.latest_closest_obstacle = speech_text

                if speech_text != self.last_spoken or now - self.last_speech_time > 8.0:
                    self.voice.speak(speech_text)
                    self.last_spoken = speech_text
                    self.last_speech_time = now
            else:
                self.latest_closest_obstacle = None
                if self.last_spoken != "clear":
                    self.voice.speak("Path is clear.")
                    self.last_spoken = "clear"
                    self.last_speech_time = now

        # -------------------------------------------------------------
        # 5. DEPTH HEATMAP & CORRIDOR VISUALIZATION
        # -------------------------------------------------------------
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        # Draw spatial corridor lines (Left | Center | Right)
        cv2.line(depth_colormap, (third_w, 0), (third_w, h), (255, 255, 255), 1)
        cv2.line(depth_colormap, (2 * third_w, 0), (2 * third_w, h), (255, 255, 255), 1)

        # Draw Wall indicator on depth map
        if self.wall_detected:
            cv2.rectangle(depth_colormap, (third_w + 10, 10), (2 * third_w - 10, 50), (0, 0, 255), -1)
            cv2.putText(depth_colormap, "WALL AHEAD", (third_w + 20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
        # Broadcast for Caregiver Dashboard
        broadcast_event("vision", self.get_vision_context())

        return frame, depth_colormap, len(detected_obstacles)

    def process_ocr(self, frame):
        """Document / Sign Text Reading Mode."""
        self.toggle_esp32_led(True)
        self.voice.speak("Reading text...")
        time.sleep(0.3)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.ocr.readtext(rgb, detail=0)
        self.toggle_esp32_led(False)

        if results:
            text = " ".join(results)
            self.add_log(f"[OCR] Text: {text}")
            self.voice.speak(f"Text says: {text}")
        else:
            self.voice.speak("No clear text detected.")

        self.mode = "NAV"

    def run(self):
        print("\n" + "="*65)
        print(" SURDAS ASSISTIVE VISION & VOICE ASSISTANT RUNNING")
        print(" Features:")
        print("   • YOLOv8 Object & Obstacle Detection")
        print("   • MiDaS Dense Depth & Wall / Barrier Detection")
        print("   • Natural Voice Assistant ('Hey Surdas')")
        print(" Controls: [N] Nav Mode | [T] Read Text | [L] Torch | [Q] Quit")
        print("="*65 + "\n")

        prev_time = time.time()

        while True:
            frame = self.stream.get_frame()
            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for ESP32-CAM Stream...", (60, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(blank, f"Connecting to {STREAM_URL}", (60, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.putText(blank, "Make sure PC is connected to Wi-Fi: SURDAS_EYES", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                self.latest_display_frame = blank.copy()
                
                if not self.headless:
                    cv2.imshow("SURDAS - Vision Monitor", blank)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    time.sleep(0.1)
                continue

            curr_time = time.time()
            fps = 1.0 / max(0.001, (curr_time - prev_time))
            prev_time = curr_time

            if self.mode == "NAV":
                rgb_view, depth_view, count = self.process_navigation(frame)
                combined = np.hstack((rgb_view, depth_view))
                display_frame = cv2.resize(combined, (1024, 400))
                
                wall_tag = " | 🧱 WALL DETECTED" if self.wall_detected else ""
                cv2.putText(display_frame, f"FPS: {int(fps)} | Objects: {count}{wall_tag}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                
                self.latest_display_frame = display_frame.copy()
                
                if not self.headless:
                    cv2.imshow("SURDAS - Vision Monitor", display_frame)

            elif self.mode == "OCR":
                self.process_ocr(frame)
                
            if not self.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.mode = "NAV"
                    self.voice.speak("Navigation mode active.")
                elif key == ord('t'):
                    self.mode = "OCR"
                elif key == ord('l'):
                    self.toggle_esp32_led(not self.led_on)
            else:
                time.sleep(0.01)

        if self.voice_assistant:
            self.voice_assistant.stop()
        self.stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    brain = SurdasBrain()
    brain.run()
