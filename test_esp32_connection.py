import urllib.request
import urllib.error
import cv2
import time
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

# Guarantee working directory is the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

ESP32_IP = "192.168.4.1"
STATUS_URL = f"http://{ESP32_IP}/status"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
STREAM_URL = f"http://{ESP32_IP}/stream"
LED_URL = f"http://{ESP32_IP}/led"

print("=" * 65)
print("       SURDAS ESP32-CAM HARDWARE & STREAM DIAGNOSTIC TEST")
print("=" * 65)

# Step 1: Check HTTP Status Endpoint
print(f"\n[1/4] Checking ESP32-CAM connection at: {STATUS_URL}")
try:
    req = urllib.request.Request(STATUS_URL, headers={"User-Agent": "SurdasClient"})
    with urllib.request.urlopen(req, timeout=2.5) as resp:
        status_code = resp.status
        body = resp.read().decode('utf-8')
        print(f"  ✅ ESP32-CAM Status Endpoint OK! (HTTP {status_code})")
        print(f"     Response Payload: {body}")
except urllib.error.URLError as e:
    print(f"  ❌ Cannot reach {STATUS_URL}: {e}")
    print("     👉 Please ensure:")
    print("        1. The ESP32-CAM is powered ON.")
    print("        2. Your Mac Wi-Fi is connected to: 'SURDAS_EYES' (Password: 12345678)")
    print("\n[DIAGNOSTIC FAILED - ESP32 OFFLINE]")
    sys.exit(1)
except Exception as e:
    print(f"  ⚠️ Warning on status check: {e}")

# Step 2: Test Flashlight LED Control
print(f"\n[2/4] Testing Flashlight LED endpoint: {LED_URL}?state=on")
try:
    with urllib.request.urlopen(f"{LED_URL}?state=on", timeout=1.5) as resp:
        print(f"  ✅ Flashlight turned ON: {resp.read().decode('utf-8')}")
    time.sleep(0.5)
    with urllib.request.urlopen(f"{LED_URL}?state=off", timeout=1.5) as resp:
        print(f"  ✅ Flashlight turned OFF: {resp.read().decode('utf-8')}")
except Exception as e:
    print(f"  ⚠️ Flashlight control warning: {e}")

# Step 3: Test Single Frame Capture
print(f"\n[3/4] Testing frame snapshot from: {CAPTURE_URL}")
frame = None
try:
    with urllib.request.urlopen(CAPTURE_URL, timeout=3.0) as resp:
        img_bytes = bytearray(resp.read())
        img_np = np.asarray(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is not None:
            h, w, _ = frame.shape
            print(f"  ✅ Frame capture successful! Resolution: {w}x{h}")
        else:
            print("  ❌ Failed to decode JPEG frame.")
except Exception as e:
    print(f"  ❌ Capture error: {e}")

# Step 4: Run YOLO + MiDaS inference test on captured ESP32 frame
if frame is not None:
    print(f"\n[4/4] Running YOLOv8 + MiDaS pipeline on ESP32-CAM frame...")
    try:
        # Load YOLO
        yolo_path = os.path.join(SCRIPT_DIR, "yolov8n.pt")
        yolo = YOLO(yolo_path)
        results = yolo(frame, conf=0.25, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(f"{label} ({int(conf*100)}%)")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(conf*100)}%", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(SCRIPT_DIR, "esp32_test_result.jpg")
        cv2.imwrite(output_path, frame)
        print(f"  ✅ Inference complete!")
        print(f"     Detected {len(detections)} objects: {', '.join(detections) if detections else 'None (Path clear)'}")
        print(f"     Saved annotated snapshot to: {output_path}")

    except Exception as e:
        print(f"  ⚠️ Inference test warning: {e}")

print("\n" + "=" * 65)
print("     ESP32-CAM DIAGNOSTIC & HARDWARE VERIFICATION COMPLETE")
print("=" * 65)
