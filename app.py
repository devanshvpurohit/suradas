import threading
import time
import cv2
from flask import Flask, render_template, Response, jsonify
from surdas_brain import SurdasBrain

app = Flask(__name__)

# Initialize brain in headless mode
brain = SurdasBrain()
brain.headless = True

# Start brain in a background thread
brain_thread = threading.Thread(target=brain.run, daemon=True)
brain_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        frame = brain.latest_display_frame
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    return jsonify({
        "logs": brain.log_history,
        "mode": brain.mode,
        "led_on": brain.led_on,
        "wall_detected": brain.wall_detected,
        "closest_obstacle": brain.latest_closest_obstacle,
        "objects": brain.latest_detected_objects
    })

@app.route('/control/<action>')
def control(action):
    if action == "toggle_nav":
        brain.mode = "NAV"
        brain.voice.speak("Navigation mode active.")
    elif action == "toggle_ocr":
        brain.mode = "OCR"
    elif action == "toggle_led":
        brain.toggle_esp32_led(not brain.led_on)
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False, threaded=True)
