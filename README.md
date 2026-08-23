# 👁️ SURDAS — Real-Time Assistive Vision & Local Voice Assistant

SURDAS is an AI-powered perception and voice assistant designed for visually impaired users. It combines real-time computer vision (YOLOv8, MiDaS Depth, Indian Banknote Recognition, EasyOCR) with a **100% offline, local voice assistant** (Silero VAD, Wake Word, Whisper STT, Command Router, Local LLM via Ollama, and TTS).

---

## 🏛️ System Architecture

```
SURDAS
                            |
             +--------------+--------------+
             |                             |
             v                             v
       Vision System                 Voice System
             |                             |
       ESP32-CAM                         Microphone
             |                             |
       OpenCV / YOLO                      VAD (Silero)
             |                             |
     Depth (MiDaS) / OCR              Speech-to-Text
             |                        (faster-whisper)
       Object Detection                    |
             |                             |
             +-------------+---------------+
                           |
                           v
                    Command Router
                           |
              +------------+-------------+
              |            |              |
              v            v              v
         Navigation       OCR           Torch
              |            |              |
              +------------+--------------+
                           |
                           v
                     Vision Context
                           |
                           v
                     Local LLM
                     (Ollama)
                           |
                           v
                     Unified TTS
                     (pyttsx3 / Piper)
                           |
                           v
                        Speaker
```

---

## 📁 Package Structure

```
surdas_ai/
├── voice/
│   ├── __init__.py
│   ├── vad.py               # Voice Activity Detection (Silero VAD + Energy Fallback)
│   ├── wakeword.py          # Wake Word Detection ("Hey Surdas" / openWakeWord)
│   ├── stt.py               # Speech to Text (faster-whisper / Whisper)
│   ├── tts.py               # Unified Text-to-Speech (Piper + pyttsx3)
│   ├── llm.py               # Local LLM integration (Ollama with live Vision Context)
│   ├── command_router.py    # Deterministic Command Router vs LLM Query Router
│   └── assistant.py         # Non-blocking audio capture & background listener
├── surdas_brain.py          # Main Unified Vision & Voice System
├── currency_detector.py     # Indian Banknote Recognition (Rs. 10 - Rs. 500)
├── surdas_esp32_cam/
│   └── surdas_esp32_cam.ino # ESP32-CAM Firmware (AP Stream, High-Res Capture, Flashlight)
├── requirements.txt
└── README.md
```

---

## 🗣️ Spoken Voice Commands

The assistant wakes up on **"Hey Surdas"** or **"Surdas"**:

### 1. Deterministic Commands (Zero Latency, No LLM required):
* **Navigation**: *"Hey Surdas, start navigation"* or *"Navigation mode"*
* **Read Text**: *"Hey Surdas, read text"* or *"Read this sign"* (automatically turns on torch, snaps image, reads aloud, and turns off torch)
* **Flashlight On**: *"Hey Surdas, turn on light"* or *"Torch on"*
* **Flashlight Off**: *"Hey Surdas, turn off light"* or *"Torch off"*
* **Fast Scene Check**: *"Hey Surdas, what do you see?"* or *"What is in front of me?"*
* **Stop / Silence**: *"Hey Surdas, stop"* or *"Silence"*
* **Status**: *"Hey Surdas, status"*

### 2. Conversational / Visual Q&A (Powered by Local Ollama):
* *"Hey Surdas, is there a chair I can sit on?"*
* *"Hey Surdas, where is the person standing?"*
* *(The local LLM is automatically provided with the live vision context: detected objects, proximity, and torch status to answer accurately.)*

---

## 🚀 How to Run

1. **(Optional) Start Ollama** (for conversational AI):
   ```bash
   ollama run llama3.2
   ```
2. **Run SURDAS**:
   ```bash
   cd /Users/devanshvpurohit/surdas_ai
   python3 surdas_brain.py
   ```

* **Keyboard Fallbacks**: `[N]` Navigation Mode, `[T]` Read Text, `[L]` Toggle Flashlight, `[Q]` Quit.
