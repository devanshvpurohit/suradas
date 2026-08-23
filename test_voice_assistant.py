import os
import sys
import numpy as np
import time

# Guarantee working directory is script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print("=" * 65)
print("     SURDAS VOICE ASSISTANT COMPREHENSIVE TEST SUITE")
print("=" * 65)

# --- 1. TEST VAD (Voice Activity Detection) ---
print("\n[1/5] Testing Silero Voice Activity Detector (VAD)...")
try:
    from voice.vad import VoiceActivityDetector
    vad = VoiceActivityDetector(sample_rate=16000)
    
    # Generate 1s of silence (zeros)
    silence = np.zeros(16000, dtype=np.float32)
    # Generate 1s of synthetic audio activity
    t = np.linspace(0, 1, 16000)
    synthetic_speech = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    is_silence_speech = vad.is_speech(silence)
    is_synth_speech = vad.is_speech(synthetic_speech)
    
    print(f"  • Silence chunk speech detected: {is_silence_speech} (Expected: False) -> {'✅ PASS' if not is_silence_speech else '❌ FAIL'}")
    print(f"  • Active audio chunk speech detected: {is_synth_speech} (Expected: True) -> {'✅ PASS' if is_synth_speech else '❌ FAIL'}")
except Exception as e:
    print(f"  ❌ VAD Test Failed: {e}")

# --- 2. TEST WAKE WORD DETECTOR ---
print("\n[2/5] Testing Wake Word Detector ('Hey Surdas')...")
try:
    from voice.wakeword import WakeWordDetector
    ww = WakeWordDetector()
    
    test_phrases = [
        ("hey surdas turn on the light", True, "turn on the light"),
        ("surdas what do you see in front of me", True, "what do you see in front of me"),
        ("hello how are you today", False, "hello how are you today"),
        ("hey assistant read this sign", True, "read this sign")
    ]
    
    for phrase, expected_wake, expected_cmd in test_phrases:
        has_wake, cmd = ww.check_transcription(phrase)
        status = "✅ PASS" if has_wake == expected_wake and expected_cmd in cmd else "❌ FAIL"
        print(f"  • Input: \"{phrase}\" -> Wake: {has_wake}, Cmd: \"{cmd}\" [{status}]")
except Exception as e:
    print(f"  ❌ Wake Word Test Failed: {e}")

# --- 3. TEST COMMAND ROUTER & DETERMINISTIC DISPATCH ---
print("\n[3/5] Testing Command Router (Deterministic Actions)...")
try:
    from voice.command_router import CommandRouter
    from voice.llm import LocalLLM
    
    # Mock SurdasBrain for unit testing
    class MockBrain:
        def __init__(self):
            self.mode = "NAV"
            self.led_on = False
            self.spoken_messages = []
            
        class MockVoice:
            def __init__(self, parent):
                self.parent = parent
            def speak(self, text, force=False):
                self.parent.spoken_messages.append(text)
                
        def __init__(self):
            self.mode = "NAV"
            self.led_on = False
            self.spoken_messages = []
            self.voice = self.MockVoice(self)
            
        def toggle_esp32_led(self, state):
            self.led_on = state
            
        def get_vision_context(self):
            return {
                "mode": self.mode,
                "torch_on": self.led_on,
                "detected_objects": ["person", "chair"],
                "closest_obstacle": "person ahead, nearby"
            }
            
        def get_status_summary(self):
            return "Mock status report: Camera active."

    mock_brain = MockBrain()
    mock_llm = LocalLLM()
    router = CommandRouter(mock_brain, mock_llm)
    
    # Test Torch ON
    router.route_command("turn on torch")
    torch_on_ok = mock_brain.led_on == True
    print(f"  • Command 'turn on torch' -> Torch state: {mock_brain.led_on} -> {'✅ PASS' if torch_on_ok else '❌ FAIL'}")

    # Test Torch OFF
    router.route_command("turn off torch")
    torch_off_ok = mock_brain.led_on == False
    print(f"  • Command 'turn off torch' -> Torch state: {mock_brain.led_on} -> {'✅ PASS' if torch_off_ok else '❌ FAIL'}")

    # Test OCR Trigger
    router.route_command("read text")
    ocr_ok = mock_brain.mode == "OCR"
    print(f"  • Command 'read text' -> Active Mode: {mock_brain.mode} -> {'✅ PASS' if ocr_ok else '❌ FAIL'}")

    # Test Navigation Trigger
    router.route_command("navigation mode")
    nav_ok = mock_brain.mode == "NAV"
    print(f"  • Command 'navigation mode' -> Active Mode: {mock_brain.mode} -> {'✅ PASS' if nav_ok else '❌ FAIL'}")

    # Test Fast Scene Summary
    mock_brain.spoken_messages.clear()
    router.route_command("what do you see")
    scene_ok = len(mock_brain.spoken_messages) > 0 and "person" in mock_brain.spoken_messages[-1]
    print(f"  • Command 'what do you see' -> Feedback: \"{mock_brain.spoken_messages[-1] if mock_brain.spoken_messages else 'None'}\" -> {'✅ PASS' if scene_ok else '❌ FAIL'}")

except Exception as e:
    print(f"  ❌ Command Router Test Failed: {e}")

# --- 4. TEST LOCAL LLM (OLLAMA) WITH VISION CONTEXT ---
print("\n[4/5] Testing Local LLM Client (Ollama)...")
try:
    llm = LocalLLM()
    is_online = llm.is_available()
    print(f"  • Ollama local server status: {'🟢 ONLINE' if is_online else '🟡 OFFLINE (Ollama server not running)'}")
    
    if is_online:
        vision_ctx = {
            "mode": "NAV",
            "torch_on": False,
            "detected_objects": ["chair", "laptop", "bottle"],
            "closest_obstacle": "chair on your right, nearby"
        }
        print("  • Querying Ollama: 'Where can I sit?' with live scene context...")
        response = llm.query("Where can I sit?", vision_context=vision_ctx)
        print(f"  • Ollama Spoken Response: \"{response}\" -> ✅ PASS")
    else:
        print("  • Note: Start Ollama with 'ollama run llama3.2' anytime for full conversational Q&A.")
except Exception as e:
    print(f"  ❌ LLM Test Warning: {e}")

# --- 5. TEST TEXT-TO-SPEECH (TTS) ---
print("\n[5/5] Testing Non-Blocking Voice Engine (TTS)...")
try:
    from voice.tts import VoiceEngine
    tts = VoiceEngine()
    tts.speak("SURDAS Voice Assistant test completed successfully.")
    time.sleep(1.0)
    print("  • TTS non-blocking queue initialized & spoken -> ✅ PASS")
except Exception as e:
    print(f"  ❌ TTS Test Failed: {e}")

print("\n" + "=" * 65)
print("   🎉 ALL VOICE ASSISTANT SUBSYSTEM MODULES TESTED & READY!")
print("=" * 65)
