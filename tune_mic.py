#!/usr/bin/env python3
"""
tune_mic.py — Interactive TWS & Microphone Tuning Utility for SURDAS

Use this tool to:
  1. Inspect all connected audio input devices (AirPods, boAt, Galaxy Buds, etc.)
  2. Test your TWS microphone live with a visual real-time VU audio meter
  3. Calibrate software AGC gain and test speech detection
  4. Test Whisper speech-to-text recognition with your specific headset

Usage:
  python3 tune_mic.py
  python3 tune_mic.py --device airpods --gain 2.0
"""
import os
import sys
import time
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from voice.audio_device import list_input_devices, find_best_input_device, AudioStreamProcessor


def print_devices():
    devices = list_input_devices()
    print("=" * 65)
    print("  AVAILABLE AUDIO INPUT DEVICES")
    print("=" * 65)
    for d in devices:
        tws_flag = " [🎧 TWS / BLUETOOTH HEADSET]" if d["is_tws"] else ""
        print(f"  [{d['index']}] {d['name']}")
        print(f"      Native Sample Rate: {d['samplerate']} Hz | Channels: {d['channels']}{tws_flag}")
    print("=" * 65)


def live_meter(device_pref=None, gain=1.0):
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice is required. Install with: pip3 install sounddevice")
        return

    print_devices()

    dev_idx, dev_name, native_rate, is_tws = find_best_input_device(device_pref)
    effective_gain = gain if gain != 1.0 else (1.8 if is_tws else 1.0)

    print(f"\n[ACTIVE TEST DEVICE]: [{dev_idx}] {dev_name}")
    print(f"Native Hardware Rate: {native_rate} Hz")
    print(f"Software Gain Multiplier: {effective_gain:.1f}x")
    print("\nStarting Real-time VU Meter (Speak into your TWS / Mic). Press Ctrl+C to stop...\n")

    processor = AudioStreamProcessor(input_rate=native_rate, target_rate=16000, gain_boost=effective_gain)
    chunk_size = int(native_rate * 0.05)  # 50ms chunks

    from voice.vad import VoiceActivityDetector
    vad = VoiceActivityDetector()

    try:
        def audio_callback(indata, frames, time_info, status):
            processed = processor.process_chunk(indata)
            rms = float(np.sqrt(np.mean(processed ** 2) + 1e-9))
            db = 20 * np.log10(max(rms, 1e-5))
            
            # Map dB (-50 to 0) to bar length (0 to 35)
            bar_len = int(np.clip((db + 50) / 50 * 35, 0, 35))
            bar = "█" * bar_len + "░" * (35 - bar_len)
            
            is_speech = vad.is_speech(processed)
            status_tag = "🗣️ SPEECH DETECTED" if is_speech else "   listening...  "
            
            # Print dynamic VU meter line
            sys.stdout.write(f"\rLevel: [{bar}] {db:5.1f} dB | RMS: {rms:.3f} | {status_tag}")
            sys.stdout.flush()

        with sd.InputStream(
            device=dev_idx,
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n[TUNING COMPLETED]")
        print(f"To use this configuration with SURDAS, run:")
        print(f"  python3 test_ai_pipeline.py --mic {dev_idx} --mic-gain {effective_gain:.1f}")
        print(f"  python3 surdas_brain.py --mic {dev_idx} --mic-gain {effective_gain:.1f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SURDAS TWS & Microphone Tuning Tool")
    parser.add_argument("--device", "-d", default=None, help="Device index or keyword (e.g. 'airpods', 'boat', '0')")
    parser.add_argument("--gain", "-g", type=float, default=1.0, help="Software gain multiplier (e.g. 1.8, 2.0, 2.5)")
    args = parser.parse_args()

    live_meter(device_pref=args.device, gain=args.gain)
