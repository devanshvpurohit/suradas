"""
audio_device.py — Universal Audio Input Manager for SURDAS
Optimized for Bluetooth TWS Earbuds (AirPods, Galaxy Buds, boAt, Sony, JBL, etc.)
and Built-in / USB Microphones.

Key features:
  • Auto-detects connected Bluetooth TWS / Headset devices
  • Records at device's native hardware sample rate & resamples cleanly to 16kHz
  • Adaptive Software AGC (Automatic Gain Control) to boost quiet TWS microphones
  • Dynamic background noise floor tracking
"""
import numpy as np
import sys
from typing import Optional, Tuple, List, Dict


# Keywords to identify Bluetooth / TWS headsets
TWS_KEYWORDS = [
    "airpod", "buds", "earbud", "bluetooth", "wireless", "headset",
    "hands-free", "tws", "wh-", "wf-", "freebuds", "tune", "boat",
    "realme", "galaxy", "noise", "boult", "sony", "jbl", "oneplus"
]


def list_input_devices() -> List[Dict]:
    """Returns a list of all available audio input devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                name = d.get("name", "Unknown")
                is_tws = any(k in name.lower() for k in TWS_KEYWORDS)
                input_devs.append({
                    "index": i,
                    "name": name,
                    "samplerate": int(d.get("default_samplerate", 16000)),
                    "channels": int(d.get("max_input_channels", 1)),
                    "is_tws": is_tws,
                })
        return input_devs
    except Exception as e:
        print(f"[AUDIO] Error querying devices: {e}")
        return []


def find_best_input_device(user_preference: Optional[str] = None) -> Tuple[Optional[int], str, int, bool]:
    """
    Find the best input device.
    Returns: (device_index, device_name, native_samplerate, is_tws)
    """
    devices = list_input_devices()
    if not devices:
        return None, "Default Microphone", 16000, False

    # 1. User specified an index or name keyword
    if user_preference:
        pref = str(user_preference).strip().lower()
        # Check if integer index
        if pref.isdigit():
            idx = int(pref)
            for d in devices:
                if d["index"] == idx:
                    return d["index"], d["name"], d["samplerate"], d["is_tws"]
        # Match by name keyword
        for d in devices:
            if pref in d["name"].lower():
                return d["index"], d["name"], d["samplerate"], d["is_tws"]

    # 2. Auto-detect any active Bluetooth / TWS device
    for d in devices:
        if d["is_tws"]:
            return d["index"], d["name"], d["samplerate"], True

    # 3. Fallback to OS default input device
    try:
        import sounddevice as sd
        default_idx = sd.default.device[0]
        for d in devices:
            if d["index"] == default_idx:
                return d["index"], d["name"], d["samplerate"], d["is_tws"]
    except Exception:
        pass

    # 4. First available input device
    first = devices[0]
    return first["index"], first["name"], first["samplerate"], first["is_tws"]


class AudioStreamProcessor:
    """
    Real-time audio signal processor for TWS & built-in mics:
      • Native-to-16kHz high-quality resampling
      • Automatic Gain Control (AGC) with soft clipping protection
      • Noise-floor tracking
    """
    def __init__(self, input_rate: int = 16000, target_rate: int = 16000, gain_boost: float = 1.0):
        self.input_rate = input_rate
        self.target_rate = target_rate
        self.gain_boost = max(0.5, float(gain_boost))
        self.noise_floor = 0.005
        self.smooth_rms = 0.01

    def process_chunk(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Processes a raw input chunk (float32):
        1. Mono extraction
        2. Resample to target_rate (16kHz) if needed
        3. Apply AGC software gain
        """
        if raw_audio is None or len(raw_audio) == 0:
            return np.zeros(0, dtype=np.float32)

        # Ensure 1D float32
        if raw_audio.ndim > 1:
            audio = raw_audio[:, 0]
        else:
            audio = raw_audio

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        # 1. Resample to 16000 Hz if hardware rate differs
        if self.input_rate != self.target_rate and len(audio) > 1:
            target_len = int(round(len(audio) * (self.target_rate / self.input_rate)))
            if target_len > 0:
                indices = np.linspace(0, len(audio) - 1, target_len)
                audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # 2. Track RMS & Noise Floor
        current_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
        self.smooth_rms = 0.9 * self.smooth_rms + 0.1 * current_rms

        if current_rms < self.noise_floor * 2.0:
            # Update background ambient noise floor slowly
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * current_rms

        # 3. Apply Adaptive Gain
        # Boost quiet microphones (TWS earbuds) cleanly
        effective_gain = self.gain_boost
        if self.smooth_rms < 0.03 and current_rms > self.noise_floor * 1.5:
            # Dynamic quiet speech boost (up to 2.5x extra for whispered/quiet TWS audio)
            boost_factor = min(2.5, 0.05 / (self.smooth_rms + 1e-4))
            effective_gain *= boost_factor

        boosted = audio * effective_gain

        # 4. Soft Limiter (tanh) to prevent harsh digital clipping
        if np.max(np.abs(boosted)) > 0.95:
            boosted = np.tanh(boosted)

        return boosted.astype(np.float32)
