"""
stt.py — Low-latency Speech-to-Text for SURDAS

Tuning for speed:
  • beam_size=1         — greedy decode, ~3× faster than beam_size=5
  • vad_filter=True     — skips silence at boundaries, shorter audio fed to model
  • temperature=0.0     — deterministic, no sampling overhead
  • condition_on_prev=False — don't carry previous context (saves memory bandwidth)
  • language="en"       — skip language detection step
"""
import os
import numpy as np

# Ensure HuggingFace Hub operates offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class SpeechToText:
    def __init__(self, model_size: str = "tiny.en"):
        self.model_size  = model_size
        self.engine_type = None
        self.model       = None
        self._init_model()

    def _init_model(self):
        # 1. faster-whisper (CTranslate2 — fastest)
        try:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading faster-whisper ({self.model_size}) [offline]...")
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    num_workers=1,
                    local_files_only=True,
                )
            except Exception:
                # Fallback to standard init if local_files_only argument format varies
                self.model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    num_workers=1,
                )
            self.engine_type = "faster-whisper"
            print("[STT] ✅ faster-whisper ready (offline mode).")
            return
        except Exception as e:
            print(f"[STT] faster-whisper unavailable: {e}")

        # 2. openai-whisper fallback
        try:
            import whisper
            print(f"[STT] Loading openai-whisper ({self.model_size}) [offline]...")
            self.model = whisper.load_model(self.model_size)
            self.engine_type = "whisper"
            print("[STT] ✅ openai-whisper ready (offline mode).")
        except Exception as e:
            print(f"[STT] whisper unavailable: {e}")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        if self.model is None or len(audio_data) == 0:
            return ""

        # Normalise to float32 [-1, 1]
        audio = (audio_data.astype(np.float32) / 32768.0
                 if audio_data.dtype == np.int16
                 else audio_data.astype(np.float32))

        try:
            if self.engine_type == "faster-whisper":
                segments, _ = self.model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,           # greedy — fastest
                    temperature=0.0,
                    vad_filter=True,       # skip silence → shorter effective audio
                    vad_parameters=dict(
                        min_silence_duration_ms=300,   # tight silence window
                        threshold=0.4,
                    ),
                    condition_on_previous_text=False,  # no context carryover overhead
                    word_timestamps=False,             # skip word-level alignment
                )
                return " ".join(s.text for s in segments).strip()

            elif self.engine_type == "whisper":
                result = self.model.transcribe(audio, language="en", fp16=False)
                return result.get("text", "").strip()

        except Exception as e:
            print(f"[STT] Transcription error: {e}")

        return ""
