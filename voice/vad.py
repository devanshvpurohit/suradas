import numpy as np
import torch
import os

class VoiceActivityDetector:
    """
    Voice Activity Detector using Silero VAD with fallback to RMS energy detection.
    """
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silero_model = None
        self._init_silero()

    def _init_silero(self):
        try:
            # Try loading cached silero VAD locally from disk first (100% offline)
            cached_dir = os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")
            if os.path.isdir(cached_dir):
                model, _ = torch.hub.load(
                    repo_or_dir=cached_dir,
                    model='silero_vad',
                    source='local',
                    trust_repo=True
                )
            else:
                model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True
                )
            self.silero_model = model
            print("[VAD] Silero VAD initialized successfully (offline mode).")
        except Exception as e:
            print(f"[VAD] Silero VAD initialization fallback ({e}). Using energy-based VAD.")
            self.silero_model = None

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Determines if an audio chunk (16kHz float32 or int16) contains speech.
        """
        if len(audio_chunk) == 0:
            return False

        # Convert to float32 normalized [-1.0, 1.0] if int16
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        if self.silero_model is not None:
            try:
                tensor = torch.from_numpy(audio_float)
                # Silero expects shape (1, N) or (N,)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                with torch.no_grad():
                    speech_prob = self.silero_model(tensor, self.sample_rate).item()
                return speech_prob > self.threshold
            except Exception:
                pass

        # Fallback: RMS Energy detection
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms > 0.02
