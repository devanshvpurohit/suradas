import os
import numpy as np

class WakeWordDetector:
    """
    Wake word detector supporting openWakeWord with fallback keyword matching.
    Target wake word: 'Hey Surdas' / 'Surdas'
    """
    def __init__(self, target_phrases=None, model_path=None):
        if target_phrases is None:
            self.target_phrases = ["hey surdas", "surdas", "hey assistant", "surdas assistant"]
        else:
            self.target_phrases = [p.lower() for p in target_phrases]

        self.oww_model = None
        self._init_openwakeword(model_path)

    def _init_openwakeword(self, model_path):
        try:
            import openwakeword
            from openwakeword.model import Model
            if model_path and os.path.exists(model_path):
                self.oww_model = Model(wakeword_models=[model_path])
                print(f"[WAKEWORD] Loaded custom openWakeWord model from: {model_path}")
            else:
                # Load default pre-trained models
                self.oww_model = Model(wakeword_models=["hey_jarvis", "alexa"])
                print("[WAKEWORD] Loaded default openWakeWord models ('hey_jarvis', 'alexa').")
        except Exception as e:
            # Fallback to post-STT keyword recognition
            self.oww_model = None
            print("[WAKEWORD] openWakeWord not loaded, using text-based wake word matching for 'Hey Surdas'.")

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process raw 16kHz audio chunk through openWakeWord if available.
        """
        if self.oww_model is not None:
            try:
                # Convert float32 [-1, 1] to int16 if needed
                if audio_chunk.dtype == np.float32:
                    audio_int16 = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_chunk
                
                prediction = self.oww_model.predict(audio_int16)
                for model_name, score in prediction.items():
                    if score > 0.5:
                        print(f"[WAKEWORD] Wake word '{model_name}' triggered (score: {score:.2f})")
                        return True
            except Exception:
                pass
        return False

    def check_transcription(self, text: str) -> tuple[bool, str]:
        """
        Checks if transcribed text contains the wake word, and returns (is_wake_word_present, remaining_command).
        """
        cleaned = text.strip().lower()
        if not cleaned:
            return False, ""

        for phrase in self.target_phrases:
            if phrase in cleaned:
                # Strip the wake phrase from the actual command
                idx = cleaned.find(phrase)
                command = cleaned[idx + len(phrase):].strip(" ,.?!")
                return True, command

        return False, cleaned
