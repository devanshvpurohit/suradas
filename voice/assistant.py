"""
assistant.py — Low-latency Voice Assistant Loop with TWS Tuning & Conversational Follow-up

Features:
  • Universal Mic & TWS Earbud Support (AirPods, boAt, Galaxy Buds, Sony, etc.)
  • Automatic hardware sample-rate discovery & native resampling to 16kHz
  • Adaptive Software AGC (Automatic Gain Control) to boost quiet TWS microphones
  • Conversational Follow-up Mode: 20-second active window after any query
  • Acoustic Echo Cancellation: ignores speaker output while AI speaks
  • Clean audio buffer flushing between utterances
"""
import numpy as np
import threading
import time
import queue
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from surdas_brain import SurdasBrain

from voice.vad import VoiceActivityDetector
from voice.wakeword import WakeWordDetector
from voice.stt import SpeechToText
from voice.llm import LocalLLM
from voice.command_router import CommandRouter
from voice.audio_device import find_best_input_device, AudioStreamProcessor


class VoiceAssistant:
    """Continuous mic listener: VAD → STT (faster-whisper) → CommandRouter."""

    SILENCE_MS             = 420    # ms of quiet before utterance is considered done
    MIN_CLIP_SEC           = 0.35   # ignore clips shorter than this
    FOLLOW_UP_WINDOW_SEC   = 20.0   # seconds to keep conversation open without wake word

    def __init__(self, brain: "SurdasBrain", sample_rate: int = 16000, mic_device: Optional[str] = None, mic_gain: float = 1.0):
        self.brain       = brain
        self.sample_rate = sample_rate
        self.chunk_ms    = 30
        self.chunk_size  = int(sample_rate * self.chunk_ms / 1000)
        self.mic_device  = mic_device
        self.mic_gain    = mic_gain

        self.vad      = VoiceActivityDetector(sample_rate=sample_rate)
        self.wakeword = WakeWordDetector()
        self.stt      = SpeechToText()

        # Reuse brain's LLM instance — one Ollama connection shared
        self.llm    = getattr(brain, "llm", None) or LocalLLM()
        self.router = CommandRouter(brain, self.llm)

        self._running                = False
        self._audio_q                = queue.Queue()
        self._processing             = False
        self._last_interaction_time  = 0.0
        self._processor              = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        threading.Thread(target=self._mic_loop, daemon=True, name="MicLoop").start()
        print("[VOICE] 🎤 Mic listener active — say 'Hey Surdas' or ask any question.")

    def stop(self):
        self._running = False

    # ── Sounddevice callback (runs in audio thread) ──────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if self._running and self._processor is not None:
            processed = self._processor.process_chunk(indata)
            self._audio_q.put(processed)

    def _flush_audio_queue(self):
        """Discard any accumulated audio chunks (e.g. from during TTS playback)."""
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    # ── Main mic loop ────────────────────────────────────────────────────────

    def _mic_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[VOICE] sounddevice not installed — voice input disabled.")
            return

        # 1. Resolve Best Audio Device (TWS auto-detect or user specified)
        dev_idx, dev_name, native_rate, is_tws = find_best_input_device(self.mic_device)

        # Apply TWS optimized gain (default 1.8x boost for quiet Bluetooth earbuds if not specified)
        effective_gain = self.mic_gain if self.mic_gain != 1.0 else (1.8 if is_tws else 1.0)
        tws_tag = " [🎧 TWS / Bluetooth Headset Mode]" if is_tws else ""
        print(f"[VOICE] 🎙️ Mic: {dev_name} (Index {dev_idx} | {native_rate}Hz | Gain {effective_gain:.1f}x){tws_tag}")

        self._processor = AudioStreamProcessor(
            input_rate=native_rate,
            target_rate=self.sample_rate,
            gain_boost=effective_gain
        )
        native_chunk_size = int(native_rate * self.chunk_ms / 1000)

        try:
            stream = sd.InputStream(
                device=dev_idx,
                samplerate=native_rate,
                channels=1,
                dtype="float32",
                blocksize=native_chunk_size,
                callback=self._audio_callback,
            )
        except Exception as e:
            print(f"[VOICE] Falling back to default system mic due to ({e})...")
            try:
                self._processor = AudioStreamProcessor(input_rate=self.sample_rate, target_rate=self.sample_rate, gain_boost=effective_gain)
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                )
            except Exception as e2:
                print(f"[VOICE] Could not open microphone ({e2}). Check macOS microphone permission.")
                return

        max_silence = int(self.SILENCE_MS / self.chunk_ms)

        with stream:
            recording      = False
            speech_buf     = []
            silence_chunks = 0
            was_speaking   = False

            while self._running:
                try:
                    chunk = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Echo suppression: do not listen to speaker output while AI is actively speaking
                is_ai_speaking = (
                    self.brain.voice.is_speaking
                    or (time.time() - self.brain.voice.last_speech_time < 0.35)
                )

                if is_ai_speaking:
                    was_speaking = True
                    recording = False
                    speech_buf = []
                    silence_chunks = 0
                    continue

                # If AI just finished speaking, flush any lingering speaker echo from the queue
                if was_speaking:
                    was_speaking = False
                    self._flush_audio_queue()
                    continue

                is_speech = self.vad.is_speech(chunk)

                if is_speech:
                    if not recording:
                        recording      = True
                        speech_buf     = []
                        silence_chunks = 0
                        print("\n[VOICE] 🔴 Listening…", end="", flush=True)
                    speech_buf.append(chunk)

                elif recording:
                    speech_buf.append(chunk)
                    silence_chunks += 1

                    if silence_chunks >= max_silence:
                        recording      = False
                        clip           = np.concatenate(speech_buf)
                        speech_buf     = []
                        silence_chunks = 0
                        print()   # newline after 🔴 Listening…

                        if len(clip) >= self.sample_rate * self.MIN_CLIP_SEC:
                            threading.Thread(
                                target=self._process_clip,
                                args=(clip,),
                                daemon=True,
                            ).start()

    # ── Utterance processing ─────────────────────────────────────────────────

    def _process_clip(self, audio: np.ndarray):
        self._processing = True
        try:
            t0 = time.time()
            print("[VOICE] ⏳ Transcribing…", end="", flush=True)
            text = self.stt.transcribe(audio, sample_rate=self.sample_rate)
            dt = time.time() - t0
            print(f" ({dt*1000:.0f} ms)")

            if not text or not text.strip():
                return

            print(f"[VOICE] 💬 Heard: \"{text}\"")
            self._route(text)

        except Exception as e:
            print(f"[VOICE] Processing error: {e}")
        finally:
            self._processing = False

    def _route(self, text: str):
        """Check wake word or active conversation window, then route command."""
        has_wake, command = self.wakeword.check_transcription(text)

        if has_wake:
            print(f"[VOICE] ✨ Wake word matched! Command: \"{command}\"")
            self._last_interaction_time = time.time()
            if command and len(command.strip()) > 1:
                self.router.route_command(command)
            else:
                self.brain.voice.speak("Yes?", force=True)
            return

        # Direct-command fast path triggers
        lower = text.lower().strip()
        direct_triggers = [
            # SURDAS hardware & modes
            "turn on light", "turn off light", "turn on torch", "turn off torch",
            "read text", "read this", "navigation mode", "what do you see",
            "stop", "describe", "quiet", "silence",
            # App launcher
            "open ", "close ", "launch ", "quit ",
            # System controls
            "volume up", "volume down", "mute", "screenshot", "lock screen",
            "what time", "what's the time", "whats the time", "time now", "the time",
            "what date", "what's the date", "whats the date", "what day", "today's date", "todays date",
            # Model management
            "list models", "available models", "switch to ", "use model",
            "current model", "which model",
            # Questions
            "what is", "what are", "what was", "what were", "what does",
            "how do", "how to", "how does", "how can", "how is", "how much", "how many", "how tall", "how far",
            "tell me", "can you", "who is", "who was", "who are", "who won",
            "where is", "where was", "where are", "why is", "why does", "why are",
            "explain", "define", "help me", "search",
        ]

        in_conversation = (time.time() - self._last_interaction_time < self.FOLLOW_UP_WINDOW_SEC)

        if in_conversation or any(k in lower for k in direct_triggers) or lower in ("time", "time now", "date", "today"):
            self._last_interaction_time = time.time()
            if in_conversation and not any(k in lower for k in direct_triggers):
                print(f"[VOICE] 💬 Conversational follow-up: \"{text}\"")
            else:
                print(f"[VOICE] ⚡ Direct command: \"{text}\"")
            self.router.route_command(text)
        else:
            print(f"[VOICE] (Ignored: \"{text}\" — say 'Hey Surdas' or ask a question)")
