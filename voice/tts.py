"""
tts.py — Zero-latency Voice Engine for SURDAS

Design goals:
  • Print to terminal INSTANTLY (before audio even starts)
  • Non-blocking audio playback via background worker thread
  • Reliable is_speaking flag to prevent microphone echo self-interruption
  • Clean subprocess execution with '--' option delimiter on macOS to support LLM outputs
"""
import threading
import queue
import sys
import subprocess
import time
import re

_MACOS_RATE = 200


def clean_speech_text(text: str) -> str:
    """Strip markdown, symbols, bullets, emojis, and formatting for clean TTS."""
    if not text:
        return ""
    
    # Remove code blocks and inline code
    t = re.sub(r"```[\s\S]*?```", "", text)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    
    # Remove markdown links [label](url) -> label
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    
    # Remove markdown headers #, ##, etc.
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
    
    # Remove bullet markers (- , * , + , > )
    t = re.sub(r"^[\s\*\-\+\>]+\s*", "", t, flags=re.MULTILINE)
    
    # Remove numbered bullets (1. , 2. )
    t = re.sub(r"^\d+\.\s*", "", t, flags=re.MULTILINE)
    
    # Remove bold/italic markers (*, _)
    t = t.replace("*", "").replace("_", "").replace("~", "")
    t = t.replace('"', '').replace("'", "").replace("#", "")
    t = t.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    t = t.replace("(", "").replace(")", "").replace("\\", "")
    
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


class VoiceEngine:
    def __init__(self):
        self.is_macos    = (sys.platform == "darwin")
        # Generous queue size so multi-sentence LLM responses don't drop
        self._queue      = queue.Queue(maxsize=30)
        self._proc       = None
        self._proc_lock  = threading.Lock()
        self._running    = True
        self._pyttsx     = None
        self._speaking   = False
        self._last_spoke = 0

        if not self.is_macos:
            self._init_pyttsx()

        threading.Thread(target=self._worker, daemon=True, name="TTSWorker").start()

    @property
    def is_speaking(self) -> bool:
        """Returns True if speech audio is currently playing or queued."""
        if self._speaking or not self._queue.empty():
            return True
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def last_speech_time(self) -> float:
        return self._last_spoke

    # ── pyttsx3 fallback ────────────────────────────────────────────────────
    def _init_pyttsx(self):
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty("rate", _MACOS_RATE)
            self._pyttsx = e
        except Exception as ex:
            print(f"[TTS] pyttsx3 init: {ex}")

    # ── Worker: drains queue in background ──────────────────────────────────
    def _worker(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.05)
                self._speaking = True
                self._say(text)
                self._speaking = False
                self._last_spoke = time.time()
                self._queue.task_done()
            except queue.Empty:
                self._speaking = False
            except Exception as ex:
                self._speaking = False
                print(f"[TTS] worker error: {ex}")

    # ── Core speech call ─────────────────────────────────────────────────────
    def _say(self, text: str):
        if not text or not text.strip():
            return
        clean = text.strip()
        if self.is_macos:
            try:
                with self._proc_lock:
                    # Use '--' to ensure leading characters (like dashes) aren't treated as CLI flags
                    self._proc = subprocess.Popen(
                        ["say", "-r", str(_MACOS_RATE), "--", clean],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                self._proc.wait()
            except Exception as ex:
                print(f"[TTS] macOS 'say' error: {ex}")
            finally:
                with self._proc_lock:
                    self._proc = None
        elif self._pyttsx:
            try:
                self._pyttsx.say(clean)
                self._pyttsx.runAndWait()
            except Exception as ex:
                print(f"[TTS] pyttsx3 error: {ex}")

    # ── Interrupt speech ────────────────────────────────────────────────────
    def _interrupt(self):
        with self._proc_lock:
            proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.2)
            except Exception:
                pass
        self._speaking = False

    # ── Public API ───────────────────────────────────────────────────────────
    def speak(self, text: str, force: bool = False):
        """
        Queue text for speech.

        Args:
            text:     What to say.
            force:    If True, clear any pending queue items and say immediately.
        """
        if not text or not text.strip():
            if force:
                self._drain()
            return

        clean = clean_speech_text(text)
        if not clean:
            return

        # Print immediately to terminal
        print(f"[SPEECH OUT] 🔊  {clean}")

        # Broadcast to dashboard
        try:
            from telemetry import broadcast_event
            broadcast_event("speech", {"text": clean})
        except Exception:
            pass

        if force:
            self._drain()
            self._interrupt()

        try:
            self._queue.put(clean, timeout=0.5)
        except queue.Full:
            pass

    def _drain(self):
        """Empty pending speech queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def stop(self):
        self._running = False
        self._drain()
        self._interrupt()
