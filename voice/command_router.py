"""
command_router.py — SURDAS Voice Command Dispatcher

Priority order:
  1. App launch / close commands          → app_launcher (instant, no LLM)
  2. SURDAS hardware / mode commands      → deterministic handlers (instant)
  3. Vision description queries           → fast heuristic (instant)
  4. General knowledge / conversation     → LocalLLM via Ollama (streamed)
"""
import re
import subprocess
from typing import TYPE_CHECKING

from voice.app_launcher import open_app, close_app, get_app_list

if TYPE_CHECKING:
    from surdas_brain import SurdasBrain
    from voice.llm import LocalLLM


# ── helpers ──────────────────────────────────────────────────────────────────

def _contains(text: str, *phrases) -> bool:
    return any(p in text for p in phrases)


def _extract_app_name(text: str, trigger: str) -> str:
    """Pull out the app name that follows a trigger word/phrase."""
    idx = text.find(trigger)
    if idx == -1:
        return ""
    after = text[idx + len(trigger):].strip()
    # Strip filler words
    for filler in ("please", "now", "for me", "quickly", "up"):
        after = after.replace(filler, "").strip()
    return after.strip()


# ── main router ──────────────────────────────────────────────────────────────

class CommandRouter:
    """
    Routes transcribed voice text to the correct handler.
    Every public method that handles a category returns True on success.
    """

    def __init__(self, brain: "SurdasBrain", llm: "LocalLLM"):
        self.brain = brain
        self.llm = llm

    # ─────────────────────────────────────────────────────────────────────────
    def route_command(self, raw_text: str) -> bool:
        text = raw_text.strip().lower()
        if not text:
            return False

        print(f"[ROUTER] ▶ '{text}'")

        # ── 1. APP LAUNCHER ───────────────────────────────────────────────────
        if self._handle_app_commands(text, raw_text):
            return True

        # ── 2. HARDWARE / MODE COMMANDS ───────────────────────────────────────
        if self._handle_hardware_commands(text):
            return True

        # ── 3. FAST VISION DESCRIPTION ────────────────────────────────────────
        if self._handle_vision_query(text):
            return True

        # ── 4. MODEL MANAGEMENT ───────────────────────────────────────────────
        if self._handle_model_commands(text):
            return True

        # ── 5. GENERAL KNOWLEDGE / CONVERSATION (Ollama LLM) ─────────────────
        self._handle_llm_query(raw_text)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Category 1 — App commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_app_commands(self, text: str, raw_text: str) -> bool:

        # OPEN triggers
        open_triggers = [
            "open ", "launch ", "start ", "run ",
            "open up ", "open the ", "launch the ", "can you open ",
        ]
        for trigger in open_triggers:
            if trigger in text:
                app_name = _extract_app_name(text, trigger)
                if not app_name:
                    continue
                success, msg = open_app(app_name)
                self.brain.voice.speak(msg)
                return True

        # CLOSE / QUIT triggers
        close_triggers = [
            "close ", "quit ", "exit ", "kill ",
            "close the ", "quit the ",
        ]
        for trigger in close_triggers:
            if trigger in text:
                app_name = _extract_app_name(text, trigger)
                if not app_name:
                    continue
                success, msg = close_app(app_name)
                self.brain.voice.speak(msg)
                return True

        # "what apps can you open?"
        if _contains(text, "what apps", "which apps", "list apps", "what can you open", "supported apps"):
            self.brain.voice.speak(get_app_list())
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 2 — Hardware / SURDAS mode commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_hardware_commands(self, text: str) -> bool:

        # Navigation mode
        if _contains(text, "navigation mode", "start navigation", "nav mode",
                     "resume navigation", "go to navigation"):
            self.brain.mode = "NAV"
            self.brain.voice.speak("Navigation mode active.")
            return True

        # OCR / Read text mode
        if _contains(text, "read text", "read this", "read sign", "read document",
                     "scan text", "ocr mode", "start reading", "what does it say"):
            self.brain.mode = "OCR"
            return True

        # Flashlight ON
        if _contains(text, "turn on light", "turn on torch", "light on", "torch on",
                     "flash on", "flashlight on", "turn on flashlight"):
            self.brain.toggle_esp32_led(True)
            self.brain.voice.speak("Flashlight on.")
            return True

        # Flashlight OFF
        if _contains(text, "turn off light", "turn off torch", "light off", "torch off",
                     "flash off", "flashlight off", "turn off flashlight"):
            self.brain.toggle_esp32_led(False)
            self.brain.voice.speak("Flashlight off.")
            return True

        # Stop / silence
        if text in ("stop", "quiet", "silence", "be quiet", "shut up", "mute", "cancel"):
            self.brain.voice.speak("", force=True)
            return True

        # System status
        if _contains(text, "system status", "connection status", "your status", "status report"):
            self.brain.voice.speak(self.brain.get_status_summary())
            return True

        # Time
        time_patterns = [
            r"\b(what('s|s| is)? (the )?time( now)?)\b",
            r"\b(what time( is it)?( now)?)\b",
            r"\b(tell me (the )?time)\b",
            r"\b(current time)\b",
            r"\b(time right now)\b",
            r"\b(time please)\b",
            r"^time$",
            r"^time now$",
        ]
        if any(re.search(pat, text) for pat in time_patterns) or _contains(text, "whats the time", "what's the time", "what is the time", "what time is it", "tell me the time", "current time", "the time now"):
            import datetime
            now = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
            self.brain.voice.speak(f"The time is {now}.", force=True)
            return True

        # Date
        date_patterns = [
            r"\b(what('s|s| is)? (the |today('s)? )?date( today| now)?)\b",
            r"\b(what day( is it)?( today)?)\b",
            r"\b(what is today)\b",
            r"\b(tell me (the )?date)\b",
            r"\b(today('s)? date)\b",
            r"^date$",
            r"^today$",
        ]
        if any(re.search(pat, text) for pat in date_patterns) or _contains(text, "whats the date", "what's the date", "what is the date", "today's date", "todays date", "what day is it", "what is today"):
            import datetime
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            self.brain.voice.speak(f"Today is {today}.", force=True)
            return True

        # Volume control
        if _contains(text, "volume up", "increase volume", "louder"):
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])
            self.brain.voice.speak("Volume increased.")
            return True

        if _contains(text, "volume down", "decrease volume", "quieter", "lower volume"):
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
            self.brain.voice.speak("Volume decreased.")
            return True

        if _contains(text, "mute volume", "mute sound"):
            subprocess.run(["osascript", "-e", "set volume with output muted"])
            self.brain.voice.speak("Muted.")
            return True

        # Screenshot
        if _contains(text, "screenshot", "take a screenshot", "capture screen"):
            subprocess.Popen(["screencapture", "-i", "/tmp/surdas_screenshot.png"])
            self.brain.voice.speak("Taking a screenshot.")
            return True

        # Lock screen
        if _contains(text, "lock screen", "lock my mac", "lock computer"):
            subprocess.Popen(["osascript", "-e",
                'tell application "System Events" to keystroke "q" using {command down, control down}'])
            self.brain.voice.speak("Locking your screen.")
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 3 — Fast vision description (no LLM latency)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_vision_query(self, text: str) -> bool:
        if not _contains(text, "what do you see", "what is in front", "describe surroundings",
                         "describe scene", "any obstacles", "is path clear", "am i safe"):
            return False

        ctx = self.brain.get_vision_context()
        objects = ctx.get("detected_objects", [])
        closest = ctx.get("closest_obstacle", None)
        wall = ctx.get("wall_ahead", False)

        if wall:
            reply = "Warning! There is a wall or barrier directly ahead."
        elif not objects:
            reply = "The path ahead looks clear. No obstacles detected."
        else:
            unique = list(dict.fromkeys(objects))  # preserve order, dedup
            obj_str = ", ".join(unique[:4])
            reply = f"I can see: {obj_str}. {closest if closest else 'Use caution.'}"

        self.brain.voice.speak(reply)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Category 4 — Model management commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_model_commands(self, text: str) -> bool:

        # "list models" / "what models do you have" / "available models"
        if _contains(text, "list models", "available models", "what models",
                     "which models", "show models", "installed models"):
            msg = self.llm.spoken_model_list()
            self.brain.voice.speak(msg)
            return True

        # "what model are you using" / "current model" / "which model"
        if _contains(text, "what model", "current model", "which model",
                     "what ai model", "which ai"):
            self.brain.voice.speak(
                f"I am using {self.llm.get_current_model()}."
            )
            return True

        # "switch to model X" / "use model X" / "change model to X"
        # / "switch model to X"
        for trigger in ("switch to ", "use model ", "change model to ",
                        "switch model to ", "use ", "load model "):
            if trigger in text:
                after = text[text.find(trigger) + len(trigger):].strip()
                # Strip "model" prefix if the user said "switch to model llama3"
                if after.startswith("model "):
                    after = after[6:]
                model_name = after.strip()
                if not model_name:
                    continue
                # Don't intercept generic "use [app]" — only model-like tokens
                # (model names contain digits, dots, or known keywords)
                import re
                if not re.search(r'[\d\.]', model_name) and model_name not in (
                    "llama", "mistral", "gemma", "phi", "qwen", "deepseek",
                    "tinyllama", "codellama", "dolphin", "vicuna", "orca",
                ):
                    continue

                success = self.llm.switch_model(model_name)
                if success:
                    self.brain.voice.speak(
                        f"Switched to {self.llm.get_current_model()}."
                    )
                    # Broadcast model change to dashboard
                    try:
                        from telemetry import broadcast_event
                        broadcast_event("model_change", {"model": self.llm.model_name})
                    except Exception:
                        pass
                else:
                    available = self.llm.get_available_models()
                    names = ", ".join(m.replace(":latest", "") for m in available[:4])
                    self.brain.voice.speak(
                        f"Model {model_name} not found. Available: {names}."
                    )
                return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 5 — General LLM query
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_llm_query(self, raw_text: str):
        if not self.llm.is_available():
            self.brain.voice.speak(
                "Ollama is not running. Start it with: ollama serve."
            )
            return

        print("[ROUTER] → Ollama LLM (streaming)…")
        vision_ctx = self.brain.get_vision_context()

        for sentence in self.llm.query(raw_text, vision_context=vision_ctx):
            if sentence:
                print(f"[LLM] {sentence}")
                self.brain.voice.speak(sentence)
