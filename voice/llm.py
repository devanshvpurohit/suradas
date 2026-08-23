import json
import urllib.request
import urllib.error
import re
from typing import Iterator, List

class LocalLLM:
    """
    Ultra-low-latency Local LLM client using Ollama.
    - Optimized parameters (small context, short generation, fast sampling)
    - Clause/sentence streaming for sub-second time-to-first-speech
    - Model selectable at runtime (CLI arg, voice command, or direct assignment)
    """

    DEFAULT_MODEL = "gemma3:1b"

    def __init__(self, model_name: str = "", host: str = "http://localhost:11434"):
        self.host = host.rstrip('/')
        self.model_name = model_name or self.DEFAULT_MODEL
        self._build_system_prompt()

    def _build_system_prompt(self):
        self.system_prompt = (
            "You are SURDAS, an AI voice assistant for visually impaired users. "
            "Give clear, factual, and concise answers in 1 to 2 spoken sentences. "
            "Never use markdown, bullet points, asterisks, or lists. Plain conversational English only."
        )

    # ── Model management ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama server is running locally."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """Fetch list of locally installed Ollama models."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return models
        except Exception as e:
            print(f"[LLM] Could not fetch model list: {e}")
            return []

    def switch_model(self, model_name: str) -> bool:
        """Switch active model. Returns True if model exists locally."""
        available = self.get_available_models()
        match = None
        for m in available:
            if model_name.lower() in m.lower() or m.lower().startswith(model_name.lower()):
                match = m
                break

        if match:
            self.model_name = match
            print(f"[LLM] ✅ Switched model → {self.model_name}")
            return True
        else:
            print(f"[LLM] ❌ Model '{model_name}' not found. Available: {available}")
            return False

    def get_current_model(self) -> str:
        return self.model_name.replace(":latest", "")

    def spoken_model_list(self) -> str:
        models = self.get_available_models()
        if not models:
            return "No models found. Make sure Ollama is running."
        clean = [m.replace(":latest", "") for m in models]
        return f"Installed models: {', '.join(clean)}."

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, user_prompt: str, vision_context: dict = None) -> Iterator[str]:
        """
        Fast stream response from Ollama.
        Yields short clauses/sentences immediately for instant TTS response.
        """
        import datetime
        now_str = datetime.datetime.now().strftime("%A, %I:%M %p")
        context_parts = [f"Time: {now_str}"]

        if vision_context:
            objects = vision_context.get("detected_objects", [])
            closest = vision_context.get("closest_obstacle", None)
            if objects:
                context_parts.append(f"Seeing: {', '.join(objects[:4])}")
            if closest:
                context_parts.append(f"Obstacle: {closest}")

        context_str = f"[{' | '.join(context_parts)}]"
        full_prompt = f"{context_str}\nUser: {user_prompt}\nAnswer:"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "system": self.system_prompt,
            "stream": True,
            "options": {
                "num_predict": 50,    # 1-2 concise sentences
                "num_ctx": 512,       # Small KV cache = much faster prompt eval
                "temperature": 0.2,   # Fast deterministic sampling
                "top_p": 0.9,
                "top_k": 20,
            }
        }

        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                buf = ""
                for line in resp:
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    chunk = data.get("response", "")
                    buf += chunk
                    
                    # Yield complete clauses or sentences as soon as ready
                    if any(p in chunk for p in ('.', '!', '?', '\n')):
                        parts = re.split(r'(?<=[.!?\n])\s+', buf)
                        for part in parts[:-1]:
                            cleaned = part.strip()
                            if cleaned:
                                yield cleaned
                        buf = parts[-1] if parts else ""
                    elif len(buf.split()) >= 9 and any(p in chunk for p in (',', ';')):
                        # Yield early clause if buffer has accumulated >8 words
                        parts = re.split(r'(?<=[,;])\s+', buf)
                        if len(parts) > 1:
                            cleaned = parts[0].strip()
                            if cleaned:
                                yield cleaned
                            buf = " ".join(parts[1:])
                
                if buf.strip():
                    yield buf.strip()

        except urllib.error.URLError:
            yield "Ollama is not running. Please start it with: ollama serve."
        except Exception as e:
            print(f"[LLM] Error: {e}")
            yield "Sorry, I could not process that."
