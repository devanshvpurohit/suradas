"""
app_launcher.py — macOS App Launcher for SURDAS Voice Commands
Supports fuzzy matching so "open chrome" finds "Google Chrome.app"
"""
import subprocess
import os
import re
from typing import Optional

# ──────────────────────────────────────────────────────────────────
# Canonical app name → macOS application bundle name mapping
# Add more entries here as needed.
# ──────────────────────────────────────────────────────────────────
APP_MAP: dict[str, str] = {
    # Browsers
    "chrome":           "Google Chrome",
    "google chrome":    "Google Chrome",
    "brave":            "Brave Browser",
    "brave browser":    "Brave Browser",
    "safari":           "Safari",
    "arc":              "Arc",
    "arc browser":      "Arc",

    # Communication
    "whatsapp":         "WhatsApp",
    "telegram":         "Telegram",

    # Productivity / Apple
    "pages":            "Pages",
    "numbers":          "Numbers",
    "keynote":          "Keynote",
    "notes":            "Notes",
    "calendar":         "Calendar",
    "reminders":        "Reminders",
    "maps":             "Maps",
    "facetime":         "FaceTime",
    "photos":           "Photos",
    "music":            "Music",
    "podcasts":         "Podcasts",
    "finder":           "Finder",
    "calculator":       "Calculator",
    "clock":            "Clock",

    # Development
    "kiro":             "Kiro",
    "arduino":          "Arduino IDE",
    "terminal":         "Terminal",
    "xcode":            "Xcode",

    # Streaming / Entertainment
    "prime video":      "Prime Video",
    "amazon prime":     "Prime Video",
    "prime":            "Prime Video",
    "roblox":           "Roblox",

    # System / Utility
    "settings":         "System Preferences",
    "system preferences": "System Preferences",
    "system settings":  "System Settings",
    "activity monitor": "Activity Monitor",
    "disk utility":     "Disk Utility",
    "mail":             "Mail",
    "messages":         "Messages",
    "siri":             "Siri",
    "spotlight":        "Spotlight",
    "app store":        "App Store",

    # AI / Other
    "ollama":           "Ollama",
    "antigravity":      "Antigravity",
}

# System actions (not real app bundles)
SYSTEM_ACTIONS: dict[str, str] = {
    "screenshot":       "screencapture -i /tmp/surdas_screenshot.png",
    "take screenshot":  "screencapture -i /tmp/surdas_screenshot.png",
}


def _fuzzy_match(query: str) -> Optional[str]:
    """Find the best app bundle name for a query string."""
    q = query.lower().strip()

    # Exact map hit
    if q in APP_MAP:
        return APP_MAP[q]

    # Partial map hit
    for key, bundle in APP_MAP.items():
        if key in q or q in key:
            return bundle

    # Check /Applications directly as fallback
    try:
        installed = os.listdir("/Applications")
        for app_file in installed:
            name = app_file.replace(".app", "")
            if q in name.lower() or name.lower() in q:
                return name
    except Exception:
        pass

    return None


def open_app(app_query: str) -> tuple[bool, str]:
    """
    Open a macOS application by fuzzy name.
    Returns (success: bool, message: str)
    """
    # Check system actions first
    action_key = app_query.lower().strip()
    if action_key in SYSTEM_ACTIONS:
        cmd = SYSTEM_ACTIONS[action_key]
        subprocess.Popen(cmd, shell=True)
        return True, f"Running {action_key}."

    bundle = _fuzzy_match(app_query)
    if not bundle:
        return False, f"I could not find an app matching {app_query}."

    try:
        subprocess.Popen(["open", "-a", bundle])
        print(f"[LAUNCHER] Opened: {bundle}")
        return True, f"Opening {bundle}."
    except Exception as e:
        print(f"[LAUNCHER] Error opening {bundle}: {e}")
        return False, f"Could not open {bundle}. {e}"


def close_app(app_query: str) -> tuple[bool, str]:
    """Close a running app by name using AppleScript."""
    bundle = _fuzzy_match(app_query) or app_query
    script = f'tell application "{bundle}" to quit'
    try:
        subprocess.run(["osascript", "-e", script], timeout=3)
        print(f"[LAUNCHER] Closed: {bundle}")
        return True, f"Closing {bundle}."
    except Exception as e:
        return False, f"Could not close {bundle}."


def get_app_list() -> str:
    """Return a spoken summary of supported apps."""
    common = ["Chrome", "WhatsApp", "Telegram", "Safari", "Arc",
              "Terminal", "Kiro", "Prime Video", "Notes", "Calculator",
              "Music", "Finder", "Messages", "Mail"]
    return "I can open: " + ", ".join(common) + ", and more."
