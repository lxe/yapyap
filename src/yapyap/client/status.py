"""Recording state for status bars.

The client keeps $XDG_RUNTIME_DIR/yapyap-state-<uid>.json up to date with
{"active": bool, "recording": bool, "updated_at": unix time}, and can signal
Waybar (SIGRTMIN+N) so a custom module re-reads it at once.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time


def state_path() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return os.environ.get("YAPYAP_STATE") or os.path.join(runtime, f"yapyap-state-{os.getuid()}.json")


class Status:
    def __init__(self, waybar_signal: int = 0):
        self.path = state_path()
        self.waybar_signal = waybar_signal
        self._lock = threading.Lock()

    def update(self, *, active: bool, recording: bool) -> None:
        payload = {"active": active, "recording": recording, "pid": os.getpid(), "updated_at": time.time()}
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp, self.path)
        if self.waybar_signal:
            # pkill scans every process (50-100 ms); keep it off the hotkey path.
            threading.Thread(
                target=subprocess.run,
                args=(["pkill", f"-RTMIN+{self.waybar_signal}", "-x", "waybar"],),
                kwargs={"check": False},
                daemon=True,
            ).start()
