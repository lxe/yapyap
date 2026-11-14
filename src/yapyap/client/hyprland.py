"""Hyprland IPC: focused window, keyboard layout groups and shortcuts.

Everything here is optional. On other compositors these calls find no
socket and return nothing, and the client falls back to what works there.
"""
from __future__ import annotations

import glob
import json
import os
import socket


def _socket_path() -> str | None:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    candidates = [f"{runtime}/hypr/{signature}/.socket.sock"] if signature else []
    candidates += sorted(glob.glob(f"{runtime}/hypr/*/.socket.sock"), key=os.path.getmtime, reverse=True)
    return next((path for path in candidates if os.path.exists(path)), None)


def request(command: str) -> str | None:
    """Send one command to Hyprland's socket; None when Hyprland is not running."""
    path = _socket_path()
    if not path:
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            client.connect(path)
            client.sendall(command.encode())
            chunks = []
            while chunk := client.recv(65536):
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", "replace")
    except OSError:
        return None


def _json(command: str):
    answer = request(f"j/{command}")
    try:
        return json.loads(answer) if answer else None
    except ValueError:
        return None


def active_window() -> tuple[str, str] | None:
    """(class, address) of the focused window, or None."""
    window = _json("activewindow")
    if not window or not window.get("address"):
        return None
    return (window.get("class") or "").lower(), window["address"]


def layout_groups() -> list[str]:
    """Layouts configured as xkb groups (input kb_layout), e.g. ["us", "ru"]."""
    devices = _json("devices") or {}
    for keyboard in devices.get("keyboards", []):
        layout = keyboard.get("layout")
        if layout:
            return [group.strip() for group in layout.split(",")]
    return []


def switch_layout_group(group: int) -> None:
    """Switch every keyboard to layout group `group`.

    All keyboards switch together because key remappers such as keyd re-emit
    our virtual keyboard's keys through a keyboard of their own.
    """
    request(f"switchxkblayout all {group}")


def send_shortcut(shortcut: str, address: str) -> None:
    """Send a shortcut such as "CTRL,V" to the window at `address`."""
    request(f"dispatch sendshortcut {shortcut},address:{address}")
