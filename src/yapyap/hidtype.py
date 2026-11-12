#!/usr/bin/env python3
"""Type text through a virtual uinput keyboard instead of pasting it.

Runs as a long-lived process: it creates one virtual keyboard at startup,
then reads transcripts from stdin, one per line, and types each one as real
key presses. Keeping the device open avoids the delay a compositor needs to
pick up a freshly created keyboard.

Characters the US layout cannot produce are first mapped to ASCII look-alikes
(curly quotes, dashes, ellipsis); anything left is handed to wtype, if
installed, which types it through the Wayland virtual-keyboard protocol.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import evdev
from evdev import UInput, ecodes as e

DEVICE_NAME = "yapyap-virtual-keyboard"

MODIFIER_KEYS = {
    e.KEY_LEFTCTRL, e.KEY_RIGHTCTRL, e.KEY_LEFTALT, e.KEY_RIGHTALT,
    e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT, e.KEY_LEFTMETA, e.KEY_RIGHTMETA,
}

_UNSHIFTED = {
    **{chr(c): getattr(e, f"KEY_{chr(c).upper()}") for c in range(ord("a"), ord("z") + 1)},
    **{str(d): getattr(e, f"KEY_{d}") for d in range(10)},
    " ": e.KEY_SPACE, "\t": e.KEY_TAB, "\n": e.KEY_ENTER,
    "-": e.KEY_MINUS, "=": e.KEY_EQUAL, "[": e.KEY_LEFTBRACE, "]": e.KEY_RIGHTBRACE,
    "\\": e.KEY_BACKSLASH, ";": e.KEY_SEMICOLON, "'": e.KEY_APOSTROPHE,
    "`": e.KEY_GRAVE, ",": e.KEY_COMMA, ".": e.KEY_DOT, "/": e.KEY_SLASH,
}
_SHIFTED = {
    **{chr(c).upper(): getattr(e, f"KEY_{chr(c).upper()}") for c in range(ord("a"), ord("z") + 1)},
    "!": e.KEY_1, "@": e.KEY_2, "#": e.KEY_3, "$": e.KEY_4, "%": e.KEY_5,
    "^": e.KEY_6, "&": e.KEY_7, "*": e.KEY_8, "(": e.KEY_9, ")": e.KEY_0,
    "_": e.KEY_MINUS, "+": e.KEY_EQUAL, "{": e.KEY_LEFTBRACE, "}": e.KEY_RIGHTBRACE,
    "|": e.KEY_BACKSLASH, ":": e.KEY_SEMICOLON, '"': e.KEY_APOSTROPHE,
    "~": e.KEY_GRAVE, "<": e.KEY_COMMA, ">": e.KEY_DOT, "?": e.KEY_SLASH,
}
KEYMAP: dict[str, tuple[int, bool]] = {
    **{ch: (code, False) for ch, code in _UNSHIFTED.items()},
    **{ch: (code, True) for ch, code in _SHIFTED.items()},
}

ASCII_LOOKALIKES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    " ": " ", " ": " ", " ": " ",
    "…": "...",
})


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def log(message: str) -> None:
    print(f"yapyap-type: {message}", file=sys.stderr, flush=True)


class Typer:
    def __init__(self) -> None:
        keys = sorted({code for code, _ in KEYMAP.values()} | {e.KEY_LEFTSHIFT})
        self.device = UInput({e.EV_KEY: keys}, name=DEVICE_NAME)
        # Events go out in bursts with a short pause between them. The kernel
        # buffers 64 events per reader (keyd, then the compositor), and big
        # bursts lose keys: 64 events per burst dropped ~30% of characters and
        # 32 occasionally dropped one, while 16 (4 plain characters) lost none
        # in 8,400 characters. That still types ~3,400 characters a second.
        self.burst_events = max(int(env_float("YAPYAP_TYPE_BURST_EVENTS", 16)), 4)
        self.burst_pause = env_float("YAPYAP_TYPE_BURST_PAUSE_MS", 1.0) / 1000.0
        self._sent = 0
        # Optional extra delay after every key event, for apps that need it.
        self.key_delay = env_float("YAPYAP_TYPE_KEY_DELAY_MS", 0.0) / 1000.0
        # Physical keyboards stay open: closing an evdev node costs ~10 ms,
        # and reopening all of them took ~300 ms per transcript.
        self.watched: dict[str, evdev.InputDevice] = {}
        self.ignored: set[str] = set()
        self._refresh_watched()
        self.modifier_wait = env_float("YAPYAP_TYPE_MODIFIER_WAIT_S", 2.0)
        self.wtype = shutil.which("wtype")
        # Compositors take a moment to register a new keyboard.
        time.sleep(env_float("YAPYAP_TYPE_SETTLE_S", 0.3))

    def close(self) -> None:
        for device in self.watched.values():
            device.close()
        self.device.close()

    def _emit(self, code: int, value: int) -> None:
        self.device.write(e.EV_KEY, code, value)
        self.device.syn()
        # Each key event plus its SYN_REPORT takes two buffer slots.
        self._sent += 2
        if self.key_delay:
            time.sleep(self.key_delay)
        elif self.burst_pause and self._sent >= self.burst_events:
            time.sleep(self.burst_pause)
            self._sent = 0

    def _tap(self, code: int, shifted: bool) -> None:
        if shifted:
            self._emit(e.KEY_LEFTSHIFT, 1)
        self._emit(code, 1)
        self._emit(code, 0)
        if shifted:
            self._emit(e.KEY_LEFTSHIFT, 0)

    def _refresh_watched(self) -> None:
        paths = set(evdev.list_devices())
        for path in list(self.watched):
            if path not in paths:
                self.watched.pop(path).close()
        # Event node numbers get reused, so forget ignored nodes that went away.
        self.ignored &= paths
        for path in paths - self.watched.keys() - self.ignored:
            try:
                device = evdev.InputDevice(path)
            except OSError:
                continue
            keys = device.capabilities().get(e.EV_KEY, [])
            if device.name == DEVICE_NAME or not MODIFIER_KEYS.intersection(keys):
                device.close()
                self.ignored.add(path)
                continue
            self.watched[path] = device

    def _modifier_held(self) -> bool:
        for path, device in list(self.watched.items()):
            try:
                if MODIFIER_KEYS.intersection(device.active_keys()):
                    return True
            except OSError:
                self.watched.pop(path).close()
        return False

    def wait_for_modifiers_released(self) -> None:
        """Wait until no physical keyboard holds a modifier.

        Typing while the dictation chord (Ctrl+Alt by default) is still held
        would turn every letter into a shortcut.
        """
        self._refresh_watched()
        deadline = time.monotonic() + self.modifier_wait
        while self._modifier_held():
            if time.monotonic() >= deadline:
                log("modifier still held; typing anyway")
                return
            time.sleep(0.005)

    def type_text(self, text: str) -> None:
        text = text.translate(ASCII_LOOKALIKES)
        pending_unicode: list[str] = []
        for ch in text:
            mapping = KEYMAP.get(ch)
            if mapping is None:
                pending_unicode.append(ch)
                continue
            if pending_unicode:
                self._type_unicode("".join(pending_unicode))
                pending_unicode.clear()
            self._tap(*mapping)
        if pending_unicode:
            self._type_unicode("".join(pending_unicode))

    def _type_unicode(self, text: str) -> None:
        # wtype goes through a different device; let our queued keys drain first.
        time.sleep(self.burst_pause)
        self._sent = 0
        if not self.wtype:
            log(f"no wtype; dropped {text!r}")
            return
        subprocess.run([self.wtype, "--", text], check=False)


def main() -> None:
    try:
        typer = Typer()
    except (OSError, evdev.UInputError) as exc:
        log(f"cannot create virtual keyboard: {exc}")
        sys.exit(1)
    log(f"ready ({DEVICE_NAME})")
    try:
        for line in sys.stdin:
            text = line.rstrip("\n")
            if not text:
                continue
            typer.wait_for_modifiers_released()
            typer.type_text(text)
            # Tell a waiting caller this line is done.
            print("done", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        typer.close()


if __name__ == "__main__":
    main()
