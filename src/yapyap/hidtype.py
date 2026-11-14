#!/usr/bin/env python3
"""Type text through a virtual uinput keyboard instead of pasting it.

Runs as a long-lived process: it creates one virtual keyboard at startup,
then reads transcripts from stdin, one per line, and types each one as real
key presses. Keeping the device open avoids the delay a compositor needs to
pick up a freshly created keyboard.

Curly quotes, dashes and ellipses are mapped to ASCII look-alikes. A line
with any other character the US layout cannot produce (Cyrillic, accents) is
typed whole by wtype, if installed, through the Wayland virtual-keyboard
protocol.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
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
        # in 8,400 characters.
        # A 4 ms pause types ~1,000 characters a second. At 1 ms (~3,400 a
        # second) Electron apps such as Claude fall behind and redraw the text
        # in two pieces; at 4 ms the text appears smoothly as it is typed.
        self.burst_events = max(int(env_float("YAPYAP_TYPE_BURST_EVENTS", 16)), 4)
        self.burst_pause = env_float("YAPYAP_TYPE_BURST_PAUSE_MS", 4.0) / 1000.0
        self._sent = 0
        self._last_emit: float | None = None
        self.max_gap = 0.0
        self.unicode_runs = 0
        self.unicode_time = 0.0
        # Optional extra delay after every key event, for apps that need it.
        self.key_delay = env_float("YAPYAP_TYPE_KEY_DELAY_MS", 0.0) / 1000.0
        # Physical keyboards stay open: closing an evdev node costs ~10 ms,
        # and reopening all of them took ~300 ms per transcript.
        self.watched: dict[str, evdev.InputDevice] = {}
        self.ignored: set[str] = set()
        self._refresh_watched()
        self.modifier_wait = env_float("YAPYAP_TYPE_MODIFIER_WAIT_S", 2.0)
        self.wtype = shutil.which("wtype")
        # wtype brings its own keymap, and under Hyprland the app only
        # switches to it when wtype's first key arrives, so that first key
        # is lost: "Если хочу" came out as "сли очу", and every streamed
        # piece lost its leading space. Each call therefore starts with a
        # Shift tap the app may drop, then waits before the real text.
        self.wtype_settle_ms = max(int(env_float("YAPYAP_WTYPE_SETTLE_MS", 30)), 0)
        # Compositors take a moment to register a new keyboard.
        time.sleep(env_float("YAPYAP_TYPE_SETTLE_S", 0.3))

    def close(self) -> None:
        for device in self.watched.values():
            device.close()
        self.device.close()

    def _emit(self, code: int, value: int) -> None:
        now = time.perf_counter()
        if self._last_emit is not None:
            self.max_gap = max(self.max_gap, now - self._last_emit)
        self._last_emit = now
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
        self._last_emit = None
        self.max_gap = 0.0
        self.unicode_runs = 0
        self.unicode_time = 0.0
        ascii_text = text.translate(ASCII_LOOKALIKES)
        if any(ch not in KEYMAP for ch in ascii_text):
            # One wtype call for the whole line keeps keymap switches, and the
            # keys apps drop around them, to one per line. wtype can type the
            # original dashes and quotes, so skip the ASCII look-alikes.
            self._type_unicode(text)
            return
        for ch in ascii_text:
            self._tap(*KEYMAP[ch])

    def _type_unicode(self, text: str) -> None:
        # wtype goes through a different device; let our queued keys drain first.
        time.sleep(self.burst_pause)
        self._sent = 0
        if not self.wtype:
            log(f"no wtype; dropped {text!r}")
            return
        start = time.perf_counter()
        subprocess.run(
            [self.wtype, "-k", "Shift_L", "-s", str(self.wtype_settle_ms), "--", text],
            check=False,
        )
        self.unicode_runs += 1
        self.unicode_time += time.perf_counter() - start
        # Do not count the wtype call as a gap between our own key events.
        self._last_emit = None


def read_lines(fd: int, lines: queue.Queue) -> None:
    """Read stdin lines, acknowledge each one at once, and queue it.

    Acknowledging on receipt lets the caller send the next streamed piece
    while earlier ones are still being typed. None marks end of input.
    """
    pending = b""
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            lines.put(None)
            return
        pending += chunk
        *complete, pending = pending.split(b"\n")
        for raw in complete:
            lines.put(raw.decode("utf-8", "replace"))
            print("done", flush=True)


def next_batch(lines: queue.Queue) -> list[str] | None:
    """Block for one line, then take every line already queued behind it."""
    first = lines.get()
    if first is None:
        return None
    batch = [first]
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            return batch
        if line is None:
            lines.put(None)
            return batch
        batch.append(line)


def main() -> None:
    try:
        typer = Typer()
    except (OSError, evdev.UInputError) as exc:
        log(f"cannot create virtual keyboard: {exc}")
        sys.exit(1)
    log(f"ready ({DEVICE_NAME})")
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=read_lines, args=(0, lines), daemon=True).start()
    try:
        while (batch := next_batch(lines)) is not None:
            # Pieces that queued up while the last batch was typed go out
            # together: one wtype call for several Russian words instead of
            # one per word.
            text = "".join(batch)
            if not text:
                continue
            start = time.perf_counter()
            typer.wait_for_modifiers_released()
            typed_at = time.perf_counter()
            typer.type_text(text)
            done_at = time.perf_counter()
            # Streamed transcripts arrive as many short pieces; log whole
            # transcripts and anything unusual, not every piece.
            if len(text) >= 40 or typer.max_gap > 0.02 or typer.unicode_runs:
                log(
                    f"typed {len(text)} chars from {len(batch)} line(s) in"
                    f" {1000 * (done_at - typed_at):.0f} ms"
                    f" (modifier wait {1000 * (typed_at - start):.0f} ms,"
                    f" longest gap {1000 * typer.max_gap:.1f} ms,"
                    f" wtype {typer.unicode_runs}x {1000 * typer.unicode_time:.0f} ms)"
                )
    except KeyboardInterrupt:
        pass
    finally:
        typer.close()

if __name__ == "__main__":
    main()
