"""Type text as key presses on a virtual uinput keyboard.

Text goes out as real key events, so it works in any app that takes keyboard
input, including terminals, remote desktops and games. Curly quotes, dashes
and ellipses become ASCII look-alikes.

Text in another script is typed as real keys too when the compositor has a
matching layout group and the typer knows that layout: under Hyprland with
`kb_layout = "us,ru"`, Russian goes out on the ЙЦУКЕН keys with the keyboard
switched to the "ru" group. Anything else is typed by wtype, if installed,
through the Wayland virtual-keyboard protocol.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time

import evdev
from evdev import UInput, ecodes as e

from . import hyprland
from .config import OutputConfig
from .session import wayland_env

log = logging.getLogger("yapyap")

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
    "«": '"', "»": '"',
})

# The xkb "ru" layout (ЙЦУКЕН) on the same physical keys as KEYMAP.
_RU_ROWS = [
    ("йцукенгшщзхъ", [e.KEY_Q, e.KEY_W, e.KEY_E, e.KEY_R, e.KEY_T, e.KEY_Y, e.KEY_U, e.KEY_I, e.KEY_O, e.KEY_P, e.KEY_LEFTBRACE, e.KEY_RIGHTBRACE]),
    ("фывапролджэ", [e.KEY_A, e.KEY_S, e.KEY_D, e.KEY_F, e.KEY_G, e.KEY_H, e.KEY_J, e.KEY_K, e.KEY_L, e.KEY_SEMICOLON, e.KEY_APOSTROPHE]),
    ("ячсмитьбю", [e.KEY_Z, e.KEY_X, e.KEY_C, e.KEY_V, e.KEY_B, e.KEY_N, e.KEY_M, e.KEY_COMMA, e.KEY_DOT]),
    ("ё", [e.KEY_GRAVE]),
]
RU_KEYMAP: dict[str, tuple[int, bool]] = {
    **{ch: (code, False) for letters, codes in _RU_ROWS for ch, code in zip(letters, codes)},
    **{ch.upper(): (code, True) for letters, codes in _RU_ROWS for ch, code in zip(letters, codes)},
    **{str(d): (getattr(e, f"KEY_{d}"), False) for d in range(10)},
    " ": (e.KEY_SPACE, False), "\t": (e.KEY_TAB, False), "\n": (e.KEY_ENTER, False),
    "!": (e.KEY_1, True), '"': (e.KEY_2, True), "№": (e.KEY_3, True), ";": (e.KEY_4, True),
    "%": (e.KEY_5, True), ":": (e.KEY_6, True), "?": (e.KEY_7, True), "*": (e.KEY_8, True),
    "(": (e.KEY_9, True), ")": (e.KEY_0, True), "-": (e.KEY_MINUS, False), "_": (e.KEY_MINUS, True),
    "=": (e.KEY_EQUAL, False), "+": (e.KEY_EQUAL, True), ".": (e.KEY_SLASH, False), ",": (e.KEY_SLASH, True),
    "\\": (e.KEY_BACKSLASH, False), "/": (e.KEY_BACKSLASH, True),
}



# Layouts the typer can type as real keys, by xkb layout name.
LAYOUT_KEYMAPS: dict[str, dict[str, tuple[int, bool]]] = {"ru": RU_KEYMAP}


class Typer:
    def __init__(self, config: OutputConfig) -> None:
        keys = sorted({code for code, _ in KEYMAP.values()} | {e.KEY_LEFTSHIFT})
        self.device = UInput({e.EV_KEY: keys}, name=DEVICE_NAME)
        # Events go out in bursts with a short pause between them. The kernel
        # buffers 64 events per reader (a key remapper such as keyd, then the
        # compositor), and bursts of 32 or more events lost keys in testing;
        # 16 (4 plain characters) lost none. A 4 ms pause types about 1,000
        # characters a second, slow enough for Electron apps to redraw the
        # text as it grows instead of in one jump.
        self.burst_events = max(int(config.burst_events), 4)
        self.burst_pause = config.burst_pause_ms / 1000.0
        # Time for keys already sent to reach the compositor before the
        # layout group changes under them.
        self.group_settle = config.group_settle_ms / 1000.0
        # wtype brings its own keymap, and the app may drop the first key
        # typed after a keymap change, so each wtype call starts with a Shift
        # tap and waits before the real text.
        self.wtype_settle_ms = max(int(config.wtype_settle_ms), 0)
        self.wtype = shutil.which("wtype")
        self._sent = 0
        # Physical keyboards stay open: closing an evdev node costs ~10 ms,
        # and reopening all of them took ~300 ms per transcript.
        self.watched: dict[str, evdev.InputDevice] = {}
        self.ignored: set[str] = set()
        self._refresh_watched()
        # Extra layouts this compositor offers that the typer can type.
        groups = hyprland.layout_groups()
        self.layouts = {
            index: LAYOUT_KEYMAPS[name] for index, name in enumerate(groups) if index and name in LAYOUT_KEYMAPS
        }
        self.group = 0
        if self.layouts:
            hyprland.switch_layout_group(0)
            log.info("typing %s as keys through layout groups", ", ".join(groups[i] for i in self.layouts))
        # Compositors take a moment to register a new keyboard.
        time.sleep(0.3)

    def close(self) -> None:
        self.restore_layout()
        for device in self.watched.values():
            device.close()
        self.device.close()

    def _emit(self, code: int, value: int) -> None:
        self.device.write(e.EV_KEY, code, value)
        self.device.syn()
        # Each key event plus its SYN_REPORT takes two buffer slots.
        self._sent += 2
        if self.burst_pause and self._sent >= self.burst_events:
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

    def wait_for_modifiers_released(self, timeout: float = 2.0) -> None:
        """Wait until no physical keyboard holds a modifier.

        Typing while the dictation chord is still held would turn every
        letter into a shortcut.
        """
        self._refresh_watched()
        deadline = time.monotonic() + timeout
        while self._modifier_held():
            if time.monotonic() >= deadline:
                log.warning("a modifier key is still held; typing anyway")
                return
            time.sleep(0.005)

    def type_text(self, text: str) -> None:
        ascii_text = text.translate(ASCII_LOOKALIKES)
        for group, keymap in self.layouts.items():
            if any(ch in keymap and ch not in KEYMAP for ch in ascii_text) and all(
                ch in keymap or ch in KEYMAP for ch in ascii_text
            ):
                self._type_with_group(ascii_text, group, keymap)
                return
        # Everything below types with the first (US) group's key positions.
        self.restore_layout()
        if all(ch in KEYMAP for ch in ascii_text):
            for ch in ascii_text:
                self._tap(*KEYMAP[ch])
        else:
            # One wtype call for the whole text keeps keymap changes, and the
            # keys apps drop around them, to one. wtype can type the original
            # dashes and quotes, so skip the ASCII look-alikes.
            self._type_wtype(text)

    def _switch_group(self, group: int) -> None:
        if self.group == group:
            return
        time.sleep(self.group_settle)
        hyprland.switch_layout_group(group)
        self.group = group
        self._sent = 0

    def _type_with_group(self, text: str, group: int, keymap: dict[str, tuple[int, bool]]) -> None:
        """Type text mixing another layout with English, switching groups.

        Characters both layouts have (digits, space, most punctuation) stay
        in whichever group is active. The keyboard stays in the other group
        afterwards, so streamed pieces need no switch each; the caller runs
        restore_layout() once typing goes idle.
        """
        for ch in text:
            if self.group == group and ch in keymap:
                self._tap(*keymap[ch])
            elif self.group != group and ch in KEYMAP:
                self._tap(*KEYMAP[ch])
            elif ch in keymap:
                self._switch_group(group)
                self._tap(*keymap[ch])
            else:
                self._switch_group(0)
                self._tap(*KEYMAP[ch])

    def restore_layout(self) -> None:
        """Put every keyboard back on the first layout group."""
        if self.layouts:
            self._switch_group(0)

    def _type_wtype(self, text: str) -> None:
        # wtype goes through a different device; let our queued keys drain first.
        time.sleep(self.burst_pause)
        self._sent = 0
        if not self.wtype:
            log.warning("cannot type %r: install wtype for characters outside the keyboard layout", text)
            return
        subprocess.run(
            [self.wtype, "-k", "Shift_L", "-s", str(self.wtype_settle_ms), "--", text],
            check=False,
            env=wayland_env(),
        )
