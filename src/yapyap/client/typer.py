"""Type text as key presses on a virtual uinput keyboard.

Text goes out as real key events, so it works in any app that takes keyboard
input, including terminals, remote desktops and games. The typer reads the
keyboard layouts the compositor uses (with libxkbcommon) and presses the keys
that produce each character, with Shift and AltGr as needed. Under Hyprland,
text in another configured layout, such as Russian with kb_layout =
"us,ru", is typed by switching the keyboards to that layout group. Anything
no layout can type goes through wtype, if installed.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import evdev
from evdev import UInput, ecodes as e

from . import hyprland
from .config import OutputConfig
from .session import wayland_env
from .xkb import RIGHT_ALT, Layout, Stroke, load_layouts

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

# Used only when libxkbcommon is missing.
US_LAYOUT = Layout("us", {ch: Stroke(code, shift) for ch, (code, shift) in KEYMAP.items()}, None)


def detect_layouts() -> tuple[list[Layout], bool]:
    """The compositor's layout groups, and whether the typer can switch between them."""
    hypr = hyprland.keyboard_layout()
    if hypr:
        layout, variant, options = hypr
    else:
        layout = os.environ.get("XKB_DEFAULT_LAYOUT", "us")
        variant = os.environ.get("XKB_DEFAULT_VARIANT", "")
        options = os.environ.get("XKB_DEFAULT_OPTIONS", "")
    try:
        layouts = load_layouts(layout, variant, options)
    except (OSError, ValueError) as exc:
        log.warning("cannot read keyboard layout %r (%s); typing with a US layout", layout, exc)
        return [US_LAYOUT], False
    return layouts, hypr is not None and len(layouts) > 1


class Typer:
    def __init__(self, config: OutputConfig, layouts: list[Layout] | None = None, can_switch: bool | None = None) -> None:
        if layouts is None:
            layouts, can_switch = detect_layouts()
        self.layouts = layouts
        self.can_switch = bool(can_switch)
        codes = {stroke.code for layout in layouts for stroke in layout.strokes.values()}
        self.device = UInput({e.EV_KEY: sorted(codes | {e.KEY_LEFTSHIFT, RIGHT_ALT})}, name=DEVICE_NAME)
        # Events go out in bursts with a short pause between them. The kernel
        # buffers 64 events per reader (a key remapper such as keyd, then the
        # compositor), and bursts of 64 events lost keys in testing; 16 (4
        # plain characters) lost none up to a 0.25 ms pause (12,600 keys a
        # second). The default 1 ms pause types about 3,700 keys a second and
        # leaves room for the compositor to stall briefly without losing keys.
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
        self.group = 0
        if self.can_switch:
            hyprland.switch_layout_group(0)
        log.info("typing with keyboard layout%s %s", "s" if len(layouts) > 1 else "",
                 ", ".join(layout.name for layout in (layouts if self.can_switch else layouts[:1])))
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

    def _tap(self, stroke: Stroke) -> None:
        if stroke.level3:
            self._emit(RIGHT_ALT, 1)
        if stroke.shift:
            self._emit(e.KEY_LEFTSHIFT, 1)
        self._emit(stroke.code, 1)
        self._emit(stroke.code, 0)
        if stroke.shift:
            self._emit(e.KEY_LEFTSHIFT, 0)
        if stroke.level3:
            self._emit(RIGHT_ALT, 0)

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

    def _stroke(self, ch: str, group: int) -> tuple[int, Stroke] | None:
        """The group and keys for one character, staying in `group` when it can."""
        groups = range(len(self.layouts)) if self.can_switch else range(1)
        order = [group, 0, *groups]
        for candidate in order:
            if candidate in groups:
                stroke = self.layouts[candidate].strokes.get(ch)
                if stroke:
                    return candidate, stroke
        return None

    def type_text(self, text: str) -> None:
        """Type text as keys; characters no layout can produce go through wtype."""
        plan: list[tuple[int, Stroke] | str] = []
        group = self.group
        for ch in text:
            # A curly quote, dash or ellipsis that no layout has becomes ASCII.
            for option in (ch, ch.translate(ASCII_LOOKALIKES)):
                steps = [self._stroke(c, group) for c in option]
                if None not in steps:
                    plan += steps
                    group = steps[-1][0]
                    break
            else:
                # Neighbouring characters for wtype go out in one call.
                if plan and isinstance(plan[-1], str):
                    plan[-1] += ch
                else:
                    plan.append(ch)
        for step in plan:
            if isinstance(step, str):
                self._type_wtype(step)
            else:
                self._switch_group(step[0])
                self._tap(step[1])

    def _switch_group(self, group: int) -> None:
        if self.group == group:
            return
        time.sleep(self.group_settle)
        hyprland.switch_layout_group(group)
        self.group = group
        self._sent = 0

    def restore_layout(self) -> None:
        """Put every keyboard back on the first layout group.

        After typing in another group the keyboard stays there, so streamed
        pieces need no switch each; the caller restores it once typing goes
        idle.
        """
        if self.can_switch:
            self._switch_group(0)

    def _type_wtype(self, text: str) -> None:
        # wtype goes through a different device; let our queued keys drain first.
        time.sleep(self.burst_pause)
        self._sent = 0
        if not self.wtype:
            log.warning("cannot type %r: install wtype for characters outside the keyboard layouts", text)
            return
        subprocess.run(
            [self.wtype, "-k", "Shift_L", "-s", str(self.wtype_settle_ms), "--", text],
            check=False,
            env=wayland_env(),
        )
