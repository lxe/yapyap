"""Push-to-talk hotkey: watch every keyboard for a held key chord (evdev).

Reading /dev/input needs membership in the `input` group.
"""
from __future__ import annotations

import logging
import select
import time
from typing import Callable

import evdev
from evdev import ecodes

from .typer import DEVICE_NAME

log = logging.getLogger("yapyap")

RESCAN_S = 2.0


def key_codes(names: list[str]) -> set[int]:
    codes = set()
    for name in names:
        code = ecodes.ecodes.get(name)
        if code is None:
            raise ValueError(f"unknown key {name!r}; use evdev names such as KEY_LEFTCTRL")
        codes.add(code)
    return codes


class ChordListener:
    """Calls on_press when every key of the chord is down, on_release when one lets go."""

    def __init__(self, chord: list[str], on_press: Callable[[], None], on_release: Callable[[], None]):
        self.chord = key_codes(chord)
        self.on_press = on_press
        self.on_release = on_release
        self.held: set[int] = set()
        self.active = False
        self.devices: dict[int, evdev.InputDevice] = {}
        self.paths: set[str] = set()

    def _scan(self) -> None:
        for path in set(evdev.list_devices()) - self.paths:
            try:
                device = evdev.InputDevice(path)
            except OSError:
                continue
            keys = device.capabilities().get(ecodes.EV_KEY, [])
            if device.name == DEVICE_NAME or not self.chord.issubset(keys):
                device.close()
                continue
            # Drop key events queued before we opened the device, so a stale
            # press cannot start a recording.
            try:
                while device.read_one():
                    pass
            except (BlockingIOError, OSError):
                pass
            self.devices[device.fd] = device
            self.paths.add(path)
            log.info("watching %s (%s)", device.name, path)

    def _handle(self, code: int, pressed: bool) -> None:
        if code not in self.chord:
            return
        if pressed:
            self.held.add(code)
        else:
            self.held.discard(code)
        if not self.active and self.held == self.chord:
            self.active = True
            self.on_press()
        elif self.active and self.held != self.chord:
            self.active = False
            self.on_release()

    def run(self) -> None:
        self._scan()
        if not self.devices:
            log.warning("no keyboard with the chord keys is readable yet; is the user in the 'input' group?")
        next_scan = time.monotonic() + RESCAN_S
        while True:
            timeout = max(0.0, next_scan - time.monotonic())
            ready, _, _ = select.select(list(self.devices), [], [], timeout)
            for fd in ready:
                device = self.devices[fd]
                try:
                    for event in device.read():
                        if event.type == ecodes.EV_KEY and event.value in (0, 1):
                            self._handle(event.code, event.value == 1)
                except OSError:
                    log.info("keyboard %s went away", device.name)
                    self.devices.pop(fd, None)
                    self.paths.discard(device.path)
            if time.monotonic() >= next_scan:
                self._scan()
                next_scan = time.monotonic() + RESCAN_S
