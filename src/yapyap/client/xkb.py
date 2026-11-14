"""Which key and modifiers type each character, for every configured layout.

Uses libxkbcommon (through ctypes, so nothing needs compiling), the library
Wayland compositors use for keyboard layouts. For each layout group it maps
characters to an evdev key code plus Shift and AltGr (ISO_Level3_Shift).
"""
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass

XKB_KEY_ISO_LEVEL3_SHIFT = 0xFE03
XKB_KEYPAD_SYMS = range(0xFF80, 0xFFBE)  # KP_* keysyms: skip the keypad, whose keys depend on Num Lock
EVDEV_OFFSET = 8  # xkb key codes are evdev codes plus 8
RIGHT_ALT = 100  # evdev KEY_RIGHTALT, AltGr in layouts that have it


@dataclass(frozen=True)
class Stroke:
    code: int  # evdev key code
    shift: bool = False
    level3: bool = False  # hold the layout's AltGr key


@dataclass
class Layout:
    name: str
    strokes: dict[str, Stroke]
    level3_code: int | None  # evdev code of the key that gives ISO_Level3_Shift


class _RuleNames(ctypes.Structure):
    _fields_ = [(name, ctypes.c_char_p) for name in ("rules", "model", "layout", "variant", "options")]


def _library():
    path = ctypes.util.find_library("xkbcommon") or "libxkbcommon.so.0"
    lib = ctypes.CDLL(path)
    lib.xkb_context_new.restype = ctypes.c_void_p
    lib.xkb_context_new.argtypes = [ctypes.c_int]
    lib.xkb_keymap_new_from_names.restype = ctypes.c_void_p
    lib.xkb_keymap_new_from_names.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RuleNames), ctypes.c_int]
    for name in ("xkb_keymap_min_keycode", "xkb_keymap_max_keycode", "xkb_keymap_num_layouts"):
        getattr(lib, name).restype = ctypes.c_uint32
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.xkb_keymap_num_levels_for_key.restype = ctypes.c_uint32
    lib.xkb_keymap_num_levels_for_key.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.xkb_keymap_key_get_syms_by_level.restype = ctypes.c_int
    lib.xkb_keymap_key_get_syms_by_level.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
    ]
    lib.xkb_keymap_key_get_mods_for_level.restype = ctypes.c_size_t
    lib.xkb_keymap_key_get_mods_for_level.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
    ]
    lib.xkb_keymap_mod_get_index.restype = ctypes.c_uint32
    lib.xkb_keymap_mod_get_index.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.xkb_keysym_to_utf32.restype = ctypes.c_uint32
    lib.xkb_keysym_to_utf32.argtypes = [ctypes.c_uint32]
    lib.xkb_keymap_unref.argtypes = [ctypes.c_void_p]
    lib.xkb_context_unref.argtypes = [ctypes.c_void_p]
    return lib


def _mod_bit(lib, keymap, name: str) -> int:
    index = lib.xkb_keymap_mod_get_index(keymap, name.encode())
    return 0 if index == 0xFFFFFFFF else 1 << index


def load_layouts(layout: str, variant: str = "", options: str = "") -> list[Layout]:
    """Character-to-key maps for each group of an xkb layout string such as "us,ru".

    Raises OSError when libxkbcommon is missing and ValueError when the
    layout does not compile.
    """
    lib = _library()
    context = lib.xkb_context_new(0)
    names = _RuleNames(b"evdev", b"pc105", layout.encode(), variant.encode(), options.encode())
    keymap = lib.xkb_keymap_new_from_names(context, ctypes.byref(names), 0)
    if not keymap:
        lib.xkb_context_unref(context)
        raise ValueError(f"xkb could not compile layout {layout!r} (variant {variant!r})")
    try:
        shift = _mod_bit(lib, keymap, "Shift")
        level3 = _mod_bit(lib, keymap, "Mod5") | _mod_bit(lib, keymap, "LevelThree")
        layouts = []
        group_names = [name.strip() for name in layout.split(",")]
        for group in range(lib.xkb_keymap_num_layouts(keymap)):
            strokes: dict[str, Stroke] = {}
            level3_code = None
            for keycode in range(lib.xkb_keymap_min_keycode(keymap), lib.xkb_keymap_max_keycode(keymap) + 1):
                code = keycode - EVDEV_OFFSET
                if code <= 0 or code > 248:
                    continue
                for level in range(lib.xkb_keymap_num_levels_for_key(keymap, keycode, group)):
                    syms = ctypes.POINTER(ctypes.c_uint32)()
                    if lib.xkb_keymap_key_get_syms_by_level(keymap, keycode, group, level, ctypes.byref(syms)) != 1:
                        continue
                    sym = syms[0]
                    if sym in XKB_KEYPAD_SYMS:
                        break
                    if level == 0 and sym == XKB_KEY_ISO_LEVEL3_SHIFT and level3_code != RIGHT_ALT:
                        level3_code = code  # prefer Right Alt, a key that real keyboards have
                    char = chr(lib.xkb_keysym_to_utf32(sym) or 0)
                    if char == "\r":
                        char = "\n"
                    if char == "\0" or (not char.isprintable() and char not in " \t\n"):
                        continue
                    masks = (ctypes.c_uint32 * 8)()
                    count = lib.xkb_keymap_key_get_mods_for_level(keymap, keycode, group, level, masks, 8)
                    for mask in sorted(masks[i] for i in range(count)):
                        if mask & ~(shift | level3):
                            continue  # needs Ctrl, Caps Lock or similar: skip
                        stroke = Stroke(code, bool(mask & shift), bool(mask & level3))
                        best = strokes.get(char)
                        if best is None or (best.shift + 2 * best.level3) > (stroke.shift + 2 * stroke.level3):
                            strokes[char] = stroke
                        break
            if level3_code != RIGHT_ALT:
                # Only Right Alt is a dependable AltGr key: key remappers may not
                # pass the virtual <LVL3> key that some layouts use instead.
                level3_code = None
                strokes = {ch: s for ch, s in strokes.items() if not s.level3}
            name = group_names[group] if group < len(group_names) else str(group)
            layouts.append(Layout(name, strokes, level3_code))
        return layouts
    finally:
        lib.xkb_keymap_unref(keymap)
        lib.xkb_context_unref(context)
