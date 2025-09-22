#!/usr/bin/env python3
"""
Cross-platform keyboard monitoring abstraction.
"""
from abc import ABC, abstractmethod
from typing import Callable, Set
import sys
import platform


class KeyboardMonitor(ABC):
    """Abstract base class for platform-specific keyboard monitoring."""
    
    @abstractmethod
    def start_monitoring(self, key_handler: Callable[[str, bool], None]) -> None:
        """
        Start monitoring keyboard events.
        
        Args:
            key_handler: Callback function that receives (key_name, is_pressed)
        """
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop monitoring keyboard events."""
        pass
    
    @abstractmethod
    def normalize_key_name(self, key) -> str:
        """Convert platform-specific key representation to standardized name."""
        pass


def create_keyboard_monitor() -> KeyboardMonitor:
    """Factory function to create appropriate keyboard monitor for current platform."""
    system = platform.system()
    
    if system == "Linux":
        from .keyboard_linux import LinuxKeyboardMonitor
        return LinuxKeyboardMonitor()
    elif system == "Darwin":  # macOS
        from .keyboard_macos import MacOSKeyboardMonitor
        return MacOSKeyboardMonitor()
    elif system == "Windows":
        from .keyboard_macos import MacOSKeyboardMonitor  # pynput works on Windows too
        return MacOSKeyboardMonitor()
    else:
        raise NotImplementedError(f"Keyboard monitoring not supported on {system}")


# Standard key name mappings for cross-platform compatibility
STANDARD_KEY_NAMES = {
    # Modifier keys
    "ctrl_l": "KEY_LEFTCTRL",
    "ctrl_r": "KEY_RIGHTCTRL", 
    "alt_l": "KEY_LEFTALT",
    "alt_r": "KEY_RIGHTALT",
    "shift_l": "KEY_LEFTSHIFT",
    "shift_r": "KEY_RIGHTSHIFT",
    "cmd": "KEY_LEFTMETA",  # macOS Command key
    "cmd_l": "KEY_LEFTMETA",
    "cmd_r": "KEY_RIGHTMETA",
    "super": "KEY_LEFTMETA",  # Linux Super key
    "super_l": "KEY_LEFTMETA",
    "super_r": "KEY_RIGHTMETA",
    
    # Function keys
    "f1": "KEY_F1",
    "f2": "KEY_F2",
    "f3": "KEY_F3",
    "f4": "KEY_F4",
    "f5": "KEY_F5",
    "f6": "KEY_F6",
    "f7": "KEY_F7",
    "f8": "KEY_F8",
    "f9": "KEY_F9",
    "f10": "KEY_F10",
    "f11": "KEY_F11",
    "f12": "KEY_F12",
    
    # Special keys
    "space": "KEY_SPACE",
    "enter": "KEY_ENTER",
    "return": "KEY_ENTER",
    "tab": "KEY_TAB",
    "esc": "KEY_ESC",
    "escape": "KEY_ESC",
    "backspace": "KEY_BACKSPACE",
    "delete": "KEY_DELETE",
    "fn": "KEY_FN",
    
    # Arrow keys
    "up": "KEY_UP",
    "down": "KEY_DOWN",
    "left": "KEY_LEFT",
    "right": "KEY_RIGHT",
}


def normalize_key_name(key_name: str) -> str:
    """Convert various key name formats to standardized evdev-style names."""
    key_lower = key_name.lower().replace("key.", "").replace("_", "").replace("-", "")
    
    # Handle single characters
    if len(key_lower) == 1 and key_lower.isalpha():
        return f"KEY_{key_lower.upper()}"
    
    # Handle numbers
    if len(key_lower) == 1 and key_lower.isdigit():
        return f"KEY_{key_lower}"
    
    # Handle special mappings
    return STANDARD_KEY_NAMES.get(key_lower, f"KEY_{key_name.upper()}")
