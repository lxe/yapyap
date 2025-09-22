#!/usr/bin/env python3
"""
Cross-platform keyboard monitoring interface.
This module provides a unified interface for keyboard monitoring across different platforms.
"""
from typing import Callable
from .keyboard_base import create_keyboard_monitor


def monitor_keyboard_events(key_handler: Callable[[str, bool], None]) -> None:
    """
    Monitor keyboard events across platforms.
    
    Args:
        key_handler: Function that receives (key_name, is_pressed) events
                    key_name is in evdev format (e.g., "KEY_LEFTCTRL", "KEY_A")
                    is_pressed is True for press, False for release
    """
    monitor = create_keyboard_monitor()
    
    # Wrapper to convert from new interface to old interface format
    def unified_handler(key_name: str, is_pressed: bool):
        # The new interface passes key names directly
        # The old interface expected (key_code, is_pressed) but we'll adapt
        key_handler(key_name, is_pressed)
    
    try:
        monitor.start_monitoring(unified_handler)
    except KeyboardInterrupt:
        print("Keyboard monitoring stopped", file=sys.stderr)
    finally:
        monitor.stop_monitoring()
