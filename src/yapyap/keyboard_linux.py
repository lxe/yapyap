#!/usr/bin/env python3
"""
Linux-specific keyboard monitoring using evdev.
"""
import sys
import select
from typing import Callable, List
from .keyboard_base import KeyboardMonitor

try:
    import evdev
    from evdev import ecodes
except ImportError:
    # Graceful fallback if evdev is not available
    evdev = None
    ecodes = None


class LinuxKeyboardMonitor(KeyboardMonitor):
    """Linux keyboard monitor using evdev."""
    
    def __init__(self):
        if evdev is None:
            raise ImportError("evdev is required for Linux keyboard monitoring")
        
        self.devices: List[evdev.InputDevice] = []
        self.device_map = {}
        self.running = False
    
    def _get_keyboard_devices(self) -> List[evdev.InputDevice]:
        """Find all keyboard input devices with EV_KEY capabilities."""
        devices = []
        for device_path in evdev.list_devices():
            try:
                device = evdev.InputDevice(device_path)
                if ecodes.EV_KEY in device.capabilities():
                    devices.append(device)
                    print(f"Found keyboard: {device_path} - {device.name}", file=sys.stderr)
                    
            except (OSError, IOError, PermissionError):
                continue
        
        if not devices:
            print("No keyboard devices found!", file=sys.stderr)
            raise RuntimeError("No keyboard devices found")
        
        return devices
    
    def normalize_key_name(self, key_code: int) -> str:
        """Convert evdev key code to standardized key name."""
        if ecodes is None:
            return f"KEY_{key_code}"
        
        key_name = ecodes.bytype[ecodes.EV_KEY].get(key_code, f"KEY_{key_code}")
        if isinstance(key_name, list):
            key_name = key_name[0]
        
        return key_name
    
    def start_monitoring(self, key_handler: Callable[[str, bool], None]) -> None:
        """Start monitoring keyboard events using evdev."""
        self.devices = self._get_keyboard_devices()
        print(f"Monitoring {len(self.devices)} keyboard device(s)", file=sys.stderr)
        
        self.device_map = {device.fd: device for device in self.devices}
        self.running = True
        
        while self.running:
            try:
                ready, _, _ = select.select(self.device_map.keys(), [], [], 0.1)  # 100ms timeout
                
                for fd in ready:
                    device = self.device_map[fd]
                    for event in device.read():
                        if event.type == ecodes.EV_KEY and event.value in (0, 1):
                            key_name = self.normalize_key_name(event.code)
                            is_pressed = event.value == 1
                            key_handler(key_name, is_pressed)
                            
            except (OSError, IOError):
                # Device disconnected, remove from monitoring
                if fd in self.device_map:
                    device = self.device_map[fd]
                    print(f"Device {device.path} disconnected", file=sys.stderr)
                    del self.device_map[fd]
                
                if not self.device_map:
                    print("All devices disconnected", file=sys.stderr)
                    break
                    
            except KeyboardInterrupt:
                print("Keyboard monitoring stopped", file=sys.stderr)
                break
    
    def stop_monitoring(self) -> None:
        """Stop monitoring keyboard events."""
        self.running = False
        
        # Close all device handles
        for device in self.devices:
            try:
                device.close()
            except:
                pass
        
        self.devices.clear()
        self.device_map.clear()
