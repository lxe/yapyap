#!/usr/bin/env python3
import sys
import evdev
import select
from evdev import ecodes

def get_keyboard_devices():
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
        sys.exit(1)
    
    return devices

def monitor_keyboard_events(key_handler):
    devices = get_keyboard_devices()
    print(f"Monitoring {len(devices)} keyboard device(s)", file=sys.stderr)
    device_map = {device.fd: device for device in devices}
    
    while True:
        try:
            ready, _, _ = select.select(device_map.keys(), [], [])
            
            for fd in ready:
                device = device_map[fd]
                for event in device.read():
                    if event.type == ecodes.EV_KEY and event.value in (0, 1):
                        key_handler(event.code, event.value == 1)
                        
        except (OSError, IOError):
            # Device disconnected, remove from monitoring
            print(f"Device {device.path} disconnected", file=sys.stderr)
            del device_map[fd]
            if not device_map:
                print("All devices disconnected", file=sys.stderr)
                return
                
        except KeyboardInterrupt:
            print("Keyboard monitoring stopped", file=sys.stderr)
            return
