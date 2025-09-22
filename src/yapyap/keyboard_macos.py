#!/usr/bin/env python3
"""
macOS/Windows keyboard monitoring using pynput.
"""
import sys
import threading
from typing import Callable, Optional
from .keyboard_base import KeyboardMonitor, normalize_key_name

try:
    from pynput import keyboard
except ImportError:
    # Graceful fallback if pynput is not available
    keyboard = None


class MacOSKeyboardMonitor(KeyboardMonitor):
    """Cross-platform keyboard monitor using pynput (works on macOS, Windows, Linux)."""
    
    def __init__(self):
        if keyboard is None:
            raise ImportError("pynput is required for macOS/Windows keyboard monitoring")
        
        self.listener: Optional[keyboard.Listener] = None
        self.running = False
        self._key_handler: Optional[Callable[[str, bool], None]] = None
        self.fn_pressed = False  # Track Fn key state manually
    
    def normalize_key_name(self, key) -> str:
        """Convert pynput key to standardized key name."""
        # Handle KeyCode objects (like Fn key)
        if hasattr(key, 'vk') and key.vk == 63:
            return "KEY_FN"
        
        # Handle special keys
        if hasattr(key, 'name'):
            key_name = key.name.lower()
        elif hasattr(key, 'char') and key.char:
            # Regular character keys
            key_name = key.char.lower()
        else:
            # Fallback
            key_name = str(key).lower()
        
        # Clean up the key name
        key_name = key_name.replace('key.', '').replace('_', '').replace('-', '')
        
        # Use our standard normalization
        return normalize_key_name(key_name)
    
    def _on_press(self, key):
        """Handle key press events from pynput."""
        import os
        debug_keys = os.getenv('DEBUG_KEYS', '').lower() in ('1', 'true', 'yes')
        
        if debug_keys:
            print(f"DEBUG: Key pressed - raw: {key}, type: {type(key)}", file=sys.stderr)
        
        if self._key_handler and self.running:
            try:
                key_name = self.normalize_key_name(key)
                if debug_keys:
                    print(f"DEBUG: Normalized key name: {key_name}", file=sys.stderr)
                
                # Special handling for Fn key - treat any Fn event as a toggle
                if key_name == "KEY_FN":
                    self.fn_pressed = not self.fn_pressed
                    if debug_keys:
                        print(f"DEBUG: Fn key toggled to: {self.fn_pressed}", file=sys.stderr)
                    self._key_handler(key_name, self.fn_pressed)
                else:
                    self._key_handler(key_name, True)  # True = pressed
            except Exception as e:
                print(f"Error handling key press: {e}", file=sys.stderr)
    
    def _on_release(self, key):
        """Handle key release events from pynput."""
        import os
        debug_keys = os.getenv('DEBUG_KEYS', '').lower() in ('1', 'true', 'yes')
        
        if debug_keys:
            print(f"DEBUG: Key released - raw: {key}, type: {type(key)}", file=sys.stderr)
        
        if self._key_handler and self.running:
            try:
                key_name = self.normalize_key_name(key)
                if debug_keys:
                    print(f"DEBUG: Normalized key name (release): {key_name}", file=sys.stderr)
                
                # Special handling for Fn key - treat any Fn event as a toggle
                if key_name == "KEY_FN":
                    self.fn_pressed = not self.fn_pressed
                    if debug_keys:
                        print(f"DEBUG: Fn key toggled to: {self.fn_pressed}", file=sys.stderr)
                    self._key_handler(key_name, self.fn_pressed)
                else:
                    self._key_handler(key_name, False)  # False = released
            except Exception as e:
                print(f"Error handling key release: {e}", file=sys.stderr)
    
    def start_monitoring(self, key_handler: Callable[[str, bool], None]) -> None:
        """Start monitoring keyboard events using pynput."""
        if self.running:
            print("Keyboard monitoring already running", file=sys.stderr)
            return
        
        self._key_handler = key_handler
        self.running = True
        
        print("Starting keyboard monitoring (pynput)", file=sys.stderr)
        
        # Check for macOS permissions
        if sys.platform == "darwin":
            print("Note: On macOS, you may need to grant accessibility permissions", file=sys.stderr)
            print("Go to System Preferences > Security & Privacy > Privacy > Accessibility", file=sys.stderr)
        
        try:
            # Create and start the listener
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress=False  # Don't suppress key events
            )
            
            # Start the listener in a separate thread
            self.listener.start()
            
            # Keep the main thread alive
            self.listener.join()
            
        except Exception as e:
            print(f"Error starting keyboard monitoring: {e}", file=sys.stderr)
            if "accessibility" in str(e).lower() or "permission" in str(e).lower():
                print("This appears to be a permissions issue.", file=sys.stderr)
                print("On macOS: Enable accessibility permissions in System Preferences", file=sys.stderr)
            raise
    
    def stop_monitoring(self) -> None:
        """Stop monitoring keyboard events."""
        self.running = False
        
        if self.listener:
            try:
                self.listener.stop()
            except:
                pass
            self.listener = None
        
        self._key_handler = None
        print("Keyboard monitoring stopped", file=sys.stderr)
