#!/usr/bin/env python3
import sys
import os
import numpy as np
import scipy.signal
import sounddevice as sd
from .keyboard import monitor_keyboard_events
from .model import create_model

def parse_chord(chord_str):
    if not chord_str:
        return {"KEY_LEFTCTRL", "KEY_LEFTALT"}
    return set(k.strip() for k in chord_str.replace(",", " ").split() if k.strip())

CHORD = parse_chord(os.environ.get("CHORD"))
WHISPER_SAMPLERATE = 16000
# Default model depends on platform - will be set by engine selection
MODEL_NAME = os.getenv('MODEL')

class VoiceRecorder:
    def __init__(self):
        self.pressed_keys = set()
        self.recording = False
        self.audio_data = None
        self.stream = None
        
        # Initialize model - choose best engine for platform
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Use MLX-optimized Parakeet on Apple Silicon
            default_engine = 'parakeet-mlx'
        else:
            # Use full NeMo Parakeet on other platforms
            default_engine = 'parakeet'
        
        engine = os.getenv('ENGINE', default_engine)
        self.model = create_model(MODEL_NAME, language="auto", engine=engine)

        # Can't default this to 16000, because some audio libs can't handle it
        self.samplerate = int(sd.query_devices(kind="input")["default_samplerate"])
        print(f"Samplerate: {self.samplerate}", file=sys.stderr)
        sd.default.samplerate = self.samplerate
        sd.default.channels = 1

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        if self.recording:
            return

        self.recording = True
        self.audio_data = []
        print("🎤 Recording...", file=sys.stderr)

        self.stream = sd.InputStream(callback=self.audio_callback)
        self.stream.start()

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)

            if self.samplerate != WHISPER_SAMPLERATE:
                print(f"🔄 Resampling from {self.samplerate}Hz to 16000Hz", file=sys.stderr)
                num_samples = int(len(audio) * WHISPER_SAMPLERATE / self.samplerate)
                audio = scipy.signal.resample(audio, num_samples)

            print("🎯 Transcribing...", file=sys.stderr)
            segments = self.model.transcribe(audio)
            text = ' '.join(segment.text for segment in segments).strip()
            
            if text:
                print(text, flush=True)
            else:
                print("(no speech detected)", file=sys.stderr)

    def handle_key(self, key_name, pressed):
        # key_name is already in the correct format (e.g., "KEY_LEFTCTRL")
        debug_keys = os.getenv('DEBUG_KEYS', '').lower() in ('1', 'true', 'yes')
        
        if debug_keys:
            print(f"DEBUG: handle_key called with key_name='{key_name}', pressed={pressed}", file=sys.stderr)
            print(f"DEBUG: CHORD={CHORD}, pressed_keys={self.pressed_keys}", file=sys.stderr)
        
        if key_name in CHORD:
            if debug_keys:
                print(f"DEBUG: Key {key_name} is in CHORD", file=sys.stderr)
            if pressed:
                self.pressed_keys.add(key_name)
                if debug_keys:
                    print(f"DEBUG: Added key, pressed_keys now: {self.pressed_keys}", file=sys.stderr)
            else:
                self.pressed_keys.discard(key_name)
                if debug_keys:
                    print(f"DEBUG: Removed key, pressed_keys now: {self.pressed_keys}", file=sys.stderr)

            if CHORD.issubset(self.pressed_keys):
                if debug_keys:
                    print(f"DEBUG: CHORD is subset of pressed_keys - starting recording", file=sys.stderr)
                self.start_recording()
            else:
                if debug_keys:
                    print(f"DEBUG: CHORD is NOT subset of pressed_keys - stopping recording", file=sys.stderr)
                self.stop_recording()
        else:
            if debug_keys:
                print(f"DEBUG: Key {key_name} is NOT in CHORD", file=sys.stderr)

def main():
    recorder = VoiceRecorder()

    print(f"Press {', '.join(CHORD)} to record voice", file=sys.stderr)

    try:
        monitor_keyboard_events(recorder.handle_key)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        if recorder.recording:
            recorder.stop_recording()
    except PermissionError as e:
        print(f"Permission denied: {e}", file=sys.stderr)
        print("Try running with sudo or add your user to the input group", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
