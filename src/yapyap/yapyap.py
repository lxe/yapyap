#!/usr/bin/env python3
import base64
import sys
import os
import ctypes
import ctypes.util
import importlib.util
import io
import json
import socket
import subprocess
import threading
import time
import tempfile

# pywhispercpp installs libwhisper.so + libggml*.so into site-packages but the linker can't find them
if not ctypes.util.find_library("whisper"):
    _spec = importlib.util.find_spec("_pywhispercpp")
    if _spec and _spec.origin:
        _site_dir = os.path.dirname(_spec.origin)
        if os.path.isfile(os.path.join(_site_dir, "libwhisper.so")):
            os.environ["LD_LIBRARY_PATH"] = _site_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import scipy.signal
import sounddevice as sd
from pywhispercpp.model import Model
from evdev import ecodes
from .ipc import SOCKET_PATH, tcp_bind_address
from .keyboard import monitor_keyboard_events

def parse_chord(chord_str):
    if not chord_str:
        return {"KEY_LEFTCTRL", "KEY_LEFTALT"}
    return set(k.strip() for k in chord_str.replace(",", " ").split() if k.strip())

CHORD = parse_chord(os.environ.get("CHORD"))
WHISPER_SAMPLERATE = 16000
MODEL_NAME = os.getenv('MODEL', 'large-v3-turbo-q8_0')
WAYBAR_SIGNAL = os.environ.get("YAPYAP_WAYBAR_SIGNAL")
SERVER_ONLY = os.environ.get("YAPYAP_SERVER_ONLY", "").lower() in {"1", "true", "yes", "on"}
TCP_BIND = tcp_bind_address()


def default_state_path():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return os.path.join(runtime_dir, f"yapyap-state-{os.getuid()}.json")
    return os.path.join(tempfile.gettempdir(), f"yapyap-state-{os.getuid()}.json")


STATE_PATH = os.environ.get("YAPYAP_STATE", default_state_path())


def listener_label():
    if TCP_BIND:
        host, port = TCP_BIND
        return f"tcp://{host}:{port}"
    return SOCKET_PATH


class RuntimeState:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.payload = {
            "active": False,
            "recording": False,
            "pid": os.getpid(),
            "socket_path": SOCKET_PATH if not TCP_BIND else None,
            "listen": listener_label(),
            "state_path": self.path,
            "model": MODEL_NAME,
            "chord": sorted(CHORD),
        }

    def update(self, **changes):
        with self.lock:
            next_payload = {**self.payload, **changes, "updated_at": time.time()}
            if next_payload == self.payload:
                return

            self.payload = next_payload
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp_path = f"{self.path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(self.payload, handle)
            os.replace(tmp_path, self.path)

        self.notify_waybar()

    def notify_waybar(self):
        if not WAYBAR_SIGNAL:
            return

        subprocess.run(
            ["pkill", f"-RTMIN+{WAYBAR_SIGNAL}", "waybar"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def load_audio_file(path):
    import soundfile as sf

    timings = {}
    start = time.perf_counter()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    print(f"Reading: {path}", file=sys.stderr)
    read_start = time.perf_counter()
    audio, samplerate = sf.read(path, dtype="float32")
    timings["read_file_s"] = time.perf_counter() - read_start

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if samplerate != WHISPER_SAMPLERATE:
        print(f"Resampling {samplerate}Hz -> {WHISPER_SAMPLERATE}Hz", file=sys.stderr)
        resample_start = time.perf_counter()
        num_samples = int(len(audio) * WHISPER_SAMPLERATE / samplerate)
        audio = scipy.signal.resample(audio, num_samples)
        timings["resample_s"] = time.perf_counter() - resample_start

    convert_start = time.perf_counter()
    audio = audio.astype(np.float32).flatten()
    timings["prepare_audio_s"] = time.perf_counter() - convert_start
    timings["load_audio_total_s"] = time.perf_counter() - start
    return audio, timings


def load_audio_bytes(audio_bytes, filename):
    import soundfile as sf

    timings = {}
    start = time.perf_counter()
    print(f"Reading uploaded audio: {filename}", file=sys.stderr)
    read_start = time.perf_counter()
    audio, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    timings["read_file_s"] = time.perf_counter() - read_start

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if samplerate != WHISPER_SAMPLERATE:
        print(f"Resampling {samplerate}Hz -> {WHISPER_SAMPLERATE}Hz", file=sys.stderr)
        resample_start = time.perf_counter()
        num_samples = int(len(audio) * WHISPER_SAMPLERATE / samplerate)
        audio = scipy.signal.resample(audio, num_samples)
        timings["resample_s"] = time.perf_counter() - resample_start

    convert_start = time.perf_counter()
    audio = audio.astype(np.float32).flatten()
    timings["prepare_audio_s"] = time.perf_counter() - convert_start
    timings["load_audio_total_s"] = time.perf_counter() - start
    timings["uploaded_bytes"] = len(audio_bytes)
    return audio, timings


class TranscriptionServer:
    def __init__(self, recorder, socket_path, tcp_bind):
        self.recorder = recorder
        self.socket_path = socket_path
        self.tcp_bind = tcp_bind
        self.server = None
        self.thread = None
        self.running = False

    def start(self):
        if self.tcp_bind:
            host, port = self.tcp_bind
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((host, port))
            endpoint_label = f"tcp://{host}:{port}"
        else:
            if os.path.exists(self.socket_path):
                try:
                    os.unlink(self.socket_path)
                except OSError as e:
                    print(f"Failed to remove stale socket {self.socket_path}: {e}", file=sys.stderr)
                    raise

            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(self.socket_path)
            os.chmod(self.socket_path, 0o600)
            endpoint_label = self.socket_path

        self.server.listen()
        self.server.settimeout(1.0)
        self.running = True
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()
        print(f"CLI socket listening on {endpoint_label}", file=sys.stderr)

    def stop(self):
        self.running = False
        if self.server:
            self.server.close()
            self.server = None
        if not self.tcp_bind and os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def serve_forever(self):
        while self.running:
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                try:
                    self.handle_connection(conn)
                except Exception as e:
                    print(f"CLI connection failed: {e}", file=sys.stderr)

    def handle_connection(self, conn):
        response = None
        try:
            request = conn.makefile("r", encoding="utf-8").readline()
            if not request:
                return
            payload = json.loads(request)
            trace = payload.get("trace", False)
            if "audio_b64" in payload:
                audio_bytes = base64.b64decode(payload["audio_b64"])
                filename = payload.get("filename", "uploaded audio")
                text, timings = self.recorder.transcribe_bytes(audio_bytes, filename, trace=trace)
            else:
                path = payload["path"]
                text, timings = self.recorder.transcribe_path(path, trace=trace)
            response = {"text": text}
            if trace:
                response["timings"] = timings
        except Exception as e:
            response = {"error": str(e)}

        if response is None:
            return

        try:
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except BrokenPipeError:
            print("CLI client disconnected before receiving response", file=sys.stderr)

class VoiceRecorder:
    def __init__(self, runtime_state):
        self.runtime_state = runtime_state
        self.pressed_keys = set()
        self.recording = False
        self.audio_data = None
        self.stream = None
        self.model_lock = threading.Lock()
        
        # Initialize Whisper model
        print(f"Loading Whisper model: {MODEL_NAME}", file=sys.stderr)
        self.model = Model(MODEL_NAME, language="auto")

        # Can't default this to 16000, because some audio libs can't handle it
        self.samplerate = int(sd.query_devices(kind="input")["default_samplerate"])
        print(f"Samplerate: {self.samplerate}", file=sys.stderr)
        sd.default.samplerate = self.samplerate
        sd.default.channels = 1

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"⚠️  Audio status: {status}", file=sys.stderr)
        if self.recording:
            self.audio_data.append(indata.copy())
            if len(self.audio_data) % 50 == 1:  # Print every ~1 sec
                print(f"📊 Captured {len(self.audio_data)} chunks", file=sys.stderr)

    def start_recording(self):
        if self.recording:
            return

        self.audio_data = []

        try:
            # Debug audio device info
            device_info = sd.query_devices(kind="input")
            print(f"🎤 Recording... Device: {device_info['name']}", file=sys.stderr)
            print(f"🎤 Device samplerate: {device_info['default_samplerate']}, channels: {device_info['max_input_channels']}", file=sys.stderr)

            self.stream = sd.InputStream(callback=self.audio_callback)
            print(f"🎤 Stream created: {self.stream}", file=sys.stderr)
            self.stream.start()
            self.recording = True
            self.runtime_state.update(active=True, recording=True)
            print(f"🎤 Stream started, active={self.stream.active}", file=sys.stderr)
        except Exception:
            self.recording = False
            self.audio_data = []
            self.runtime_state.update(active=True, recording=False)
            if self.stream:
                self.stream.close()
                self.stream = None
            raise

    def stop_recording(self):
        print(f"🛑 stop_recording called, self.recording={self.recording}", file=sys.stderr)
        if not self.recording:
            return

        self.recording = False
        self.runtime_state.update(active=True, recording=False)

        if self.stream:
            print("🛑 Aborting stream...", file=sys.stderr)
            self.stream.abort()  # abort() doesn't wait for buffers, unlike stop()
            print("🛑 Stream aborted", file=sys.stderr)
            self.stream.close()
            print("🛑 Stream closed", file=sys.stderr)
            self.stream = None

        print(f"🛑 audio_data chunks: {len(self.audio_data) if self.audio_data else 0}", file=sys.stderr)

        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            print(f"🛑 Audio shape: {audio.shape}, dtype: {audio.dtype}", file=sys.stderr)
            print(f"🛑 Audio min: {audio.min():.4f}, max: {audio.max():.4f}", file=sys.stderr)

            if self.samplerate != WHISPER_SAMPLERATE:
                print(f"🔄 Resampling from {self.samplerate}Hz to 16000Hz", file=sys.stderr)
                num_samples = int(len(audio) * WHISPER_SAMPLERATE / self.samplerate)
                audio = scipy.signal.resample(audio, num_samples)
                print(f"🛑 After resample shape: {audio.shape}", file=sys.stderr)

            # Ensure audio is float32 and 1D
            audio = audio.flatten().astype(np.float32)
            print(f"🛑 Final audio shape: {audio.shape}, dtype: {audio.dtype}", file=sys.stderr)

            print("🎯 Transcribing...", file=sys.stderr)
            try:
                text, _ = self.transcribe_audio(audio)

                if text:
                    print(text, flush=True)
                else:
                    print("(no speech detected)", file=sys.stderr)
            except Exception as e:
                print(f"❌ Transcription error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    def handle_key(self, key_code, pressed):
        key_name = ecodes.bytype[ecodes.EV_KEY][key_code]
        if isinstance(key_name, list):
            key_name = key_name[0]

        if key_name in CHORD:
            if pressed:
                self.pressed_keys.add(key_name)
            else:
                self.pressed_keys.discard(key_name)

            print(f"🔑 Key: {key_name}, pressed={pressed}, active={self.pressed_keys}", file=sys.stderr)

            if CHORD.issubset(self.pressed_keys):
                self.start_recording()
            else:
                self.stop_recording()

    def transcribe_audio(self, audio, trace=False):
        start = time.perf_counter()
        with self.model_lock:
            inference_start = time.perf_counter()
            segments = list(self.model.transcribe(audio))
        inference_time = time.perf_counter() - inference_start
        print(f"🛑 Got {len(segments)} segments", file=sys.stderr)
        text = " ".join(segment.text for segment in segments).strip()
        timings = {
            "inference_s": inference_time,
            "transcribe_audio_total_s": time.perf_counter() - start,
        }
        return text, timings

    def transcribe_path(self, path, trace=False):
        print("🎯 Transcribing file request...", file=sys.stderr)
        start = time.perf_counter()
        audio, load_timings = load_audio_file(path)
        text, transcribe_timings = self.transcribe_audio(audio, trace=trace)
        timings = {
            **load_timings,
            **transcribe_timings,
            "server_total_s": time.perf_counter() - start,
        }
        return text, timings

    def transcribe_bytes(self, audio_bytes, filename, trace=False):
        print("🎯 Transcribing uploaded file request...", file=sys.stderr)
        start = time.perf_counter()
        audio, load_timings = load_audio_bytes(audio_bytes, filename)
        text, transcribe_timings = self.transcribe_audio(audio, trace=trace)
        timings = {
            **load_timings,
            **transcribe_timings,
            "server_total_s": time.perf_counter() - start,
        }
        return text, timings
def main():
    runtime_state = RuntimeState(STATE_PATH)
    recorder = VoiceRecorder(runtime_state)
    server = TranscriptionServer(recorder, SOCKET_PATH, TCP_BIND)
    server.start()
    runtime_state.update(active=True, recording=False)

    if SERVER_ONLY:
        print(f"Running in server-only mode on {listener_label()}", file=sys.stderr)
    else:
        print(f"Press {', '.join(CHORD)} to record voice", file=sys.stderr)

    try:
        if SERVER_ONLY:
            while True:
                time.sleep(3600)
        else:
            monitor_keyboard_events(recorder.handle_key)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        if recorder.recording:
            recorder.stop_recording()
    except PermissionError as e:
        print(f"Permission denied: {e}", file=sys.stderr)
        print("Try running with sudo or add your user to the input group", file=sys.stderr)
        sys.exit(1)
    finally:
        runtime_state.update(active=False, recording=False)
        server.stop()

if __name__ == "__main__":
    main()
