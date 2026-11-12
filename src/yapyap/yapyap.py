#!/usr/bin/env python3
import base64
import os
import sys
import ctypes
import ctypes.util
import importlib.util
import io
import json
import shutil
import signal
import socket
import subprocess
import threading
import time
import tempfile

REMOTE_HOST = os.environ.get("YAPYAP_REMOTE_HOST", "").strip() or None
INPUT_DEVICE = os.environ.get("YAPYAP_INPUT_DEVICE", "").strip() or None

# pywhispercpp installs libwhisper.so + libggml*.so into site-packages but the linker can't find them
if not REMOTE_HOST and not ctypes.util.find_library("whisper"):
    _spec = importlib.util.find_spec("_pywhispercpp")
    if _spec and _spec.origin:
        _site_dir = os.path.dirname(_spec.origin)
        if os.path.isfile(os.path.join(_site_dir, "libwhisper.so")):
            os.environ["LD_LIBRARY_PATH"] = _site_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import scipy.signal
import sounddevice as sd
from evdev import ecodes
from .ipc import SOCKET_PATH, request_transcription, tcp_bind_address
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
REMOTE_UNREACHABLE_TEXT = "[remove server unreachable]"


def _pactl_default_source_name():
    try:
        result = subprocess.run(
            ["pactl", "get-default-source"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip() or None


def _pactl_source_description(source_name):
    properties = _pactl_source_properties(source_name)
    return properties.get("Description")


def _pactl_source_properties(source_name):
    if not source_name:
        return {}

    try:
        result = subprocess.run(
            ["pactl", "list", "sources"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    in_source = False
    in_properties = False
    properties = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source #"):
            in_source = False
            in_properties = False
            continue
        if stripped == f"Name: {source_name}":
            in_source = True
            continue
        if not in_source:
            continue
        if stripped.startswith("Description: "):
            properties["Description"] = stripped.removeprefix("Description: ").strip()
            continue
        if stripped == "Properties:":
            in_properties = True
            continue
        if in_properties and " = " in stripped:
            key, value = stripped.split(" = ", 1)
            properties[key] = value.strip('"')

    return properties


def _pactl_default_source_alsa_device():
    properties = _pactl_source_properties(_pactl_default_source_name())
    if properties.get("api.alsa.pcm.stream") != "capture":
        return None

    card = properties.get("api.alsa.pcm.card") or properties.get("alsa.card")
    device = properties.get("alsa.device")
    if card is None or device is None:
        return None

    return f"hw:{card},{device}"


def _query_input_device(device):
    try:
        sd.query_devices(device, kind="input")
    except (ValueError, sd.PortAudioError):
        return None
    return device


def _default_source_sounddevice_device():
    properties = _pactl_source_properties(_pactl_default_source_name())
    if properties.get("api.alsa.pcm.stream") != "capture":
        return None

    card = properties.get("api.alsa.pcm.card") or properties.get("alsa.card")
    device = properties.get("alsa.device")
    candidates = []

    if card is not None:
        # Some UCM profiles expose a default Line/Mic source whose direct hw
        # node is unavailable or silent after reboot, while the card's base
        # capture node is the actual working microphone.
        candidates.extend(f"hw:{card},{idx}" for idx in range(4))
    if card is not None and device is not None:
        candidates.append(f"hw:{card},{device}")

    source_description = properties.get("Description")
    if source_description:
        candidates.append(source_description)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if _query_input_device(candidate):
            return candidate

    return None


def _sounddevice_input_devices():
    return [
        device
        for device in sd.query_devices()
        if int(device.get("max_input_channels", 0)) > 0
    ]


def _sounddevice_input_names():
    return [device["name"] for device in _sounddevice_input_devices()]


def _pactl_default_source_sounddevice_name():
    source_name = _pactl_default_source_name()
    source_description = _pactl_source_description(source_name)
    input_names = _sounddevice_input_names()
    if source_description and source_description in input_names:
        return source_description

    if source_name:
        source_tail = source_name.rsplit(".", 1)[-1].replace("_", " ").lower()
        for name in input_names:
            if source_tail and source_tail in name.lower():
                return name

    alsa_device = _pactl_default_source_alsa_device()
    if alsa_device:
        alsa_suffix = f"({alsa_device})"
        for name in input_names:
            if alsa_suffix in name:
                return name

    return None


def _pipewire_source_id(source_name):
    if not source_name:
        return None

    try:
        result = subprocess.run(
            ["pw-dump"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    for item in payload:
        if item.get("type") != "PipeWire:Interface:Node":
            continue
        props = item.get("info", {}).get("props", {})
        if props.get("media.class") != "Audio/Source":
            continue
        if props.get("node.name") == source_name:
            serial = props.get("object.serial")
            if serial is not None:
                return str(serial)
            return source_name

    return None


def resolve_input_device(configured_device):
    if is_default_input_device(configured_device):
        return _default_source_sounddevice_device() or None

    if configured_device.lower() == "alsa-default-source":
        return _pactl_default_source_alsa_device() or None

    return configured_device


def is_default_input_device(configured_device):
    return not configured_device or configured_device.lower() in {"default", "pulse-default"}


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
            "remote_host": REMOTE_HOST,
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

        # pkill scans every process (~50-100 ms); run it in the background so
        # stopping a recording does not wait for it before transcribing.
        threading.Thread(
            target=subprocess.run,
            args=(["pkill", f"-RTMIN+{WAYBAR_SIGNAL}", "waybar"],),
            kwargs={
                "check": False,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            },
            daemon=True,
        ).start()


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


def load_pcm16_bytes(audio_bytes, samplerate):
    timings = {}
    start = time.perf_counter()
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    timings["decode_pcm_s16le_s"] = time.perf_counter() - start
    audio = (audio.astype(np.float32) / 32768.0).flatten()

    if samplerate != WHISPER_SAMPLERATE:
        print(f"Resampling {samplerate}Hz -> {WHISPER_SAMPLERATE}Hz", file=sys.stderr)
        resample_start = time.perf_counter()
        num_samples = int(len(audio) * WHISPER_SAMPLERATE / samplerate)
        audio = scipy.signal.resample(audio, num_samples)
        timings["resample_s"] = time.perf_counter() - resample_start

    timings["prepare_audio_s"] = 0.0
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
            if payload.get("audio_encoding") == "pcm_s16le":
                audio_bytes = base64.b64decode(payload["audio_b64"])
                samplerate = int(payload["samplerate"])
                text, timings = self.recorder.transcribe_pcm16(audio_bytes, samplerate, trace=trace)
            elif "audio_b64" in payload:
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
    def __init__(self, runtime_state, enable_input=True):
        self.runtime_state = runtime_state
        self.enable_input = enable_input
        self.pressed_keys = set()
        self.recording = False
        self.audio_data = None
        self.stream = None
        self.capture_process = None
        self.capture_thread = None
        self.audio_backend = "pulse"
        self.model_lock = threading.Lock()
        self.samplerate = None
        self.input_device = None
        self.remote_host = REMOTE_HOST
        self.model = None
        
        if self.remote_host:
            print(f"Using remote transcription server: {self.remote_host}", file=sys.stderr)
        else:
            from pywhispercpp.model import Model

            # Initialize Whisper model
            print(f"Loading Whisper model: {MODEL_NAME}", file=sys.stderr)
            self.model = Model(MODEL_NAME, language="auto")

        if self.enable_input:
            try:
                self.refresh_input_device(force_log=True)
            except (ValueError, sd.PortAudioError) as e:
                print(f"Input device unavailable at startup: {e}", file=sys.stderr)

    def refresh_input_device(self, force_log=False):
        # Re-resolve "default" on each recording so reconnects/card renumbering
        # do not leave the service pinned to a stale hw:N,M device.
        if is_default_input_device(INPUT_DEVICE) and shutil.which("pw-record"):
            self.audio_backend = "pipewire"
            source_name = _pactl_default_source_name()
            next_input_device = _pipewire_source_id(source_name) or source_name
            next_samplerate = WHISPER_SAMPLERATE
            device_name = _pactl_source_description(source_name) or source_name or "PipeWire default source"

            if force_log or next_input_device != self.input_device or next_samplerate != self.samplerate:
                print(f"Samplerate: {next_samplerate}", file=sys.stderr)
                print(f"Input device: {device_name}", file=sys.stderr)

            self.input_device = next_input_device
            self.samplerate = next_samplerate
            return {
                "name": device_name,
                "default_samplerate": next_samplerate,
                "max_input_channels": 1,
            }

        self.audio_backend = "sounddevice"
        next_input_device = resolve_input_device(INPUT_DEVICE)
        try:
            device_info = sd.query_devices(next_input_device, kind="input")
        except (ValueError, sd.PortAudioError) as e:
            if next_input_device is None:
                raise

            print(
                f"Input device {next_input_device!r} unavailable ({e}); falling back to sounddevice default",
                file=sys.stderr,
            )
            next_input_device = None
            device_info = sd.query_devices(kind="input")
        next_samplerate = int(device_info["default_samplerate"])

        if force_log or next_input_device != self.input_device or next_samplerate != self.samplerate:
            print(f"Samplerate: {next_samplerate}", file=sys.stderr)
            print(f"Input device: {device_info['name']}", file=sys.stderr)

        self.input_device = next_input_device
        self.samplerate = next_samplerate
        return device_info

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
            device_info = self.refresh_input_device()
            print(f"🎤 Recording... Device: {device_info['name']}", file=sys.stderr)
            print(f"🎤 Device samplerate: {device_info['default_samplerate']}, channels: {device_info['max_input_channels']}", file=sys.stderr)

            self.recording = True
            self.runtime_state.update(active=True, recording=True)
            if self.audio_backend == "pipewire":
                self.start_pipewire_capture()
            elif self.audio_backend == "pulse":
                self.start_pulse_capture()
            else:
                self.stream = sd.InputStream(
                    device=self.input_device,
                    channels=1,
                    samplerate=self.samplerate,
                    callback=self.audio_callback,
                )
                print(f"🎤 Stream created: {self.stream}", file=sys.stderr)
                self.stream.start()
                print(f"🎤 Stream started, active={self.stream.active}", file=sys.stderr)
        except Exception:
            self.recording = False
            self.audio_data = []
            self.runtime_state.update(active=True, recording=False)
            if self.stream:
                self.stream.close()
                self.stream = None
            if self.capture_process:
                self.capture_process.terminate()
                self.capture_process = None
            raise

    def read_pulse_capture(self):
        if not self.capture_process or not self.capture_process.stdout:
            return

        fd = self.capture_process.stdout.fileno()
        while self.capture_process:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            self.audio_data.append(np.frombuffer(chunk, dtype=np.float32).copy())
            if len(self.audio_data) % 50 == 1:
                print(f"📊 Captured {len(self.audio_data)} chunks", file=sys.stderr)

    def start_pipewire_capture(self):
        command = [
            "pw-record",
            "--raw",
            "--format=f32",
            f"--rate={WHISPER_SAMPLERATE}",
            "--channels=1",
            "--latency=20ms",
        ]
        if self.input_device:
            command.extend(["--target", self.input_device])
        command.append("-")

        self.capture_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.capture_thread = threading.Thread(target=self.read_pulse_capture, daemon=True)
        self.capture_thread.start()
        print("🎤 PipeWire capture started", file=sys.stderr)

    def start_pulse_capture(self):
        command = [
            "parec",
            "--raw",
            "--format=float32le",
            f"--rate={WHISPER_SAMPLERATE}",
            "--channels=1",
            "--latency-msec=20",
            "--process-time-msec=10",
        ]
        self.capture_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.capture_thread = threading.Thread(target=self.read_pulse_capture, daemon=True)
        self.capture_thread.start()
        print("🎤 Pulse/PipeWire capture started", file=sys.stderr)

    def stop_recording(self):
        print(f"🛑 stop_recording called, self.recording={self.recording}", file=sys.stderr)
        if not self.recording:
            return

        self.recording = False
        self.runtime_state.update(active=True, recording=False)

        if self.capture_process:
            print("🛑 Stopping PipeWire/Pulse capture...", file=sys.stderr)
            self.capture_process.terminate()
            try:
                self.capture_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.capture_process.kill()
                self.capture_process.wait(timeout=2)
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            self.capture_process = None
            self.capture_thread = None
            print("🛑 PipeWire/Pulse capture stopped", file=sys.stderr)

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
                if self.remote_host:
                    print(REMOTE_UNREACHABLE_TEXT, flush=True)
                else:
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
                try:
                    self.start_recording()
                except (ValueError, sd.PortAudioError) as e:
                    print(f"❌ Could not start recording: {e}", file=sys.stderr)
                    self.runtime_state.update(active=True, recording=False)
            else:
                self.stop_recording()

    def transcribe_audio(self, audio, trace=False):
        if self.remote_host:
            return self.transcribe_audio_remote(audio, trace=trace)

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

    def transcribe_audio_remote(self, audio, trace=False):
        encode_start = time.perf_counter()
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        payload = {
            "trace": trace,
            "audio_b64": base64.b64encode(pcm16.tobytes()).decode("ascii"),
            "audio_encoding": "pcm_s16le",
            "samplerate": WHISPER_SAMPLERATE,
            "channels": 1,
        }
        encode_time = time.perf_counter() - encode_start
        response, client_timings = request_transcription(payload, host=self.remote_host)
        if "error" in response:
            raise RuntimeError(response["error"])

        timings = {
            "encode_pcm_s16le_s": encode_time,
            **client_timings,
        }
        if trace:
            timings.update(response.get("timings", {}))
        return response.get("text", ""), timings

    def transcribe_path(self, path, trace=False):
        print("🎯 Transcribing file request...", file=sys.stderr)
        if self.remote_host:
            with open(path, "rb") as handle:
                payload = {
                    "trace": trace,
                    "audio_b64": base64.b64encode(handle.read()).decode("ascii"),
                    "filename": os.path.basename(path),
                }
            response, client_timings = request_transcription(payload, host=self.remote_host)
            if "error" in response:
                raise RuntimeError(response["error"])
            timings = {**client_timings}
            if trace:
                timings.update(response.get("timings", {}))
            return response.get("text", ""), timings

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
        if self.remote_host:
            payload = {
                "trace": trace,
                "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                "filename": filename,
            }
            response, client_timings = request_transcription(payload, host=self.remote_host)
            if "error" in response:
                raise RuntimeError(response["error"])
            timings = {**client_timings}
            if trace:
                timings.update(response.get("timings", {}))
            return response.get("text", ""), timings

        start = time.perf_counter()
        audio, load_timings = load_audio_bytes(audio_bytes, filename)
        text, transcribe_timings = self.transcribe_audio(audio, trace=trace)
        timings = {
            **load_timings,
            **transcribe_timings,
            "server_total_s": time.perf_counter() - start,
        }
        return text, timings

    def transcribe_pcm16(self, audio_bytes, samplerate, trace=False):
        print("🎯 Transcribing uploaded pcm16 request...", file=sys.stderr)
        start = time.perf_counter()
        audio, load_timings = load_pcm16_bytes(audio_bytes, samplerate)
        text, transcribe_timings = self.transcribe_audio(audio, trace=trace)
        timings = {
            **load_timings,
            **transcribe_timings,
            "server_total_s": time.perf_counter() - start,
        }
        return text, timings
def main():
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))
    runtime_state = RuntimeState(STATE_PATH)
    recorder = VoiceRecorder(runtime_state, enable_input=not SERVER_ONLY)
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
