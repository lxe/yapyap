# Yap Yap

Push-to-talk dictation for Linux. Hold a key chord, speak, and release: the words appear in the focused window as real key presses, typed while the server is still transcribing the end of the sentence.

https://github.com/user-attachments/assets/208f949a-3d26-46b5-9b64-52b273e1b00b

Yap Yap has two parts. The **server** runs [Whisper](https://github.com/openai/whisper) on a GPU or CPU and serves the OpenAI transcription API, so it can live on another machine on the network and any OpenAI-compatible tool can use it too. The **client** runs on the Linux desktop: it records while the chord is held, streams the audio to the server and types the text as it arrives.

| Server hardware | Engine | Model | First words after release* |
|---|---|---|---|
| Intel Arc 140V (integrated) | [OpenVINO](https://github.com/openvinotoolkit/openvino.genai) | whisper-large-v3-turbo FP16 | about 250 ms |
| NVIDIA GPU | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) with CUDA | large-v3-turbo Q8 | depends on the GPU |
| Any x86-64 or ARM CPU | whisper.cpp | large-v3-turbo Q8, or a smaller model | seconds, depending on cores |

\*Time from releasing the chord to the first typed word, for a sentence of 5 to 20 seconds. The rest of the sentence follows at the speed of decoding.

## Server

### Docker

Docker Compose builds and runs the server. Clone the repository on the machine that will transcribe, then create a `.env` file next to `compose.yaml` with an API token that clients will send:

```bash
git clone https://github.com/lxe/yapyap && cd yapyap
echo "YAPYAP_TOKEN=$(openssl rand -hex 24)" > .env
```

Start the service that matches the hardware. On an Intel GPU, the container needs the host's group ID of the GPU render node as well:

```bash
# Intel GPU (Arc, Iris Xe, Core Ultra)
echo "RENDER_GID=$(stat -c %g /dev/dri/renderD128)" >> .env
docker compose --profile intel up -d

# NVIDIA GPU (needs the NVIDIA Container Toolkit)
docker compose --profile nvidia up -d

# CPU only
docker compose --profile cpu up -d
```

The first start downloads the model (about 1.6 GB) into the `models` volume, and the server answers on port 8122 once `curl http://localhost:8122/health` returns `{"status": "ok"}`. Logs come from `docker compose --profile intel logs -f`, with the profile name of the running service.

To use a different model, set `YAPYAP_MODEL` in `.env`: a Hugging Face repository with an OpenVINO export for the Intel service (for example `OpenVINO/whisper-small-fp16-ov`), or a [whisper.cpp model name](https://huggingface.co/ggerganov/whisper.cpp) for the others (for example `small-q8_0`).

### Without Docker

The server is a Python package with one extra for each engine. With [uv](https://docs.astral.sh/uv/):

```bash
# Intel GPU: needs Intel's GPU compute runtime (intel-opencl-icd / intel-compute-runtime)
uv tool install "yapyap[openvino] @ git+https://github.com/lxe/yapyap"
yapyap-server --engine openvino --device GPU

# NVIDIA GPU: builds whisper.cpp against the installed CUDA toolkit
GGML_CUDA=1 uv tool install --no-binary-package pywhispercpp "yapyap[whispercpp] @ git+https://github.com/lxe/yapyap"
yapyap-server --engine whispercpp
```

`yapyap-server --help` lists the options. Each one can also come from an environment variable (`YAPYAP_ENGINE`, `YAPYAP_MODEL`, `YAPYAP_DEVICE`, `YAPYAP_HOST`, `YAPYAP_PORT`, `YAPYAP_TOKEN` or `YAPYAP_TOKEN_FILE`), which suits a systemd unit. The server listens on 127.0.0.1 by default; `--host 0.0.0.0` opens it to the network, and the server then warns unless a token is set.

### API

The server implements the transcription endpoint of the [OpenAI audio API](https://platform.openai.com/docs/api-reference/audio/createTranscription):

```bash
curl http://localhost:8122/v1/audio/transcriptions \
  -H "Authorization: Bearer $YAPYAP_TOKEN" \
  -F file=@speech.wav -F stream=true
```

The multipart form takes `file` and the optional fields `language` (such as `en`; detected when absent), `prompt` (text that hints at vocabulary or a mix of languages), `stream` and `response_format` (`json` or `text`). With `stream=true`, the answer is a stream of server-sent events, one `transcript.text.delta` for each piece of text as the model decodes it and a final `transcript.text.done` with the whole transcript. Audio in 16 kHz mono 16-bit WAV is read directly, and the Docker images include ffmpeg for other formats. `GET /health` needs no token, and `GET /v1/models` names the loaded model.

## Client

The client needs Linux with Wayland, Python 3.12, PipeWire or PulseAudio, and read access to `/dev/input` for the key chord, which comes with membership in the `input` group (log out and back in after this):

```bash
sudo usermod -aG input $USER
```

Install the client and its systemd user service:

```bash
uv tool install "yapyap[client] @ git+https://github.com/lxe/yapyap"
yapyap install
```

`yapyap install` writes `~/.config/yapyap/config.toml` with every setting at its default, and a user service that starts the client at login. Point the client at the server in that file:

```toml
[server]
url = "http://192.168.1.20:8122"
token = "the YAPYAP_TOKEN from the server's .env"
```

Then `yapyap check` confirms that the client reaches the server, and `systemctl --user restart yapyap` applies the change. Hold Left Ctrl + Left Alt, speak and release to dictate. `journalctl --user -u yapyap -f` shows each recording with its timings.

### Typing

By default the client types the transcript through a virtual keyboard, so the text works anywhere that keyboard input works, including terminals, remote desktop clients and games. Characters outside the US keyboard layout go through [wtype](https://github.com/atx/wtype), so install wtype for accented letters and other scripts.

Russian types as fast as English on [Hyprland](https://hyprland.org) when Russian is a second keyboard layout. The client then switches the keyboard to the Russian layout for Cyrillic and back afterwards:

```
input {
    kb_layout = us,ru
}
```

With `mode = "paste"` in the `[output]` section, the client pastes each transcript into the focused window instead (Hyprland only), and with `mode = "stdout"` it prints one line per transcript for other programs to read:

```bash
yapyap run | while read -r line; do notify-send "$line"; done
```

### Settings

All settings live in `~/.config/yapyap/config.toml`, which `yapyap install` creates with comments. These are the ones most worth changing:

| Section | Setting | Default | Meaning |
|---|---|---|---|
| `server` | `language` | detect | Language code such as `en` or `ru`, which skips detection |
| `server` | `prompt` | none | Words or a sample sentence that steer spelling and mixed-language speech |
| `hotkey` | `chord` | `["KEY_LEFTCTRL", "KEY_LEFTALT"]` | Keys to hold, by their [evdev names](https://github.com/torvalds/linux/blob/master/include/uapi/linux/input-event-codes.h) |
| `capture` | `target` | default source | PipeWire or PulseAudio source to record from |
| `output` | `mode` | `type` | `type`, `paste` or `stdout` |
| `ducking` | `enabled` | `false` | Lower other applications' volume while recording |
| `status` | `waybar_signal` | `0` | Signal number for a Waybar custom module to refresh on |

The client writes its state to `$XDG_RUNTIME_DIR/yapyap-state-<uid>.json` (`{"active": true, "recording": false}`) for status bars.

### Files

`yapyap transcribe` sends audio files to the server and prints the text:

```bash
yapyap transcribe meeting.ogg --language en
```

## Development

```bash
uv sync --extra client --extra server
uv run pytest
```

## License

MIT
