# yapyap - Fast and simple push to talk dictation

https://github.com/user-attachments/assets/208f949a-3d26-46b5-9b64-52b273e1b00b

When you press and hold a specific key combination (`left ctrl + alt` by default), it records audio and automatically transcribes it to text using [whisper.cpp](https://github.com/ggml-org/whisper.cpp), then flushes the transcription to stdout, so you can do whatever with it, like:

```bash
yapyap | while read l; do wl-copy -- $l && hyprctl dispatch sendshortcut 'CTRL+SHIFT,V,'; done
```

For now, it only works on Linux due to evdev requirement.

## TODO:

 - [ ] Mac support
 - [ ] Windows support

## Requirements

On Linux the user must be part of the `input` group to access keyboard devices:

```bash
sudo usermod -a -G input $USER
```

Log out and log back in for the group change to take effect.

## Installation

Get [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you have an NVIDIA card and [cuda toolkit](https://developer.nvidia.com/cuda-downloads) installed:

```bash
uv tool install --reinstall -Caccel=cuda -Crepair=false git+https://github.com/lxe/yapyap
```

For CPU only (not recommended):

```bash
uv tool install --reinstall git+https://github.com/lxe/yapyap
```

To install the background user service and managed wrapper:

```bash
yapyap-install-service
```

That command writes:

- `~/.config/systemd/user/yapyap.service`
- `~/.local/bin/yapyap-runner`
- `~/.config/yapyap/env` (only if it does not already exist)

Then it runs `systemctl --user daemon-reload`, enables `yapyap.service`, and restarts it.
Re-run `yapyap-install-service` after upgrading `yapyap` to refresh the generated files.

## Usage

Press a chord, (`KEY_LEFTCTRL,KEY_LEFTALT` by default), speak, release the keys, and it will flush the transcription to stdout:

```bash
yapyap
```

If you are using the user service, the generated runner copies each transcript to the clipboard and, on Hyprland, pastes it into the active window. Outside that setup, run `yapyap` directly and pipe stdout into whatever action you want.

While `yapyap` is running, `yapyap-cli` can send file transcription requests to the already-loaded model:

```bash
yapyap-cli file.ogg
```

`yapyap-cli` requires a running `yapyap` process and will fail if the background socket is not available.

For a remote server, use TCP and `yapyap-cli` will upload the audio file contents instead of sending a local path:

```bash
yapyap-cli --host 192.168.0.2:8122 --trace file.ogg
```

The managed service uses `~/.config/yapyap/env` for overrides. Common settings:

```bash
CHORD=KEY_LEFTCTRL,KEY_LEFTALT
MODEL=large-v3-turbo-q8_0
YAPYAP_WAYBAR_SIGNAL=11
YAPYAP_SERVER_ONLY=1
YAPYAP_TCP_HOST=0.0.0.0
YAPYAP_TCP_PORT=8122
YAPYAP_REMOTE_HOST=192.168.0.102:8122
YAPYAP_INPUT_DEVICE=pulse-default
YAPYAP_DUCK_AUDIO=1
YAPYAP_DUCK_VOLUME=0.50
YAPYAP_DUCK_FADE_OUT_MS=120
YAPYAP_DUCK_FADE_IN_MS=180
YAPYAP_AUTO_PASTE=1
YAPYAP_PASTE_DELAY=0.05
```

With audio ducking enabled, active application playback fades to the configured
fraction of its current per-app volume while the recording chord is held, then
fades back to the exact prior volume when the chord is released. Muted and
paused streams are not changed.

If `YAPYAP_REMOTE_HOST` is set on a listening client, local microphone capture stays local but transcription is forwarded to that remote server. In that mode `yapyap` does not load a local Whisper model; if the remote server cannot be reached, the listening pipeline emits `[remove server unreachable]`.

After editing that file, reload the service:

```bash
systemctl --user restart yapyap
```

Useful service commands:

```bash
systemctl --user status yapyap
journalctl --user -u yapyap -f
```

You can change the [key combination](https://gitlab.freedesktop.org/libevdev/evtest/-/blob/master/evtest.c?ref_type=heads#L224):

```bash
CHORD=KEY_FN yapyap
```

By default it's using If you're on CPU and it's slow, try a different model:

```bash
MODEL=tiny.en-q8_0 yapyap
```  

You can perform actions on the output:

```bash
yapyap | while read l; do echo "do whatever you want with $l"; done
```

Practical example: copy and paste anywhere on hyprland:

```bash
yapyap | while read l; do wl-copy -- $l && hyprctl dispatch sendshortcut 'CTRL+SHIFT,V,'; done
```

## License

MIT
