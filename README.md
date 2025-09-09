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

## Usage

Press a chord, (`KEY_LEFTCTRL,KEY_LEFTALT` by default), speak, release the keys, and it will flush the transcription to stdout:

```bash
yapyap
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
