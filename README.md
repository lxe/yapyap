# Yap Yap

Push-to-talk dictation for Linux. Hold Left Ctrl + Left Alt, speak, and let go: your words get typed into whatever window has focus, starting about a quarter second later.

https://github.com/user-attachments/assets/208f949a-3d26-46b5-9b64-52b273e1b00b

Whisper runs on a server, which can be your own machine or another one on your network with a GPU. The server speaks the OpenAI transcription API, so other apps can use it too.

## Features

**Fast.** Time from letting go of the keys until the first words are typed, and until the whole transcript is in, with the default model:

| Server GPU | Engine | 5 s of speech | 10 s | 20 s |
|---|---|---|---|---|
| NVIDIA RTX 5090 | whisper.cpp, Q8 | 76 ms | 90 ms | 106 ms |
| NVIDIA RTX 3090 | whisper.cpp, Q8 | 216 ms | 234 ms | 263 ms |
| Intel Arc 140V (integrated) | OpenVINO, FP16 | 216 ms, all in 0.27 s | 231 ms, all in 0.35 s | 225 ms, all in 0.47 s |

On NVIDIA GPUs the whole transcript arrives with the first words. On the Arc, the server streams text while Whisper decodes it, so typing starts before the transcript is complete. Each figure is the median of five runs of `yapyap bench`, with the 3090 and the Arc reached over the local network.

**Multilingual.** The default model is [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), which covers 99 languages and detects the language of each recording. I've used it every day for over a year. I've tested newer models, including Qwen3-ASR and Parakeet, and none of them matched it on both speed and accuracy on the Arc.

**Real key presses.** Yap Yap types through a virtual keyboard, the same way a USB keyboard would. No paste shortcuts are involved, and the text lands in any window that accepts typing, including terminals, remote desktops and games.

## Install

1. **Start the server** on the machine that will transcribe. Pick the line for its hardware:

   ```bash
   git clone https://github.com/lxe/yapyap && cd yapyap
   echo "YAPYAP_TOKEN=$(openssl rand -hex 24)" > .env

   # Intel GPU
   echo "RENDER_GID=$(stat -c %g /dev/dri/renderD128)" >> .env
   docker compose --profile intel up -d
   # NVIDIA GPU
   docker compose --profile nvidia up -d
   # CPU only
   docker compose --profile cpu up -d
   ```

2. **Install the client** on your Linux desktop. It needs Wayland, PipeWire, and your user in the `input` group:

   ```bash
   sudo usermod -aG input $USER   # then log out and back in
   uv tool install "yapyap[client] @ git+https://github.com/lxe/yapyap"
   yapyap install
   ```

3. **Connect them.** Put the server's address and the token from its `.env` into `~/.config/yapyap/config.toml`:

   ```toml
   [server]
   url = "http://192.168.1.20:8122"
   token = "..."
   ```

   Then run `yapyap check` and `systemctl --user restart yapyap`.

## Use it

Hold the keys, talk, and let go. Yap Yap reads your keyboard layouts, so it types any language that one of them covers. On [Hyprland](https://hyprland.org) you can configure up to four layouts, such as `kb_layout = us,ru,de,gr`, and Yap Yap switches to the right layout for each word. Characters that none of your layouts can type, such as Chinese or letters that need a dead key, go through [wtype](https://github.com/atx/wtype) if it's installed.

`yapyap transcribe file.ogg` transcribes an audio file.

## More

- [Server](docs/SERVER.md): installing without Docker, choosing a model, the API
- [Client](docs/CLIENT.md): settings, paste and stdout modes, status bars

## License

MIT
