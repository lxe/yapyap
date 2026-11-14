# Yap Yap

Push-to-talk dictation for Linux. Hold Left Ctrl + Left Alt, speak, and let go: your words get typed into whatever window has focus, starting about a quarter second later.

https://github.com/user-attachments/assets/208f949a-3d26-46b5-9b64-52b273e1b00b

Whisper runs on a server, which can be your own machine or another one on your network with a GPU. The server speaks the OpenAI transcription API, so other apps can use it too.

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

Hold the keys, talk, release. Text goes out as real key presses, so it works in terminals, browsers, remote desktops and games. For accented letters and other scripts, install [wtype](https://github.com/atx/wtype). On [Hyprland](https://hyprland.org), you can add Russian as a second keyboard layout (`kb_layout = us,ru`), and Russian will type as fast as English.

`yapyap transcribe file.ogg` transcribes an audio file.

## More

- [Server](docs/SERVER.md): installing without Docker, choosing a model, the API
- [Client](docs/CLIENT.md): settings, paste and stdout modes, status bars

## License

MIT
