# Client

`yapyap install` creates `~/.config/yapyap/config.toml` with every setting at its default, plus a systemd user service. After you edit the file, restart the service with `systemctl --user restart yapyap`. Logs are in `journalctl --user -u yapyap -f`.

| Setting | Default | Meaning |
|---|---|---|
| `server.url` | `http://127.0.0.1:8122` | Server address |
| `server.token` | none | The server's `YAPYAP_TOKEN` |
| `server.language` | detect | Language code such as `en`, to skip detection |
| `server.prompt` | none | Words or a sample sentence that guide spelling and mixed languages |
| `hotkey.chord` | `["KEY_LEFTCTRL", "KEY_LEFTALT"]` | Keys to hold, by [evdev name](https://github.com/torvalds/linux/blob/master/include/uapi/linux/input-event-codes.h) |
| `capture.target` | default source | PipeWire source to record from |
| `output.mode` | `type` | `type`, `paste` or `stdout` |
| `output.clipboard` | `true` | Also copy each transcript to the clipboard |
| `output.burst_pause_ms` | `1.0` | Pause after every 16 key events. `0.5` types about twice as fast; `4.0` lets slow apps redraw smoothly |
| `ducking.enabled` | `false` | Turn down other apps' audio while you talk |
| `status.waybar_signal` | `0` | Refresh a Waybar module on this signal |

## Output modes

`type` presses keys on a virtual keyboard. The text appears as fast as Whisper decodes it, at about 3,000 characters a second. Yap Yap uses [libxkbcommon](https://xkbcommon.org) to work out which keys type each character in your layouts, including Shift and AltGr. On Hyprland it also switches between layout groups. `paste` copies each transcript and presses the paste shortcut in the focused window, on Hyprland only. `stdout` prints one line per transcript:

```bash
yapyap run | while read -r line; do notify-send "$line"; done
```

## Status bars

The client keeps `$XDG_RUNTIME_DIR/yapyap-state-<uid>.json` up to date, for example `{"active": true, "recording": false}`. A Waybar custom module can read that file and refresh on `status.waybar_signal`.
