"""Client settings, read from ~/.config/yapyap/config.toml."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "yapyap" / "config.toml"


@dataclass
class ServerConfig:
    url: str = "http://127.0.0.1:8122"
    token: str = ""
    language: str = ""  # empty: detect per recording
    prompt: str = ""
    timeout: float = 60.0


@dataclass
class HotkeyConfig:
    chord: list[str] = field(default_factory=lambda: ["KEY_LEFTCTRL", "KEY_LEFTALT"])


@dataclass
class CaptureConfig:
    target: str = ""  # PipeWire/PulseAudio source; empty: the default source


@dataclass
class OutputConfig:
    mode: str = "type"  # type, paste or stdout
    clipboard: bool = True  # also copy every transcript to the clipboard
    burst_events: int = 16
    burst_pause_ms: float = 1.0
    wtype_settle_ms: float = 30.0
    group_settle_ms: float = 12.0


@dataclass
class DuckingConfig:
    enabled: bool = False
    volume: float = 0.5
    fade_out_ms: int = 120
    fade_in_ms: int = 180


@dataclass
class StatusConfig:
    waybar_signal: int = 0  # 0: do not signal Waybar


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ducking: DuckingConfig = field(default_factory=DuckingConfig)
    status: StatusConfig = field(default_factory=StatusConfig)


def load_config(path: Path | None = None) -> Config:
    """Read the config file; missing sections and keys keep their defaults."""
    path = path or config_path()
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    config = Config()
    for section in fields(Config):
        values = data.get(section.name, {})
        target = getattr(config, section.name)
        known = {f.name for f in fields(target)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"{path}: unknown setting(s) in [{section.name}]: {', '.join(sorted(unknown))}")
        for key, value in values.items():
            setattr(target, key, value)
    unknown_sections = set(data) - {f.name for f in fields(Config)}
    if unknown_sections:
        raise ValueError(f"{path}: unknown section(s): {', '.join(sorted(unknown_sections))}")
    if config.output.mode not in {"type", "paste", "stdout"}:
        raise ValueError(f"{path}: [output] mode must be type, paste or stdout")
    return config


DEFAULT_CONFIG = """\
# Yap Yap client settings. Every setting is optional; the values shown are
# the defaults. Restart the service after editing:
#   systemctl --user restart yapyap

[server]
url = "http://127.0.0.1:8122"   # a yapyap-server or any OpenAI-compatible endpoint
token = ""                      # API token, if the server requires one
language = ""                   # e.g. "en"; empty detects the language per recording
prompt = ""                     # optional text that hints vocabulary or languages

[hotkey]
chord = ["KEY_LEFTCTRL", "KEY_LEFTALT"]   # hold to record, release to transcribe

[capture]
target = ""                     # PipeWire source name; empty uses the default source

[output]
mode = "type"                   # type: press keys; paste: clipboard + paste shortcut; stdout: print
clipboard = true                # also copy each transcript to the clipboard

[ducking]
enabled = false                 # lower other apps' audio while recording
volume = 0.5

[status]
waybar_signal = 0               # send SIGRTMIN+N to Waybar on state changes; 0 disables
"""
