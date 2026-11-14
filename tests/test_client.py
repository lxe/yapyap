import queue
import time
from unittest import mock

import pytest

from yapyap.audio import trailing_pad_seconds
from yapyap.client import output as output_module
from yapyap.client.config import DEFAULT_CONFIG, Config, OutputConfig, load_config
from yapyap.client.hotkey import ChordListener


def test_trailing_pad_stops_at_the_30_second_window():
    assert trailing_pad_seconds(5.0) == 2.0
    assert trailing_pad_seconds(29.0) == pytest.approx(1.0)
    assert trailing_pad_seconds(45.0) == 2.0
    assert trailing_pad_seconds(59.5) == pytest.approx(0.5)


def test_default_config_file_matches_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(DEFAULT_CONFIG)
    assert load_config(path) == Config()
    assert load_config(tmp_path / "missing.toml") == Config()


def test_config_overrides_and_rejects_typos(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nurl = "http://box:8122"\n[output]\nmode = "paste"\n')
    config = load_config(path)
    assert config.server.url == "http://box:8122"
    assert config.output.mode == "paste"
    assert config.hotkey.chord == ["KEY_LEFTCTRL", "KEY_LEFTALT"]
    path.write_text('[server]\nurll = "x"\n')
    with pytest.raises(ValueError, match="urll"):
        load_config(path)


def test_chord_fires_once_on_full_press_and_on_first_release():
    events = []
    listener = ChordListener(["KEY_LEFTCTRL", "KEY_LEFTALT"], lambda: events.append("press"), lambda: events.append("release"))
    ctrl, alt = sorted(listener.chord)
    listener._handle(ctrl, True)
    assert events == []
    listener._handle(alt, True)
    listener._handle(alt, True)  # held, no repeat
    listener._handle(ctrl, False)
    listener._handle(alt, False)
    assert events == ["press", "release"]


def test_unknown_chord_key_is_rejected():
    with pytest.raises(ValueError, match="KEY_NOPE"):
        ChordListener(["KEY_NOPE"], lambda: None, lambda: None)


class FakeTyper:
    def __init__(self, config):
        self.typed = []
        self.restored = 0

    def wait_for_modifiers_released(self):
        pass

    def type_text(self, text):
        time.sleep(0.05)
        self.typed.append(text)

    def restore_layout(self):
        self.restored += 1

    def close(self):
        pass


def test_type_output_batches_pieces_that_queue_up():
    with mock.patch("yapyap.client.typer.Typer", FakeTyper), mock.patch.object(output_module, "copy_to_clipboard"):
        out = output_module.TypeOutput(OutputConfig())
        out.begin()
        for piece in ["Вот", " сейчас", " я", " говорю"]:
            out.piece(piece)
        out.finish("Вот сейчас я говорю")
        time.sleep(0.5)
        out.close()
    assert "".join(out.typer.typed) == "Вот сейчас я говорю"
    assert len(out.typer.typed) <= 2  # pieces that queue up go out together
    assert out.typer.restored >= 1  # back to the first layout group once idle


def test_type_output_types_the_transcript_when_nothing_streamed():
    with mock.patch("yapyap.client.typer.Typer", FakeTyper), mock.patch.object(output_module, "copy_to_clipboard") as copy:
        out = output_module.TypeOutput(OutputConfig())
        out.begin()
        out.finish("Hello there.")
        time.sleep(0.2)
        out.close()
    assert out.typer.typed == ["Hello there."]
    copy.assert_called_once_with("Hello there.")
