from unittest import mock

from evdev import ecodes as e

from yapyap.client import typer as typer_module
from yapyap.client.typer import KEYMAP, RU_KEYMAP, Typer


def make_typer(layouts=None):
    """A Typer without a real uinput device or compositor."""
    typer = Typer.__new__(Typer)
    typer.device = mock.Mock()
    typer.burst_events = 16
    typer.burst_pause = 0
    typer.group_settle = 0
    typer.wtype_settle_ms = 30
    typer.wtype = "/usr/bin/wtype"
    typer._sent = 0
    typer.layouts = layouts or {}
    typer.group = 0
    return typer


def typed(typer, groups):
    """Decode the key events a typer wrote, using the keymap of the group active at each event."""
    reverse = {group: {(code, shift): ch for ch, (code, shift) in keymap.items()} for group, keymap in groups.items()}
    shift, out = False, []
    for group, code, value in typer.events:
        if code == e.KEY_LEFTSHIFT:
            shift = value == 1
        elif value == 1:
            out.append(reverse[group][(code, shift)])
    return "".join(out)


def record(typer):
    typer.events = []
    typer.device.write.side_effect = lambda _type, code, value: typer.events.append((typer.group, code, value))


def test_types_ascii_and_maps_lookalikes():
    typer = make_typer()
    record(typer)
    with mock.patch.object(typer_module.subprocess, "run") as run:
        typer.type_text('Hello, World! It’s 3:45 — "q" & (p) {b} ~/t_ u|p? <ok> a…')
    run.assert_not_called()
    assert typed(typer, {0: KEYMAP}) == 'Hello, World! It\'s 3:45 - "q" & (p) {b} ~/t_ u|p? <ok> a...'


def test_other_characters_go_to_wtype_in_one_call():
    typer = make_typer()
    with mock.patch.object(typer_module.subprocess, "run") as run:
        typer.type_text("café — ok")
    typer.device.write.assert_not_called()
    argv = run.call_args.args[0]
    # A Shift tap the app may drop, the settle pause, then the original text.
    assert argv == ["/usr/bin/wtype", "-k", "Shift_L", "-s", "30", "--", "café — ok"]


def test_russian_is_typed_as_keys_in_its_layout_group():
    typer = make_typer(layouts={1: RU_KEYMAP})
    record(typer)
    switches = []

    def switch(group):
        switches.append(group)

    with mock.patch.object(typer_module.hyprland, "switch_layout_group", side_effect=switch), \
            mock.patch.object(typer_module.subprocess, "run") as run:
        typer.type_text("Вышел зайчик погулять, OK. Ёж — 2.")
        typer.type_text(" Дальше")
        assert switches == [1, 0, 1]  # stays in "ru" between streamed pieces
        typer.restore_layout()
        typer.type_text(" and English.")
    run.assert_not_called()
    assert switches == [1, 0, 1, 0]
    assert typed(typer, {0: KEYMAP, 1: RU_KEYMAP}) == "Вышел зайчик погулять, OK. Ёж - 2. Дальше and English."


def test_bursts_pause_between_groups_of_events():
    typer = make_typer()
    typer.burst_pause = 0.004
    with mock.patch.object(typer_module.time, "sleep") as sleep:
        typer.type_text("abcdefgh")  # 8 characters, 4 events each
    assert sleep.call_count == 2
