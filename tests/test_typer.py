from unittest import mock

import pytest
from evdev import ecodes as e

from yapyap.client import typer as typer_module
from yapyap.client.config import OutputConfig
from yapyap.client.typer import Typer
from yapyap.client.xkb import RIGHT_ALT, load_layouts

try:
    LAYOUTS = load_layouts("us,ru,de,gr")
except OSError:  # libxkbcommon is not installed
    LAYOUTS = None
needs_xkb = pytest.mark.skipif(LAYOUTS is None, reason="libxkbcommon is not installed")


def make_typer(layouts, can_switch=True):
    """A Typer with a fake uinput device and no compositor."""
    with mock.patch.object(typer_module, "UInput"), mock.patch.object(typer_module.time, "sleep"), \
            mock.patch.object(typer_module.hyprland, "switch_layout_group"), \
            mock.patch.object(Typer, "_refresh_watched"):
        typer = Typer(OutputConfig(burst_pause_ms=0, group_settle_ms=0), layouts, can_switch)
    typer.wtype = "/usr/bin/wtype"
    typer.events = []
    typer.device.write.side_effect = lambda _type, code, value: typer.events.append((typer.group, code, value))
    return typer


def typed(typer):
    """Decode written key events with the layout of the group active at each event."""
    reverse = [
        {(s.code, s.shift, s.level3): ch for ch, s in sorted(layout.strokes.items(), reverse=True)}
        for layout in typer.layouts
    ]
    shift = level3 = False
    out = []
    for group, code, value in typer.events:
        if code == e.KEY_LEFTSHIFT:
            shift = value == 1
        elif code == RIGHT_ALT:
            level3 = value == 1
        elif value == 1:
            out.append(reverse[group][(code, shift, level3)])
    return "".join(out)


@needs_xkb
def test_types_english_and_maps_lookalikes():
    typer = make_typer(LAYOUTS[:1], can_switch=False)
    with mock.patch.object(typer_module.subprocess, "run") as run:
        typer.type_text('Hello, World! It’s 3:45 — "q" & (p) {b} ~/t_ u|p? <ok> a…')
    run.assert_not_called()
    assert typed(typer) == 'Hello, World! It\'s 3:45 - "q" & (p) {b} ~/t_ u|p? <ok> a...'


@needs_xkb
def test_any_configured_layout_is_typed_as_keys():
    typer = make_typer(LAYOUTS)
    switches = []
    with mock.patch.object(typer_module.hyprland, "switch_layout_group", side_effect=switches.append), \
            mock.patch.object(typer_module.subprocess, "run") as run:
        typer.type_text("Вышел зайчик, OK. Größe 5 €. Γεια σου!")
        typer.restore_layout()
    run.assert_not_called()
    assert typed(typer) == "Вышел зайчик, OK. Größe 5 €. Γεια σου!"
    assert switches == [1, 0, 2, 3, 0]  # ru, back for "OK", de, gr, then restored


@needs_xkb
def test_stays_in_a_group_between_streamed_pieces():
    typer = make_typer(LAYOUTS)
    switches = []
    with mock.patch.object(typer_module.hyprland, "switch_layout_group", side_effect=switches.append):
        typer.type_text("Привет")
        typer.type_text(", мир.")
    assert switches == [1]
    assert typed(typer) == "Привет, мир."


@needs_xkb
def test_without_group_switching_other_scripts_go_to_wtype():
    typer = make_typer(LAYOUTS, can_switch=False)
    with mock.patch.object(typer_module.subprocess, "run") as run:
        typer.type_text("Привет \u2014 мир")
    # Russian runs go to wtype; the space and the dash (as "-") are typed as keys.
    sent = [c.args[0][-1] for c in run.call_args_list]
    assert sent == ["Привет", "мир"]
    assert run.call_args_list[0].args[0][:6] == ["/usr/bin/wtype", "-k", "Shift_L", "-s", "30", "--"]
    assert typed(typer) == " - "


@needs_xkb
def test_dead_key_letters_go_to_wtype_and_the_rest_is_typed():
    typer = make_typer(LAYOUTS)
    with mock.patch.object(typer_module.subprocess, "run") as run, \
            mock.patch.object(typer_module.hyprland, "switch_layout_group"):
        typer.type_text("Καλημέρα")
    assert [c.args[0][-1] for c in run.call_args_list] == ["έ"]
    assert typed(typer) == "Καλημρα"


@needs_xkb
def test_bursts_pause_between_groups_of_events():
    typer = make_typer(LAYOUTS[:1], can_switch=False)
    typer.burst_pause = 0.004
    with mock.patch.object(typer_module.time, "sleep") as sleep:
        typer.type_text("abcdefgh")  # 8 characters, 4 events each
    assert sleep.call_count == 2
