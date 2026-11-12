from unittest import mock

from evdev import ecodes as e

from yapyap import hidtype


def decode(events):
    """Turn (code, value) key events back into text using the US layout."""
    reverse = {(code, shifted): ch for ch, (code, shifted) in hidtype.KEYMAP.items()}
    shift = False
    out = []
    for code, value in events:
        if code == e.KEY_LEFTSHIFT:
            shift = value == 1
        elif value == 1:
            out.append(reverse[(code, shift)])
    return "".join(out)


def make_typer():
    typer = hidtype.Typer.__new__(hidtype.Typer)
    typer.device = mock.Mock()
    typer.key_delay = 0
    typer.burst_events = 32
    typer._sent = 0
    typer.burst_pause = 0
    typer.wtype = "/usr/bin/wtype"
    return typer


def test_types_ascii_and_lookalikes():
    typer = make_typer()
    text = 'Hello, World! It’s 3:45 — "q" & (p) {b} ~t_ u|p? <ok> a…'
    typer.type_text(text)
    events = [(c.args[1], c.args[2]) for c in typer.device.write.call_args_list]
    assert decode(events) == 'Hello, World! It\'s 3:45 - "q" & (p) {b} ~t_ u|p? <ok> a...'
    assert all(c.args[0] == e.EV_KEY for c in typer.device.write.call_args_list)


def test_unmappable_characters_go_to_wtype_in_order():
    typer = make_typer()
    calls = []
    typer.device.write.side_effect = lambda _t, code, value: calls.append(("key", code, value))
    with mock.patch.object(hidtype.subprocess, "run", side_effect=lambda argv, check: calls.append(("wtype", argv[-1]))):
        typer.type_text("café ok")
    wtype_at = calls.index(("wtype", "é"))
    before = [(c, v) for kind, *rest in calls[:wtype_at] if kind == "key" for c, v in [rest]]
    after = [(c, v) for kind, *rest in calls[wtype_at + 1:] if kind == "key" for c, v in [rest]]
    assert decode(before) == "caf"
    assert decode(after) == " ok"
