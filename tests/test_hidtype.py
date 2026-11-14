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
    typer._last_emit = None
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


def test_lines_with_non_us_characters_go_to_wtype_whole():
    typer = make_typer()
    typer.wtype_settle_ms = 15
    runs = []
    with mock.patch.object(hidtype.subprocess, "run", side_effect=lambda argv, check: runs.append(argv)):
        typer.type_text(" хочу говорить \u2014 по-русски")
        typer.type_text("caf\u00e9 ok")
    # One call per line, with the settle pause, and the original dash kept.
    assert runs == [
        ["/usr/bin/wtype", "-s", "15", "--", " хочу говорить \u2014 по-русски"],
        ["/usr/bin/wtype", "-s", "15", "--", "caf\u00e9 ok"],
    ]
    typer.device.write.assert_not_called()


def test_english_with_curly_quotes_stays_on_the_keyboard():
    typer = make_typer()
    with mock.patch.object(hidtype.subprocess, "run") as run:
        typer.type_text("It\u2019s \u201cfine\u201d")
    run.assert_not_called()
    events = [(c.args[1], c.args[2]) for c in typer.device.write.call_args_list]
    assert decode(events) == 'It\'s "fine"'
