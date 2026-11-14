import json
import subprocess
import unittest
from types import SimpleNamespace

from yapyap.client.ducking import PulseAudioDucker, _playback_streams


def sink_input(index, volumes, *, corked=False, mute=False, name="Music"):
    return {
        "index": index,
        "corked": corked,
        "mute": mute,
        "volume": {
            f"channel{channel}": {"value": volume}
            for channel, volume in enumerate(volumes)
        },
        "properties": {"application.name": name},
    }


class FakePactl:
    def __init__(self):
        self.streams = [sink_input(42, [65536, 32768], name="Player")]
        self.volume_commands = []
        self.commands = []
        self.restore_props = True
        self.remembered_volumes = [65536, 32768]

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        if command == ["wpctl", "settings", "node.stream.restore-props"]:
            return SimpleNamespace(stdout=f"Value: {str(self.restore_props).lower()}\n")
        if command[:3] == [
            "wpctl",
            "settings",
            "node.stream.restore-props",
        ]:
            self.restore_props = command[3] == "true"
            return SimpleNamespace(stdout="")
        if command[:5] == ["pactl", "-f", "json", "list", "sink-inputs"]:
            return SimpleNamespace(stdout=json.dumps(self.streams))
        if command[:2] == ["pactl", "set-sink-input-volume"]:
            self.volume_commands.append(command)
            volumes = [int(value) for value in command[3:]]
            self.streams[0]["volume"] = {
                f"channel{channel}": {"value": volume}
                for channel, volume in enumerate(volumes)
            }
            if self.restore_props:
                self.remembered_volumes = volumes
            return SimpleNamespace(stdout="")
        raise subprocess.CalledProcessError(1, command)

    def remove_stream(self):
        self.streams = []

    def recreate_stream(self):
        self.streams = [sink_input(43, self.remembered_volumes, name="Player")]


class AudioDuckingTests(unittest.TestCase):
    def test_parser_skips_paused_and_muted_streams(self):
        payload = json.dumps(
            [
                sink_input(1, [65536], name="Playing"),
                sink_input(2, [65536], corked=True, name="Paused"),
                sink_input(3, [65536], mute=True, name="Muted"),
            ]
        )

        streams = _playback_streams(payload)

        self.assertEqual([stream.application_name for stream in streams], ["Playing"])

    def test_duck_and_restore_preserve_each_channel(self):
        pactl = FakePactl()
        ducker = PulseAudioDucker(
            enabled=True,
            volume_ratio=0.50,
            fade_out_ms=30,
            fade_in_ms=30,
            run=pactl,
            sleep=lambda _seconds: None,
        )

        ducker.duck()
        ducker._worker.join(timeout=1)
        self.assertEqual(
            pactl.volume_commands[-1],
            ["pactl", "set-sink-input-volume", "42", "32768", "16384"],
        )

        ducker.restore(wait=True)
        self.assertEqual(
            pactl.volume_commands[-1],
            ["pactl", "set-sink-input-volume", "42", "65536", "32768"],
        )

    def test_temporary_duck_does_not_replace_remembered_app_volume(self):
        pactl = FakePactl()
        ducker = PulseAudioDucker(
            enabled=True,
            volume_ratio=0.50,
            fade_out_ms=0,
            fade_in_ms=0,
            run=pactl,
            sleep=lambda _seconds: None,
        )
        ducker._wpctl_available = True

        ducker.duck()
        ducker._worker.join(timeout=1)
        pactl.remove_stream()
        ducker.restore(wait=True)
        pactl.recreate_stream()

        self.assertEqual(pactl.remembered_volumes, [65536, 32768])
        self.assertEqual(
            [stream["volume"] for stream in pactl.streams],
            [
                {
                    "channel0": {"value": 65536},
                    "channel1": {"value": 32768},
                }
            ],
        )
        self.assertTrue(pactl.restore_props)
        self.assertIn(
            ["wpctl", "settings", "node.stream.restore-props", "false"],
            pactl.commands,
        )


if __name__ == "__main__":
    unittest.main()
