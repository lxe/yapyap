"""Audio helpers shared by the client and the server.

Yap Yap moves audio as 16 kHz mono 16-bit WAV, the format Whisper works in.
"""
from __future__ import annotations

import io
import wave

SAMPLE_RATE = 16000


def pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def trailing_pad_seconds(duration: float, pad: float = 2.0, window: float = 30.0) -> float:
    """Seconds of silence to append before transcription.

    Whisper often drops a final sentence that ends right at the end of the
    audio, and trailing silence brings it back. Whisper also works in
    30-second windows, so padding that pushes a clip just past a window
    boundary costs a whole extra window; the pad then stops at the boundary.
    """
    position = duration % window
    if duration > 0 and position + pad > window:
        return window - position
    return pad
