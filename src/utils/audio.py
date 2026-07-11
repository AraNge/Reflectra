from __future__ import annotations

import io
from pathlib import Path


def audio_payload_for_llm(
    path: str | Path,
    clip_seconds: float | None = 15.0,
) -> tuple[bytes, str]:
    path = Path(path)

    if clip_seconds is None or clip_seconds <= 0:
        return path.read_bytes(), path.suffix.lower().removeprefix(".")

    try:
        import soundfile as sf

        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
        if duration <= clip_seconds or info.frames <= 0:
            return path.read_bytes(), path.suffix.lower().removeprefix(".")

        offset = max((duration - clip_seconds) / 2.0, 0.0)
        start = int(offset * info.samplerate)
        frames = int(clip_seconds * info.samplerate)
        samples, sample_rate = sf.read(
            str(path),
            start=start,
            frames=frames,
            always_2d=False,
        )
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        return buffer.getvalue(), "wav"
    except Exception:
        return path.read_bytes(), path.suffix.lower().removeprefix(".")
