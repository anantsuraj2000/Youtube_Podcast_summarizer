"""Language check: decide which ASR route an audio file takes, or reject it as out of scope."""
import os
import subprocess
import tempfile

from app.config import LANGUAGE_NAMES, SUPPORTED_LANGUAGE_CODES
from app.errors import UnsupportedLanguageError

PROBE_SECONDS = 30
NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def _duration_seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _cut_probe(audio_path: str) -> str:
    """Cuts a ~30 s clip starting 20% into the audio (skips intro music / silence)."""
    duration = _duration_seconds(audio_path)
    start = max(0.0, min(duration * 0.2, duration - PROBE_SECONDS))
    fd, probe_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.1f}", "-t", str(PROBE_SECONDS),
         "-i", audio_path, "-ac", "1", "-ar", "16000", "-b:a", "64k", probe_path],
        capture_output=True,
    )
    return probe_path


def detect_language(audio_path: str, groq_client) -> str:
    """Returns 'en' / 'hi' / 'kn', or raises UnsupportedLanguageError."""
    probe = _cut_probe(audio_path)
    try:
        with open(probe, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                file=(os.path.basename(probe), f.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
    finally:
        if os.path.exists(probe):
            os.remove(probe)

    raw = resp.get("language") if isinstance(resp, dict) else getattr(resp, "language", None)
    name = (raw or "").strip().lower()
    code = NAME_TO_CODE.get(name) or (name if name in SUPPORTED_LANGUAGE_CODES else None)
    if code is None:
        shown = name.title() if name else "unknown"
        raise UnsupportedLanguageError(shown, [LANGUAGE_NAMES[c] for c in SUPPORTED_LANGUAGE_CODES])
    return code