import os
from groq import Groq
from pydub import AudioSegment
from app.config import GROQ_API_KEY
from app.language import detect_language

client = Groq(api_key=GROQ_API_KEY)

MAX_UPLOAD_BYTES = 24 * 1024 * 1024
SEGMENT_MS = 15 * 60 * 1000       # 15-minute segments for oversized audio
MIN_SEGMENT_MS = 1000             # drop trailing slivers the API would reject


def _transcribe_file(path: str) -> str:
    """Sends one audio file (must be under the upload cap) to Groq Whisper."""
    with open(path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(path), audio_file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return response.strip()


def _split_audio(audio_path: str) -> list[str]:
    """Splits audio into 15-minute MP3 segments (64 kbps) next to the original."""
    audio = AudioSegment.from_file(audio_path)
    base = os.path.splitext(audio_path)[0]
    segment_paths = []
    for idx, start in enumerate(range(0, len(audio), SEGMENT_MS)):
        segment = audio[start:start + SEGMENT_MS]
        if len(segment) < MIN_SEGMENT_MS:
            continue
        seg_path = f"{base}_part{idx:03d}.mp3"
        segment.export(seg_path, format="mp3", bitrate="64k")
        segment_paths.append(seg_path)
    return segment_paths


def _transcribe_groq(audio_path: str) -> str: #transcribe_audio
    """
    Transcribes audio with Groq Whisper. Files above ~24 MB are split into
    15-minute segments, transcribed one by one, and joined in order.
    """
    if os.path.getsize(audio_path) <= MAX_UPLOAD_BYTES:
        return _transcribe_file(audio_path)

    segment_paths = _split_audio(audio_path)
    try:
        parts = [_transcribe_file(p) for p in segment_paths]
    finally:
        for p in segment_paths:
            try:
                os.remove(p)
            except OSError:
                pass
    return " ".join(part for part in parts if part)

def transcribe_with_language(audio_path: str) -> tuple[str, str]:
    """Language check, then transcription. Returns (transcript, language_code)."""
    lang = detect_language(audio_path, client)
    if lang == "en":
        return _transcribe_groq(audio_path), lang
    from app.indic_asr import transcribe_indic          # lazy: torch only loads when needed
    return transcribe_indic(audio_path, lang), lang


def transcribe_audio(audio_path: str) -> str:
    """Transcript only (kept for evaluate.py / test scripts)."""
    return transcribe_with_language(audio_path)[0]