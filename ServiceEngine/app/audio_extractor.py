import yt_dlp
import os
import uuid

# Folder where extracted audio files will be temporarily stored
AUDIO_DIR = "temp_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def extract_audio_from_youtube(youtube_url: str) -> str:
    """
    Downloads audio-only stream from a YouTube URL and converts it
    to 16kHz mono WAV using FFmpeg (via yt-dlp's postprocessor).
    Returns the path to the resulting WAV file.
    """
    file_id = str(uuid.uuid4())
    output_template = os.path.join(AUDIO_DIR, file_id)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", "16000",   # sample rate: 16kHz
            "-ac", "1",       # mono channel
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    wav_path = output_template + ".wav"

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio extraction failed, expected file not found: {wav_path}")

    return wav_path