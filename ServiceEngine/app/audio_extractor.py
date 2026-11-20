import yt_dlp
import os
import subprocess
import uuid

AUDIO_DIR = "temp_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


class AudioExtractionError(Exception):
    """Raised when a YouTube URL can't be turned into usable audio."""
    pass


def extract_audio_from_youtube(youtube_url: str) -> str:
    if not youtube_url or not youtube_url.strip():
        raise AudioExtractionError("No URL provided.")

    file_id = str(uuid.uuid4())
    output_template = os.path.join(AUDIO_DIR, file_id)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
        {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1", "-b:a", "64k"],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        if "private video" in msg:
            raise AudioExtractionError("This video is private and can't be accessed.")
        elif "age" in msg or "sign in to confirm" in msg:
            raise AudioExtractionError("This video is age-restricted and can't be processed.")
        elif "unavailable" in msg or "not available" in msg:
            raise AudioExtractionError("This video is unavailable (deleted, region-blocked, or removed).")
        elif "unsupported url" in msg or "is not a valid url" in msg:
            raise AudioExtractionError("That doesn't look like a valid YouTube URL.")
        else:
            raise AudioExtractionError(f"Could not download audio: {e}")
    except Exception as e:
        raise AudioExtractionError(f"Unexpected error during audio extraction: {e}")

    audio_path = output_template + ".mp3"
    if not os.path.exists(audio_path):
        raise AudioExtractionError("Audio file was not created — extraction may have silently failed.")

    if os.path.getsize(audio_path) < 5000:
        os.remove(audio_path)
        raise AudioExtractionError("Extracted audio is too short or empty to process.")

    return audio_path


def get_video_title(youtube_url: str) -> str:
    try:
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("title", "Untitled video")
    except Exception:
        return "Untitled video"

def get_video_duration(youtube_url: str) -> float | None:
    """Video length in seconds (None if it can't be read)."""
    try:
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(youtube_url, download=False).get("duration")
    except Exception:
        return None

def prepare_uploaded_audio(src_path: str) -> str:
    """
    Converts a user-uploaded audio/video file to 16 kHz mono 64 kbps MP3 (same format the
    YouTube path produces), so the transcriber's 24 MB check and 15-minute splitting apply.
    """
    out_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
    cmd = ["ffmpeg", "-y", "-i", src_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise AudioExtractionError("ffmpeg is not installed or not on PATH.")

    if proc.returncode != 0 or not os.path.exists(out_path):
        raise AudioExtractionError("Could not read that file as audio. It may be corrupted or not an audio file.")
    if os.path.getsize(out_path) < 5000:
        os.remove(out_path)
        raise AudioExtractionError("Uploaded audio is too short or empty to process.")
    return out_path