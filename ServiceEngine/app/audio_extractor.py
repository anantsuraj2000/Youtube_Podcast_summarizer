import yt_dlp
import os
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
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
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

    wav_path = output_template + ".wav"
    if not os.path.exists(wav_path):
        raise AudioExtractionError("Audio file was not created — extraction may have silently failed.")

    # Guard against near-empty audio (e.g. a 2-second clip, a corrupted stream)
    if os.path.getsize(wav_path) < 5000:  # ~5KB, well below any real speech clip
        os.remove(wav_path)
        raise AudioExtractionError("Extracted audio is too short or empty to process.")

    return wav_path


def get_video_title(youtube_url: str) -> str:
    try:
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info.get("title", "Untitled video")
    except Exception:
        return "Untitled video"