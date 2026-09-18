from groq import Groq
from app.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)


def transcribe_audio(wav_path: str) -> str:
    """
    Sends a WAV file to Groq's hosted Whisper endpoint and returns
    the full transcript as plain text.
    """
    with open(wav_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=(wav_path, audio_file.read()),
            model="whisper-large-v3",
            response_format="text",
        )

    return response