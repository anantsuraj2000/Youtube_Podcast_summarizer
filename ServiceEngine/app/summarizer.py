from transformers import pipeline

from app.chunking import chunk_transcript
from app.config import CHUNK_MAX_CHARS_BART
from app.errors import SummarizationError

_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

MIN_TRANSCRIPT_CHARS = 50  # BART fails/produces garbage on near-empty input


def extractive_summarize(transcript: str, max_length: int = 130, min_length: int = 30) -> str:
    if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
        raise SummarizationError(
            f"Transcript too short to summarize ({len(transcript.strip()) if transcript else 0} characters). "
            "The video may have no speech, or transcription failed silently."
        )

    summaries = []
    for chunk in chunk_transcript(transcript, max_chunk_chars=CHUNK_MAX_CHARS_BART):
        try:
            safe_min = min(min_length, max(5, len(chunk) // 4))
            result = _summarizer(chunk, max_length=max_length, min_length=safe_min,
                                 do_sample=False, truncation=True)
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            raise SummarizationError(f"BART summarization failed on a chunk: {e}")

    if not summaries:
        raise SummarizationError("No valid content chunks produced a summary.")

    return " ".join(summaries)