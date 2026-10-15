from transformers import pipeline

_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

MIN_TRANSCRIPT_CHARS = 50  # BART fails/produces garbage on near-empty input


class SummarizationError(Exception):
    pass


def extractive_summarize(transcript: str, max_length: int = 130, min_length: int = 30) -> str:
    if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
        raise SummarizationError(
            f"Transcript too short to summarize ({len(transcript.strip()) if transcript else 0} characters). "
            "The video may have no speech, or transcription failed silently."
        )

    max_chunk_chars = 3000
    chunks = [transcript[i:i + max_chunk_chars] for i in range(0, len(transcript), max_chunk_chars)]

    summaries = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < MIN_TRANSCRIPT_CHARS:
            continue  # skip tiny trailing fragments that would break BART
        try:
            # min_length can't exceed input length for very short chunks — clamp it
            safe_min = min(min_length, max(5, len(chunk) // 4))
            result = _summarizer(chunk, max_length=max_length, min_length=safe_min, do_sample=False)
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            raise SummarizationError(f"BART summarization failed on a chunk: {e}")

    if not summaries:
        raise SummarizationError("No valid content chunks produced a summary.")

    return " ".join(summaries)