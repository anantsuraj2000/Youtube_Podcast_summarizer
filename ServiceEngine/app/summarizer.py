from transformers import pipeline

# Loaded once when the module is imported — avoids reloading the model on every call
_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def extractive_summarize(transcript: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Generates a summary from the transcript using BART.
    Handles transcripts longer than the model's input limit by
    chunking and summarizing each chunk separately.
    """
    max_chunk_chars = 3000  # rough safe limit for BART's token window
    chunks = [transcript[i:i + max_chunk_chars] for i in range(0, len(transcript), max_chunk_chars)]

    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) == 0:
            continue
        result = _summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)