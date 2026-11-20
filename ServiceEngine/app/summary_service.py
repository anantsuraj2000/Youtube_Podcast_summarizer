from app.config import SUMMARIZER_BACKEND


def summarize(transcript: str, backend: str | None = None, language: str = "en") -> str:
    backend = (backend or SUMMARIZER_BACKEND).lower()
    if language != "en":
        backend = "llm" 
    if backend == "llm":
        from app.llm_summarizer import llm_summarize
        return llm_summarize(transcript)
    if backend == "bart":
        from app.summarizer import extractive_summarize
        return extractive_summarize(transcript)
    raise ValueError(f"Unknown SUMMARIZER_BACKEND '{backend}' (use 'llm' or 'bart').")