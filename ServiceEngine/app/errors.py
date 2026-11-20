class SummarizationError(Exception):
    """Raised when a transcript cannot be summarized (any backend)."""
    pass
class UnsupportedLanguageError(Exception):
    """Raised when the audio's language is outside the supported set (English, Hindi, Kannada)."""

    def __init__(self, language_name: str, supported: list[str]):
        self.language_name = language_name
        super().__init__(
            f"Audio language ({language_name}) is out of scope. "
            f"Supported languages: {', '.join(supported)}."
        )