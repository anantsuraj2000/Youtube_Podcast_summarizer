"""Translates a finished summary into English, Hindi or Kannada with an LLM."""
import re

from app.config import TRANSLATION_PROVIDER
from app.errors import SummarizationError
from app.llm_summarizer import _call_llm, _parallel

TRANSLATION_LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
PART_CHARS = 800      # long summaries are translated in parts of about this size, in parallel

_SYSTEM = (
    "You are a professional translator. Translate the text faithfully and naturally. "
    "Keep names of people, places, organizations and products, numbers and units unchanged "
    "(you may transliterate names into the target script). Do not add, remove or explain anything. "
    "Output only the translation."
)


class TranslationError(SummarizationError):
    pass


def _split_parts(text: str, max_chars: int = PART_CHARS) -> list[str]:
    """Splits text into parts of whole sentences (handles the Devanagari danda '।' too)."""
    parts, current = [], ""
    for sent in re.split(r"(?<=[.!?\u0964])\s+", text.strip()):
        while len(sent) > max_chars:                       # sentence without punctuation: hard split
            if current:
                parts.append(current)
                current = ""
            parts.append(sent[:max_chars])
            sent = sent[max_chars:]
        if current and len(current) + 1 + len(sent) > max_chars:
            parts.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        parts.append(current)
    return parts


def translate_text(text: str, target: str, source: str = "en") -> str:
    if target not in TRANSLATION_LANGUAGES:
        raise TranslationError(f"Unsupported target language '{target}'.")
    if target == source or not text.strip():
        return text

    src, dst = TRANSLATION_LANGUAGES.get(source, source), TRANSLATION_LANGUAGES[target]
    parts = _split_parts(text)
    total = len(parts)

    def translate_part(item) -> str:
        i, part = item
        note = "" if total == 1 else (
            f" It is part {i + 1} of {total} of one longer text: translate only this part, "
            "with no introduction or summary."
        )
        prompt = f"Translate this {src} text into {dst}.{note}\n\nTEXT:\n{part}"
        return _call_llm(_SYSTEM, prompt, TRANSLATION_PROVIDER)

    return " ".join(_parallel(translate_part, list(enumerate(parts))))