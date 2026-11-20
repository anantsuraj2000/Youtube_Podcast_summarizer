import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
# "bart" (Phase 1 baseline) or "llm" (hierarchical LLM)
SUMMARIZER_BACKEND = os.getenv("SUMMARIZER_BACKEND", "bart").lower()
# LLM provider for the "llm" backend: "gemini" or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_PARALLEL = int(os.getenv("LLM_MAX_PARALLEL", "3"))          # simultaneous LLM calls
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", LLM_PROVIDER).lower()
# Model names change often, so keep them in .env
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
print("All ok! API keys loaded successfully.")
# --- Speech recognition languages ---
# English: Groq Whisper large-v3. Indian languages: local fine-tuned Whisper models
# (Hugging Face id, or a local folder such as an unzipped AI4Bharat IndicWhisper model).
INDIC_MODELS = {
    "hi": os.getenv("INDIC_HI_MODEL", "vasista22/whisper-hindi-medium"),
    "kn": os.getenv("INDIC_KN_MODEL", "vasista22/whisper-kannada-medium"),
}
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
SUPPORTED_LANGUAGE_CODES = ["en", *INDIC_MODELS.keys()]

# Sentence overlap between consecutive transcript chunks
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))   # use 2 or 3; 0 disables

# Size limits per backend (characters, excluding nothing: overlap is included)
CHUNK_MAX_CHARS_BART = int(os.getenv("CHUNK_MAX_CHARS_BART", "2600"))   # was ~3000; lowered so chunk + overlap fits BART's 1024-token input
CHUNK_MAX_CHARS_LLM  = int(os.getenv("CHUNK_MAX_CHARS_LLM", "12000"))
SRL_BACKEND = os.getenv("SRL_BACKEND", "spacy").lower()   # "spacy" | "llm" | "off"