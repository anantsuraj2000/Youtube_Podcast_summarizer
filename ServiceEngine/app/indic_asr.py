"""Local transcription for Indian languages with fine-tuned Whisper models (CPU or GPU)."""
from app.config import INDIC_MODELS

_pipes = {}   # language code -> loaded pipeline (first call downloads + loads the model)


def _get_pipe(lang: str):
    if lang not in _pipes:
        import torch
        from transformers import pipeline

        pipe = pipeline(
            "automatic-speech-recognition",
            model=INDIC_MODELS[lang],
            chunk_length_s=30,                       # Whisper window; handles long audio
            device=0 if torch.cuda.is_available() else -1,
        )
        # Force the language/task decoder prompt, as in the models' cards
        pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
            language=lang, task="transcribe"
        )
        _pipes[lang] = pipe
    return _pipes[lang]


def transcribe_indic(audio_path: str, lang: str) -> str:
    """Transcribes an audio file in the given language code ('hi', 'kn')."""
    return _get_pipe(lang)(audio_path)["text"].strip()