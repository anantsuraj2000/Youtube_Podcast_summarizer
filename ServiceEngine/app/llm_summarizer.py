import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.chunking import chunk_transcript
from app.errors import SummarizationError
from app.config import (
    CHUNK_MAX_CHARS_LLM, GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_LLM_MODEL,
    LLM_MAX_OUTPUT_TOKENS, LLM_MAX_PARALLEL, LLM_PROVIDER, LLM_TIMEOUT_SECONDS,
)

CHUNK_CHARS = CHUNK_MAX_CHARS_LLM   # ~3k tokens per chunk (set in config.py)
MAX_REDUCE_CHARS = 15000   # if all notes exceed this, condense them in another round
MAX_LEVELS = 4             # safety stop for the reduce loop
MAX_RETRIES = 3            # attempts per call (the Groq SDK's own retries are switched off)
MIN_TRANSCRIPT_CHARS = 50


class LLMSummarizationError(SummarizationError):
    pass


# ---------------- provider layer: all API-specific details live here ----------------
_gemini_client = None
_groq_client = None
_client_lock = threading.Lock()      # clients are created lazily, possibly from several threads


def _call_gemini(system: str, prompt: str) -> str:
    global _gemini_client
    with _client_lock:
        if _gemini_client is None:
            if not GEMINI_API_KEY:
                raise LLMSummarizationError("GEMINI_API_KEY is missing in .env (provider=gemini).")
            from google import genai  # pip install -U google-genai
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    interaction = _gemini_client.interactions.create(
        model=GEMINI_MODEL,
        system_instruction=system,
        input=prompt,
        generation_config={"thinking_level": "low"},
    )
    return interaction.output_text


def _call_groq(system: str, prompt: str) -> str:
    global _groq_client
    with _client_lock:
        if _groq_client is None:
            from groq import Groq
            # max_retries=0: retrying is done once, in _call_llm, instead of stacking two retry loops
            _groq_client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)

    kwargs = dict(
        model=GROQ_LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=LLM_MAX_OUTPUT_TOKENS,   # Groq's default is only 1024 (reasoning included)
    )
    if GROQ_LLM_MODEL.startswith("openai/gpt-oss"):
        # reasoning model: think briefly and don't send the reasoning back (faster, fewer tokens)
        kwargs["extra_body"] = {"reasoning_effort": "low", "include_reasoning": False}

    resp = _groq_client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise LLMSummarizationError(
            "The model hit its output limit before finishing. Raise LLM_MAX_OUTPUT_TOKENS in .env."
        )
    return choice.message.content


def _call_llm(system: str, prompt: str, provider: str | None = None) -> str:
    """Calls the provider (default: LLM_PROVIDER), retrying with backoff on rate limits / timeouts."""
    name = (provider or LLM_PROVIDER).lower()
    call = {"gemini": _call_gemini, "groq": _call_groq}.get(name)
    if call is None:
        raise LLMSummarizationError(f"Unknown LLM provider '{name}' (use 'gemini' or 'groq').")

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            text = call(system, prompt)
            if text and text.strip():
                return text.strip()
            last_err = LLMSummarizationError("The model returned an empty response.")
        except LLMSummarizationError:
            raise
        except Exception as e:                      # 429s, timeouts, 5xx ...
            last_err = e
        if attempt < MAX_RETRIES - 1:               # no pointless wait after the last attempt
            time.sleep(2 ** attempt * 2)            # 2s, 4s
    raise LLMSummarizationError(
        f"LLM call failed after {MAX_RETRIES} attempts ({type(last_err).__name__}): {last_err}"
    )


def _parallel(fn, items: list) -> list:
    """Runs fn over items with up to LLM_MAX_PARALLEL threads; results keep the input order."""
    if len(items) <= 1 or LLM_MAX_PARALLEL <= 1:
        return [fn(i) for i in items]
    pool = ThreadPoolExecutor(max_workers=LLM_MAX_PARALLEL)
    try:
        return list(pool.map(fn, items))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)   # on an error, queued calls are dropped


# ---------------- prompts ----------------
SYSTEM = (
    "You summarize speech transcripts of videos and podcasts. Use only information in the "
    "text you are given; never add outside facts. Ignore filler, greetings, sponsor reads and "
    "requests to like, subscribe or comment. Write in the same language as the transcript."
)


def _target_words(transcript: str) -> int:
    """About 1 summary word per 10 transcript words, kept between 70 and 250."""
    return max(70, min(250, len(transcript.split()) // 10))


def _notes_for_chunk(chunk: str, idx: int, total: int) -> str:
    prompt = (
        f"This is part {idx} of {total} of one transcript.\n"
        "Write concise notes (4-8 bullet points) with the main points of THIS part only. "
        "The first sentences may repeat the end of the previous part, as context only: "
        "do not note points from them again.\n"
        "Keep concrete details: rules, steps, numbers, names, conclusions.\n\n"
        f"TRANSCRIPT PART:\n{chunk}"
    )
    return _call_llm(SYSTEM, prompt)


def _condense_notes(notes: str) -> str:
    prompt = (
        "These are notes from consecutive parts of one transcript, in order. "
        "Merge them into shorter notes (at most 8 bullet points) that keep every distinct "
        "main point, rule, step, number and name.\n\n"
        f"NOTES:\n{notes}"
    )
    return _call_llm(SYSTEM, prompt)


def _final_summary(material: str, target_words: int, from_notes: bool) -> str:
    source = "notes from consecutive parts of a video, in order" if from_notes else "a video transcript"
    prompt = (
        f"Below are {source}. Write ONE coherent summary of about {target_words} words as plain "
        "prose (no bullets, headings or preamble). Cover the whole video from start to end, not "
        "just the beginning, and make sure every main point or rule it states is included.\n\n"
        f"{'NOTES' if from_notes else 'TRANSCRIPT'}:\n{material}"
    )
    return _call_llm(SYSTEM, prompt)


# ---------------- public API ----------------
def llm_summarize_with_stats(transcript: str) -> tuple[str, dict]:
    """Returns (summary, stats) with chunk_count, reduce_levels, llm_calls."""
    if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
        raise LLMSummarizationError(
            f"Transcript too short to summarize ({len(transcript.strip()) if transcript else 0} characters). "
            "The video may have no speech, or transcription failed silently."
        )

    target = _target_words(transcript)
    chunks = chunk_transcript(transcript, max_chunk_chars=CHUNK_CHARS)
    stats = {"chunk_count": len(chunks), "reduce_levels": 0, "llm_calls": 0}

    if len(chunks) == 1:
        stats["llm_calls"] = 1
        return _final_summary(chunks[0], target, from_notes=False), stats

    # MAP: chunk -> notes (chunks are independent, so they run in parallel)
    total = len(chunks)
    notes = _parallel(lambda ic: _notes_for_chunk(ic[1], ic[0] + 1, total), list(enumerate(chunks)))
    stats["llm_calls"] += total

    # REDUCE: condense in batches until everything fits one final prompt
    while sum(len(n) for n in notes) > MAX_REDUCE_CHARS and stats["reduce_levels"] < MAX_LEVELS:
        batches, current, size = [], [], 0
        for n in notes:
            if current and size + len(n) > MAX_REDUCE_CHARS // 2:
                batches.append(current)
                current, size = [], 0
            current.append(n)
            size += len(n)
        if current:
            batches.append(current)
        if len(batches) >= len(notes):
            break
        notes = _parallel(lambda b: _condense_notes("\n".join(b)), batches)
        stats["llm_calls"] += len(batches)
        stats["reduce_levels"] += 1

    # FINAL
    stats["llm_calls"] += 1
    return _final_summary("\n\n".join(notes), target, from_notes=True), stats


def llm_summarize(transcript: str) -> str:
    return llm_summarize_with_stats(transcript)[0]