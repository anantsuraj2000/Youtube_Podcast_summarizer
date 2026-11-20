from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import os, uuid, threading

from app.audio_extractor import (
    extract_audio_from_youtube, get_video_title, prepare_uploaded_audio, AudioExtractionError, AUDIO_DIR,
)
from app.transcriber import transcribe_audio,transcribe_with_language
from app.errors import SummarizationError,UnsupportedLanguageError
from app.summary_service import summarize
from app.refiner import refine_summary
from app.config import LANGUAGE_NAMES
from app.translator import TRANSLATION_LANGUAGES, translate_text
from app.srl import safe_extract_roles

app = FastAPI()

# In-memory job store (fine for Phase 1 / single-user demo; not persistent across restarts)
jobs = {}

ALLOWED_AUDIO_EXTS = {"mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4", "webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024   # keep in sync with MAX_AUDIO_MB in WebApp/common.py


class JobRequest(BaseModel):
    source: str  # YouTube URL


def run_pipeline(job_id: str, url: str | None = None, upload_path: str | None = None):
    """Runs the pipeline for a YouTube URL, or for an uploaded file saved at upload_path."""
    try:
        jobs[job_id]["status"] = "extracting_audio"
        if upload_path:
            try:
                audio_path = prepare_uploaded_audio(upload_path)
            finally:
                if os.path.exists(upload_path):
                    os.remove(upload_path)          # raw upload no longer needed
        else:
            audio_path = extract_audio_from_youtube(url)

        jobs[job_id]["status"] = "transcribing"
        try:
            transcript, language = transcribe_with_language(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)               # audio is no longer needed after ASR

        jobs[job_id]["language"] = LANGUAGE_NAMES.get(language, language)

        jobs[job_id]["status"] = "summarizing"
        summary = summarize(transcript, language=language)

        jobs[job_id]["status"] = "refining"
        if language == "en":
            result = refine_summary(summary)
        else:                                   # spaCy en_core_web_sm is English-only
            result = {"summary": summary, "entities": [],
                      "roles": safe_extract_roles(summary, language)}

        jobs[job_id]["status"] = "done"
        jobs[job_id]["transcript"] = transcript
        jobs[job_id]["summary"] = result["summary"]
        jobs[job_id]["entities"] = result["entities"]
        jobs[job_id]["roles"] = result.get("roles", [])

    except UnsupportedLanguageError as e:
        jobs[job_id]["status"] = "unsupported_language"
        jobs[job_id]["language"] = e.language_name
        jobs[job_id]["error"] = str(e)

    except (AudioExtractionError, SummarizationError) as e:
        # Known, expected failure — show the clean message
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

    except Exception as e:
        # Unexpected failure (e.g. Groq/network error) — still surface it, don't crash the thread
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = f"Unexpected error: {e}"


@app.post("/api/jobs")
def create_job(request: JobRequest):
    job_id = str(uuid.uuid4())
    try:
        title = get_video_title(request.source)
    except Exception:
        title = "Untitled video"

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "source": request.source,
        "title": title,
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, request.source))
    thread.start()

    return jobs[job_id]


@app.post("/api/jobs/upload")
def create_upload_job(file: UploadFile = File(...), title: str = Form("")):
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if ext not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}")

    job_id = str(uuid.uuid4())
    upload_path = os.path.join(AUDIO_DIR, f"upload_{job_id}.{ext}")

    # Stream to disk in 1 MB pieces so a big file never sits fully in memory
    size, too_big = 0, False
    with open(upload_path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                too_big = True
                break
            out.write(chunk)
    if too_big:
        os.remove(upload_path)
        raise HTTPException(status_code=413,
                            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    display_title = title.strip() or os.path.splitext(file.filename)[0]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "source": file.filename,
        "title": display_title,
    }
    threading.Thread(target=run_pipeline, args=(job_id,), kwargs={"upload_path": upload_path}).start()
    return jobs[job_id]

@app.post("/api/jobs/{job_id}/translate")
def translate_job(job_id: str, target: str = "en"):
    """Returns the job's summary in English (en), Hindi (hi) or Kannada (kn). Cached per language."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="The summary isn't ready yet.")
    if target not in TRANSLATION_LANGUAGES:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported language '{target}'. Use one of: {', '.join(TRANSLATION_LANGUAGES)}.")

    source = job.get("summary_language", "en")       # language the summary was written in
    cache = job.setdefault("translations", {source: job["summary"]})
    if target not in cache:
        try:
            cache[target] = translate_text(job["summary"], target, source)
        except Exception as e:                       # provider/rate-limit problems -> clear message
            raise HTTPException(status_code=502, detail=f"Translation failed: {e}")
    return {"job_id": job_id, "language": target, "text": cache[target]}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return jobs.get(job_id, {"error": "job not found"})