from fastapi import FastAPI
from pydantic import BaseModel
import uuid, threading

from app.audio_extractor import extract_audio_from_youtube
from app.transcriber import transcribe_audio
from app.summarizer import extractive_summarize
from app.refiner import refine_summary

app = FastAPI()

# In-memory job store (fine for Phase 1 / single-user demo; not persistent across restarts)
jobs = {}


class JobRequest(BaseModel):
    source: str  # YouTube URL only, for now


def run_pipeline(job_id: str, url: str):
    try:
        jobs[job_id]["status"] = "extracting_audio"
        wav_path = extract_audio_from_youtube(url)

        jobs[job_id]["status"] = "transcribing"
        transcript = transcribe_audio(wav_path)

        jobs[job_id]["status"] = "summarizing"
        summary = extractive_summarize(transcript)

        jobs[job_id]["status"] = "refining"
        result = refine_summary(summary)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["transcript"] = transcript
        jobs[job_id]["summary"] = result["summary"]
        jobs[job_id]["entities"] = result["entities"]

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/api/jobs")
def create_job(request: JobRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"job_id": job_id, "status": "queued", "source": request.source}

    # Runs in a background thread so the API responds immediately
    thread = threading.Thread(target=run_pipeline, args=(job_id, request.source))
    thread.start()

    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return jobs.get(job_id, {"error": "job not found"})