"""Shared WebApp settings and helpers."""
import streamlit as st
import httpx

API_BASE = "http://localhost:8000"
MAX_JOBS = 5                      # keep only the most recent jobs in the session
AUDIO_TYPES = ["mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4", "webm"]
MAX_AUDIO_MB = 200                # Streamlit's default upload limit; raise via server.maxUploadSize

# Flip to True once the backend has POST /api/jobs/upload (see pages/upload.py -> submit_audio_job)
AUDIO_BACKEND_READY = True


def add_job(job: dict, source_type: str) -> None:
    """Stores a job in the session, newest last, and drops the oldest beyond MAX_JOBS."""
    job["source_type"] = source_type           # "youtube" or "audio_file"
    st.session_state.jobs[job["job_id"]] = job
    while len(st.session_state.jobs) > MAX_JOBS:
        del st.session_state.jobs[next(iter(st.session_state.jobs))]

# ---------------------------------------------------------------- summary language selector
SUMMARY_LANGUAGES = {"English": "en", "Hindi": "hi", "Kannada": "kn"}   # English is the default


def _fetch_translation(job_id: str, code: str) -> str:
    r = httpx.post(f"{API_BASE}/api/jobs/{job_id}/translate", params={"target": code}, timeout=180.0)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()["text"]


def show_summary(job_id: str, meta: dict, ctx: str) -> None:
    """Shows a finished job's summary with an English / Hindi / Kannada selector (English default)."""
    choice = st.radio("Summary language", list(SUMMARY_LANGUAGES), horizontal=True,
                      key=f"lang_{ctx}_{job_id}", index=0)
    code = SUMMARY_LANGUAGES[choice]
    source = meta.get("summary_language", "en")

    if code == source:                                   # already in that language: no call needed
        st.write(meta.get("summary"))
        return

    cache = st.session_state.setdefault("translations", {})
    key = (job_id, code)
    if key not in cache:
        try:
            with st.spinner(f"Translating to {choice}..."):
                cache[key] = _fetch_translation(job_id, code)
        except httpx.ConnectError:
            st.error(f"Can't reach the backend at {API_BASE}. Is uvicorn running?")
            return
        except httpx.TimeoutException:
            st.error("Translation is taking too long. Please try again in a moment.")
            return
        except Exception as e:
            st.error(f"Could not translate: {e}")
            return
    st.write(cache[key])

def show_roles(meta: dict) -> None:
    """Table of who did what (semantic roles) for a finished job."""
    roles = meta.get("roles") or []
    if not roles:
        return
    rows = [{
        "Who": r.get("agent") or "—",
        "Action": ("not " if r.get("negated") else "") + (r.get("predicate") or ""),
        "To whom / what": r.get("patient") or "—",
        "When": r.get("time") or "—",
        "Where": r.get("location") or "—",
    } for r in roles]
    with st.expander(f"🧩 Who did what ({len(rows)} events)"):
        st.dataframe(rows, hide_index=True)
