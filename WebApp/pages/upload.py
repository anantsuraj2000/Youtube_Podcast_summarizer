import httpx
import streamlit as st

from common import API_BASE, AUDIO_BACKEND_READY, AUDIO_TYPES, MAX_AUDIO_MB, add_job

BACKEND_DOWN_MSG = (
    "Can't reach the backend at " + API_BASE + ". "
    "Start it with: uvicorn app.main:app --reload --port 8000"
)


# ---------------------------------------------------------------- backend calls
def submit_youtube_job(url: str) -> dict:
    r = httpx.post(f"{API_BASE}/api/jobs", json={"source": url}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def submit_audio_job(uploaded_file, title: str) -> dict:
    """
    Sends a local audio file to the backend.

    CONTRACT for the backend (to build later):
        POST /api/jobs/upload   (multipart/form-data)
            file  : the audio file
            title : optional display name
        -> 200 JSON, same shape as POST /api/jobs:
           {"job_id": ..., "status": "queued", "source": <filename>, "title": ...}
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(),
                      uploaded_file.type or "application/octet-stream")}
    r = httpx.post(f"{API_BASE}/api/jobs/upload", files=files,
                   data={"title": title}, timeout=600.0)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- tabs
def _youtube_tab():
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("🚀 Start Summarization", disabled=not url, key="start_youtube"):
        try:
            with st.spinner("Submitting job..."):
                job = submit_youtube_job(url.strip())
            add_job(job, "youtube")
            st.success(f"Job queued: `{job['job_id']}` — see the Jobs page.")
        except httpx.ConnectError:
            st.error(BACKEND_DOWN_MSG)
        except httpx.HTTPError as e:
            st.error(f"Backend error: {e}")


def _audio_tab():
    st.caption(f"Upload a podcast or recording ({', '.join(AUDIO_TYPES)}), up to {MAX_AUDIO_MB} MB.")
    f = st.file_uploader("Audio file", type=AUDIO_TYPES, key="audio_file")

    title = ""
    too_big = False
    if f is not None:
        size_mb = f.size / (1024 * 1024)
        too_big = size_mb > MAX_AUDIO_MB
        st.markdown(f"**{f.name}** · {size_mb:.1f} MB")
        if too_big:
            st.error(f"File is {size_mb:.0f} MB; the limit is {MAX_AUDIO_MB} MB.")
        else:
            st.audio(f)                                            # play-back preview
            title = st.text_input("Title (optional)", value=f.name.rsplit(".", 1)[0],
                                  key="audio_title")

    if st.button("🚀 Start Summarization", disabled=(f is None or too_big), key="start_audio"):
        if not AUDIO_BACKEND_READY:
            st.info("The audio upload form is ready, but it isn't connected to the backend yet. "
                    "Nothing was sent.")
            return
        try:
            with st.spinner("Uploading and submitting job..."):
                job = submit_audio_job(f, title.strip() or f.name)
            add_job(job, "audio_file")
            st.success(f"Job queued: `{job['job_id']}` — see the Jobs page.")
        except httpx.ConnectError:
            st.error(BACKEND_DOWN_MSG)
        except httpx.HTTPError as e:
            st.error(f"Backend error: {e}")


# ---------------------------------------------------------------- page
def app():
    st.markdown("## 🎬 Summarize a YouTube Video or Audio File")
    st.caption("Choose a source, then start. Progress shows on the Jobs page.")
    st.caption("Supported audio languages: English, Hindi, Kannada. Progress shows on the Jobs page.")
    tab_yt, tab_audio = st.tabs(["🔗 YouTube link", "🎧 Audio file"])
    with tab_yt:
        _youtube_tab()
    with tab_audio:
        _audio_tab()