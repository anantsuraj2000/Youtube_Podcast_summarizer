import streamlit as st
import httpx

def app():
    st.markdown("## 🎬 Summarize a YouTube Video")
    st.caption("Phase 1: Cascaded extractive summarization ")
    st.write("")

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("🚀 Start Summarization", disabled=not url):
        payload = {"source": url}
        with st.spinner("Submitting job..."):
            r = httpx.post("http://localhost:8000/api/jobs", json=payload, timeout=60.0)
            job = r.json()
            st.session_state.jobs[job["job_id"]] = job
            st.success(f"Job queued: `{job['job_id']}`")