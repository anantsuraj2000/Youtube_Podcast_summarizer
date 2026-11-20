import httpx
import streamlit as st

from common import API_BASE,show_summary,show_roles
from style import status_color

SOURCE_LABELS = {"youtube": "🔗 YouTube", "audio_file": "🎧 Audio file"}


def poll_job(job_id):
    r = httpx.get(f"{API_BASE}/api/jobs/{job_id}", timeout=30.0)
    return r.json()


def app():
    st.markdown("## 📋 Jobs")

    if not st.session_state.jobs:
        st.info("No jobs yet — go to Upload to submit a YouTube URL or an audio file.")
        return

    # Newest first; only the last 5 jobs are kept (see common.add_job)
    for job_id, meta in reversed(list(st.session_state.jobs.items())):
        status = meta.get("status", "unknown")
        color = status_color(status)
        source = SOURCE_LABELS.get(meta.get("source_type"), "")
        title = meta.get("title", "Untitled")
        language = meta.get("language")
        lang_label = f" · 🌐 {language}" if language else ""
        status_text = "out of scope" if status == "unsupported_language" else status

        st.markdown(f"""
        <div class="job-card">
            <b>{title}</b> &nbsp;<small>{source} · {job_id[:8]}...</small>
            <span class="status-badge" style="background-color:{color}">{status}</span>
        </div>
        """, unsafe_allow_html=True)

        if status not in ("done", "failed"):
            if st.button("🔄 Refresh", key=f"refresh_{job_id}"):
                try:
                    updated = poll_job(job_id)
                    updated["source_type"] = meta.get("source_type")   # keep our own field
                    st.session_state.jobs[job_id] = updated
                except httpx.HTTPError:
                    st.error("Can't reach the backend to refresh this job.")
                else:
                    st.rerun()

        if status == "done":
            with st.expander("📄 View Summary & Entities", expanded=True):
                st.markdown("**Summary:**")
                show_summary(job_id, meta, "jobs")
                st.markdown("**Entities found:**")
                st.write(meta.get("entities"))
            show_roles(meta)          # <-- new line, aligned with the `with`, not inside it

        if status == "failed":
            st.error(meta.get("error"))