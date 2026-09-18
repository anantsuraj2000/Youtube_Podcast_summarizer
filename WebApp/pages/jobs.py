import httpx, streamlit as st
from style import status_color

def poll_job(job_id):
    r = httpx.get(f"http://localhost:8000/api/jobs/{job_id}")
    return r.json()

def app():
    st.markdown("## 📋 Jobs")

    if not st.session_state.jobs:
        st.info("No jobs yet — go to Upload to submit a YouTube URL.")
        return

    for job_id, meta in st.session_state.jobs.items():
        status = meta.get("status", "unknown")
        color = status_color(status)

        st.markdown(f"""
        <div class="job-card">
            <b>Job:</b> {job_id[:8]}...
            <span class="status-badge" style="background-color:{color}">{status}</span>
        </div>
        """, unsafe_allow_html=True)

        if status not in ("done", "failed"):
            if st.button("🔄 Refresh", key=f"refresh_{job_id}"):
                st.session_state.jobs[job_id] = poll_job(job_id)
                st.rerun()

        if status == "done":
            with st.expander("📄 View Summary & Entities", expanded=True):
                st.markdown("**Summary:**")
                st.write(meta.get("summary"))
                st.markdown("**Entities found:**")
                st.write(meta.get("entities"))

        if status == "failed":
            st.error(meta.get("error"))