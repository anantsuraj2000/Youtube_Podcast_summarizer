import streamlit as st

def app():
    st.markdown("## 🕘 History")

    if not st.session_state.jobs:
        st.info("No videos summarized yet.")
        return

    # Most recent 5 jobs, newest first
    recent_jobs = list(st.session_state.jobs.items())[-5:][::-1]

    for job_id, meta in recent_jobs:
        title = meta.get("title", "Untitled video")
        status = meta.get("status")

        # Styled as a clickable link using a button
        if st.button(f"🔗 {title}", key=f"history_{job_id}"):
            st.session_state["selected_history_job"] = job_id

        # If this job is the one currently selected, show its summary right below
        if st.session_state.get("selected_history_job") == job_id:
            if status == "done":
                with st.container(border=True):
                    st.markdown(f"**Video:** {title}")
                    st.markdown(f"**URL:** {meta.get('source')}")
                    st.markdown("**Summary:**")
                    st.write(meta.get("summary"))
                    st.markdown("**Entities:**")
                    st.write(meta.get("entities"))
            elif status == "failed":
                st.error(meta.get("error"))
            else:
                st.info(f"Still processing... (status: {status})")

        st.divider()