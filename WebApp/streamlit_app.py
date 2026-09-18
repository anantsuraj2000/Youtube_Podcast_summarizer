import streamlit as st

st.set_page_config(page_title="AI Summarizer", layout="wide")
if "jobs" not in st.session_state:
    st.session_state["jobs"] = {}   # job_id -> metadata

from style import inject_custom_css
inject_custom_css()
st.sidebar.title("AI YouTube & Podcast Summarizer")
page = st.sidebar.radio("Navigate", ["Upload", "Jobs", "Detail", "Admin"])

if page == "Upload":
    import pages.upload as upload
    upload.app()
elif page == "Jobs":
    import pages.jobs as jobs_page
    jobs_page.app()
# elif page == "Detail":
#     import pages.3_Detail as detail
#     detail.app()
# else:
#     import pages.4_Admin as admin
#     admin.app()
