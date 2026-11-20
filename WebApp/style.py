import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
        /* Overall app background */
        .stApp {
            background-color: #F7F9FC;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #12213B;
        }
        section[data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] h1 {
            color: #FF7A59 !important;
            font-size: 1.3rem;
        }

        /* Headers */
        h1, h2, h3 {
            color: #12213B;
        }

        /* Primary buttons */
        div.stButton > button {
            background-color: #1E88A8;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
        }
        div.stButton > button:hover {
            background-color: #FF7A59;
            color: white;
        }

        /* Text inputs */
        div[data-baseweb="input"] {
            border-radius: 8px;
        }

        /* Success/info boxes */
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        /* Card-like containers */
        .job-card {
            background-color: white;
            border: 1px solid #E3E8EF;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 6px rgba(18, 33, 59, 0.08);
        }
        .status-badge {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)


def status_color(status: str) -> str:
    colors = {
        "queued": "#5C6B7A",
        "extracting_audio": "#1E88A8",
        "transcribing": "#1E88A8",
        "summarizing": "#1E88A8",
        "refining": "#1E88A8",
        "done": "#2E9E5B",
        "failed": "#E5484D",
        "unsupported_language": "#C98A00",
    }
    return colors.get(status, "#5C6B7A")