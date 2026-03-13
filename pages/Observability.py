import streamlit as st

st.set_page_config(page_title="Observability", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #050816; color: #E6EAF2; }
    [data-testid="stHeader"] { background: rgba(5, 8, 22, 0.92); }
    .block-container { padding-top: 1rem; max-width: 100%; }
    .obs-title { font-size: 1.55rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.2rem; }
    .obs-subtitle { color: #94A3B8; font-size: 0.95rem; margin-bottom: 1rem; }
    .status-chip {
        padding: 10px 12px;
        border: 1px solid rgba(148,163,184,0.15);
        background: rgba(15,23,42,0.96);
        border-radius: 12px;
        color: #CBD5E1;
        font-size: 0.92rem;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="obs-title">Observability</div>', unsafe_allow_html=True)
st.markdown('<div class="obs-subtitle">Use LangSmith directly for the exact dashboard.</div>', unsafe_allow_html=True)

dashboard_url = "https://smith.langchain.com/o/951a4048-d833-49a0-a0bf-4706a63d91f0/dashboards/projects/39722a41-5c8b-4462-a2a1-7292be5c09a0"

st.markdown(
    """
    <div class="status-chip">
        Inline embedding is blocked by LangSmith security headers, so open the dashboard in a new tab for the exact real-time view.
    </div>
    """,
    unsafe_allow_html=True,
)

st.link_button("Open LangSmith Dashboard", dashboard_url, use_container_width=True)

st.code(dashboard_url, language="text")
