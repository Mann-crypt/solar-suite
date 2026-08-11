import streamlit as st

st.set_page_config(
    page_title="Solar Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Idle-reload script (10 min) ──────────────────────────────────────────────
st.components.v1.html("""<script>
let t;
function r(){clearTimeout(t);t=setTimeout(()=>location.reload(),120000);}
["mousemove","mousedown","keydown","scroll","touchstart"].forEach(e=>document.addEventListener(e,r));
r();
</script>""", height=0)

# ── Global sidebar branding ───────────────────────────────────────────────────

# Aeromal logout — only visible when logged in
if st.session_state.get("aeromal_auth", False):
    if st.sidebar.button("🚪 Logout Aeromal", use_container_width=True):
        st.session_state.aeromal_auth = False
        st.rerun()
    st.sidebar.divider()

st.sidebar.markdown("""
<div style='text-align:center;color:gray;font-size:13px;line-height:2;'>
Developed and Maintained by:<br><b style='color:#e5e7eb;'>Manjot Singh</b><br>
Script Writer:<br><b style='color:#e5e7eb;'>Tushar Sharma</b><br>
Challenger:<br><b style='color:#e5e7eb;'>Aarav Sharma</b><br>
Tester:<br><b style='color:#e5e7eb;'>Jatin Chaturvedi</b><br>
Improviser:<br><b style='color:#e5e7eb;'>Ujala Agrahari</b><br>
Suggested by:<br><b style='color:#e5e7eb;'>Garima Bajetha</b>
</div>
""", unsafe_allow_html=True)

# ── Home page content ─────────────────────────────────────────────────────────
st.markdown("""
<h1 style='background:linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:48px;font-weight:800;'>⚡ Solar Suite</h1>
<p style='color:#8b96a8;font-size:18px;'>Forecast Correction Platform</p>
""", unsafe_allow_html=True)

st.divider()

st.markdown(
    """
    <style>
    .tool-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 24px;
        min-height: 190px;
        margin-bottom: 10px;
    }

    .tool-card h3 {
        margin-top: 0;
        color: white;
    }

    .tool-card p {
        color: #8b96a8;
        font-size: 14px;
        line-height: 1.6;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 45px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3 = st.columns(3)

# ==========================================================
# LOSS CORRECTION
# ==========================================================

with col1:

    st.markdown(
        """
        <div class="tool-card">
            <h4>⛅ Loss Correction</h4>
            <p>
                Differential evolution optimization for Fixed and
                Tracking plants. Supports cluster and non-cluster
                workbooks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Open Loss Correction →",
        key="home_loss_correction",
        use_container_width=True,
    ):
        st.switch_page("pages/1_Loss_Correction.py")


# ==========================================================
# RT CORRECTION
# ==========================================================

with col2:

    st.markdown(
        """
        <div class="tool-card">
            <h4>⏰ RT Correction</h4>
            <p>
                Parabolic ramp-profile fitting for real-time
                generation correction with key time block lookup.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Open RT Correction →",
        key="home_rt_correction",
        use_container_width=True,
    ):
        st.switch_page("pages/2_RT_Correction.py")


# ==========================================================
# AEROMAL
# ==========================================================

with col3:

    st.markdown(
        """
        <div class="tool-card">
            <h4>🐱‍🏍 Aeromal</h4>
            <p>
                Curtailment shaping and 95th-percentile profile
                generation using Savitzky-Golay filtering.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Open Aeromal →",
        key="home_aeromal",
        use_container_width=True,
    ):
        st.switch_page("pages/3_Aeromal.py")
