import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Aeromal — Solar Suite",
    page_icon="⚡",
    layout="wide",
)

AEROMAL_PASSWORD = os.environ.get("AEROMAL_PASSWORD", "")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("""
<h1 style='text-align:center;
background:linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-size:40px;font-weight:800;'>
⚡ Solar Suite
</h1>
<p style='text-align:center;color:gray;font-size:14px;'>
Forecast Correction Platform
</p>
""", unsafe_allow_html=True)

st.sidebar.divider()

if st.session_state.get("aeromal_auth", False):
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.aeromal_auth = False
        st.rerun()


# ==========================================================
# AUTH GATE
# ==========================================================

if "aeromal_auth" not in st.session_state:
    st.session_state.aeromal_auth = False

if not st.session_state.aeromal_auth:
    st.title("🔒 Access bas bade logo ke paas hai")
    password = st.text_input("Enter Password", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        if AEROMAL_PASSWORD and password == AEROMAL_PASSWORD:
            st.session_state.aeromal_auth = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    st.stop()


# ==========================================================
# CACHED HEAVY COMPUTATIONS
# All inputs are plain tuples/scalars so Streamlit can hash them.
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_percentile(power_tuple, days):
    """95th percentile profile — only reruns when data changes."""
    power = np.array(power_tuple, dtype=float)
    a     = power.reshape(days, 96)
    return tuple(np.percentile(a, 95, axis=0).tolist())


@st.cache_data(show_spinner=False)
def cached_best_shift(profile_tuple):
    """Symmetry shift scan — cached, runs once per profile."""
    profile    = np.array(profile_tuple)
    best_err   = np.inf
    best_shift = 0
    for i in range(96):
        sh  = np.roll(profile, -i)
        sym = (profile + sh[::-1]) / 2
        err = np.sqrt(np.mean((profile - sym) ** 2))
        if err < best_err:
            best_err   = err
            best_shift = i
    return int(best_shift)


@st.cache_data(show_spinner=False)
def cached_no_curtailment(power_tuple, days, window, power_availability):
    """Full no-curtailment pipeline — cached per param combination."""
    ap = np.array(cached_percentile(power_tuple, days))
    s  = savgol_filter(ap, window_length=window, polyorder=3)

    best_shift = cached_best_shift(tuple(s.tolist()))

    alpha = 0.50
    sh    = np.roll(s, -best_shift)
    sym   = alpha * s + (1 - alpha) * sh[::-1]

    thr = 0.1
    idx = np.where(ap > thr)[0]
    if len(idx) > 0:
        st_i, en_i = idx[0], idx[-1]
        blend = 8
        w = np.linspace(1, 0, blend)
        sym[st_i+1:st_i+1+blend] = (
            w * ap[st_i+1:st_i+1+blend]
            + (1-w) * sym[st_i+1:st_i+1+blend]
        )
        w = np.linspace(0, 1, blend)
        sym[en_i-blend:en_i] = (
            w * ap[en_i-blend:en_i]
            + (1-w) * sym[en_i-blend:en_i]
        )

    s   = savgol_filter(ap, window_length=11, polyorder=3)
    sym = savgol_filter(sym, window_length=11, polyorder=3)
    s   = np.where(np.clip(s,   0, None) < 0.1, 0, np.clip(s,   0, None))
    sym = np.where(np.clip(sym, 0, None) < 0.1, 0, np.clip(sym, 0, None))
    s   *= power_availability / 100
    sym *= power_availability / 100

    return tuple(ap.tolist()), tuple(s.tolist()), tuple(sym.tolist()), best_shift


@st.cache_data(show_spinner=False)
def cached_sym_shift_nc(s_tuple, shift):
    """Apply a manual shift override in no-curtailment mode."""
    s     = np.array(s_tuple)
    alpha = 0.50
    sh    = np.roll(s, -shift)
    sym   = alpha * s + (1 - alpha) * sh[::-1]
    sym   = savgol_filter(sym, window_length=11, polyorder=3)
    sym   = np.where(np.clip(sym, 0, None) < 0.1, 0, np.clip(sym, 0, None))
    return tuple(sym.tolist())


@st.cache_data(show_spinner=False)
def cached_curtailment(power_tuple, days, peak_cap, target_width,
                       window, power_availability):
    """Full curtailment pipeline — cached per param combination."""
    ap = np.array(cached_percentile(power_tuple, days)) * 1.03

    y = ap.copy()
    n = len(y)
    para = np.zeros(n)

    ys   = savgol_filter(y, 7, 2)
    grad = np.gradient(ys)

    left_peak  = np.argmax(ys[:n//2])
    left_start = np.argmax(grad[:left_peak])
    m1, c1 = np.polyfit(np.arange(left_start, left_peak), ys[left_start:left_peak], 1)

    right_peak = np.argmax(ys[n//2:]) + n//2
    threshold  = 0.02 * np.max(ys)
    active_idx = np.where(ys > threshold)[0]
    right_end  = active_idx[-1]
    m2, c2 = np.polyfit(np.arange(right_peak, right_end), ys[right_peak:right_end], 1)

    A    = (1/m2) - (1/m1)
    B    = (c1/m1) - (c2/m2)
    trip = max(0, (target_width - B) / A)

    peak_left_idx  = int(round((trip - c1) / m1))
    peak_right_idx = int(round((trip - c2) / m2))

    if peak_cap >= trip:
        for i in range(n):
            val = m1*i + c1
            para[i] = min(val, trip)
            if val >= trip: peak_left_idx = i; break
        right_curve = np.zeros(n)
        for i in range(n-1, -1, -1):
            val = m2*i + c2
            right_curve[i] = min(val, trip)
            if val >= trip: peak_right_idx = i; break
        para  = np.maximum(para, right_curve)
        width = peak_right_idx - peak_left_idx
        dome_height = max(20, 0.12 * trip)
        xc    = np.linspace(-1, 1, width)
        shape = np.sqrt(np.maximum(0, 1 - xc**2))
        dome  = trip + dome_height * shape
        dome[0] = trip; dome[-1] = trip
        para[peak_left_idx:peak_right_idx] = dome
        para = savgol_filter(para, window, 3)
        para = np.clip(para, 0, None)
        para[:left_start] = ys[:left_start]
        para[right_end:]  = ys[right_end:]
        para = savgol_filter(para, 7, 3) * power_availability / 100
        para = np.clip(para, 0, None)
        para = np.where(para < 0.2, 0, para)
    else:
        for i in range(n):
            val = m1*i + c1
            para[i] = val
            if val >= peak_cap: peak_left_idx = i; break
        right_curve = np.zeros(n)
        for i in range(n-1, -1, -1):
            val = m2*i + c2
            right_curve[i] = val
            if val >= peak_cap: peak_right_idx = i; break
        para = np.maximum(para, right_curve)
        para[peak_left_idx:peak_right_idx] = peak_cap
        para = np.clip(para, 0, peak_cap)
        para = savgol_filter(para, window, 3)
        para[:left_start] = ys[:left_start]
        para[right_end:]  = ys[right_end:]
        para = savgol_filter(para, 7, 3) * power_availability / 100
        para = np.clip(para, 0, peak_cap)
        para = np.where(para < 1, 0, para)

    best_shift = cached_best_shift(tuple(para.tolist()))
    return tuple(ap.tolist()), tuple(para.tolist()), best_shift


@st.cache_data(show_spinner=False)
def cached_sym_shift_c(final_smooth_tuple, shift):
    """Apply a manual shift in curtailment mode — instant."""
    fs  = np.array(final_smooth_tuple)
    sh  = np.roll(fs, -int(shift))
    sym = (fs + sh[::-1]) / 2
    return tuple(sym.tolist())


# ==========================================================
# PAGE
# ==========================================================

st.title("Kaha hai Aeromal ka khauf?!!")

# ── Session defaults ──────────────────────────────────────
if "am_power" not in st.session_state:
    st.session_state.am_power = [0.0] * 96

# ── Data editor ───────────────────────────────────────────
input_df = pd.DataFrame({"Power": st.session_state.am_power})

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    key="cam_editor",
)

power_list = pd.to_numeric(
    edited_df.iloc[:, 0], errors="coerce"
).fillna(0).tolist()

# Persist without triggering extra reruns
if power_list != st.session_state.am_power:
    st.session_state.am_power = power_list

power_tuple = tuple(power_list)
n_rows      = len(power_list)

if n_rows == 0:
    st.stop()

if n_rows % 96 != 0:
    st.error(f"Number of rows must be divisible by 96. Current: {n_rows}")
    st.stop()

days = n_rows // 96

# ── Mode toggle ───────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stToggle"]{
    background:#1f2937;border:2px solid #0072ff;
    border-radius:14px;padding:14px 18px;margin-bottom:15px;
}
div[data-testid="stToggle"] label{
    font-size:18px !important;font-weight:700 !important;
}
</style>""", unsafe_allow_html=True)

curtailment = st.toggle("⚡ Curtailment Mode", value=False)


# ==========================================================
# NO CURTAILMENT  — params in a form, heavy work cached
# ==========================================================

if not curtailment:

    with st.form("am_nc_form"):
        col1, col2, col3 = st.columns(3)
        window = col1.number_input(
            "Window Length", min_value=5, max_value=31, step=2, value=11)
        power_availability = col2.number_input(
            "Power Availability (%)", min_value=0, max_value=1000, value=100)
        apply_btn = st.form_submit_button(
            "Apply", use_container_width=True, type="primary")

    # Compute (cached — instant if params unchanged)
    ap_t, s_t, sym_t, best_shift = cached_no_curtailment(
        power_tuple, days, window, power_availability)

    with st.form("am_nc_shift_form"):
        col1, col2, _ = st.columns(3)
        shift = col1.number_input(
            "Shift", min_value=0, max_value=95, value=best_shift, step=1)
        shift_btn = st.form_submit_button(
            "Apply Shift", use_container_width=True)

    # Apply shift override if user changed it
    if shift != best_shift:
        sym_t = cached_sym_shift_nc(s_t, shift)

    ap  = list(ap_t)
    s   = list(s_t)
    sym = list(sym_t)
    x   = list(range(96))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sym, name="Sym Profile",
                              line=dict(color="#00c6ff", width=4)))
    fig.add_trace(go.Scatter(x=x, y=s,   name="Profile",
                              line=dict(color="#22c55e", width=4)))
    fig.add_trace(go.Scatter(x=x, y=ap,  name="95th Percentile",
                              line=dict(color="#ef4444", width=4)))
    fig.update_layout(
        height=550, hovermode="x unified", template="streamlit",
        xaxis_title="Block", yaxis_title="Power",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Generated Curve")
    st.dataframe(
        pd.DataFrame({"Percentile": ap, "Profile": s, "Sym Profile": sym}),
        use_container_width=True, hide_index=True,
    )


# ==========================================================
# CURTAILMENT  — params in a form, heavy work cached
# ==========================================================

else:

    if not np.any(np.array(power_list) > 0):
        st.warning("Please enter Power values to continue.")
        st.stop()

    st.subheader("Parameters")

    with st.form("am_c_form"):
        col1, col2 = st.columns(2)
        power_availability = col1.number_input(
            "Power Availability (%)", value=100, step=1)
        peak_cap = col1.number_input(
            "Peak Cap", value=int(max(power_list)), step=1)
        target_width = col2.number_input(
            "Target Width", value=25, step=1)
        window = col2.slider(
            "Window Length", 5, 31, 11, step=2)
        apply_btn = st.form_submit_button(
            "Apply", use_container_width=True, type="primary")

    # Compute profile (cached — instant if params unchanged)
    ap_t, fs_t, best_shift = cached_curtailment(
        power_tuple, days, peak_cap, target_width, window, power_availability)

    with st.form("am_c_shift_form"):
        col1, _, _ = st.columns(3)
        shift = col1.number_input(
            "Shift", min_value=0, max_value=95, value=best_shift, step=1)
        shift_btn = st.form_submit_button(
            "Apply Shift", use_container_width=True)

    # Apply shift (cached — instant)
    sym_t = cached_sym_shift_c(fs_t, shift)

    ap  = list(ap_t)
    fs  = list(fs_t)
    sym = list(sym_t)
    x   = list(range(96))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ap,  name="Generation",
                              line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x, y=fs,  name="Profile",
                              line=dict(color="#00c6ff", width=3)))
    fig.add_trace(go.Scatter(x=x, y=sym, name="Sym Profile",
                              line=dict(color="#0072ff", width=3)))
    fig.update_layout(
        height=550, hovermode="x unified", template="streamlit",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        pd.DataFrame({"Power": ap, "Profile": fs, "Sym Profile": sym}),
        use_container_width=True,
    )
