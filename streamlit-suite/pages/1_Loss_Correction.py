import io
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Loss Correction — Solar Suite",
    page_icon="⚡",
    layout="wide",
)


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


# ==========================================================
# STYLE
# ==========================================================

st.markdown("""
<style>
div[data-testid="metric-container"]{
    background:#111827;
    border:1px solid #1f2937;
    border-radius:10px;
    padding:12px 20px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# QUOTES
# ==========================================================

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice...",
    "🌟 Aapke face pr toh Modiji se bhi jyda glow hai..",
    "😁 Horaha hai benstokes Kaan mai ghusjao insaan ke...",
    "😗 Muskuraiye aap MAL mai hai...",
    "🥱 Hum na hote toh Operations ka kya hota?..",
    "😎 6:30 hote hi Billu MAL se faraar...",
    "😇 Guruji ne ek baat kahi thi....",
    "🎼 Karna hai kuchh kaam M se gaao...",
    "😠 Nahi karni Loss Correction, Now what to do?...",
    "💸 Iss Job ko chhod or chhod kar ameer ho..",
]


# ==========================================================
# CACHED EXCEL READERS  — parse once per file, never again
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_workbook(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    cluster = "Fixed-CL1" in xls.sheet_names
    sheet = "Fixed-CL1" if cluster else "Fixed"
    ghi_cols = (
        ["CL1-GHI", "CL2-GHI", "CL3-GHI", "CL4-GHI", "CL5-GHI"]
        if cluster else ["GHI_Forecast"]
    )
    df = xls.parse(sheet_name=sheet, header=1)
    df.columns = df.columns.str.strip()
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce").fillna(0)
    null_idx = df[df["Date"].isna()].index
    if len(null_idx):
        df = df.iloc[: df.index.get_loc(null_idx[0])]
    df = df.iloc[:96]
    return cluster, ghi_cols, df[ghi_cols + ["Actual"]].copy()


@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes):
    # Try header rows 0, 1, 2 until we find one with numeric efficiency data
    for header_row in [1, 2, 0]:
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Area & Efficiency",
            header=header_row,
            usecols=range(8),
        )
        df.columns = df.columns.str.strip()
        df = df[df["Module Type"].notna()].copy()
        # Force numeric on efficiency and area columns
        for col in df.columns:
            if col != "Module Type":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Standard PV Efficiency (%)"])
        if len(df) > 0 and df["Standard PV Efficiency (%)"].notna().any():
            return df.reset_index(drop=True)
    # Fallback — return whatever we got from header=1
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1, usecols=range(8),
    )
    df.columns = df.columns.str.strip()
    return df[df["Module Type"].notna()].copy()


@st.cache_data(show_spinner=False)
def read_weights(file_bytes):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2, usecols=[12, 13, 14, 15, 16],
    )


@st.cache_data(show_spinner=False)
def read_forecast_config(file_bytes):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config", header=8,
    )
    return float(df.loc[0, "Lat"])


@st.cache_data(show_spinner=False)
def read_tilt(file_bytes):
    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle", header=7,
    )
    df.columns = df.columns.str.strip()
    df["Fixed"] = df["Fixed"].fillna(0)
    df = df[df["Fixed"] != 0].dropna(how="all", axis=1)
    df = df.rename(columns={"Unnamed: 2": "Month_Num", "Unnamed: 3": "Month"})
    return df.set_index("Month")["Fixed"].to_dict()


@st.cache_data(show_spinner=False)
def read_backend_cal(file_bytes, sheet):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)


@st.cache_data(show_spinner=False)
def read_tracking_sheet(file_bytes):
    return pd.read_excel(
        io.BytesIO(file_bytes), sheet_name="Tracking", header=1,
    )


@st.cache_data(show_spinner=False)
def read_calculation_sheet(file_bytes, sheet):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=1)
    df.columns = df.columns.str.strip()
    null_idx = df[df["Date"].isna()].index
    if len(null_idx):
        df = df.iloc[: df.index.get_loc(null_idx[0])]
    return df.iloc[:96].copy()


# ==========================================================
# CACHED HEAVY COMPUTATIONS — only rerun when data changes
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_geometry(lat, tilt_dict_key, tilt_val, n=96):
    """
    Returns scalar solar geometry values for today.
    tilt_dict_key distinguishes fixed-tilt (month lookup) vs tracking (0).
    """
    today = pd.Timestamp.today().normalize()
    first = today.replace(month=1, day=1)
    days = (today - first).days + 1
    declination = 23.45 * np.sin(np.radians(360 * (284 + days) / 365))
    elevation = 90 - lat + declination
    month = today.strftime("%B")
    if tilt_dict_key and month in tilt_dict_key:
        tilt_b = tilt_dict_key[month]
    else:
        tilt_b = tilt_val
    apb = elevation + tilt_b
    sin_apb = np.sin(np.radians(apb))
    sin_a   = np.sin(np.radians(elevation))
    return sin_apb, sin_a


@st.cache_data(show_spinner=False)
def cached_optimize_loss(
    area_std_eff,    # tuple — hashable
    area_m2,         # tuple
    actual_tuple,    # tuple
    poa_tuple,       # tuple of tuples
    cluster,
    weight_tuple,    # tuple or None
):
    """
    Vectorised peak-match loss scan.
    All inputs are plain tuples so Streamlit can hash them.
    """
    std_eff = np.array(area_std_eff, dtype=float)
    area    = np.array(area_m2,      dtype=float)
    actual  = np.array(actual_tuple, dtype=float)

    # Guard against empty or all-NaN data
    std_eff = std_eff[~np.isnan(std_eff)]
    area    = area[~np.isnan(area)]
    if len(std_eff) == 0 or len(area) == 0 or actual.max() == 0:
        return 0.0

    actual_peak = actual.max()
    losses    = np.arange(0, std_eff.min() + 0.01, 0.1)
    best_loss = 0.0
    best_err  = np.inf

    for loss in losses:
        eff_area = area * (std_eff - loss) / 100
        if cluster:
            weights = np.array(weight_tuple)
            pred = np.zeros(len(actual))
            for i in range(5):
                pred += np.array(poa_tuple[i]) * eff_area[i] * weights[i] / 1e6
        else:
            pred = np.array(poa_tuple[0]) * eff_area.sum() / 1e6
        err = abs(actual_peak - pred.max())
        if err < best_err:
            best_err  = err
            best_loss = loss

    return float(best_loss)


@st.cache_data(show_spinner=False)
def cached_tracking_forecast(
    ghi_tuple,        # tuple of tuples
    blocks_tuple,     # tuple
    weights_tuple,    # tuple
    DHI, start, end, maximum, east, west,
):
    """Final tracking curve — cached so tweaking params is instant."""
    blocks = np.array(blocks_tuple)
    m1 = 90 / (start - 1 - maximum)
    m2 = 90 / (end + 1 - maximum)

    zenith = np.where(
        blocks <= maximum,
        np.minimum(89, m1 * (blocks - maximum)),
        np.minimum(89, m2 * (blocks - maximum)),
    )
    panel = np.where(
        blocks < maximum,
        np.minimum(zenith, abs(east)),
        np.where((blocks > maximum) & (zenith > west), west, zenith),
    )
    cos_a = np.clip(np.cos(np.radians(panel)), 1e-6, None)

    forecast = np.zeros(len(blocks))
    for ghi_t, w in zip(ghi_tuple, weights_tuple):
        ghi = np.array(ghi_t)
        dhi = ghi * DHI / 100
        forecast += ((ghi - dhi) / cos_a) * w / 1e6

    return forecast


# ==========================================================
# DE OPTIMISATION  (not cached — runs once, stored in session)
# ==========================================================

def run_de(actual, blocks_tuple, ghi_tuple, weights_tuple):
    blocks   = np.array(blocks_tuple)
    mask     = actual != 0
    actual_m = actual[mask]

    bounds = [(0,10),(0,30),(65,80),(44,60),(0,70),(0,70)]

    def objective(x):
        try:
            DHI, s, e, m, east, west = (int(round(v)) for v in x)
            if s >= m or m >= e: return 1e9
            m1 = 90/(s-1-m); m2 = 90/(e+1-m)
            zenith = np.where(
                blocks<=m, np.minimum(89,m1*(blocks-m)), np.minimum(89,m2*(blocks-m)))
            panel  = np.where(
                blocks<m, np.minimum(zenith,abs(east)),
                np.where((blocks>m)&(zenith>west),west,zenith))
            cos_a  = np.clip(np.cos(np.radians(panel)),1e-6,None)
            pred   = np.zeros_like(blocks,dtype=float)
            for ghi_t, w in zip(ghi_tuple, weights_tuple):
                ghi  = np.array(ghi_t)
                dhi  = ghi * DHI / 100
                pred += ((ghi-dhi)/cos_a)*w/1e6
            pred = pred[mask]
            if not len(pred) or np.isnan(pred).any() or actual_m.max()==0: return 1e9
            peak = actual_m.max()
            return (0.80*np.mean(np.abs(actual_m-pred))/peak
                    +0.10*abs(peak-pred.max())/peak
                    +0.10*abs(actual_m.sum()-pred.sum())/actual_m.sum())
        except Exception:
            return 1e9

    progress = st.progress(0)
    status   = st.empty()
    state    = {"gen": 0, "quote": None}
    MAX      = 100

    def cb(xk, convergence):
        state["gen"] += 1
        progress.progress(min(state["gen"]/MAX, 1.0))
        if state["gen"] % 20 == 1 or state["quote"] is None:
            choices = [q for q in QUOTES if q != state["quote"]]
            state["quote"] = random.choice(choices)
        status.info(f"{state['quote']}\n\nGeneration {state['gen']} / {MAX}")

    status.info(random.choice(QUOTES))
    with st.spinner("Ho raha hai aap tab tak saath waale se baat karlo...🗣"):
        result = differential_evolution(
            objective, bounds=bounds, strategy="best1bin",
            maxiter=MAX, popsize=15, tol=0.001,
            mutation=(0.5,1), recombination=0.7,
            seed=42, polish=True, workers=1, callback=cb,
        )
    progress.empty()
    status.success("✅ Dekha Kitni Jaldi Hogaya!")
    return np.round(result.x).astype(int)


# ==========================================================
# HELPERS
# ==========================================================

def make_chart(forecast, actual):
    x = np.arange(1, len(actual)+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=forecast, mode="lines", name="Forecast",
                             line={"color":"#00c6ff","width":3}))
    fig.add_trace(go.Scatter(x=x, y=actual,   mode="lines", name="Actual",
                             line={"color":"#ef4444","width":3}))
    fig.update_layout(
        title="Forecast vs Actual Power", template="streamlit",
        height=500, hovermode="x unified",
        xaxis={"title":"15 Minute Block","dtick":4},
        yaxis={"title":"Power (MW)"},
        legend={"orientation":"h","y":1.08,"x":0},
        margin={"l":20,"r":20,"t":60,"b":20},
    )
    return fig


def show_efficiency(area_df):
    cols = ["Module Type","Standard PV Efficiency (%)","Efficiency Losses(%)","Net Efficiency (%)","Total area(m2)"]
    disp = area_df[cols].copy()
    num  = disp.select_dtypes(include=np.number).columns
    disp[num] = disp[num].round(2)
    with st.expander("🔍 View Efficiency Calculations"):
        st.dataframe(disp, use_container_width=True, hide_index=True)


def apply_efficiency(area_df, loss):
    df = area_df.copy()
    df["Efficiency Losses(%)"] = loss
    df["Net Efficiency (%)"]   = df["Standard PV Efficiency (%)"] - loss
    df["Eff Area"]             = df["Total area(m2)"] * df["Net Efficiency (%)"] / 100
    return df


# ==========================================================
# PAGE START
# ==========================================================

st.title("Pakima Pakam Ravi, 3-4 Loss Correction kar chuke hai!! 😎")

uploaded_file = st.file_uploader("Yaha Feko!!", type=["xlsx"], key="lc_uploader")

if uploaded_file is None:
    st.info("Pehle File toh upload karo!!!")
    st.stop()

file_bytes = uploaded_file.getvalue()

# Clear stale session when a new file is loaded
if st.session_state.get("lc_last_file") != uploaded_file.name:
    for k in ["lc_params","lc_run","lc_last_file"]:
        st.session_state.pop(k, None)
    st.session_state["lc_last_file"] = uploaded_file.name
    st.rerun()

is_cluster, ghi_cols, raw_df = detect_workbook(file_bytes)

# ── Input editor ──────────────────────────────────────────
st.subheader("Input Data")
original_df = raw_df.copy()
edited_df   = st.data_editor(raw_df, use_container_width=True,
                              hide_index=True, num_rows="fixed", key="lc_editor")
for col in ghi_cols:
    edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce").fillna(0)
edited_df["Actual"] = pd.to_numeric(edited_df["Actual"], errors="coerce").fillna(0)
edited_df = edited_df.iloc[:96].reset_index(drop=True)

changed = edited_df.ne(original_df.fillna(0)).any(axis=1)
if changed.any():
    st.toast(f"✨ {changed.sum()} rows updated successfully!", icon="✅")

plant_type = st.pills("Select Plant Type", ["🏗️ Fixed","🔄 Tracking"], default="🏗️ Fixed")

if "lc_run" not in st.session_state:
    st.session_state.lc_run = False

if st.button("🚀 Dabao magar pyaar se!!", use_container_width=True, type="primary"):
    st.session_state.lc_run = True
    st.session_state.pop("lc_params", None)

if not st.session_state.lc_run:
    st.stop()

# ── Common data ───────────────────────────────────────────
area_df = read_area_efficiency(file_bytes)
st.write("Area DF shape:", area_df.shape)
st.write("Efficiency col:", area_df["Standard PV Efficiency (%)"].tolist())
lat     = read_forecast_config(file_bytes)
sheet   = "Fixed-CL1" if is_cluster else "Fixed"
df_fix  = read_calculation_sheet(file_bytes, sheet)

# Align lengths before applying editor data
# Use edited_df directly — it already has all the right columns
# from detect_workbook, cleaned and numeric. No need to merge with df_fix.
edited_df = edited_df.iloc[:96].reset_index(drop=True)

for col in ghi_cols:
    edited_df[col] = pd.to_numeric(edited_df[col], errors="coerce").fillna(0)
edited_df["Actual"] = pd.to_numeric(edited_df["Actual"], errors="coerce").fillna(0)

# df_fix only needed for extra columns (Date, geometry cols etc.)
# Overwrite its GHI + Actual from the editor so user edits are reflected
n = min(len(df_fix), len(edited_df))
df_fix = df_fix.iloc[:n].reset_index(drop=True)

for col in ghi_cols:
    df_fix[col] = edited_df[col].values[:n]
df_fix["Actual"] = edited_df["Actual"].values[:n]

actual = df_fix["Actual"].to_numpy(float)

# ==========================================================
# FIXED PLANT
# ==========================================================

if plant_type == "🏗️ Fixed":

    tilt_dict = read_tilt(file_bytes)

    # Geometry — cached (only depends on lat + today's date)
    sin_apb, sin_a = cached_geometry(lat, tilt_dict, 0)

    # POA per cluster / single
    suffixes = ["","-CL2","-CL3","-CL4","-CL5"] if is_cluster else [""]
    poa_list = []
    for col, suf in zip(ghi_cols, suffixes):
        ghi = df_fix[col].to_numpy(float)
        poa = (ghi * sin_apb) / sin_a
        poa_list.append(poa)

    if is_cluster:
        weight_df    = read_weights(file_bytes)
        weights_raw  = tuple(float(weight_df[f"CL-{i}"].iloc[0]) for i in range(1,6))
        poa_tuple    = tuple(tuple(p) for p in poa_list)
        best_loss    = cached_optimize_loss(
            tuple(area_df["Standard PV Efficiency (%)"].tolist()),
            tuple(area_df["Total area(m2)"].tolist()),
            tuple(actual.tolist()),
            poa_tuple, True, weights_raw,
        )
        area_df_eff  = apply_efficiency(area_df, best_loss)
        forecast     = np.zeros(len(df_fix))
        for i in range(5):
            forecast += poa_list[i] * area_df_eff["Eff Area"].iloc[i] * weights_raw[i] / 1e6
    else:
        poa_tuple    = (tuple(poa_list[0]),)
        best_loss    = cached_optimize_loss(
            tuple(area_df["Standard PV Efficiency (%)"].tolist()),
            tuple(area_df["Total area(m2)"].tolist()),
            tuple(actual.tolist()),
            poa_tuple, False, None,
        )
        area_df_eff  = apply_efficiency(area_df, best_loss)
        forecast     = poa_list[0] * area_df_eff["Eff Area"].sum() / 1e6

    st.metric("Efficiency Loss", f"{best_loss:.2f}%")
    show_efficiency(area_df_eff)
    st.plotly_chart(make_chart(forecast, actual), use_container_width=True)


# ==========================================================
# TRACKING PLANT
# ==========================================================

else:

    # Geometry — tilt=0 for tracking
    sin_apb, sin_a = cached_geometry(lat, {}, 0)

    ghi_arrays = [df_fix[col].to_numpy(float) for col in ghi_cols]

    if is_cluster:
        weight_df   = read_weights(file_bytes)
        weights_raw = tuple(float(weight_df[f"CL-{i}"].iloc[0]) for i in range(1,6))
        poa_list    = [(ghi * sin_apb) / sin_a for ghi in ghi_arrays]
        poa_tuple   = tuple(tuple(p) for p in poa_list)

        best_loss = cached_optimize_loss(
            tuple(area_df["Standard PV Efficiency (%)"].tolist()),
            tuple(area_df["Total area(m2)"].tolist()),
            tuple(actual.tolist()),
            poa_tuple, True, weights_raw,
        )
        area_df_eff = apply_efficiency(area_df, best_loss)
        weights_eff = tuple(
            float(area_df_eff["Eff Area"].iloc[i]) * weights_raw[i]
            for i in range(5)
        )
        block_df    = read_backend_cal(file_bytes, "Backend Cal CL1")

    else:
        poa         = (df_fix["GHI_Forecast"].to_numpy(float) * sin_apb) / sin_a
        poa_tuple   = (tuple(poa),)
        best_loss   = cached_optimize_loss(
            tuple(area_df["Standard PV Efficiency (%)"].tolist()),
            tuple(area_df["Total area(m2)"].tolist()),
            tuple(actual.tolist()),
            poa_tuple, False, None,
        )
        area_df_eff = apply_efficiency(area_df, best_loss)
        weights_eff = (float(area_df_eff["Eff Area"].sum()),)
        block_df    = read_backend_cal(file_bytes, "Backend Cal")

    tracking_sheet = read_tracking_sheet(file_bytes)
    blocks         = block_df["Block No."].to_numpy(float)
    blocks_tuple   = tuple(blocks.tolist())
    ghi_tuple      = tuple(tuple(g.tolist()) for g in ghi_arrays)

    # ── DE optimisation — runs once, stored in session ────
    if "lc_params" not in st.session_state:
        best = run_de(actual, blocks_tuple, ghi_tuple, weights_eff)
        st.session_state.lc_params = {
            "DHI":   int(best[0]), "start": int(best[1]),
            "end":   int(best[2]), "max":   int(best[3]),
            "east":  int(best[4]), "west":  int(best[5]),
            "loss":  float(best_loss),
        }

    params = st.session_state.lc_params

    # ── Parameter inputs inside a form — zero reruns while typing ──
    st.subheader("Optimized Parameters")
    st.caption("Adjust values then click Recalculate — chart updates instantly without rerunning the optimizer.")

    with st.form("lc_params_form"):
        best_loss_input = st.number_input(
            "Efficiency Loss (%)", step=0.1, value=float(params["loss"]))
        c1,c2,c3 = st.columns(3)
        DHI     = c1.number_input("DHI (%)",        step=1, value=params["DHI"])
        start   = c2.number_input("Starting Block",  step=1, value=params["start"])
        end     = c3.number_input("Ending Block",    step=1, value=params["end"])
        c1,c2,c3 = st.columns(3)
        maximum = c1.number_input("Max Block",   step=1, value=params["max"])
        east    = c2.number_input("East Limit",  step=1, value=params["east"])
        west    = c3.number_input("West Limit",  step=1, value=params["west"])
        recalc  = st.form_submit_button(
            "🔄 Recalculate", use_container_width=True, type="primary")

    # Update stored params when user recalculates
    if recalc:
        st.session_state.lc_params.update({
            "loss": best_loss_input, "DHI": DHI,
            "start": start, "end": end, "max": maximum,
            "east": east, "west": west,
        })

    p = st.session_state.lc_params

    # ── Updated efficiency ────────────────────────────────
    area_df_eff = apply_efficiency(area_df, p["loss"])
    show_efficiency(area_df_eff)

    # ── Final forecast — fully cached, instant on param changes ──
    forecast = cached_tracking_forecast(
        ghi_tuple, blocks_tuple, weights_eff,
        p["DHI"], p["start"], p["end"], p["max"], p["east"], p["west"],
    )

    tracking_sheet["Fixed Power=I*Ƞ*A"] = forecast
    st.plotly_chart(
        make_chart(tracking_sheet["Fixed Power=I*Ƞ*A"].values, actual),
        use_container_width=True,
    )
