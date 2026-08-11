import io
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Loss Correction — Solar Suite", layout="wide")

# ── Idle-reload script (10 min) ──────────────────────────────────────────────
st.components.v1.html("""<script>
let t;
function r(){clearTimeout(t);t=setTimeout(()=>location.reload(),600000);}
["mousemove","mousedown","keydown","scroll","touchstart"].forEach(e=>document.addEventListener(e,r));
r();
</script>""", height=0)

st.sidebar.markdown("""
<h1 style='text-align:center;
background:linear-gradient(90deg,#00c6ff,#0072ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
font-size:40px;font-weight:800;'>
⚡ Solar Suite
</h1>
<p style='text-align:center;color:gray;font-size:14px;'>Forecast Correction Platform</p>
""", unsafe_allow_html=True)
st.sidebar.divider()


# ── Page style ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{font-size:13px;}
div[data-testid="metric-container"]{
    background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px 20px;
}
</style>""", unsafe_allow_html=True)

st.title("Pakima Pakam Ravi, 3-4 Loss Correction kar chuke hai!! 😎")

# ── Quotes ───────────────────────────────────────────────────────────────────
QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?..",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice🧊 Dudh bangya Very Nice - Dudh Dudh Dudh Dudh...",
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

# ── Cached workbook readers ──────────────────────────────────────────────────
# Each function is cached by file bytes — Streamlit won't re-parse the Excel
# unless the actual file content changes. This is the fix for the freezing.

@st.cache_data(show_spinner=False)
def detect_workbook(file_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    is_cluster = "Fixed-CL1" in xls.sheet_names
    sheet = "Fixed-CL1" if is_cluster else "Fixed"
    ghi_cols = ["CL1-GHI","CL2-GHI","CL3-GHI","CL4-GHI","CL5-GHI"] if is_cluster else ["GHI_Forecast"]
    df = xls.parse(sheet_name=sheet, header=[1])
    df.columns = df.columns.str.strip()
    df["Actual"] = df["Actual"].fillna(0)
    null_idx = df[df["Date"].isna()].index
    if len(null_idx):
        df = df.iloc[: df.index.get_loc(null_idx[0])]
    df = df.iloc[:96].copy()
    return is_cluster, ghi_cols, df[ghi_cols + ["Actual"]].copy()


@st.cache_data(show_spinner=False)
def read_area_efficiency(file_bytes: bytes, usecols=None):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Area & Efficiency",
                       header=[1], usecols=usecols if usecols else range(8))
    df.columns = df.columns.str.strip()
    null_idx = df[df["Module Type"].isna()].index
    return df.iloc[: df.index.get_loc(null_idx[0])].copy()


@st.cache_data(show_spinner=False)
def read_forecast_config(file_bytes: bytes):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Forecast Config", header=[8])
    return float(df.loc[0, "Lat"])


@st.cache_data(show_spinner=False)
def read_tilt(file_bytes: bytes):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Config Tilt Angle", header=[7])
    df.columns = df.columns.str.strip()
    df["Fixed"] = df["Fixed"].fillna(0)
    null_idx = df[df["Fixed"] == 0].index
    df = df.iloc[: df.index.get_loc(null_idx[0])].dropna(how="all", axis=1)
    df = df.rename(columns={"Unnamed: 2": "Month_Num", "Unnamed: 3": "Month"})
    return df.set_index("Month")["Fixed"].to_dict()


@st.cache_data(show_spinner=False)
def read_weights(file_bytes: bytes):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name="Area & Efficiency",
                         header=2, usecols=[12,13,14,15,16])


@st.cache_data(show_spinner=False)
def read_backend_cal(file_bytes: bytes, sheet: str):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)


@st.cache_data(show_spinner=False)
def read_tracking_sheet(file_bytes: bytes):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name="Tracking", header=[1])


# ── Geometry helpers ─────────────────────────────────────────────────────────

def add_solar_geometry(df, lat, tilt_b):
    today = pd.Timestamp.today()
    first = today.replace(month=1, day=1).normalize()
    df["Date"] = today
    df["Declination Angle ∆"] = 23.45 * np.sin(
        np.radians(360 * (284 + (df["Date"] - first).dt.days + 1) / 365))
    df["Elevation angle a"] = 90 - lat + df["Declination Angle ∆"]
    df["Tilt Angle b"] = tilt_b if isinstance(tilt_b, (int, float)) else \
        df["Date"].dt.strftime("%B").map(tilt_b)
    df["a+b"] = df["Elevation angle a"] + df["Tilt Angle b"]
    df["SIN(a+b)"] = np.sin(np.radians(df["a+b"]))
    df["Sin(a)"] = np.sin(np.radians(df["Elevation angle a"]))
    return df


def add_poa_cluster(df):
    for cl, suffix in zip(["CL1","CL2","CL3","CL4","CL5"],
                           ["", "-CL2", "-CL3", "-CL4", "-CL5"]):
        ghi_col = f"{cl}-GHI"
        df[f"GHI*sin(a){suffix}"] = df[ghi_col] * df["Sin(a)"]
        df[f"GHI*sin(a+b){suffix}"] = df[ghi_col] * df["SIN(a+b)"]
        df[f"POA fixed{suffix}"] = df[f"GHI*sin(a+b){suffix}"] / df["Sin(a)"]
    return df


# ── Chart ────────────────────────────────────────────────────────────────────

def make_chart(forecast, actual):
    x = np.arange(1, 97)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=forecast, mode="lines", name="Forecast",
                             line=dict(color="#00c6ff", width=3)))
    fig.add_trace(go.Scatter(x=x, y=actual, mode="lines", name="Actual",
                             line=dict(color="#ef4444", width=3)))
    fig.update_layout(title="Forecast vs Actual Power", template="plotly_dark",
                      height=500, hovermode="x unified",
                      xaxis=dict(title="15 Minute Block", dtick=4),
                      yaxis=dict(title="Power (MW)"),
                      legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(l=20, r=20, t=60, b=20),
                      paper_bgcolor="#111827", plot_bgcolor="#111827")
    return fig


# ── Upload ───────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Yaha Feko!!", type=["xlsx"], key="lc_uploader")

if uploaded_file is None:
    st.info("Pehle File toh upload karo!!!")
    st.stop()

file_bytes = uploaded_file.read()

# Clear state when a new file is uploaded
if st.session_state.get("lc_last_file") != uploaded_file.name:
    for key in ["lc_params","lc_run","lc_loss","lc_dhi","lc_start",
                "lc_end","lc_max","lc_east","lc_west","lc_edited"]:
        st.session_state.pop(key, None)
    st.session_state["lc_last_file"] = uploaded_file.name
    st.rerun()

is_cluster, ghi_cols, raw_df = detect_workbook(file_bytes)

# ── Input data editor ────────────────────────────────────────────────────────
st.subheader("Input Data")
original_df = raw_df.copy()
edited_df = st.data_editor(raw_df, use_container_width=True, hide_index=True,
                            num_rows="fixed", key="lc_editor")
edited_df[ghi_cols] = edited_df[ghi_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
edited_df["Actual"] = pd.to_numeric(edited_df["Actual"], errors="coerce").fillna(0)
edited_df = edited_df.iloc[:96].reset_index(drop=True)

changed = (edited_df.ne(original_df.fillna(0))).any(axis=1)
if changed.any():
    st.toast(f"✨ {changed.sum()} rows updated successfully!", icon="✅")

# ── Plant type ───────────────────────────────────────────────────────────────
plant_type = st.pills("Select Plant Type", ["🏗️ Fixed", "🔄 Tracking"], default="🏗️ Fixed")

if "lc_run" not in st.session_state:
    st.session_state.lc_run = False

if st.button("🚀 Dabao magar pyaar se!!", use_container_width=True, type="primary"):
    st.session_state.pop("lc_params", None)
    st.session_state.lc_run = True

if not st.session_state.lc_run:
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTER — FIXED
# ═══════════════════════════════════════════════════════════════════════════════
if is_cluster and plant_type == "🏗️ Fixed":
    df_ae = read_area_efficiency(file_bytes)
    df_w  = read_weights(file_bytes)
    lat   = read_forecast_config(file_bytes)
    month_lookup = read_tilt(file_bytes)

    df_fix = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Fixed-CL1", header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    null_idx = df_fix[df_fix["Date"].isna()].index
    df_fix = df_fix.iloc[: df_fix.index.get_loc(null_idx[0])]
    df_fix[ghi_cols] = edited_df[ghi_cols].values
    df_fix["Actual"] = edited_df["Actual"].values
    df_fix = df_fix.iloc[:96].copy()

    df_fix = add_solar_geometry(df_fix, lat, month_lookup)
    df_fix = add_poa_cluster(df_fix)

    max_loss = df_ae["Standard PV Efficiency (%)"].min()
    results = []
    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df_ae["Efficiency Losses(%)"] = loss
        df_ae["Net Efficiency (%)"] = df_ae["Standard PV Efficiency (%)"] - loss
        df_w2 = pd.DataFrame({f"CL-{i}": ((df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100)
                               * df_w[f"CL-{i}"].values[0:1] for i in range(1,6)})
        total = sum((df_fix[f"POA fixed{s}"] * np.sum(df_w2[f"CL-{i}"])) / 1e6
                    for i,s in zip(range(1,6),["","-CL2","-CL3","-CL4","-CL5"]))
        results.append({"Efficiency Loss (%)": loss,
                         "Peak Error": abs(df_fix["Actual"].max() - total.max())})
    best_loss = pd.DataFrame(results).loc[lambda d: d["Peak Error"].idxmin(), "Efficiency Loss (%)"]

    df_ae["Efficiency Losses(%)"] = best_loss
    df_ae["Net Efficiency (%)"] = df_ae["Standard PV Efficiency (%)"] - best_loss
    df_w2 = pd.DataFrame({f"CL-{i}": ((df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100)
                           * df_w[f"CL-{i}"].values[0:1] for i in range(1,6)})
    total = sum((df_fix[f"POA fixed{s}"] * np.sum(df_w2[f"CL-{i}"])) / 1e6
                for i,s in zip(range(1,6),["","-CL2","-CL3","-CL4","-CL5"]))

    st.metric("Efficiency Loss", f"{best_loss:.2f}%")
    disp = df_ae[["Module Type","Standard PV Efficiency (%)","Efficiency Losses(%)","Net Efficiency (%)","Total area(m2)"]].copy()
    disp[disp.select_dtypes("number").columns] = disp.select_dtypes("number").round(2)
    with st.expander("🔍 View Efficiency Calculations"):
        st.dataframe(disp, use_container_width=True, hide_index=True)
    st.plotly_chart(make_chart(total.values, df_fix["Actual"].values), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTER — TRACKING (DE optimization)
# ═══════════════════════════════════════════════════════════════════════════════
elif is_cluster and plant_type == "🔄 Tracking":
    df_ae = read_area_efficiency(file_bytes)
    df_w  = read_weights(file_bytes)
    lat   = read_forecast_config(file_bytes)

    df_fix = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Fixed-CL1", header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    null_idx = df_fix[df_fix["Date"].isna()].index
    df_fix = df_fix.iloc[: df_fix.index.get_loc(null_idx[0])]
    df_fix[ghi_cols] = edited_df[ghi_cols].values
    df_fix["Actual"] = edited_df["Actual"].values
    df_fix = df_fix.iloc[:96].copy()
    df_fix = add_solar_geometry(df_fix, lat, 0)   # tracking: tilt = 0
    df_fix = add_poa_cluster(df_fix)

    max_loss = df_ae["Standard PV Efficiency (%)"].min()
    results = []
    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df_ae["Efficiency Losses(%)"] = loss
        df_ae["Net Efficiency (%)"] = df_ae["Standard PV Efficiency (%)"] - loss
        df_w2 = pd.DataFrame({f"CL-{i}": ((df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100)
                               * df_w[f"CL-{i}"].values[0:1] for i in range(1,6)})
        total = sum((df_fix[f"POA fixed{s}"] * np.sum(df_w2[f"CL-{i}"])) / 1e6
                    for i,s in zip(range(1,6),["","-CL2","-CL3","-CL4","-CL5"]))
        results.append({"Efficiency Loss (%)": loss,
                         "Peak Error": abs(df_fix["Actual"].max() - total.max())})
    best_loss = pd.DataFrame(results).loc[lambda d: d["Peak Error"].idxmin(), "Efficiency Loss (%)"]
    df_ae["Efficiency Losses(%)"] = best_loss
    df_ae["Net Efficiency (%)"] = df_ae["Standard PV Efficiency (%)"] - best_loss
    df_w2 = pd.DataFrame({f"CL-{i}": ((df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100)
                           * df_w[f"CL-{i}"].values[0:1] for i in range(1,6)})

    backend_list = [read_backend_cal(file_bytes, f"Backend Cal CL{i}") for i in range(1,6)]
    df_trac = read_tracking_sheet(file_bytes)

    actual     = df_fix["Actual"].to_numpy(dtype=np.float64)
    mask       = actual != 0
    actual_m   = actual[mask]
    blocks     = backend_list[0]["Block No."].to_numpy(dtype=np.float64)
    ghi_arrays = [df_fix[col].to_numpy(dtype=np.float64) for col in ghi_cols]
    weight_sum = np.array([df_w2[f"CL-{i}"].sum() for i in range(1,6)], dtype=np.float64)

    def objective_cl_track(x):
        try:
            DHI,s,e,m,east,west = (int(round(v)) for v in x)
            if s >= m or m >= e: return 1e9
            m1 = 90 / (s - 1 - m); m2 = 90 / (e + 1 - m)
            zenith = np.where(blocks<=m, np.minimum(89, m1*(blocks-m)), np.minimum(89, m2*(blocks-m)))
            panel  = np.where(blocks<m, np.minimum(zenith, abs(east)),
                              np.where((blocks>m)&(zenith>west), west, zenith))
            cos_a  = np.clip(np.cos(np.radians(panel)), 1e-6, None)
            pred   = np.zeros_like(blocks)
            for i,ghi in enumerate(ghi_arrays):
                dhi = ghi * DHI / 100
                pred = pred + (((ghi - dhi) / cos_a) * weight_sum[i]) / 1e6
            pred = pred[mask]
            if np.isnan(pred).any() or np.isinf(pred).any() or actual_m.max()==0: return 1e9
            return (0.80*np.mean(np.abs(actual_m-pred))/actual_m.max()
                    + 0.10*abs(actual_m.max()-pred.max())/actual_m.max()
                    + 0.10*abs(actual_m.sum()-pred.sum())/actual_m.sum())
        except Exception:
            return 1e9

    bounds = [(0,10),(0,30),(65,80),(44,60),(0,70),(0,70)]

    if "lc_params" not in st.session_state:
        progress = st.progress(0)
        status   = st.empty()
        last_q   = {"t": None}
        gen      = {"n": 0}
        MAX_ITER = 100

        def cb(xk, convergence):
            gen["n"] += 1
            progress.progress(gen["n"] / MAX_ITER)
            if gen["n"] % 20 == 1:
                q = random.choice([q for q in QUOTES if q != last_q["t"]])
                last_q["t"] = q
            status.info(f"{last_q['t']}\n\nGeneration {gen['n']} / {MAX_ITER}")
            return False

        status.info(random.choice(QUOTES))
        with st.spinner("Ho raha hai aap tab tak saath waale se baat karlo...🗣"):
            res = differential_evolution(objective_cl_track, bounds=bounds,
                                         strategy="best1bin", maxiter=MAX_ITER, popsize=15,
                                         tol=0.001, mutation=(0.5,1), recombination=0.7,
                                         seed=42, polish=True, workers=1, callback=cb)
        progress.empty(); status.success("✅ Dekha Kitni Jaldi Hogaya!")
        best = np.round(res.x).astype(int)
        st.session_state.lc_params = dict(loss=float(best_loss),
            dhi=int(best[0]),start=int(best[1]),end=int(best[2]),
            max=int(best[3]),east=int(best[4]),west=int(best[5]))
        for k,v in st.session_state.lc_params.items():
            st.session_state[f"lc_{k}"] = v

    p = st.session_state.lc_params
    for k,v in p.items():
        if f"lc_{k}" not in st.session_state:
            st.session_state[f"lc_{k}"] = v

    st.subheader("Optimized Parameters")
    best_loss = st.number_input("Efficiency Loss (%)", step=0.1, key="lc_loss")
    c1,c2,c3 = st.columns(3)
    DHI   = c1.number_input("DHI (%)",       step=1, key="lc_dhi")
    s_blk = c2.number_input("Starting Block", step=1, key="lc_start")
    e_blk = c3.number_input("Ending Block",   step=1, key="lc_end")
    c1,c2,c3 = st.columns(3)
    m_blk = c1.number_input("Max Block",  step=1, key="lc_max")
    east  = c2.number_input("East Limit", step=1, key="lc_east")
    west  = c3.number_input("West Limit", step=1, key="lc_west")

    df_ae["Efficiency Losses(%)"] = best_loss
    df_ae["Net Efficiency (%)"] = df_ae["Standard PV Efficiency (%)"] - best_loss
    df_w2 = pd.DataFrame({f"CL-{i}": ((df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100)
                           * df_w[f"CL-{i}"].values[0:1] for i in range(1,6)})
    weights = df_w2.sum()

    disp = df_ae[["Module Type","Standard PV Efficiency (%)","Efficiency Losses(%)","Net Efficiency (%)","Total area(m2)"]].copy()
    disp[disp.select_dtypes("number").columns] = disp.select_dtypes("number").round(2)
    with st.expander("🔍 View Efficiency Calculations"):
        st.dataframe(disp, use_container_width=True, hide_index=True)

    m1 = 90 / (s_blk - 1 - m_blk); m2 = 90 / (e_blk + 1 - m_blk)
    blocks = backend_list[0]["Block No."]
    zenith = np.where(blocks<=m_blk, np.minimum(89,m1*(blocks-m_blk)), np.minimum(89,m2*(blocks-m_blk)))
    panel  = np.where(blocks<m_blk, np.minimum(zenith,abs(east)),
                      np.where((blocks>m_blk)&(zenith>west), west, zenith))
    cos_a  = np.clip(np.cos(np.radians(panel)), 1e-6, None)
    forecast = np.zeros(len(df_fix))
    for i,col in enumerate(ghi_cols, start=1):
        ghi = df_fix[col].to_numpy()
        dhi = ghi * DHI / 100
        forecast = forecast + (((ghi - dhi) / cos_a) * weights[f"CL-{i}"]) / 1e6
    df_trac["Fixed Power=I*Ƞ*A"] = forecast
    st.plotly_chart(make_chart(df_trac["Fixed Power=I*Ƞ*A"].values, df_fix["Actual"].values), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# NON-CLUSTER — FIXED
# ═══════════════════════════════════════════════════════════════════════════════
elif not is_cluster and plant_type == "🏗️ Fixed":
    df_ae = read_area_efficiency(file_bytes)
    lat   = read_forecast_config(file_bytes)
    month_lookup = read_tilt(file_bytes)

    df_fix = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Fixed", header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    null_idx = df_fix[df_fix["Date"].isna()].index
    df_fix = df_fix.iloc[: df_fix.index.get_loc(null_idx[0])].copy()
    df_fix["GHI_Forecast"] = edited_df["GHI_Forecast"].values[:len(df_fix)]
    df_fix["Actual"]       = edited_df["Actual"].values[:len(df_fix)]
    df_fix["Actual"]       = df_fix["Actual"].fillna(0)

    df_fix = add_solar_geometry(df_fix, lat, month_lookup)
    df_fix["GHI*sin(a)"]   = df_fix["GHI_Forecast"] * df_fix["Sin(a)"]
    df_fix["GHI*sin(a+b)"] = df_fix["GHI_Forecast"] * df_fix["SIN(a+b)"]
    df_fix["POA fixed"]    = df_fix["GHI*sin(a+b)"] / df_fix["Sin(a)"]

    max_loss = df_ae["Standard PV Efficiency (%)"].min()
    results = []
    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df_ae["Efficiency Losses(%)"] = loss
        df_ae["Net Efficiency (%)"]   = df_ae["Standard PV Efficiency (%)"] - loss
        df_ae["Eff Area"] = (df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100
        pred = (df_fix["POA fixed"] * df_ae["Eff Area"].sum()) / 1e6
        results.append({"Efficiency Loss (%)": loss,
                         "Peak Error": abs(df_fix["Actual"].max() - pred.max())})
    best_loss = pd.DataFrame(results).loc[lambda d: d["Peak Error"].idxmin(), "Efficiency Loss (%)"]
    df_ae["Efficiency Losses(%)"] = best_loss
    df_ae["Net Efficiency (%)"]   = df_ae["Standard PV Efficiency (%)"] - best_loss
    df_ae["Eff Area"] = (df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100
    df_fix["Fixed Power=I*Ƞ*A"] = (df_fix["POA fixed"] * df_ae["Eff Area"].sum()) / 1e6

    st.metric("Efficiency Loss", f"{best_loss:.2f}%")
    disp = df_ae[["Module Type","Standard PV Efficiency (%)","Efficiency Losses(%)","Net Efficiency (%)","Total area(m2)"]].copy()
    disp[disp.select_dtypes("number").columns] = disp.select_dtypes("number").round(2)
    with st.expander("🔍 View Efficiency Calculations"):
        st.dataframe(disp, use_container_width=True, hide_index=True)
    st.plotly_chart(make_chart(df_fix["Fixed Power=I*Ƞ*A"].values, df_fix["Actual"].values), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# NON-CLUSTER — TRACKING (DE optimization)
# ═══════════════════════════════════════════════════════════════════════════════
elif not is_cluster and plant_type == "🔄 Tracking":
    df_ae = read_area_efficiency(file_bytes)
    lat   = read_forecast_config(file_bytes)

    df_fix = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Fixed", header=[1])
    df_fix.columns = df_fix.columns.str.strip()
    null_idx = df_fix[df_fix["Date"].isna()].index
    df_fix = df_fix.iloc[: df_fix.index.get_loc(null_idx[0])].copy()
    df_fix["GHI_Forecast"] = edited_df["GHI_Forecast"].values[:len(df_fix)]
    df_fix["Actual"]       = edited_df["Actual"].values[:len(df_fix)]
    df_fix["Actual"]       = df_fix["Actual"].fillna(0)

    df_fix = add_solar_geometry(df_fix, lat, 0)   # tracking: tilt=0
    df_fix["GHI*sin(a)"]   = df_fix["GHI_Forecast"] * df_fix["Sin(a)"]
    df_fix["GHI*sin(a+b)"] = df_fix["GHI_Forecast"] * df_fix["SIN(a+b)"]
    df_fix["POA fixed"]    = df_fix["GHI*sin(a+b)"] / df_fix["Sin(a)"]

    max_loss = df_ae["Standard PV Efficiency (%)"].min()
    results = []
    for loss in np.arange(0, max_loss + 0.01, 0.1):
        df_ae["Efficiency Losses(%)"] = loss
        df_ae["Net Efficiency (%)"]   = df_ae["Standard PV Efficiency (%)"] - loss
        df_ae["Eff Area"] = (df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100
        pred = (df_fix["POA fixed"] * df_ae["Eff Area"].sum()) / 1e6
        results.append({"Efficiency Loss (%)": loss,
                         "Peak Error": abs(df_fix["Actual"].max() - pred.max())})
    best_loss = pd.DataFrame(results).loc[lambda d: d["Peak Error"].idxmin(), "Efficiency Loss (%)"]
    df_ae["Efficiency Losses(%)"] = best_loss
    df_ae["Net Efficiency (%)"]   = df_ae["Standard PV Efficiency (%)"] - best_loss
    df_ae["Eff Area"] = (df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100

    df_bcal = read_backend_cal(file_bytes, "Backend Cal")
    df_trac = read_tracking_sheet(file_bytes)

    actual  = df_fix["Actual"].to_numpy(dtype=np.float64)
    mask    = actual != 0
    actual_m = actual[mask]
    blocks  = df_bcal["Block No."].to_numpy(dtype=np.float64)
    ghi     = df_fix["GHI_Forecast"].to_numpy(dtype=np.float64)
    eff_area = df_ae["Eff Area"].sum()

    def objective_nc_track(x):
        DHI,s,e,m,east,west = (int(round(v)) for v in x)
        if s >= m or m >= e: return 1e9
        m1 = 90/(s-1-m); m2 = 90/(e+1-m)
        dhi = ghi * DHI / 100
        zenith = np.where(blocks<=m, np.minimum(89,m1*(blocks-m)), np.minimum(89,m2*(blocks-m)))
        panel  = np.where(blocks<m, np.minimum(zenith,abs(east)),
                          np.where((blocks>m)&(zenith>west), west, zenith))
        cos_a  = np.cos(np.radians(panel))
        pred   = ((ghi - dhi) / cos_a) * eff_area / 1e6
        pred   = pred[mask]
        act    = df_fix["Actual"].values[mask]
        return (0.80*np.mean(np.abs(act-pred))/act.max()
                + 0.10*abs(act.max()-pred.max())/act.max()
                + 0.10*abs(act.sum()-pred.sum())/act.sum())

    bounds = [(0,10),(0,30),(65,80),(44,60),(0,70),(0,70)]

    if "lc_params" not in st.session_state:
        progress = st.progress(0)
        status   = st.empty()
        last_q   = {"t": None}
        gen      = {"n": 0}
        MAX_ITER = 100

        def cb(xk, convergence):
            gen["n"] += 1
            progress.progress(gen["n"] / MAX_ITER)
            if gen["n"] % 20 == 1:
                q = random.choice([q for q in QUOTES if q != last_q["t"]])
                last_q["t"] = q
            status.info(f"{last_q['t']}\n\nGeneration {gen['n']} / {MAX_ITER}")
            return False

        status.info(random.choice(QUOTES))
        with st.spinner("Ho raha hai aap tab tak saath waale se baat karlo...🗣"):
            res = differential_evolution(objective_nc_track, bounds=bounds,
                                         strategy="best1bin", maxiter=MAX_ITER, popsize=15,
                                         tol=0.001, mutation=(0.5,1), recombination=0.7,
                                         seed=42, polish=True, workers=1, callback=cb)
        progress.empty(); status.success("✅ Dekha Kitni Jaldi Hogaya!")
        best = np.round(res.x).astype(int)
        st.session_state.lc_params = dict(loss=float(best_loss),
            dhi=int(best[0]),start=int(best[1]),end=int(best[2]),
            max=int(best[3]),east=int(best[4]),west=int(best[5]))
        for k,v in st.session_state.lc_params.items():
            st.session_state[f"lc_{k}"] = v

    p = st.session_state.lc_params
    for k,v in p.items():
        if f"lc_{k}" not in st.session_state:
            st.session_state[f"lc_{k}"] = v

    st.subheader("Optimized Parameters")
    best_loss = st.number_input("Efficiency Loss (%)", step=0.1, key="lc_loss")
    c1,c2,c3 = st.columns(3)
    DHI   = c1.number_input("DHI (%)",        step=1, key="lc_dhi")
    s_blk = c2.number_input("Starting Block",  step=1, key="lc_start")
    e_blk = c3.number_input("Ending Block",    step=1, key="lc_end")
    c1,c2,c3 = st.columns(3)
    m_blk = c1.number_input("Max Block",   step=1, key="lc_max")
    east  = c2.number_input("East Limit",  step=1, key="lc_east")
    west  = c3.number_input("West Limit",  step=1, key="lc_west")

    df_ae["Efficiency Losses(%)"] = best_loss
    df_ae["Net Efficiency (%)"]   = df_ae["Standard PV Efficiency (%)"] - best_loss
    df_ae["Eff Area"] = (df_ae["Total area(m2)"] * df_ae["Net Efficiency (%)"]) / 100

    disp = df_ae[["Module Type","Standard PV Efficiency (%)","Efficiency Losses(%)","Net Efficiency (%)","Total area(m2)"]].copy()
    disp[disp.select_dtypes("number").columns] = disp.select_dtypes("number").round(2)
    with st.expander("🔍 View Efficiency Calculations"):
        st.dataframe(disp, use_container_width=True, hide_index=True)

    m1 = 90/(s_blk-1-m_blk); m2 = 90/(e_blk+1-m_blk)
    dhi_v = ghi * DHI / 100.0
    zenith = np.where(blocks<=m_blk, np.minimum(89.,m1*(blocks-m_blk)), np.minimum(89.,m2*(blocks-m_blk)))
    panel  = np.where(blocks<m_blk, np.minimum(zenith,abs(east)),
                      np.where((blocks>m_blk)&(zenith>west), west, zenith))
    cos_a  = np.clip(np.cos(np.radians(panel)), 1e-6, None)
    forecast = ((ghi - dhi_v) / cos_a) * df_ae["Eff Area"].sum() / 1e6
    df_trac["Fixed Power=I*Ƞ*A"] = forecast
    st.plotly_chart(make_chart(df_trac["Fixed Power=I*Ƞ*A"].values, df_fix["Actual"].values), use_container_width=True)
