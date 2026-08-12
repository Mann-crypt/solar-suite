import io
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import differential_evolution

st.set_page_config(
    page_title="Loss Correction - Solar Suite",
    page_icon="⚡",
    layout="wide",
)

QUOTES = [
    "☕ Vo kehte the kya ho tum, aaj hum kehte hai tum kya ho be?",
    "🌦 Aapka mann nahi kar raha bahar jaane ka?",
    "😊 Jinke ghar sheeshe ke bane hote hai vo basement mai kapde change krte h...",
    "😋 Aromatic Rose Latte with Frothy Milk pine ka mann hor hai na...",
    "🥛 Garmi mai daalo dudh mai Ice, dudh bangya Very Nice...",
    "🌟 Aapke face pr toh Modiji se bhi jyada glow hai...",
    "😁 Horaha hai benstokes, kaan mai ghusjao insaan ke...",
    "😗 Muskuraiye, aap MAL mai hai...",
    "🥱 Hum na hote toh Operations ka kya hota?",
    "😎 6:30 hote hi Billu MAL se faraar...",
    "😇 Guruji ne ek baat kahi thi...",
    "🎼 Karna hai kuchh kaam M se gaao...",
    "😠 Nahi karni Loss Correction, now what to do?",
    "💸 Iss Job ko chhod or chhod kar ameer ho...",
]

DE_BOUNDS = [
    (0, 10),
    (0, 30),
    (65, 80),
    (44, 60),
    (0, 70),
    (0, 70),
]


def file_hash(data):
    return hashlib.md5(data).hexdigest()


def numeric_array(series):
    return pd.to_numeric(
        series, errors="coerce"
    ).fillna(0).to_numpy(dtype=np.float64)


def apply_efficiency(area_df, loss):
    df = area_df.copy()
    df["Efficiency Losses(%)"] = float(loss)
    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"] - float(loss)
    )
    df["Eff Area"] = (
        df["Total area(m2)"]
        * df["Net Efficiency (%)"]
        / 100
    )
    return df


def show_efficiency(df):
    cols = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
    ]
    disp = df[cols].copy()
    nums = disp.select_dtypes(include=np.number).columns
    disp[nums] = disp[nums].round(2)

    with st.expander("🔍 View Efficiency Calculations"):
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
        )


@st.cache_data(show_spinner=False)
def load_workbook(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = tuple(xls.sheet_names)

    cluster = "Fixed-CL1" in sheets
    fixed_sheet = "Fixed-CL1" if cluster else "Fixed"

    ghi_cols = (
        ["CL1-GHI", "CL2-GHI", "CL3-GHI", "CL4-GHI", "CL5-GHI"]
        if cluster
        else ["GHI_Forecast"]
    )

    fixed = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=fixed_sheet,
        header=1,
    )
    fixed.columns = fixed.columns.astype(str).str.strip()

    required = ghi_cols + ["Actual"]
    missing = [c for c in required if c not in fixed.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {fixed_sheet}: {missing}"
        )

    if "Date" in fixed.columns:
        nulls = fixed[fixed["Date"].isna()].index
        if len(nulls):
            fixed = fixed.iloc[:fixed.index.get_loc(nulls[0])]

    fixed = fixed.iloc[:96].copy().reset_index(drop=True)

    for col in required:
        fixed[col] = pd.to_numeric(
            fixed[col], errors="coerce"
        ).fillna(0)

    area = None
    for header in (1, 2, 0):
        try:
            temp = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name="Area & Efficiency",
                header=header,
                usecols=range(8),
            )
        except Exception:
            continue

        temp.columns = temp.columns.astype(str).str.strip()

        needed = {
            "Module Type",
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        }

        if not needed.issubset(temp.columns):
            continue

        temp = temp[temp["Module Type"].notna()].copy()

        for col in temp.columns:
            if col != "Module Type":
                temp[col] = pd.to_numeric(
                    temp[col], errors="coerce"
                )

        temp = temp.dropna(
            subset=[
                "Standard PV Efficiency (%)",
                "Total area(m2)",
            ]
        )

        temp = temp[
            temp["Standard PV Efficiency (%)"].between(1, 50)
        ]

        if len(temp):
            area = temp.reset_index(drop=True)
            break

    if area is None:
        raise ValueError(
            "Could not read Area & Efficiency sheet."
        )

    weights = None
    if cluster:
        wdf = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Area & Efficiency",
            header=2,
            usecols=[12, 13, 14, 15, 16],
        )
        weights = tuple(
            float(wdf[f"CL-{i}"].iloc[0])
            for i in range(1, 6)
        )

    cfg = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )
    if "Lat" not in cfg.columns:
        raise ValueError("Lat column missing from Forecast Config.")
    lat = float(cfg.loc[0, "Lat"])

    tilt = {}
    try:
        tdf = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Config Tilt Angle",
            header=7,
        )
        tdf.columns = tdf.columns.astype(str).str.strip()
        tdf = tdf.dropna(how="all", axis=1)
        tdf = tdf.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month",
            }
        )
        if "Fixed" in tdf.columns:
            tdf["Fixed"] = tdf["Fixed"].fillna(0)
        if "Month" in tdf.columns and "Fixed" in tdf.columns:
            tdf = tdf[tdf["Fixed"] != 0]
            tilt = tdf.set_index("Month")["Fixed"].to_dict()
    except Exception:
        tilt = {}

    backend_sheet = (
        "Backend Cal CL1"
        if cluster
        else "Backend Cal"
    )

    backend = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=backend_sheet,
    )

    if "Block No." not in backend.columns:
        raise ValueError(
            f"Block No. missing from {backend_sheet}."
        )

    blocks = pd.to_numeric(
        backend["Block No."],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=np.float64)[:96]

    return {
        "cluster": cluster,
        "ghi_cols": tuple(ghi_cols),
        "fixed": fixed,
        "area": area,
        "weights": weights,
        "lat": lat,
        "tilt": tilt,
        "blocks": blocks,
    }


def solar_geometry(lat, tilt=0, tracking=False):
    today = pd.Timestamp.today()
    first = today.replace(month=1, day=1)
    day_num = (today - first).days + 1

    declination = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + day_num) / 365
            )
        )
    )

    elevation = 90 - lat + declination
    actual_tilt = 0 if tracking else tilt

    sin_a = np.sin(np.radians(elevation))
    sin_ab = np.sin(
        np.radians(elevation + actual_tilt)
    )

    if abs(sin_a) < 1e-9:
        sin_a = 1e-9

    return sin_ab, sin_a


@st.cache_data(show_spinner=False)
def optimize_loss(
    std_eff,
    area,
    actual,
    poa_matrix,
    cluster,
    weights,
):
    std_eff = np.asarray(std_eff, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    poa = np.asarray(poa_matrix, dtype=np.float64)

    if actual.size == 0 or actual.max() <= 0:
        return 0.0

    losses = np.arange(
        0,
        std_eff.min() + 0.0001,
        0.1,
    )

    eff_area = (
        area[None, :]
        * (std_eff[None, :] - losses[:, None])
        / 100
    )

    if cluster:
        weights = np.asarray(weights, dtype=np.float64)
        predictions = np.einsum(
            "lm,mb,m->lb",
            eff_area,
            poa,
            weights,
        ) / 1e6
    else:
        predictions = (
            eff_area.sum(axis=1)[:, None]
            * poa[0][None, :]
            / 1e6
        )

    peak_error = np.abs(
        actual.max()
        - predictions.max(axis=1)
    )

    return float(
        losses[np.argmin(peak_error)]
    )


def tracking_forecast(
    ghi_matrix,
    weights,
    dhi,
    start,
    end,
    maximum,
    east,
    west,
):
    blocks = np.arange(
        1,
        ghi_matrix.shape[1] + 1,
        dtype=np.float64,
    )

    if not start < maximum < end:
        return np.zeros(
            ghi_matrix.shape[1],
            dtype=np.float64,
        )

    m1 = 90 / (start - 1 - maximum)
    m2 = 90 / (end + 1 - maximum)

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1 * (blocks - maximum),
        ),
        np.minimum(
            89,
            m2 * (blocks - maximum),
        ),
    )

    panel = np.where(
        blocks < maximum,
        np.minimum(zenith, abs(east)),
        np.where(
            (blocks > maximum) & (zenith > west),
            west,
            zenith,
        ),
    )

    cos_a = np.clip(
        np.cos(np.radians(panel)),
        1e-6,
        None,
    )

    return np.sum(
        ghi_matrix
        * (
            1 - dhi / 100
        )[None, :]
        / cos_a[None, :]
        * weights[:, None],
        axis=0,
    ) / 1e6


def build_tracking_objective(
    actual,
    ghi_matrix,
    weights,
):
    actual = np.asarray(actual, dtype=np.float64)
    ghi_matrix = np.asarray(ghi_matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    blocks = np.arange(
        1,
        len(actual) + 1,
        dtype=np.float64,
    )

    mask = actual != 0
    actual_m = actual[mask]

    peak = actual_m.max()
    energy = actual_m.sum()

    def objective(x):
        DHI, start, end, maximum, east, west = (
            np.rint(x).astype(int)
        )

        if not start < maximum < end:
            return 1e9

        m1 = 90 / (start - 1 - maximum)
        m2 = 90 / (end + 1 - maximum)

        zenith = np.where(
            blocks <= maximum,
            np.minimum(
                89,
                m1 * (blocks - maximum),
            ),
            np.minimum(
                89,
                m2 * (blocks - maximum),
            ),
        )

        panel = np.where(
            blocks < maximum,
            np.minimum(
                zenith,
                abs(east),
            ),
            np.where(
                (blocks > maximum) & (zenith > west),
                west,
                zenith,
            ),
        )

        cos_a = np.clip(
            np.cos(np.radians(panel)),
            1e-6,
            None,
        )

        pred = np.sum(
            ghi_matrix
            * (
                1 - DHI / 100
            )[None, :]
            / cos_a[None, :]
            * weights[:, None],
            axis=0,
        ) / 1e6

        pred = pred[mask]

        if pred.size == 0 or not np.isfinite(pred).all():
            return 1e9

        block_error = (
            np.mean(np.abs(actual_m - pred))
            / peak
        )

        peak_error = (
            abs(peak - pred.max())
            / peak
        )

        energy_error = (
            abs(energy - pred.sum())
            / energy
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    return objective


def run_tracking_de(actual, ghi_matrix, weights):
    objective = build_tracking_objective(
        actual,
        ghi_matrix,
        weights,
    )

    progress = st.progress(0)
    status = st.empty()

    MAX_ITER = 30
    state = {"generation": 0}

    def callback(xk, convergence):
        state["generation"] += 1

        # Do not update Streamlit UI on every generation.
        if (
            state["generation"] == 1
            or state["generation"] % 5 == 0
        ):
            progress.progress(
                min(
                    state["generation"] / MAX_ITER,
                    1.0,
                )
            )
            status.info(
                f"Optimizing... "
                f"Generation {state['generation']} / {MAX_ITER}"
            )

        return False

    try:
        with st.spinner(
            "Tracking optimization chal rahi hai..."
        ):
            result = differential_evolution(
                objective,
                bounds=DE_BOUNDS,
                strategy="best1bin",
                maxiter=MAX_ITER,
                popsize=8,
                tol=0.003,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=False,
                workers=1,
                callback=callback,
            )
    finally:
        progress.empty()

    status.success("✅ Optimization complete.")

    return np.rint(result.x).astype(int)


def make_chart(forecast, actual, title):
    forecast = np.asarray(forecast, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    x = np.arange(
        1,
        len(actual) + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                color="#00c6ff",
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                color="#ef4444",
                width=3,
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="streamlit",
        height=480,
        hovermode="x unified",
        xaxis=dict(
            title="15 Minute Block",
            dtick=4,
        ),
        yaxis=dict(
            title="Power (MW)",
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ==========================================================
# UI
# ==========================================================

st.sidebar.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

st.title(
    "Pakima Pakam Ravi, 3-4 Loss Correction kar chuke hai!! 😎"
)

uploaded = st.file_uploader(
    "Yaha Feko!!",
    type=["xlsx"],
    key="lc_uploader",
)

if uploaded is None:
    st.info("Pehle File toh upload karo!!!")
    st.stop()

file_bytes = uploaded.getvalue()

file_id = (
    uploaded.name,
    len(file_bytes),
    file_hash(file_bytes),
)

# ----------------------------------------------------------
# Reset only for a genuinely new file.
# IMPORTANT: no st.rerun() here.
# ----------------------------------------------------------

if st.session_state.get("lc_file_id") != file_id:
    st.session_state["lc_file_id"] = file_id
    st.session_state["lc_run"] = False
    st.session_state["lc_params"] = None
    st.session_state["lc_workbook"] = None
    st.session_state["lc_editor_version"] = 0

try:
    if st.session_state["lc_workbook"] is None:
        with st.spinner("Workbook ek baar read ho raha hai..."):
            st.session_state["lc_workbook"] = load_workbook(
                file_bytes
            )
except Exception as e:
    st.error(f"Workbook read nahi ho paya: {e}")
    st.stop()

wb = st.session_state["lc_workbook"]

st.subheader("Input Data")

raw_df = wb["fixed"].copy()
ghi_cols = list(wb["ghi_cols"])

edited_df = st.data_editor(
    raw_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key=f"lc_editor_{st.session_state['lc_editor_version']}",
)

plant_type = st.pills(
    "Select Plant Type",
    ["🏗️ Fixed", "🔄 Tracking"],
    default="🏗️ Fixed",
)

if st.button(
    "🚀 Dabao magar pyaar se!!",
    use_container_width=True,
    type="primary",
):
    st.session_state["lc_run"] = True
    st.session_state["lc_params"] = None

if not st.session_state["lc_run"]:
    st.info(
        "Input edit karne ke baad calculation start karne ke liye button dabao."
    )
    st.stop()

actual = numeric_array(edited_df["Actual"])[:96]

ghi_arrays = [
    numeric_array(edited_df[col])[:96]
    for col in ghi_cols
]

n = min(
    len(actual),
    *(len(x) for x in ghi_arrays),
)

actual = actual[:n]
ghi_arrays = [
    x[:n]
    for x in ghi_arrays
]

if n == 0:
    st.error("No valid calculation data.")
    st.stop()

area_df = wb["area"]
cluster = wb["cluster"]
weights_raw = wb["weights"]
lat = wb["lat"]

# ==========================================================
# FIXED
# ==========================================================

if plant_type == "🏗️ Fixed":

    month = pd.Timestamp.today().strftime("%B")
    tilt = float(
        wb["tilt"].get(month, 0)
    )

    sin_ab, sin_a = solar_geometry(
        lat,
        tilt,
        False,
    )

    poa = np.asarray(
        [
            ghi * sin_ab / sin_a
            for ghi in ghi_arrays
        ],
        dtype=np.float64,
    )

    best_loss = optimize_loss(
        tuple(
            area_df["Standard PV Efficiency (%)"]
        ),
        tuple(
            area_df["Total area(m2)"]
        ),
        tuple(actual),
        tuple(
            tuple(row)
            for row in poa
        ),
        cluster,
        tuple(weights_raw) if cluster else tuple(),
    )

    st.subheader("Efficiency")

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(best_loss),
        step=0.1,
        key="fixed_loss",
    )

    area_eff = apply_efficiency(
        area_df,
        loss,
    )

    show_efficiency(area_eff)

    if cluster:
        weights = np.asarray(
            weights_raw,
            dtype=np.float64,
        )

        forecast = np.sum(
            poa
            * (
                area_eff["Eff Area"].to_numpy()
                * weights
            )[:, None],
            axis=0,
        ) / 1e6

    else:
        forecast = (
            poa[0]
            * area_eff["Eff Area"].sum()
            / 1e6
        )

    st.metric(
        "Optimized Efficiency Loss",
        f"{best_loss:.2f}%",
    )

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
            "Fixed Plant - Forecast vs Actual",
        ),
        use_container_width=True,
    )

# ==========================================================
# TRACKING
# ==========================================================

else:

    st.subheader("Tracking Optimization")

    sin_ab, sin_a = solar_geometry(
        lat,
        0,
        True,
    )

    ghi_matrix = np.asarray(
        ghi_arrays,
        dtype=np.float64,
    )

    poa_matrix = (
        ghi_matrix
        * sin_ab
        / sin_a
    )

    best_loss = optimize_loss(
        tuple(
            area_df["Standard PV Efficiency (%)"]
        ),
        tuple(
            area_df["Total area(m2)"]
        ),
        tuple(actual),
        tuple(
            tuple(row)
            for row in poa_matrix
        ),
        cluster,
        tuple(weights_raw) if cluster else tuple(),
    )

    # ------------------------------------------------------
    # FIRST VISIT: no automatic DE
    # ------------------------------------------------------

    if st.session_state["lc_params"] is None:

        st.metric(
            "Initial Efficiency Loss",
            f"{best_loss:.2f}%",
        )

        st.warning(
            "Tracking optimization heavy calculation hai. "
            "Ye tabhi chalega jab aap neeche button dabayenge."
        )

        if st.button(
            "🧠 Run Tracking Optimization",
            type="primary",
            use_container_width=True,
        ):
            initial_area = apply_efficiency(
                area_df,
                best_loss,
            )

            if cluster:
                de_weights = (
                    initial_area["Eff Area"].to_numpy()
                    * np.asarray(weights_raw)
                )
            else:
                de_weights = np.asarray(
                    [initial_area["Eff Area"].sum()]
                )

            try:
                best = run_tracking_de(
                    actual,
                    ghi_matrix,
                    de_weights,
                )

                st.session_state["lc_params"] = {
                    "DHI": int(best[0]),
                    "start": int(best[1]),
                    "end": int(best[2]),
                    "max": int(best[3]),
                    "east": int(best[4]),
                    "west": int(best[5]),
                    "loss": float(best_loss),
                }

                # This rerun is intentional: it happens once,
                # after optimization, not during normal editing.
                st.rerun()

            except Exception as e:
                st.error(
                    f"Tracking optimization failed: {e}"
                )

        st.stop()

    params = st.session_state["lc_params"]

    # ------------------------------------------------------
    # MANUAL PARAMETERS
    # ------------------------------------------------------

    st.caption(
        "Parameters change karne par DE dobara run nahi hoga."
    )

    with st.form("tracking_params_form"):

        loss_input = st.number_input(
            "Efficiency Loss (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(params["loss"]),
            step=0.1,
        )

        c1, c2, c3 = st.columns(3)

        dhi = c1.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(params["DHI"]),
            step=1,
        )

        start = c2.number_input(
            "Starting Block",
            min_value=1,
            max_value=95,
            value=int(params["start"]),
            step=1,
        )

        end = c3.number_input(
            "Ending Block",
            min_value=2,
            max_value=96,
            value=int(params["end"]),
            step=1,
        )

        c1, c2, c3 = st.columns(3)

        maximum = c1.number_input(
            "Max Block",
            min_value=1,
            max_value=95,
            value=int(params["max"]),
            step=1,
        )

        east = c2.number_input(
            "East Limit",
            min_value=0,
            max_value=70,
            value=int(params["east"]),
            step=1,
        )

        west = c3.number_input(
            "West Limit",
            min_value=0,
            max_value=70,
            value=int(params["west"]),
            step=1,
        )

        recalculate = st.form_submit_button(
            "🔄 Recalculate",
            use_container_width=True,
            type="primary",
        )

    if recalculate:

        if not (
            start < maximum < end
        ):
            st.error(
                "Condition must be: Starting Block < Max Block < Ending Block"
            )
            st.stop()

        st.session_state["lc_params"] = {
            "DHI": int(dhi),
            "start": int(start),
            "end": int(end),
            "max": int(maximum),
            "east": int(east),
            "west": int(west),
            "loss": float(loss_input),
        }

        params = st.session_state["lc_params"]

    # ------------------------------------------------------
    # FINAL FORECAST
    # ------------------------------------------------------

    area_eff = apply_efficiency(
        area_df,
        params["loss"],
    )

    show_efficiency(area_eff)

    if cluster:
        effective_weights = (
            area_eff["Eff Area"].to_numpy()
            * np.asarray(weights_raw)
        )
    else:
        effective_weights = np.asarray(
            [area_eff["Eff Area"].sum()]
        )

    forecast = tracking_forecast(
        ghi_matrix,
        effective_weights,
        params["DHI"],
        params["start"],
        params["end"],
        params["max"],
        params["east"],
        params["west"],
    )

    st.plotly_chart(
        make_chart(
            forecast,
            actual,
            "Tracking Plant - Forecast vs Actual",
        ),
        use_container_width=True,
    )

    daylight = actual > 0

    if daylight.any():

        a = actual[daylight]
        p = forecast[daylight]

        mae = np.mean(np.abs(a - p))
        rmse = np.sqrt(np.mean((a - p) ** 2))

        peak_error = (
            abs(a.max() - p.max())
            / a.max()
            * 100
        )

        energy_error = (
            abs(a.sum() - p.sum())
            / a.sum()
            * 100
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        c3.metric("Peak Error", f"{peak_error:.2f}%")
        c4.metric("Energy Error", f"{energy_error:.2f}%")
