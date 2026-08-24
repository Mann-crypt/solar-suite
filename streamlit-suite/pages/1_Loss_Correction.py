# ============================================================
# LOSS CORRECTION
# Compact + Stable + Cloud Safe
#
# Heavy calculations run ONLY when RUN is clicked.
# Normal Streamlit reruns never run optimization.
# ============================================================

import io
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Loss Correction",
    page_icon="☀️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.stButton > button {
    font-weight: 650;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

SHEETS_REQUIRED = [
    "Fixed-C11",
    "Area & Efficiency",
    "Forecast Config",
    "Config Tilt Angle",
    "Result",
]

TRACKING_BOUNDS = [
    (0, 10),     # DHI
    (10, 30),    # GHI Starting Block
    (65, 80),    # GHI Ending Block
    (47, 53),    # GHI Max Block
    (10, 70),    # East
    (10, 70),    # West
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "lc_file_id": None,
    "lc_result": None,
    "lc_input_hash": None,
    "lc_editor_version": 0,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# BASIC HELPERS
# ============================================================

def num(x):
    """Always return float NumPy array."""
    if isinstance(x, pd.Series):
        return (
            pd.to_numeric(x, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

    if isinstance(x, pd.DataFrame):
        return (
            x.apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

    return np.asarray(x, dtype=float)


def scalar(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clean_name(x):
    return (
        str(x)
        .replace("*", "")
        .replace("\n", " ")
        .strip()
    )


def file_id(uploaded):
    b = uploaded.getvalue()
    return (
        uploaded.name,
        len(b),
        hashlib.md5(b).hexdigest(),
    )


def input_hash(df):
    return hashlib.md5(
        pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()
    ).hexdigest()


# ============================================================
# METRICS
# ============================================================

def metrics(actual, forecast):

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    n = min(len(actual), len(forecast))

    actual = actual[:n]
    forecast = forecast[:n]

    mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "Actual power contains no valid non-zero values."
        )

    a = actual[mask]
    f = forecast[mask]

    peak = float(a.max())
    energy = float(a.sum())

    if peak <= 0 or energy <= 0:
        raise ValueError(
            "Actual peak/energy must be greater than zero."
        )

    block_error = (
        np.mean(np.abs(a - f))
        / peak
    )

    peak_error = (
        abs(peak - f.max())
        / peak
    )

    energy_error = (
        abs(energy - f.sum())
        / energy
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return {
        "actual_peak": peak,
        "forecast_peak": float(f.max()),
        "block_error": float(block_error),
        "peak_error": float(peak_error),
        "energy_error": float(energy_error),
        "score": float(score),
    }


# ============================================================
# WORKBOOK READER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def read_workbook(file_bytes):

    bio = io.BytesIO(file_bytes)
    excel = pd.ExcelFile(bio)
    sheets = excel.sheet_names

    missing = [
        s for s in SHEETS_REQUIRED
        if s not in sheets
    ]

    if missing:
        raise ValueError(
            "Missing required sheet(s): "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    area_raw = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
    )

    area = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area.columns = [
        clean_name(c)
        for c in area.columns
    ]

    if "Standard PV Efficiency (%)" not in area.columns:
        raise ValueError(
            "Column 'Standard PV Efficiency (%)' "
            "not found in Area & Efficiency."
        )

    # VCast effective-area locations
    fixed_weights = num(
        area_raw.iloc[2:7, 15]
    )

    tracking_weights = num(
        area_raw.iloc[28:33, 15]
    )

    if len(fixed_weights) != 5:
        raise ValueError(
            "Could not read 5 Fixed effective-area values "
            "from Area & Efficiency."
        )

    if len(tracking_weights) != 5:
        raise ValueError(
            "Could not read 5 Tracking effective-area values "
            "from Area & Efficiency."
        )

    standard_efficiency = num(
        area["Standard PV Efficiency (%)"]
    )[:5]

    if len(standard_efficiency) != 5:
        raise ValueError(
            "Could not read 5 Standard PV Efficiency values."
        )

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    config.columns = [
        clean_name(c)
        for c in config.columns
    ]

    lat_col = next(
        (
            c for c in config.columns
            if str(c).strip().lower() == "lat"
        ),
        None,
    )

    if lat_col is None:
        raise ValueError(
            "Column 'Lat' not found in Forecast Config."
        )

    lat_values = pd.to_numeric(
        config[lat_col],
        errors="coerce",
    ).dropna()

    if lat_values.empty:
        raise ValueError(
            "Latitude value is invalid."
        )

    latitude = float(lat_values.iloc[0])

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    tilt_raw = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    tilt_raw.columns = [
        clean_name(c)
        for c in tilt_raw.columns
    ]

    # First try direct Month column
    month_col = next(
        (
            c for c in tilt_raw.columns
            if str(c).strip().lower() == "month"
        ),
        None,
    )

    # VCast workbook commonly has month in Unnamed:3
    if month_col is None:
        candidates = [
            c for c in tilt_raw.columns
            if "month" in str(c).lower()
        ]

        if candidates:
            month_col = candidates[0]

    # Last fallback: find a column containing month numbers
    if month_col is None:
        for c in tilt_raw.columns:
            vals = pd.to_numeric(
                tilt_raw[c],
                errors="coerce",
            )

            valid = vals[
                vals.between(1, 12)
            ]

            if len(valid) >= 3:
                month_col = c
                break

    fixed_tilt_col = next(
        (
            c for c in tilt_raw.columns
            if str(c).strip().lower() == "fixed"
        ),
        None,
    )

    if fixed_tilt_col is None:
        raise ValueError(
            "Column 'Fixed' not found in Config Tilt Angle."
        )

    if month_col is None:
        raise ValueError(
            "Month column missing in Config Tilt Angle."
        )

    # Convert month information
    month_num = pd.to_numeric(
        tilt_raw[month_col],
        errors="coerce",
    )

    # If month is written as names, convert names
    if month_num.notna().sum() == 0:

        month_names = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        month_num = (
            tilt_raw[month_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(month_names)
        )

    tilt_values = pd.to_numeric(
        tilt_raw[fixed_tilt_col],
        errors="coerce",
    )

    tilt_lookup = {
        int(m): float(t)
        for m, t in zip(
            month_num,
            tilt_values,
        )
        if pd.notna(m) and pd.notna(t)
    }

    if not tilt_lookup:
        raise ValueError(
            "No valid Fixed tilt values found."
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=range(6),
    )

    result.columns = [
        "Block",
        *GHI_COLS,
    ]

    result["Block"] = pd.to_numeric(
        result["Block"],
        errors="coerce",
    )

    result = result[
        result["Block"].notna()
    ].copy()

    for c in GHI_COLS:
        result[c] = pd.to_numeric(
            result[c],
            errors="coerce",
        ).fillna(0)

    # --------------------------------------------------------
    # FIXED-C11
    # --------------------------------------------------------

    fixed = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    fixed.columns = [
        clean_name(c)
        for c in fixed.columns
    ]

    if "Date" not in fixed.columns:
        raise ValueError(
            "Column 'Date' not found in Fixed-C11."
        )

    if "Actual" not in fixed.columns:
        raise ValueError(
            "Column 'Actual' not found in Fixed-C11."
        )

    fixed["Date"] = pd.to_datetime(
        fixed["Date"],
        errors="coerce",
    )

    fixed = fixed[
        fixed["Date"].notna()
    ].copy()

    fixed.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    n = min(
        len(fixed),
        len(result),
    )

    if n <= 0:
        raise ValueError(
            "No aligned data rows found."
        )

    fixed = fixed.iloc[:n].copy()
    result = result.iloc[:n].copy()

    actual = num(
        fixed["Actual"]
    )[:n]

    blocks = num(
        result["Block"]
    )[:n]

    ghi = result[
        GHI_COLS
    ].to_numpy(dtype=float)[:n]

    dates = fixed["Date"].iloc[:n]

    # --------------------------------------------------------
    # SOLAR GEOMETRY
    # --------------------------------------------------------

    first_date = pd.Timestamp(
        "2025-01-01"
    )

    day_offset = (
        dates - first_date
    ).dt.days.to_numpy(
        dtype=float
    )

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_offset
                    + 1
                )
                / 365
            )
        )
    )

    elevation = (
        90
        - latitude
        + declination
    )

    months = dates.dt.month.to_numpy()

    tilt = np.array(
        [
            tilt_lookup.get(
                int(m),
                0.0,
            )
            for m in months
        ],
        dtype=float,
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_ab = np.sin(
        np.radians(
            elevation + tilt
        )
    )

    safe_sin_a = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a,
    )

    geometry_factor = (
        sin_ab / safe_sin_a
    )

    fixed_poa = (
        ghi
        * geometry_factor[:, None]
    )

    # --------------------------------------------------------
    # TRACKING SHEET
    # --------------------------------------------------------

    tracking_available = (
        "Tracking" in sheets
    )

    tracking_df = None

    if tracking_available:
        tracking_df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Tracking",
            header=1,
        )

        tracking_df.columns = [
            clean_name(c)
            for c in tracking_df.columns
        ]

        tracking_df = tracking_df.iloc[
            :n
        ].copy()

    return {
        "area": area,
        "fixed_weights": fixed_weights,
        "tracking_weights": tracking_weights,
        "standard_efficiency": standard_efficiency,
        "latitude": latitude,
        "tilt_lookup": tilt_lookup,
        "dates": dates,
        "blocks": blocks,
        "ghi": ghi,
        "actual": actual,
        "fixed_poa": fixed_poa,
        "tracking_available": tracking_available,
        "tracking_df": tracking_df,
    }


# ============================================================
# REBUILD POA AFTER USER EDIT
# ============================================================

def rebuild_poa(
    original_ghi,
    original_poa,
    edited_ghi,
):
    """
    Preserve the workbook's solar geometry while allowing
    user-edited GHI to flow into POA.
    """

    original_ghi = np.asarray(
        original_ghi,
        dtype=float,
    )

    original_poa = np.asarray(
        original_poa,
        dtype=float,
    )

    edited_ghi = np.asarray(
        edited_ghi,
        dtype=float,
    )

    n = min(
        len(original_ghi),
        len(original_poa),
        len(edited_ghi),
    )

    original_ghi = original_ghi[:n]
    original_poa = original_poa[:n]
    edited_ghi = edited_ghi[:n]

    factor = np.divide(
        original_poa,
        original_ghi,
        out=np.zeros_like(
            original_poa
        ),
        where=(
            np.abs(original_ghi)
            > 1e-12
        ),
    )

    return (
        edited_ghi
        * factor
    )


# ============================================================
# FIXED FORECAST
# ============================================================

def fixed_forecast(
    fixed_poa,
    standard_efficiency,
    fixed_weights,
    loss,
):

    std = np.asarray(
        standard_efficiency,
        dtype=float,
    )

    weights = np.asarray(
        fixed_weights,
        dtype=float,
    )

    poa = np.asarray(
        fixed_poa,
        dtype=float,
    )

    n = min(
        poa.shape[0],
        len(std),
        len(weights),
    )

    poa = poa[:n]
    std = std[:n]
    weights = weights[:n]

    net = np.maximum(
        std - float(loss),
        0,
    )

    factor = np.divide(
        net,
        std,
        out=np.zeros_like(net),
        where=std != 0,
    )

    final_weights = (
        weights * factor
    )

    power_matrix = (
        poa
        * final_weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(axis=1)
    )

    return (
        forecast,
        power_matrix,
        net,
        final_weights,
    )


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def optimize_fixed_cached(
    std_tuple,
    weight_tuple,
    poa_tuple,
    actual_tuple,
):

    std = np.asarray(
        std_tuple,
        dtype=float,
    )

    weights = np.asarray(
        weight_tuple,
        dtype=float,
    )

    poa = np.asarray(
        poa_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    n = min(
        len(actual),
        poa.shape[0],
    )

    actual = actual[:n]
    poa = poa[:n]

    mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No valid Actual values for Fixed optimization."
        )

    actual_day = actual[mask]

    actual_peak = actual_day.max()
    actual_energy = actual_day.sum()

    max_loss = float(
        np.min(std)
    )

    rows = []

    # Vectorized enough for only ~100 loss values
    for loss in np.arange(
        0,
        max_loss + 0.0001,
        0.1,
    ):

        forecast, _, _, _ = fixed_forecast(
            poa,
            std,
            weights,
            loss,
        )

        f = forecast[mask]

        predicted_peak = f.max()

        peak_error = abs(
            actual_peak
            - predicted_peak
        )

        block_error = (
            np.mean(
                np.abs(
                    actual_day - f
                )
            )
            / actual_peak
        )

        energy_error = (
            abs(
                actual_energy
                - f.sum()
            )
            / actual_energy
        )

        score = (
            0.80 * block_error
            + 0.10 * (
                peak_error
                / actual_peak
            )
            + 0.10 * energy_error
        )

        rows.append({
            "Error %": round(
                float(loss),
                1,
            ),
            "Actual Peak": float(
                actual_peak
            ),
            "Predicted Peak": float(
                predicted_peak
            ),
            "Peak Error": float(
                peak_error
            ),
            "Peak Error (%)": float(
                peak_error
                / actual_peak
                * 100
            ),
            "Block Error": float(
                block_error
            ),
            "Energy Error": float(
                energy_error
            ),
            "Overall Score": float(
                score
            ),
        })

    table = pd.DataFrame(rows)

    if table.empty:
        raise ValueError(
            "Fixed optimization returned no results."
        )

    # IMPORTANT:
    # Preserve original VCast logic:
    # choose minimum Peak Error.
    best = table.loc[
        table["Peak Error"].idxmin()
    ]

    best_loss = float(
        best["Error %"]
    )

    forecast, power_matrix, net, final_weights = (
        fixed_forecast(
            poa,
            std,
            weights,
            best_loss,
        )
    )

    return {
        "loss": best_loss,
        "forecast": forecast,
        "power_matrix": power_matrix,
        "net_efficiency": net,
        "weights": final_weights,
        "table": table,
    }


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking(
    blocks,
    ghi,
    weights,
    dhi,
    start,
    end,
    maximum,
    east,
    west,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    ghi = np.asarray(
        ghi,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if not (
        start < maximum < end
    ):
        return None

    d1 = (
        start - 1 - maximum
    )

    d2 = (
        end + 1 - maximum
    )

    if d1 == 0 or d2 == 0:
        return None

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1 * (
                blocks - maximum
            ),
        ),
        np.minimum(
            89,
            m2 * (
                blocks - maximum
            ),
        ),
    )

    panel = np.where(
        blocks < maximum,
        np.minimum(
            zenith,
            abs(east),
        ),
        np.where(
            (
                (blocks > maximum)
                & (zenith > west)
            ),
            west,
            zenith,
        ),
    )

    cos_alpha = np.clip(
        np.cos(
            np.radians(panel)
        ),
        1e-6,
        None,
    )

    dhi_part = (
        ghi * float(dhi) / 100
    )

    dni = (
        ghi
        - dhi_part
    ) / cos_alpha[:, None]

    power_matrix = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = (
        power_matrix.sum(axis=1)
    )

    return (
        forecast,
        power_matrix,
        zenith,
        panel,
        dni,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def optimize_tracking_cached(
    blocks_tuple,
    ghi_tuple,
    weights_tuple,
    actual_tuple,
):

    blocks = np.asarray(
        blocks_tuple,
        dtype=float,
    )

    ghi = np.asarray(
        ghi_tuple,
        dtype=float,
    )

    weights = np.asarray(
        weights_tuple,
        dtype=float,
    )

    actual = np.asarray(
        actual_tuple,
        dtype=float,
    )

    n = min(
        len(blocks),
        len(actual),
        ghi.shape[0],
    )

    blocks = blocks[:n]
    ghi = ghi[:n]
    actual = actual[:n]

    mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    if not mask.any():
        raise ValueError(
            "No valid Actual values for Tracking optimization."
        )

    a = actual[mask]

    peak = a.max()
    energy = a.sum()

    def objective(x):

        p = np.rint(x).astype(int)

        result = calculate_tracking(
            blocks,
            ghi,
            weights,
            int(p[0]),
            int(p[1]),
            int(p[2]),
            int(p[3]),
            int(p[4]),
            int(p[5]),
        )

        if result is None:
            return 1e9

        forecast = result[0]

        if not np.all(
            np.isfinite(forecast)
        ):
            return 1e9

        f = forecast[mask]

        if len(f) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(a - f)
            )
            / peak
        )

        peak_error = (
            abs(
                peak - f.max()
            )
            / peak
        )

        energy_error = (
            abs(
                energy - f.sum()
            )
            / energy
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    result = differential_evolution(
        objective,
        bounds=TRACKING_BOUNDS,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.rint(
        result.x
    ).astype(int)

    score = objective(best)

    return (
        tuple(best.tolist()),
        float(result.fun),
        float(score),
    )


# ============================================================
# GRAPH
# ============================================================

def make_graph(
    blocks,
    actual,
    forecast,
    name,
):

    n = min(
        len(blocks),
        len(actual),
        len(forecast),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=blocks[:n],
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=blocks[:n],
            y=forecast[:n],
            mode="lines",
            name=name,
            line=dict(width=2),
        )
    )

    fig.update_layout(
        height=470,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
        xaxis_title="Block",
        yaxis_title="Power (MW)",
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title("☀️ Solar Loss Correction")
st.caption(
    "Fixed / Tracking | Automatic optimization runs only on demand"
)


# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload Solar Excel Workbook",
    type=["xlsx", "xls"],
)

if uploaded is None:
    st.info(
        "Upload the Solar Excel workbook to begin."
    )
    st.stop()

file_bytes = uploaded.getvalue()
current_id = file_id(uploaded)


# ============================================================
# NEW FILE RESET
# ============================================================

if (
    st.session_state.lc_file_id
    != current_id
):

    st.session_state.lc_file_id = current_id
    st.session_state.lc_result = None
    st.session_state.lc_input_hash = None
    st.session_state.lc_editor_version += 1


# ============================================================
# LOAD WORKBOOK
# ============================================================

try:

    with st.spinner(
        "Reading workbook..."
    ):
        wb = read_workbook(
            file_bytes
        )

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )
    st.stop()


# ============================================================
# BASIC INFORMATION
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Workbook",
        "VCast",
    )

with c2:
    st.metric(
        "Data Rows",
        len(wb["actual"]),
    )

with c3:
    st.metric(
        "Tracking",
        "Available"
        if wb["tracking_available"]
        else "Not Available",
    )


# ============================================================
# INPUT TABLE
# ============================================================

st.subheader("📥 Input Data")

input_df = pd.DataFrame(
    {
        "Date": wb["dates"].dt.date,
        "Actual": wb["actual"],
    }
)

for i, col in enumerate(GHI_COLS):
    input_df[col] = wb["ghi"][:, i]

editor_key = (
    "lc_editor_"
    + str(
        st.session_state.lc_editor_version
    )
)

edited = st.data_editor(
    input_df,
    key=editor_key,
    hide_index=True,
    num_rows="fixed",
    height=330,
    width="stretch",
    disabled=["Date"],
    column_config={
        "Date": st.column_config.DateColumn(
            "Date",
            format="DD-MM-YYYY",
        ),
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.4f",
        ),
        **{
            c: st.column_config.NumberColumn(
                c,
                format="%.4f",
            )
            for c in GHI_COLS
        },
)


# ============================================================
# CLEAN EDITED DATA
# ============================================================

edited = edited.copy()

edited["Actual"] = pd.to_numeric(
    edited["Actual"],
    errors="coerce",
).fillna(0)

for c in GHI_COLS:
    edited[c] = pd.to_numeric(
        edited[c],
        errors="coerce",
).fillna(0)

actual = edited[
    "Actual"
].to_numpy(dtype=float)

ghi = edited[
    GHI_COLS
].to_numpy(dtype=float)


# ============================================================
# INPUT CHANGE DETECTION
# ============================================================

current_input_hash = input_hash(
    edited[
        ["Actual", *GHI_COLS]
    ]
)

input_changed = (
    st.session_state.lc_input_hash is not None
    and
    current_input_hash
    != st.session_state.lc_input_hash
)

if (
    input_changed
    and
    st.session_state.lc_result is not None
):
    st.warning(
        "Input data was edited after the last calculation. "
        "Click **RUN LOSS CORRECTION** to recalculate."
    )


# ============================================================
# CORRECTION TYPE
# ============================================================

st.subheader("🌞 Correction Type")

plant = st.segmented_control(
    "Correction Type",
    ["Fixed", "Tracking"],
    default="Fixed",
    selection_mode="single",
    width="stretch",
    label_visibility="collapsed",
)


# ============================================================
# RUN BUTTON
# ============================================================

run = st.button(
    "🚀 RUN LOSS CORRECTION",
    type="primary",
    width="stretch",
)


# ============================================================
# HEAVY CALCULATION
# ONLY THIS BLOCK RUNS OPTIMIZATION
# ============================================================

if run:

    try:

        if len(actual) == 0:
            raise ValueError(
                "No input data available."
            )

        if not np.any(
            actual != 0
        ):
            raise ValueError(
                "Actual power contains no non-zero values."
            )

        with st.spinner(
            "Running loss correction..."
        ):

            # ------------------------------------------------
            # REBUILD POA FROM EDITED GHI
            # ------------------------------------------------

            edited_poa = rebuild_poa(
                wb["ghi"],
                wb["fixed_poa"],
                ghi,
            )

            result = {
                "plant": plant,
                "actual": actual.copy(),
                "ghi": ghi.copy(),
                "poa": edited_poa.copy(),
                "blocks": wb["blocks"][
                    :len(actual)
                ].copy(),
            }

            # ------------------------------------------------
            # FIXED
            # ------------------------------------------------

            if plant == "Fixed":

                fixed = optimize_fixed_cached(
                    tuple(
                        wb[
                            "standard_efficiency"
                        ].tolist()
                    ),
                    tuple(
                        wb[
                            "fixed_weights"
                        ].tolist()
                    ),
                    tuple(
                        edited_poa.tolist()
                    ),
                    tuple(
                        actual.tolist()
                    ),
                )

                result.update({
                    "fixed": fixed,
                    "metrics": metrics(
                        actual,
                        fixed["forecast"],
                    ),
                })

            # ------------------------------------------------
            # TRACKING
            # ------------------------------------------------

            else:

                if not wb[
                    "tracking_available"
                ]:
                    raise ValueError(
                        "Tracking sheet was not found "
                        "in this workbook."
                    )

                tracking = (
                    optimize_tracking_cached(
                        tuple(
                            wb[
                                "blocks"
                            ].tolist()
                        ),
                        tuple(
                            ghi.tolist()
                        ),
                        tuple(
                            wb[
                                "tracking_weights"
                            ].tolist()
                        ),
                        tuple(
                            actual.tolist()
                        ),
                    )
                )

                params = tracking[0]

                tracking_output = (
                    calculate_tracking(
                        wb["blocks"],
                        ghi,
                        wb[
                            "tracking_weights"
                        ],
                        *params,
                    )
                )

                if tracking_output is None:
                    raise ValueError(
                        "Optimized Tracking parameters "
                        "are invalid."
                    )

                result.update({
                    "tracking": tracking,
                    "tracking_output": tracking_output,
                    "metrics": metrics(
                        actual,
                        tracking_output[0],
                    ),
                })

            # ------------------------------------------------
            # SAVE EVERYTHING
            # ------------------------------------------------

            st.session_state.lc_result = result
            st.session_state.lc_input_hash = (
                current_input_hash
            )

    except Exception as e:

        st.session_state.lc_result = None

        st.error(
            f"Calculation failed: {e}"
        )


# ============================================================
# NO RESULT
# ============================================================

if st.session_state.lc_result is None:

    st.info(
        "Edit the input if required, select Fixed or "
        "Tracking, then click **RUN LOSS CORRECTION**."
    )

    st.stop()


# ============================================================
# RESULT
# ============================================================

result = st.session_state.lc_result

if result["plant"] != plant:

    st.warning(
        "Plant type changed. Click **RUN LOSS CORRECTION** "
        "to calculate the new plant type."
    )

    st.stop()


actual = result["actual"]


# ============================================================
# FIXED RESULT
# ============================================================

if plant == "Fixed":

    fixed = result["fixed"]

    st.subheader(
        "🏗️ Fixed Loss Correction"
    )

    # --------------------------------------------------------
    # PARAMETER
    # --------------------------------------------------------

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=float(
            np.min(
                wb[
                    "standard_efficiency"
                ]
            )
        ),
        value=float(
            fixed["loss"]
        ),
        step=0.1,
        format="%.1f",
        key="lc_fixed_loss",
    )

    # Cheap calculation only
    forecast, power_matrix, net_eff, weights = (
        fixed_forecast(
            result["poa"],
            wb["standard_efficiency"],
            wb["fixed_weights"],
            loss,
        )
    )

    m = metrics(
        actual,
        forecast,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "Efficiency Loss",
            f"{loss:.1f}%",
        )

    with b:
        st.metric(
            "Actual Peak",
            f"{m['actual_peak']:.4f}",
        )

    with c:
        st.metric(
            "Forecast Peak",
            f"{m['forecast_peak']:.4f}",
        )

    with d:
        st.metric(
            "Peak Error",
            f"{m['peak_error'] * 100:.3f}%",
        )

    # --------------------------------------------------------
    # OPTIMIZATION TABLE
    # --------------------------------------------------------

    with st.expander(
        "📊 Fixed Optimization Results"
    ):

        st.dataframe(
            fixed["table"],
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    with st.expander(
        "📋 Final Parameters"
    ):

        final_parameters = pd.DataFrame({
            "Parameter": [
                "Plant",
                "Efficiency Loss (%)",
                "Actual Peak",
                "Forecast Peak",
                "Peak Error (%)",
                "Block Error",
                "Energy Error",
                "Overall Score",
            ],
            "Value": [
                "Fixed",
                loss,
                m["actual_peak"],
                m["forecast_peak"],
                m["peak_error"] * 100,
                m["block_error"],
                m["energy_error"],
                m["score"],
            ],
        })

        st.dataframe(
            final_parameters,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Actual vs Fixed Forecast"
    )

    fig = make_graph(
        result["blocks"],
        actual,
        forecast,
        "Fixed Forecast",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


# ============================================================
# TRACKING RESULT
# ============================================================

else:

    tracking = result["tracking"]
    params = tracking[0]

    (
        dhi,
        start,
        end,
        maximum,
        east,
        west,
    ) = params

    st.subheader(
        "🔄 Tracking Loss Correction"
    )

    # --------------------------------------------------------
    # EDITABLE PARAMETERS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi = st.number_input(
            "DHI (%)",
            0,
            100,
            int(dhi),
            1,
            key="lc_dhi",
        )

        start = st.number_input(
            "GHI Starting Block",
            0,
            95,
            int(start),
            1,
            key="lc_start",
        )

    with c2:

        end = st.number_input(
            "GHI Ending Block",
            1,
            96,
            int(end),
            1,
            key="lc_end",
        )

        maximum = st.number_input(
            "GHI Max Block",
            0,
            95,
            int(maximum),
            1,
            key="lc_max",
        )

    with c3:

        east = st.number_input(
            "Tracking East Limit",
            0,
            90,
            int(east),
            1,
            key="lc_east",
        )

        west = st.number_input(
            "Tracking West Limit",
            0,
            90,
            int(west),
            1,
            key="lc_west",
        )

    if not (
        start < maximum < end
    ):
        st.error(
            "Tracking parameters must satisfy: "
            "Starting Block < Max Block < Ending Block."
        )
        st.stop()

    # --------------------------------------------------------
    # CHEAP TRACKING CALCULATION
    # NO OPTIMIZATION
    # --------------------------------------------------------

    tracking_output = calculate_tracking(
        result["blocks"],
        result["ghi"],
        wb["tracking_weights"],
        int(dhi),
        int(start),
        int(end),
        int(maximum),
        int(east),
        int(west),
    )

    if tracking_output is None:
        st.error(
            "Invalid Tracking parameters."
        )
        st.stop()

    forecast = tracking_output[0]

    m = metrics(
        actual,
        forecast,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    with a:
        st.metric(
            "DHI",
            f"{int(dhi)}%",
        )

    with b:
        st.metric(
            "Actual Peak",
            f"{m['actual_peak']:.4f}",
        )

    with c:
        st.metric(
            "Forecast Peak",
            f"{m['forecast_peak']:.4f}",
        )

    with d:
        st.metric(
            "Peak Error",
            f"{m['peak_error'] * 100:.3f}%",
        )

    # --------------------------------------------------------
    # FINAL PARAMETERS
    # --------------------------------------------------------

    with st.expander(
        "📋 Final Tracking Parameters"
    ):

        tracking_parameters = pd.DataFrame({
            "Parameter": [
                "DHI (%)",
                "GHI Starting Block",
                "GHI Ending Block",
                "GHI Max Block",
                "Tracking East Limit",
                "Tracking West Limit",
                "Actual Peak",
                "Forecast Peak",
                "Peak Error (%)",
                "Block Error",
                "Energy Error",
                "Overall Score",
            ],
            "Value": [
                int(dhi),
                int(start),
                int(end),
                int(maximum),
                int(east),
                int(west),
                m["actual_peak"],
                m["forecast_peak"],
                m["peak_error"] * 100,
                m["block_error"],
                m["energy_error"],
                m["score"],
            ],
        })

        st.dataframe(
            tracking_parameters,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    st.subheader(
        "📈 Actual vs Tracking Forecast"
    )

    fig = make_graph(
        result["blocks"],
        actual,
        forecast,
        "Tracking Forecast",
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )
