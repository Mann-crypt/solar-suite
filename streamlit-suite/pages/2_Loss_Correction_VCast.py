# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# STREAMLIT APPLICATION
#
# CLEAN + COMPACT VERSION
#
# IMPORTANT:
# Error % is applied ONLY ONCE.
# Differential Evolution runs ONLY on Run Calculation.
# Editable parameters do NOT rerun optimization.
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Forecast Correction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 650;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 13px 16px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 3px;
    }

    .metric-value {
        font-size: 22px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 5px;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
    }

    .stButton > button {
        border-radius: 8px;
        min-height: 40px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "calculated": False,
    "calculation_data": None,
    "plant_type": "Fixed",
    "file_key": None,
    "input_editor": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BASIC HELPERS
# ============================================================

def numeric(values):
    """
    Safely convert Series / ndarray / list / scalar to
    numeric NumPy array.

    Fixes:
    AttributeError:
    'numpy.ndarray' object has no attribute 'fillna'
    """
    if isinstance(values, pd.Series):
        return (
            pd.to_numeric(values, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )

    return np.nan_to_num(
        pd.to_numeric(
            np.asarray(values),
            errors="coerce",
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float)


def numeric_series(values):
    """Return a clean numeric Series."""
    return pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).fillna(0.0)


def read_excel(uploaded_file, **kwargs):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, **kwargs)


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return df


def trim_at_first_null(df, column):
    df = df.copy()

    if column not in df.columns:
        return df

    null_rows = df[df[column].isna()].index

    if len(null_rows):
        first_position = df.index.get_loc(null_rows[0])
        df = df.iloc[:first_position].copy()

    return df


# ============================================================
# LOAD AREA & EFFICIENCY
# ============================================================

def load_area_efficiency(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "S.No.")

    numeric_cols = [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df


# ============================================================
# CLUSTER TABLE
# ============================================================

def load_cluster_table(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "Clusters")

    return df.reset_index(drop=True)


# ============================================================
# LOAD GHI
# ============================================================

def load_ghi(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df = df.fillna(0)

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    for col in ghi_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    return df


# ============================================================
# LOAD LATITUDE
# ============================================================

def load_latitude(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = pd.to_numeric(
        df.loc[0, "Lat"],
        errors="coerce",
    )

    if pd.isna(lat):
        raise ValueError("Latitude could not be read.")

    return float(lat)


# ============================================================
# LOAD TILT
# ============================================================

def load_tilt(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df = clean_columns(df)

    df = trim_at_first_null(
        df,
        "Fixed",
    )

    df = df.dropna(
        how="all",
        axis=1,
    )

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if "Month" not in df.columns or "Fixed" not in df.columns:
        raise ValueError(
            "Month / Fixed tilt columns could not be found."
        )

    return (
        df.set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# LOAD FIXED DATA
# ============================================================

def load_fixed_data(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df = clean_columns(df)
    df = trim_at_first_null(df, "Date")

    if "Actual" not in df.columns:
        raise ValueError(
            "Actual column not found in Fixed-C11."
        )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(drop=True)


# ============================================================
# CREATE EDITABLE INPUT DATA
# ============================================================

def create_input_editor(df_ghi, df_fix):

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    n = max(
        len(df_ghi),
        len(df_fix),
    )

    editor = pd.DataFrame(
        index=range(n)
    )

    for col in ghi_cols:

        values = (
            df_ghi[col].to_numpy()
            if col in df_ghi.columns
            else np.zeros(len(df_ghi))
        )

        editor[col] = pd.Series(
            values,
            index=range(len(values)),
        )

    actual = (
        df_fix["Actual"].to_numpy()
        if "Actual" in df_fix.columns
        else np.zeros(len(df_fix))
    )

    editor["Actual"] = pd.Series(
        actual,
        index=range(len(actual)),
    )

    editor.insert(
        0,
        "Block",
        np.arange(1, n + 1),
    )

    return editor


# ============================================================
# APPLY EDITED INPUT DATA
# ============================================================

def apply_input_editor(
    editor,
    df_ghi,
    df_fix,
):

    df_ghi = df_ghi.copy()
    df_fix = df_fix.copy()

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    ghi_length = len(df_ghi)

    for col in ghi_cols:

        if col in editor.columns:

            values = numeric(
                editor[col].iloc[:ghi_length]
            )

            df_ghi[col] = values

    actual_length = len(df_fix)

    if "Actual" in editor.columns:

        actual_values = numeric(
            editor["Actual"].iloc[:actual_length]
        )

        df_fix["Actual"] = actual_values

    return df_ghi, df_fix


# ============================================================
# PREPARE FIXED GEOMETRY
# ============================================================

def prepare_fixed_geometry(
    df_fix,
    df_ghi,
    lat,
    month_lookup,
):

    df = df_fix.copy()

    today = pd.Timestamp.today()

    df["Date"] = today

    first_date = (
        today
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    day_of_year = (
        df["Date"]
        - first_date
    ).dt.days + 1

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_of_year)
                / 365
            )
        )
    )

    df["Elevation angle a"] = (
        90
        - lat
        + df["Declination Angle ∆"]
    )

    df["Tilt Angle b"] = (
        df["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    df["a+b"] = (
        df["Elevation angle a"]
        + df["Tilt Angle b"]
    )

    df["SIN(a+b)"] = np.sin(
        np.radians(
            df["a+b"]
        )
    )

    df["Sin(a)"] = np.sin(
        np.radians(
            df["Elevation angle a"]
        )
    )

    sin_a = df["Sin(a)"].replace(
        0,
        np.nan,
    )

    ghi_map = {
        "C11": "GHI C11",
        "C12": "GHI C12",
        "C13": "GHI C13",
        "C14": "GHI C14",
        "C15": "GHI C15",
    }

    for cluster, ghi_col in ghi_map.items():

        if ghi_col not in df_ghi.columns:
            ghi = np.zeros(len(df))

        else:
            ghi = numeric(
                df_ghi[ghi_col]
            )[:len(df)]

        if len(ghi) < len(df):
            ghi = np.pad(
                ghi,
                (
                    0,
                    len(df) - len(ghi),
                ),
            )

        df[
            f"GHI*sin(a)-{cluster}"
            if cluster != "C11"
            else "GHI*sin(a)"
        ] = (
            ghi
            * df["Sin(a)"].to_numpy()
        )

        poa_name = (
            "POA fixed"
            if cluster == "C11"
            else f"POA Fixed-{cluster}"
        )

        df[
            f"GHI*sin(a+b)-{cluster}"
            if cluster != "C11"
            else "GHI*sin(a+b)"
        ] = (
            ghi
            * df["SIN(a+b)"].to_numpy()
        )

        df[poa_name] = (
            df[
                f"GHI*sin(a+b)-{cluster}"
                if cluster != "C11"
                else "GHI*sin(a+b)"
            ]
            / sin_a
        )

    return df


# ============================================================
# EFFECTIVE AREA
#
# ERROR % IS APPLIED ONLY HERE.
# ============================================================

def calculate_effective_area(
    df_original,
    df_w_original,
    error,
):

    df = df_original.copy()
    df_w = df_w_original.copy()

    df["Error %"] = float(error)

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - float(error)
    )

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        df.groupby("Clusters")["Eff Area"]
        .sum()
    )

    df_w["Eff Area(m2)"] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return df, df_w


# ============================================================
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    df_fix,
    df_w,
):

    result = df_fix.copy()

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    power_cols = []

    for i, poa_col in enumerate(poa_cols):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        area = pd.to_numeric(
            df_w.iloc[i]["Eff Area(m2)"],
            errors="coerce",
        )

        if pd.isna(area):
            area = 0.0

        result[power_col] = (
            result[poa_col]
            * float(area)
            / 1_000_000
        )

        power_cols.append(
            power_col
        )

    result[
        "Total Power (CL1+CL2+…)"
    ] = result[
        power_cols
    ].sum(axis=1)

    return result


# ============================================================
# ERROR OPTIMIZATION
# ============================================================

def optimize_error(
    df_original,
    df_w_original,
    df_fix,
):

    actual = numeric(
        df_fix["Actual"]
    )

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    best_error = 0.0
    best_peak_error = np.inf

    rows = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, df_w = calculate_effective_area(
            df_original,
            df_w_original,
            error,
        )

        calculated = calculate_fixed_power(
            df_fix,
            df_w,
        )

        forecast = numeric(
            calculated[
                "Total Power (CL1+CL2+…)"
            ]
        )

        calculated_peak = (
            forecast.max()
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
        )

        rows.append(
            {
                "Error %": round(
                    error,
                    1,
                ),
                "Calculated Peak":
                    calculated_peak,
                "Actual Peak":
                    actual_peak,
                "Peak Error":
                    peak_error,
                "Peak Error %":
                    peak_error_pct,
            }
        )

        if peak_error < best_peak_error:
            best_peak_error = peak_error
            best_error = error

    return (
        round(
            best_error,
            1,
        ),
        pd.DataFrame(rows),
    )


# ============================================================
# TRACKING DATA
# ============================================================

def load_tracking_data(
    uploaded_file,
):

    backend_list = []

    for cluster in [
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
    ]:

        backend_list.append(
            read_excel(
                uploaded_file,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    df_trac = read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac = clean_columns(
        df_trac
    )

    return backend_list, df_trac


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    cl_weights = numeric(
        df_w.iloc[:5, 1]
    )

    ghi_matrix = np.column_stack(
        [
            numeric(
                df_ghi[col]
            )
            for col in ghi_cols
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    )

    actual_full = numeric(
        df_fix["Actual"]
    )

    n = min(
        len(blocks),
        len(ghi_matrix),
        len(actual_full),
    )

    blocks = blocks[:n]
    ghi_matrix = ghi_matrix[:n]
    actual_full = actual_full[:n]

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    def objective(x):

        DHI = int(round(x[0]))
        start = int(round(x[1]))
        end = int(round(x[2]))
        maximum = int(round(x[3]))
        east = int(round(x[4]))
        west = int(round(x[5]))

        if not (
            start < maximum < end
        ):
            return 1e9

        denominator_1 = (
            start - 1 - maximum
        )

        denominator_2 = (
            end + 1 - maximum
        )

        if denominator_1 == 0 or denominator_2 == 0:
            return 1e9

        m1 = 90 / denominator_1
        m2 = 90 / denominator_2

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

        dhi = (
            ghi_matrix
            * DHI
            / 100
        )

        dni = (
            ghi_matrix - dhi
        ) / cos_alpha[:, None]

        prediction_full = (
            dni @ cl_weights
        ) / 1_000_000

        if (
            np.isnan(prediction_full).any()
            or np.isinf(prediction_full).any()
        ):
            return 1e9

        prediction = (
            prediction_full[mask]
        )

        block_error = (
            np.mean(
                np.abs(
                    actual - prediction
                )
            )
            / actual_max
        )

        peak_error = (
            abs(
                actual_max
                - prediction.max()
            )
            / actual_max
        )

        energy_error = (
            abs(
                actual_sum
                - prediction.sum()
            )
            / actual_sum
        )

        return (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

    return (
        objective,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    (
        objective,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
    ) = create_tracking_objective(
        backend_list,
        df_ghi,
        df_fix,
        df_w,
    )

    bounds = [
        (0, 10),
        (10, 30),
        (65, 80),
        (47, 53),
        (10, 70),
        (10, 70),
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=40,
        popsize=15,
        tol=0.001,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
    )

    best = np.round(
        result.x
    ).astype(int)

    parameters = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
    }

    return (
        parameters,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
        result.fun,
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    ghi_matrix,
    cl_weights,
    DHI,
    start,
    end,
    maximum,
    east,
    west,
):

    denominator_1 = (
        start - 1 - maximum
    )

    denominator_2 = (
        end + 1 - maximum
    )

    if denominator_1 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    if denominator_2 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

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

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix - dhi
    ) / cos_alpha[:, None]

    forecast = (
        dni @ cl_weights
    ) / 1_000_000

    return forecast


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = numeric(actual)
    forecast = numeric(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

    actual_peak = (
        actual.max()
        if len(actual)
        else 0
    )

    forecast_peak = (
        forecast.max()
        if len(forecast)
        else 0
    )

    peak_error_pct = (
        abs(
            forecast_peak
            - actual_peak
        )
        / actual_peak
        * 100
        if actual_peak
        else np.nan
    )

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
        "Peak Error %": peak_error_pct,
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    actual = numeric(actual)
    forecast = numeric(forecast)

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                width=2.2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2.2,
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
        ),
        height=430,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=30,
            r=20,
            t=55,
            b=30,
        ),
        xaxis_title="Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Forecast correction with automatic optimization and editable inputs"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# INPUT FILE
# ============================================================

st.markdown(
    '<div class="section-title">📁 Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)


if uploaded_file is None:

    st.info(
        "Upload the solar Excel file to begin."
    )

    st.stop()


# ============================================================
# FILE CHANGE DETECTION
# ============================================================

current_file_key = (
    uploaded_file.name,
    uploaded_file.size,
)

if (
    st.session_state.file_key
    != current_file_key
):

    st.session_state.file_key = (
        current_file_key
    )

    st.session_state.calculated = False
    st.session_state.calculation_data = None
    st.session_state.input_editor = None


# ============================================================
# LOAD RAW INPUT DATA FOR EDITOR
# ============================================================

try:

    if st.session_state.input_editor is None:

        raw_ghi = load_ghi(
            uploaded_file
        )

        raw_fix = load_fixed_data(
            uploaded_file
        )

        st.session_state.input_editor = (
            create_input_editor(
                raw_ghi,
                raw_fix,
            )
        )

except Exception as e:

    st.error(
        f"Unable to load input data: {e}"
    )

    st.stop()


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">🌱 Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.segmented_control(
    "Plant Type",
    ["Fixed", "Tracking"],
    horizontal=True,
    index=(
        0
        if st.session_state.plant_type == "Fixed"
        else 1
    ),
    label_visibility="collapsed",
)

st.session_state.plant_type = (
    plant_type
)


# ============================================================
# EDITABLE INPUT DATA
# ============================================================

st.markdown(
    '<div class="section-title">📝 Input Data</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Edit GHI and Actual values below. "
    "The edited values are used when you run the calculation."
)

edited_input = st.data_editor(
    st.session_state.input_editor,
    use_container_width=True,
    height=360,
    hide_index=True,
    disabled=["Block"],
    column_config={
        "Block": st.column_config.NumberColumn(
            "Block",
            disabled=True,
        ),
        "GHI C11": st.column_config.NumberColumn(
            "GHI C11",
            format="%.3f",
        ),
        "GHI C12": st.column_config.NumberColumn(
            "GHI C12",
            format="%.3f",
        ),
        "GHI C13": st.column_config.NumberColumn(
            "GHI C13",
            format="%.3f",
        ),
        "GHI C14": st.column_config.NumberColumn(
            "GHI C14",
            format="%.3f",
        ),
        "GHI C15": st.column_config.NumberColumn(
            "GHI C15",
            format="%.3f",
        ),
        "Actual": st.column_config.NumberColumn(
            "Actual",
            format="%.3f",
        ),
    },
    key="solar_input_editor",
)

st.session_state.input_editor = (
    edited_input
)


# ============================================================
# RUN CALCULATION
# ============================================================

st.markdown("")

run_calculation = st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
)


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if run_calculation:

    try:

        with st.spinner(
            "Running optimization and calculation..."
        ):

            # ------------------------------------------------
            # LOAD COMMON DATA
            # ------------------------------------------------

            df_original = (
                load_area_efficiency(
                    uploaded_file
                )
            )

            df_w_original = (
                load_cluster_table(
                    uploaded_file
                )
            )

            df_ghi = load_ghi(
                uploaded_file
            )

            df_fix_raw = load_fixed_data(
                uploaded_file
            )

            # ------------------------------------------------
            # APPLY USER EDITED GHI + ACTUAL
            # ------------------------------------------------

            (
                df_ghi,
                df_fix_raw,
            ) = apply_input_editor(
                edited_input,
                df_ghi,
                df_fix_raw,
            )

            # ------------------------------------------------
            # LATITUDE / TILT
            # ------------------------------------------------

            lat = load_latitude(
                uploaded_file
            )

            month_lookup = load_tilt(
                uploaded_file
            )

            # ------------------------------------------------
            # FIXED GEOMETRY
            # ------------------------------------------------

            df_fix = prepare_fixed_geometry(
                df_fix_raw,
                df_ghi,
                lat,
                month_lookup,
            )

            # ------------------------------------------------
            # ERROR OPTIMIZATION
            # ------------------------------------------------

            (
                best_error,
                error_results,
            ) = optimize_error(
                df_original,
                df_w_original,
                df_fix,
            )

            # ------------------------------------------------
            # APPLY ERROR ONCE
            # ------------------------------------------------

            (
                df_final,
                df_w_final,
            ) = calculate_effective_area(
                df_original,
                df_w_original,
                best_error,
            )

            # ------------------------------------------------
            # FIXED FORECAST
            # ------------------------------------------------

            fixed_final = calculate_fixed_power(
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # TRACKING
            # ------------------------------------------------

            (
                backend_list,
                df_trac,
            ) = load_tracking_data(
                uploaded_file
            )

            (
                tracking_parameters,
                blocks,
                ghi_matrix,
                actual_tracking,
                cl_weights,
                tracking_score,
            ) = optimize_tracking(
                backend_list,
                df_ghi,
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # TRACKING FORECAST
            # ------------------------------------------------

            tracking_forecast = (
                calculate_tracking_forecast(
                    blocks,
                    ghi_matrix,
                    cl_weights,
                    tracking_parameters["DHI"],
                    tracking_parameters[
                        "GHI Starting Block"
                    ],
                    tracking_parameters[
                        "GHI Ending Block"
                    ],
                    tracking_parameters[
                        "GHI Max Block"
                    ],
                    tracking_parameters[
                        "Tracking East Limit"
                    ],
                    tracking_parameters[
                        "Tracking West Limit"
                    ],
                )
            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            st.session_state.calculation_data = {

                "df_original":
                    df_original,

                "df_w_original":
                    df_w_original,

                "df_final":
                    df_final,

                "df_w_final":
                    df_w_final,

                "df_ghi":
                    df_ghi,

                "df_fix":
                    df_fix,

                "fixed_final":
                    fixed_final,

                "backend_list":
                    backend_list,

                "df_trac":
                    df_trac,

                "blocks":
                    blocks,

                "ghi_matrix":
                    ghi_matrix,

                "actual_tracking":
                    actual_tracking,

                "cl_weights":
                    cl_weights,

                "best_error":
                    best_error,

                "tracking_parameters":
                    tracking_parameters,

                "tracking_score":
                    tracking_score,

                "tracking_forecast":
                    tracking_forecast,

                "lat":
                    lat,

                "month_lookup":
                    month_lookup,

                "error_results":
                    error_results,
            }

            st.session_state.calculated = True

        st.success(
            "Calculation completed successfully."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# STOP UNTIL CALCULATION
# ============================================================

if not st.session_state.calculated:

    st.info(
        "Edit the input data if required, then click "
        "**Run Automatic Calculation**."
    )

    st.stop()


# ============================================================
# STORED DATA
# ============================================================

data = (
    st.session_state.calculation_data
)


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ERROR %
# ============================================================

if "error_value" not in st.session_state:

    st.session_state.error_value = float(
        data["best_error"]
    )

error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(
        st.session_state.error_value
    ),
    step=0.1,
    format="%.1f",
)

st.session_state.error_value = (
    error_value
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                params["DHI"]
            ),
            step=1,
            key="dhi_parameter",
        )

        start_value = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                params[
                    "GHI Starting Block"
                ]
            ),
            step=1,
            key="start_parameter",
        )

    with c2:

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                params[
                    "GHI Ending Block"
                ]
            ),
            step=1,
            key="end_parameter",
        )

        max_value = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                params[
                    "GHI Max Block"
                ]
            ),
            step=1,
            key="max_parameter",
        )

    with c3:

        east_value = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                params[
                    "Tracking East Limit"
                ]
            ),
            step=1,
            key="east_parameter",
        )

        west_value = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                params[
                    "Tracking West Limit"
                ]
            ),
            step=1,
            key="west_parameter",
        )

else:

    dhi_value = None
    start_value = None
    end_value = None
    max_value = None
    east_value = None
    west_value = None


# ============================================================
# RECALCULATE USING EDITABLE PARAMETERS
#
# IMPORTANT:
# Differential Evolution is NOT called again here.
# ============================================================

try:

    (
        df_final,
        df_w_final,
    ) = calculate_effective_area(
        data["df_original"],
        data["df_w_original"],
        error_value,
    )

    fixed_final = calculate_fixed_power(
        data["df_fix"],
        df_w_final,
    )

    tracking_forecast = data[
        "tracking_forecast"
    ]

    if plant_type == "Tracking":

        cl_weights = numeric(
            df_w_final.iloc[:5, 1]
        )

        tracking_forecast = (
            calculate_tracking_forecast(
                data["blocks"],
                data["ghi_matrix"],
                cl_weights,
                int(dhi_value),
                int(start_value),
                int(end_value),
                int(max_value),
                int(east_value),
                int(west_value),
            )
        )

except Exception as e:

    st.error(
        f"Parameter calculation failed: {e}"
    )

    st.stop()


# ============================================================
# FINAL FORECAST
# ============================================================

if plant_type == "Fixed":

    actual = numeric(
        data["df_fix"]["Actual"]
    )

    forecast = numeric(
        fixed_final[
            "Total Power (CL1+CL2+…)"
        ]
    )

    title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    actual = numeric(
        data["actual_tracking"]
    )

    forecast = numeric(
        tracking_forecast
    )

    title = (
        "Tracking Plant | Actual vs Forecast"
    )


# ============================================================
# METRICS
# ============================================================

metrics = calculate_metrics(
    actual,
    forecast,
)


# ============================================================
# RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2 = st.columns(2)

with m1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Actual Peak
            </div>
            <div class="metric-value">
                {metrics["Actual Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Forecast Peak
            </div>
            <div class="metric-value">
                {metrics["Forecast Peak"]:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORECAST GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    build_graph(
        actual,
        forecast,
        title,
    ),
    use_container_width=True,
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)


# ============================================================
# FINAL STATE
# ============================================================

data["df_final"] = df_final
data["df_w_final"] = df_w_final
data["fixed_final"] = fixed_final

if plant_type == "Tracking":

    data["tracking_forecast"] = (
        tracking_forecast
    )

    data["tracking_parameters"] = {
        "DHI": int(dhi_value),
        "GHI Starting Block":
            int(start_value),
        "GHI Ending Block":
            int(end_value),
        "GHI Max Block":
            int(max_value),
        "Tracking East Limit":
            int(east_value),
        "Tracking West Limit":
            int(west_value),
    }

st.session_state.calculation_data = data
