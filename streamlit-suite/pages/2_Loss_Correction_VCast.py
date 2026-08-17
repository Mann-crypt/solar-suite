# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# STREAMLIT SINGLE PAGE APPLICATION
# ============================================================

import io
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")


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
# CONSTANTS
# ============================================================

REQUIRED_SHEETS_COMMON = [
    "Area & Efficiency",
    "Forecast Config",
    "Config Tilt Angle",
    "Result",
]

REQUIRED_FIXED_SHEETS = [
    "Fixed-C11",
]

REQUIRED_TRACKING_SHEETS = [
    "Backend Cal C11",
    "Backend Cal C12",
    "Backend Cal C13",
    "Backend Cal C14",
    "Backend Cal C15",
    "Tracking",
]


CLUSTERS = ["C11", "C12", "C13", "C14", "C15"]

GHI_COLUMNS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

POWER_COLUMNS = [
    "CL1 Power",
    "CL2 Power",
    "CL3 Power",
    "CL4 Power",
    "CL5 Power",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #666;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background: #fafafa;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e1e1e1;
        padding: 10px;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "Fixed and Tracking plant forecast correction using efficiency "
    "loss and solar geometry optimization."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Input & Plant Configuration</div>',
    unsafe_allow_html=True,
)

input_col1, input_col2 = st.columns([2, 1])

with input_col1:

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=["xlsx", "xls"],
        help="Upload the solar forecasting workbook.",
    )

with input_col2:

    plant_type = st.segmented_control(
        "Plant Type",
        options=["Fixed", "Tracking"],
        default="Fixed",
    )


if uploaded_file is None:

    st.info(
        "Upload the Excel workbook above to start the calculation."
    )

    st.stop()


if plant_type is None:
    plant_type = "Fixed"


# ============================================================
# READ WORKBOOK
# ============================================================

try:

    excel_file = pd.ExcelFile(uploaded_file)

    available_sheets = excel_file.sheet_names

except Exception as e:

    st.error(f"Unable to read workbook: {e}")
    st.stop()


# ============================================================
# SHEET VALIDATION
# ============================================================

required_sheets = REQUIRED_SHEETS_COMMON.copy()

if plant_type == "Fixed":
    required_sheets += REQUIRED_FIXED_SHEETS

else:
    required_sheets += REQUIRED_TRACKING_SHEETS


missing_sheets = [
    sheet
    for sheet in required_sheets
    if sheet not in available_sheets
]


if missing_sheets:

    st.error("The following required sheets are missing:")

    for sheet in missing_sheets:
        st.write(f"• `{sheet}`")

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    """Clean Excel column names."""

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    return df


def remove_empty_rows(df, key_column=None):
    """Remove rows after the first empty key column."""

    df = df.copy()

    if key_column and key_column in df.columns:

        mask = df[key_column].isna()

        if mask.any():

            first_empty = np.where(mask.to_numpy())[0][0]

            df = df.iloc[:first_empty]

    return df.reset_index(drop=True)


def safe_numeric(series):
    """Convert a series to numeric."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def get_required_column(df, possible_names):
    """Find a column from a list of possible names."""

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def align_length(*arrays):
    """Return arrays trimmed to the same minimum length."""

    lengths = [
        len(arr)
        for arr in arrays
    ]

    n = min(lengths)

    return [
        np.asarray(arr)[:n]
        for arr in arrays
    ]


def calculate_metrics(actual, forecast):

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    n = min(
        len(actual),
        len(forecast),
    )

    actual = actual[:n]
    forecast = forecast[:n]

    valid = (
        np.isfinite(actual)
        & np.isfinite(forecast)
    )

    actual = actual[valid]
    forecast = forecast[valid]

    if len(actual) == 0:

        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "Peak Error": np.nan,
            "Peak Error %": np.nan,
            "Energy Error %": np.nan,
            "R²": np.nan,
        }

    error = actual - forecast

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    actual_peak = np.max(actual)
    forecast_peak = np.max(forecast)

    peak_error = abs(
        forecast_peak - actual_peak
    )

    if actual_peak != 0:

        peak_error_pct = (
            peak_error
            / abs(actual_peak)
            * 100
        )

    else:

        peak_error_pct = np.nan

    actual_energy = np.sum(
        np.maximum(actual, 0)
    )

    forecast_energy = np.sum(
        np.maximum(forecast, 0)
    )

    if actual_energy != 0:

        energy_error_pct = (
            abs(
                forecast_energy
                - actual_energy
            )
            / actual_energy
            * 100
        )

    else:

        energy_error_pct = np.nan

    if len(actual) > 1:

        ss_res = np.sum(
            (actual - forecast) ** 2
        )

        ss_tot = np.sum(
            (actual - np.mean(actual)) ** 2
        )

        if ss_tot != 0:

            r2 = 1 - (
                ss_res / ss_tot
            )

        else:

            r2 = np.nan

    else:

        r2 = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Peak Error": peak_error,
        "Peak Error %": peak_error_pct,
        "Energy Error %": energy_error_pct,
        "R²": r2,
    }


# ============================================================
# LOAD AREA & EFFICIENCY
# ============================================================

@st.cache_data(show_spinner=False)
def load_area_efficiency(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = clean_columns(df)

    df = remove_empty_rows(
        df,
        "S.No.",
    )

    return df


@st.cache_data(show_spinner=False)
def load_cluster_table(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df = clean_columns(df)

    cluster_col = get_required_column(
        df,
        ["Clusters", "Cluster"],
    )

    if cluster_col is None:

        raise ValueError(
            "Cluster column was not found in "
            "'Area & Efficiency'."
        )

    df = remove_empty_rows(
        df,
        cluster_col,
    )

    df = df.iloc[:5].copy()

    return df


# ============================================================
# LOAD DATA
# ============================================================

file_bytes = uploaded_file.getvalue()


try:

    df_area = load_area_efficiency(
        file_bytes
    )

    df_cluster = load_cluster_table(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Error loading Area & Efficiency data: {e}"
    )

    st.stop()


# ============================================================
# CHECK AREA COLUMNS
# ============================================================

required_area_columns = [
    "Standard PV Efficiency (%)",
    "Error %",
    "Total area (m2)",
    "Clusters",
]


missing_area_columns = [
    col
    for col in required_area_columns
    if col not in df_area.columns
]


if missing_area_columns:

    st.error(
        "Missing columns in Area & Efficiency:"
    )

    for col in missing_area_columns:
        st.write(f"• `{col}`")

    st.stop()


# ============================================================
# COMMON CONFIGURATION
# ============================================================

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:

    auto_optimize_error = st.checkbox(
        "Automatically optimize Error %",
        value=True,
    )

with config_col2:

    error_min = st.number_input(
        "Error % Minimum",
        min_value=-20.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
    )

with config_col3:

    error_max = st.number_input(
        "Error % Maximum",
        min_value=-20.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
    )


error_step = st.number_input(
    "Error % Step",
    min_value=0.01,
    max_value=5.0,
    value=0.1,
    step=0.01,
)


if error_max < error_min:

    st.error(
        "Error % Maximum must be greater than or equal to Minimum."
    )

    st.stop()


# ============================================================
# AREA CALCULATION
# ============================================================

def calculate_effective_area(
    df,
    error_percent,
):

    data = df.copy()

    data["Standard PV Efficiency (%)"] = safe_numeric(
        data["Standard PV Efficiency (%)"]
    )

    data["No of Module"] = safe_numeric(
        data["No of Module"]
    )

    data["Area of 1 Module (m2)"] = safe_numeric(
        data["Area of 1 Module (m2)"]
    )

    data["Error %"] = error_percent

    data["Net Efficiency (%)"] = (
        data["Standard PV Efficiency (%)"]
        - error_percent
    )

    data["Total area (m2)"] = (
        data["No of Module"]
        * data["Area of 1 Module (m2)"]
    )

    data["Eff Area"] = (
        data["Net Efficiency (%)"]
        * data["Total area (m2)"]
        / 100
    )

    cluster_area = (
        data.groupby("Clusters")["Eff Area"]
        .sum()
    )

    cluster_df = df_cluster.copy()

    cluster_column = get_required_column(
        cluster_df,
        ["Clusters", "Cluster"],
    )

    cluster_df["Eff Area(m2)"] = (
        cluster_df[cluster_column]
        .map(cluster_area)
        .fillna(0)
    )

    return data, cluster_df


# ============================================================
# LOAD GHI
# ============================================================

@st.cache_data(show_spinner=False)
def load_ghi(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df = clean_columns(df)

    df = df.fillna(0)

    return df


try:

    df_ghi = load_ghi(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Error reading Result sheet: {e}"
    )

    st.stop()


missing_ghi = [
    col
    for col in GHI_COLUMNS
    if col not in df_ghi.columns
]


if missing_ghi:

    st.error(
        "Missing GHI columns:"
    )

    for col in missing_ghi:
        st.write(f"• `{col}`")

    st.stop()


# ============================================================
# LOAD LATITUDE
# ============================================================

@st.cache_data(show_spinner=False)
def load_latitude(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    df = clean_columns(df)

    if "Lat" not in df.columns:

        raise ValueError(
            "Lat column not found in Forecast Config."
        )

    lat = pd.to_numeric(
        df.loc[0, "Lat"],
        errors="coerce",
    )

    if pd.isna(lat):

        raise ValueError(
            "Latitude is not a valid number."
        )

    return float(lat)


try:

    latitude = load_latitude(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Error reading latitude: {e}"
    )

    st.stop()


# ============================================================
# LOAD TILT
# ============================================================

@st.cache_data(show_spinner=False)
def load_tilt(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df = clean_columns(df)

    if "Fixed" not in df.columns:

        raise ValueError(
            "'Fixed' column not found in Config Tilt Angle."
        )

    df = remove_empty_rows(
        df,
        "Fixed",
    )

    month_col = None

    for col in ["Month", "Months"]:

        if col in df.columns:

            month_col = col
            break

    if month_col is None:

        possible_month_columns = [
            col
            for col in df.columns
            if "month" in col.lower()
        ]

        if possible_month_columns:

            month_col = possible_month_columns[0]

    if month_col is None:

        raise ValueError(
            "Month column not found in Config Tilt Angle."
        )

    lookup = {}

    for _, row in df.iterrows():

        month = str(
            row[month_col]
        ).strip()

        tilt = pd.to_numeric(
            row["Fixed"],
            errors="coerce",
        )

        if pd.notna(tilt):

            lookup[month] = float(tilt)

    return lookup


try:

    month_lookup = load_tilt(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Error reading tilt angle configuration: {e}"
    )

    st.stop()


# ============================================================
# LOAD ACTUAL
# ============================================================

@st.cache_data(show_spinner=False)
def load_actual(file_bytes):

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
    )

    df = clean_columns(df)

    df = remove_empty_rows(
        df,
        "Date",
    )

    if "Actual" not in df.columns:

        raise ValueError(
            "'Actual' column not found in Fixed-C11."
        )

    actual = safe_numeric(
        df["Actual"]
    ).fillna(0)

    if "Date" in df.columns:

        dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
        )

    else:

        dates = pd.Series(
            pd.Timestamp.today(),
            index=df.index,
        )

    return (
        dates.reset_index(drop=True),
        actual.reset_index(drop=True),
    )


try:

    dates, actual_series = load_actual(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Error reading Actual data: {e}"
    )

    st.stop()


# ============================================================
# PREPARE SOLAR GEOMETRY
# ============================================================

def prepare_solar_geometry(
    dates,
    latitude,
    month_lookup,
):

    dates = pd.to_datetime(
        dates,
        errors="coerce",
    )

    if dates.isna().all():

        dates = pd.Series(
            pd.Timestamp.today(),
            index=np.arange(len(dates)),
        )

    dates = dates.ffill().bfill()

    day_of_year = (
        dates.dt.dayofyear
    )

    declination = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_of_year
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

    month_names = dates.dt.month_name()

    tilt = month_names.map(
        month_lookup
    )

    tilt = pd.to_numeric(
        tilt,
        errors="coerce",
    )

    tilt = tilt.fillna(
        tilt.median()
        if tilt.notna().any()
        else 0
    )

    a_plus_b = (
        elevation
        + tilt
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    sin_a_plus_b = np.sin(
        np.radians(a_plus_b)
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Declination Angle ∆": declination,
            "Elevation angle a": elevation,
            "Tilt Angle b": tilt,
            "a+b": a_plus_b,
            "SIN(a+b)": sin_a_plus_b,
            "Sin(a)": sin_a,
        }
    )


geometry = prepare_solar_geometry(
    dates,
    latitude,
    month_lookup,
)


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    ghi_df,
    geometry,
    cluster_df,
):

    n = min(
        len(ghi_df),
        len(geometry),
    )

    ghi_df = ghi_df.iloc[:n].copy()
    geometry = geometry.iloc[:n].copy()

    sin_a = geometry["Sin(a)"].to_numpy(
        dtype=float
    )

    sin_a_plus_b = geometry[
        "SIN(a+b)"
    ].to_numpy(
        dtype=float
    )

    safe_sin_a = np.where(
        np.abs(sin_a) < 1e-6,
        np.nan,
        sin_a,
    )

    forecast_clusters = {}

    for i, ghi_col in enumerate(
        GHI_COLUMNS
    ):

        cluster_name = CLUSTERS[i]

        ghi = safe_numeric(
            ghi_df[ghi_col]
        ).fillna(0).to_numpy()

        poa = (
            ghi
            * sin_a_plus_b
            / safe_sin_a
        )

        poa = np.nan_to_num(
            poa,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        area = float(
            cluster_df.iloc[i]["Eff Area(m2)"]
        )

        power = (
            poa
            * area
            / 1_000_000
        )

        forecast_clusters[
            cluster_name
        ] = power

    forecast_df = pd.DataFrame(
        forecast_clusters
    )

    forecast_df["Total Forecast"] = (
        forecast_df.sum(axis=1)
    )

    return forecast_df


# ============================================================
# ERROR OPTIMIZATION
# ============================================================

def optimize_error_fixed(
    df_area,
    ghi_df,
    geometry,
    cluster_df,
    actual,
    error_min,
    error_max,
    error_step,
):

    errors = np.arange(
        error_min,
        error_max + error_step / 2,
        error_step,
    )

    results = []

    for error in errors:

        _, current_cluster = (
            calculate_effective_area(
                df_area,
                error,
            )
        )

        forecast_df = (
            calculate_fixed_forecast(
                ghi_df,
                geometry,
                current_cluster,
            )
        )

        forecast = (
            forecast_df["Total Forecast"]
            .to_numpy()
        )

        actual_arr, forecast = align_length(
            actual.to_numpy(),
            forecast,
        )

        metrics = calculate_metrics(
            actual_arr,
            forecast,
        )

        results.append(
            {
                "Error %": error,
                "Calculated Peak": forecast.max()
                if len(forecast)
                else np.nan,
                "Actual Peak": actual_arr.max()
                if len(actual_arr)
                else np.nan,
                "Peak Error": metrics[
                    "Peak Error"
                ],
                "Peak Error %": metrics[
                    "Peak Error %"
                ],
            }
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "Peak Error"
    )

    best_error = float(
        result_df.iloc[0]["Error %"]
    )

    return best_error, result_df


# ============================================================
# TRACKING DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_tracking_data(file_bytes):

    backend = []

    for cluster in CLUSTERS:

        sheet = f"Backend Cal {cluster}"

        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=sheet,
        )

        df = clean_columns(df)

        backend.append(df)

    tracking = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Tracking",
        header=1,
    )

    tracking = clean_columns(
        tracking
    )

    return backend, tracking


# ============================================================
# TRACKING CALCULATION
# ============================================================

def calculate_tracking_forecast(
    ghi_matrix,
    blocks,
    weights,
    DHI,
    start_block,
    end_block,
    max_block,
    east_limit,
    west_limit,
):

    blocks = np.asarray(
        blocks,
        dtype=float,
    )

    if not (
        start_block
        < max_block
        < end_block
    ):

        raise ValueError(
            "GHI Starting Block must be less than "
            "GHI Max Block and GHI Max Block must "
            "be less than GHI Ending Block."
        )

    denominator_1 = (
        start_block
        - 1
        - max_block
    )

    denominator_2 = (
        end_block
        + 1
        - max_block
    )

    if abs(denominator_1) < 1e-9:
        raise ValueError(
            "Invalid GHI Starting Block / Max Block combination."
        )

    if abs(denominator_2) < 1e-9:
        raise ValueError(
            "Invalid GHI Ending Block / Max Block combination."
        )

    m1 = 90 / denominator_1

    m2 = 90 / denominator_2

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1 * (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89,
            m2 * (
                blocks
                - max_block
            ),
        ),
    )

    zenith = np.clip(
        zenith,
        0,
        89,
    )

    panel = np.where(
        blocks < max_block,

        np.minimum(
            zenith,
            abs(east_limit),
        ),

        np.where(
            (
                (blocks > max_block)
                & (
                    zenith
                    > west_limit
                )
            ),

            west_limit,

            zenith,
        ),
    )

    panel = np.clip(
        panel,
        0,
        89,
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    dhi = (
        ghi_matrix
        * DHI
        / 100
    )

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    dni = np.maximum(
        dni,
        0,
    )

    forecast = (
        dni @ weights
    ) / 1_000_000

    forecast = np.nan_to_num(
        forecast,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def tracking_objective(
    x,
    ghi_matrix,
    blocks,
    weights,
    actual,
    daylight_threshold,
):

    DHI = int(
        round(x[0])
    )

    start_block = int(
        round(x[1])
    )

    end_block = int(
        round(x[2])
    )

    max_block = int(
        round(x[3])
    )

    east_limit = int(
        round(x[4])
    )

    west_limit = int(
        round(x[5])
    )

    try:

        forecast, _, _ = (
            calculate_tracking_forecast(
                ghi_matrix,
                blocks,
                weights,
                DHI,
                start_block,
                end_block,
                max_block,
                east_limit,
                west_limit,
            )
        )

    except Exception:

        return 1e9

    actual, forecast = align_length(
        actual,
        forecast,
    )

    ghi_reference = (
        np.max(
            ghi_matrix,
            axis=1,
        )
    )

    ghi_reference = ghi_reference[
        :len(actual)
    ]

    mask = (
        ghi_reference
        > daylight_threshold
    ) & (
        actual
        > 0
    )

    if mask.sum() < 3:

        mask = actual > 0

    if mask.sum() < 3:

        return 1e9

    actual_day = actual[mask]
    forecast_day = forecast[mask]

    actual_max = np.max(
        actual_day
    )

    actual_sum = np.sum(
        actual_day
    )

    if actual_max <= 0:
        return 1e9

    if actual_sum <= 0:
        return 1e9

    block_error = (
        np.mean(
            np.abs(
                actual_day
                - forecast_day
            )
        )
        / actual_max
    )

    peak_error = (
        abs(
            actual_max
            - np.max(forecast_day)
        )
        / actual_max
    )

    energy_error = (
        abs(
            actual_sum
            - np.sum(forecast_day)
        )
        / actual_sum
    )

    score = (
        0.80 * block_error
        + 0.10 * peak_error
        + 0.10 * energy_error
    )

    return float(score)


# ============================================================
# MAIN CALCULATION
# ============================================================

run_button = st.button(
    "🚀 Run Forecast Correction",
    type="primary",
    use_container_width=True,
)


if not run_button:

    st.info(
        "Configure the parameters above and click "
        "**Run Forecast Correction**."
    )

    st.stop()


# ============================================================
# EFFECTIVE AREA / ERROR
# ============================================================

with st.spinner(
    "Preparing plant efficiency and effective area..."
):

    try:

        # ----------------------------------------------------
        # Manual Error %
        # ----------------------------------------------------

        if auto_optimize_error:

            with st.status(
                "Optimizing Error %...",
                expanded=False,
            ):

                best_error, error_results = (
                    optimize_error_fixed(
                        df_area,
                        df_ghi,
                        geometry,
                        df_cluster,
                        actual_series,
                        error_min,
                        error_max,
                        error_step,
                    )
                )

        else:

            best_error = st.number_input(
                "Manual Error %",
                min_value=-20.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
            )

            error_results = pd.DataFrame()

        df_final, cluster_final = (
            calculate_effective_area(
                df_area,
                best_error,
            )
        )

    except Exception as e:

        st.error(
            f"Error during efficiency calculation: {e}"
        )

        st.stop()


# ============================================================
# RESULTS HEADER
# ============================================================

st.markdown(
    '<div class="section-title">2. Optimization Result</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

tracking_parameters = None


if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">'
        "Tracking Parameters"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "These parameters are editable. "
        "You can either optimize them automatically "
        "or enter the values manually."
    )

    tracking_mode = st.segmented_control(
        "Tracking Parameter Mode",
        options=[
            "Optimize Automatically",
            "Manual",
        ],
        default="Optimize Automatically",
    )

    if tracking_mode == "Manual":

        p1, p2, p3 = st.columns(3)
        p4, p5, p6 = st.columns(3)

        with p1:

            DHI = st.number_input(
                "DHI (%)",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
            )

        with p2:

            start_block = st.number_input(
                "GHI Starting Block",
                min_value=1,
                max_value=95,
                value=20,
                step=1,
            )

        with p3:

            end_block = st.number_input(
                "GHI Ending Block",
                min_value=1,
                max_value=95,
                value=75,
                step=1,
            )

        with p4:

            max_block = st.number_input(
                "GHI Max Block",
                min_value=1,
                max_value=95,
                value=48,
                step=1,
            )

        with p5:

            east_limit = st.number_input(
                "Tracking East Limit",
                min_value=0,
                max_value=89,
                value=45,
                step=1,
            )

        with p6:

            west_limit = st.number_input(
                "Tracking West Limit",
                min_value=0,
                max_value=89,
                value=45,
                step=1,
            )

        if not (
            start_block
            < max_block
            < end_block
        ):

            st.error(
                "Parameter rule: "
                "Starting Block < Max Block < Ending Block"
            )

            st.stop()

        tracking_parameters = {
            "DHI": int(DHI),
            "GHI Starting Block": int(start_block),
            "GHI Ending Block": int(end_block),
            "GHI Max Block": int(max_block),
            "Tracking East Limit": int(east_limit),
            "Tracking West Limit": int(west_limit),
        }

    else:

        opt1, opt2, opt3 = st.columns(3)

        with opt1:

            dhi_min = st.number_input(
                "DHI Minimum",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )

            dhi_max = st.number_input(
                "DHI Maximum",
                min_value=0,
                max_value=100,
                value=10,
                step=1,
            )

        with opt2:

            start_min = st.number_input(
                "Starting Block Minimum",
                min_value=1,
                max_value=95,
                value=10,
                step=1,
            )

            start_max = st.number_input(
                "Starting Block Maximum",
                min_value=1,
                max_value=95,
                value=30,
                step=1,
            )

        with opt3:

            end_min = st.number_input(
                "Ending Block Minimum",
                min_value=1,
                max_value=95,
                value=65,
                step=1,
            )

            end_max = st.number_input(
                "Ending Block Maximum",
                min_value=1,
                max_value=95,
                value=80,
                step=1,
            )

        opt4, opt5, opt6 = st.columns(3)

        with opt4:

            max_min = st.number_input(
                "Max Block Minimum",
                min_value=1,
                max_value=95,
                value=47,
                step=1,
            )

            max_max = st.number_input(
                "Max Block Maximum",
                min_value=1,
                max_value=95,
                value=53,
                step=1,
            )

        with opt5:

            east_min = st.number_input(
                "East Limit Minimum",
                min_value=0,
                max_value=89,
                value=10,
                step=1,
            )

            east_max = st.number_input(
                "East Limit Maximum",
                min_value=0,
                max_value=89,
                value=70,
                step=1,
            )

        with opt6:

            west_min = st.number_input(
                "West Limit Minimum",
                min_value=0,
                max_value=89,
                value=10,
                step=1,
            )

            west_max = st.number_input(
                "West Limit Maximum",
                min_value=0,
                max_value=89,
                value=70,
                step=1,
            )

        if not (
            dhi_min <= dhi_max
            and start_min <= start_max
            and end_min <= end_max
            and max_min <= max_max
            and east_min <= east_max
            and west_min <= west_max
        ):

            st.error(
                "Optimization minimum values must be "
                "less than or equal to maximum values."
            )

            st.stop()


# ============================================================
# FIXED FORECAST
# ============================================================

if plant_type == "Fixed":

    with st.spinner(
        "Calculating Fixed plant forecast..."
    ):

        try:

            forecast_df = (
                calculate_fixed_forecast(
                    df_ghi,
                    geometry,
                    cluster_final,
                )
            )

        except Exception as e:

            st.error(
                f"Fixed forecast calculation failed: {e}"
            )

            st.stop()

    tracking_parameters = None


# ============================================================
# TRACKING FORECAST
# ============================================================

else:

    try:

        backend_list, df_tracking = (
            load_tracking_data(
                file_bytes
            )
        )

    except Exception as e:

        st.error(
            f"Tracking data could not be loaded: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Cluster weights
    # --------------------------------------------------------

    try:

        weight_column = (
            cluster_final.columns[-1]
        )

        weights = pd.to_numeric(
            cluster_final[
                "Eff Area(m2)"
            ],
            errors="coerce",
        ).fillna(0).to_numpy(
            dtype=float
        )

        if len(weights) < 5:

            raise ValueError(
                "Less than 5 cluster effective-area values found."
            )

        weights = weights[:5]

    except Exception as e:

        st.error(
            f"Unable to prepare cluster weights: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    block_column = None

    for backend in backend_list:

        if "Block No." in backend.columns:

            block_column = "Block No."
            break

    if block_column is None:

        st.error(
            "'Block No.' column was not found "
            "in Backend Cal sheets."
        )

        st.stop()

    blocks = safe_numeric(
        backend_list[0][
            block_column
        ]
    ).fillna(
        np.arange(
            len(
                backend_list[0]
            )
        )
    ).to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # GHI Matrix
    # --------------------------------------------------------

    try:

        ghi_matrix = np.column_stack(
            [
                safe_numeric(
                    df_ghi[col]
                ).fillna(0).to_numpy(
                    dtype=float
                )
                for col in GHI_COLUMNS
            ]
        )

    except Exception as e:

        st.error(
            f"Unable to create GHI matrix: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Align everything
    # --------------------------------------------------------

    n = min(
        len(blocks),
        len(ghi_matrix),
        len(actual_series),
    )

    blocks = blocks[:n]

    ghi_matrix = ghi_matrix[:n]

    actual_array = (
        actual_series
        .to_numpy(
            dtype=float
        )[:n]
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    if tracking_mode == "Optimize Automatically":

        bounds = [
            (
                dhi_min,
                dhi_max,
            ),
            (
                start_min,
                start_max,
            ),
            (
                end_min,
                end_max,
            ),
            (
                max_min,
                max_max,
            ),
            (
                east_min,
                east_max,
            ),
            (
                west_min,
                west_max,
            ),
        ]

        optimization_col1, optimization_col2 = (
            st.columns(2)
        )

        with optimization_col1:

            max_iterations = st.number_input(
                "Optimization Iterations",
                min_value=5,
                max_value=200,
                value=40,
                step=5,
            )

        with optimization_col2:

            population_size = st.number_input(
                "Population Size",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
            )

        daylight_threshold = st.number_input(
            "Daylight GHI Threshold",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
        )

        with st.spinner(
            "Optimizing tracking parameters..."
        ):

            try:

                result = differential_evolution(
                    tracking_objective,
                    bounds=bounds,
                    args=(
                        ghi_matrix,
                        blocks,
                        weights,
                        actual_array,
                        daylight_threshold,
                    ),
                    strategy="best1bin",
                    maxiter=int(
                        max_iterations
                    ),
                    popsize=int(
                        population_size
                    ),
                    tol=0.001,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=42,
                    polish=True,
                    workers=1,
                )

            except Exception as e:

                st.error(
                    f"Tracking optimization failed: {e}"
                )

                st.stop()

        best_tracking = np.round(
            result.x
        ).astype(int)

        tracking_parameters = {
            "DHI": int(
                best_tracking[0]
            ),
            "GHI Starting Block": int(
                best_tracking[1]
            ),
            "GHI Ending Block": int(
                best_tracking[2]
            ),
            "GHI Max Block": int(
                best_tracking[3]
            ),
            "Tracking East Limit": int(
                best_tracking[4]
            ),
            "Tracking West Limit": int(
                best_tracking[5]
            ),
        }

        st.success(
            "Tracking optimization completed."
        )

    else:

        daylight_threshold = st.number_input(
            "Daylight GHI Threshold",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
        )

    # --------------------------------------------------------
    # Final Tracking Forecast
    # --------------------------------------------------------

    try:

        forecast, zenith, panel = (
            calculate_tracking_forecast(
                ghi_matrix,
                blocks,
                weights,
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

    except Exception as e:

        st.error(
            f"Final tracking calculation failed: {e}"
        )

        st.stop()

    forecast_df = pd.DataFrame(
        {
            "C11": (
                ghi_matrix[:, 0]
                * (
                    1
                    - tracking_parameters[
                        "DHI"
                    ] / 100
                )
                / np.maximum(
                    np.cos(
                        np.radians(
                            panel
                        )
                    ),
                    1e-6,
                )
                * weights[0]
                / 1_000_000
            ),

            "C12": (
                ghi_matrix[:, 1]
                * (
                    1
                    - tracking_parameters[
                        "DHI"
                    ] / 100
                )
                / np.maximum(
                    np.cos(
                        np.radians(
                            panel
                        )
                    ),
                    1e-6,
                )
                * weights[1]
                / 1_000_000
            ),

            "C13": (
                ghi_matrix[:, 2]
                * (
                    1
                    - tracking_parameters[
                        "DHI"
                    ] / 100
                )
                / np.maximum(
                    np.cos(
                        np.radians(
                            panel
                        )
                    ),
                    1e-6,
                )
                * weights[2]
                / 1_000_000
            ),

            "C14": (
                ghi_matrix[:, 3]
                * (
                    1
                    - tracking_parameters[
                        "DHI"
                    ] / 100
                )
                / np.maximum(
                    np.cos(
                        np.radians(
                            panel
                        )
                    ),
                    1e-6,
                )
                * weights[3]
                / 1_000_000
            ),

            "C15": (
                ghi_matrix[:, 4]
                * (
                    1
                    - tracking_parameters[
                        "DHI"
                    ] / 100
                )
                / np.maximum(
                    np.cos(
                        np.radians(
                            panel
                        )
                    ),
                    1e-6,
                )
                * weights[4]
                / 1_000_000
            ),
        }
    )

    forecast_df["Total Forecast"] = (
        forecast
    )


# ============================================================
# FINAL METRICS
# ============================================================

actual_final, forecast_final = (
    align_length(
        actual_series.to_numpy(
            dtype=float
        ),
        forecast_df[
            "Total Forecast"
        ].to_numpy(
            dtype=float
        ),
    )
)

metrics = calculate_metrics(
    actual_final,
    forecast_final,
)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "3. Final Performance"
    "</div>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:

    st.metric(
        "Error %",
        f"{best_error:.2f}%",
    )

with k2:

    st.metric(
        "Actual Peak",
        f"{np.max(actual_final):,.3f}",
    )

with k3:

    st.metric(
        "Forecast Peak",
        f"{np.max(forecast_final):,.3f}",
    )

with k4:

    st.metric(
        "Peak Error %",
        f"{metrics['Peak Error %']:.2f}%",
    )

with k5:

    st.metric(
        "Energy Error %",
        f"{metrics['Energy Error %']:.2f}%",
    )

with k6:

    r2_value = metrics["R²"]

    if pd.notna(r2_value):

        r2_text = f"{r2_value:.4f}"

    else:

        r2_text = "N/A"

    st.metric(
        "R²",
        r2_text,
    )


# ============================================================
# TRACKING PARAMETER RESULTS
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">'
        "4. Final Tracking Parameters"
        "</div>",
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)
    p4, p5, p6 = st.columns(3)

    with p1:

        st.metric(
            "DHI",
            tracking_parameters["DHI"],
        )

    with p2:

        st.metric(
            "GHI Starting Block",
            tracking_parameters[
                "GHI Starting Block"
            ],
        )

    with p3:

        st.metric(
            "GHI Ending Block",
            tracking_parameters[
                "GHI Ending Block"
            ],
        )

    with p4:

        st.metric(
            "GHI Max Block",
            tracking_parameters[
                "GHI Max Block"
            ],
        )

    with p5:

        st.metric(
            "East Tracking Limit",
            tracking_parameters[
                "Tracking East Limit"
            ],
        )

    with p6:

        st.metric(
            "West Tracking Limit",
            tracking_parameters[
                "Tracking West Limit"
            ],
        )


# ============================================================
# FORECAST GRAPH
# ============================================================

graph_section_number = (
    5
    if plant_type == "Tracking"
    else 4
)

st.markdown(
    f'<div class="section-title">'
    f"{graph_section_number}. Forecast vs Actual"
    "</div>",
    unsafe_allow_html=True,
)


fig, ax = plt.subplots(
    figsize=(15, 6)
)

n_plot = min(
    len(actual_final),
    len(forecast_final),
)

x = np.arange(
    n_plot
)

ax.plot(
    x,
    forecast_final[:n_plot],
    label="Forecast",
    linewidth=2,
)

ax.plot(
    x,
    actual_final[:n_plot],
    label="Actual",
    linewidth=2,
)

ax.set_xlabel(
    "15-Minute Block"
)

ax.set_ylabel(
    "Power"
)

ax.set_title(
    f"{plant_type} Plant - Forecast vs Actual"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)


# ============================================================
# TRACKING ANGLE GRAPH
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">'
        "6. Tracking Angles"
        "</div>",
        unsafe_allow_html=True,
    )

    angle_df = pd.DataFrame(
        {
            "Block": blocks,
            "Zenith Angle": zenith,
            "Panel Angle": panel,
        }
    )

    fig_angle, ax_angle = plt.subplots(
        figsize=(15, 5)
    )

    ax_angle.plot(
        angle_df["Block"],
        angle_df["Zenith Angle"],
        label="Zenith Angle",
        linewidth=2,
    )

    ax_angle.plot(
        angle_df["Block"],
        angle_df["Panel Angle"],
        label="Panel Angle",
        linewidth=2,
    )

    ax_angle.set_xlabel(
        "Block"
    )

    ax_angle.set_ylabel(
        "Angle (°)"
    )

    ax_angle.set_title(
        "Tracking / Zenith Angle Profile"
    )

    ax_angle.grid(
        True,
        alpha=0.3,
    )

    ax_angle.legend()

    st.pyplot(
        fig_angle,
        use_container_width=True,
    )

    plt.close(fig_angle)


# ============================================================
# ERROR OPTIMIZATION TABLE
# ============================================================

if auto_optimize_error and not error_results.empty:

    st.markdown(
        '<div class="section-title">'
        "Error % Optimization"
        "</div>",
        unsafe_allow_html=True,
    )

    display_error_results = (
        error_results
        .sort_values(
            "Error %"
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        display_error_results,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Error vs Peak Error graph
    # --------------------------------------------------------

    fig_error, ax_error = plt.subplots(
        figsize=(14, 5)
    )

    ax_error.plot(
        display_error_results[
            "Error %"
        ],
        display_error_results[
            "Peak Error %"
        ],
        marker="o",
        linewidth=1.5,
    )

    ax_error.axvline(
        best_error,
        linestyle="--",
        linewidth=1.5,
        label=f"Best Error = {best_error:.2f}%",
    )

    ax_error.set_xlabel(
        "Error %"
    )

    ax_error.set_ylabel(
        "Peak Error %"
    )

    ax_error.set_title(
        "Error % vs Peak Error %"
    )

    ax_error.grid(
        True,
        alpha=0.3,
    )

    ax_error.legend()

    st.pyplot(
        fig_error,
        use_container_width=True,
    )

    plt.close(fig_error)


# ============================================================
# EFFECTIVE AREA TABLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Effective Area by Cluster"
    "</div>",
    unsafe_allow_html=True,
)

area_display = cluster_final.copy()

st.dataframe(
    area_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# AREA & EFFICIENCY TABLE
# ============================================================

with st.expander(
    "View Area & Efficiency Calculation"
):

    display_columns = [
        col
        for col in [
            "S.No.",
            "Clusters",
            "Standard PV Efficiency (%)",
            "Error %",
            "Net Efficiency (%)",
            "No of Module",
            "Area of 1 Module (m2)",
            "Total area (m2)",
            "Eff Area",
        ]
        if col in df_final.columns
    ]

    st.dataframe(
        df_final[
            display_columns
        ],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FORECAST TABLE
# ============================================================

with st.expander(
    "View Forecast Data"
):

    forecast_display = forecast_df.copy()

    forecast_display.insert(
        0,
        "Block",
        np.arange(
            len(forecast_display)
        ),
    )

    forecast_display["Actual"] = (
        actual_series
        .to_numpy()[
            :len(
                forecast_display
            )
        ]
    )

    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Final Summary"
    "</div>",
    unsafe_allow_html=True,
)

summary_data = {
    "Plant Type": plant_type,
    "Latitude": latitude,
    "Error %": best_error,
    "Actual Peak": np.max(
        actual_final
    ),
    "Forecast Peak": np.max(
        forecast_final
    ),
    "Peak Error": metrics[
        "Peak Error"
    ],
    "Peak Error %": metrics[
        "Peak Error %"
    ],
    "Energy Error %": metrics[
        "Energy Error %"
    ],
    "MAE": metrics[
        "MAE"
    ],
    "RMSE": metrics[
        "RMSE"
    ],
    "R²": metrics[
        "R²"
    ],
}


if plant_type == "Tracking":

    summary_data.update(
        tracking_parameters
    )


summary_df = pd.DataFrame(
    [summary_data]
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Download Results"
    "</div>",
    unsafe_allow_html=True,
)


output_buffer = io.BytesIO()


with pd.ExcelWriter(
    output_buffer,
    engine="openpyxl",
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

    cluster_final.to_excel(
        writer,
        sheet_name="Cluster Effective Area",
        index=False,
    )

    df_final.to_excel(
        writer,
        sheet_name="Area Efficiency",
        index=False,
    )

    forecast_display.to_excel(
        writer,
        sheet_name="Forecast",
        index=False,
    )

    if plant_type == "Tracking":

        angle_df.to_excel(
            writer,
            sheet_name="Tracking Angles",
            index=False,
        )

    if auto_optimize_error and not error_results.empty:

        error_results.to_excel(
            writer,
            sheet_name="Error Optimization",
            index=False,
        )


output_buffer.seek(0)


st.download_button(
    label="📥 Download Complete Excel Report",
    data=output_buffer,
    file_name=(
        f"Solar_Forecast_Correction_"
        f"{plant_type}.xlsx"
    ),
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True,
)


# ============================================================
# END
# ============================================================

st.success(
    f"{plant_type} plant forecast correction completed successfully."
)
