# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
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

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #777;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 650;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #f7f8fa;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e6e8eb;
    }

    .metric-label {
        font-size: 13px;
        color: #777;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 3px;
    }

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


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

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

POWER_COLS = [
    "C11 Power",
    "C12 Power",
    "C13 Power",
    "C14 Power",
    "C15 Power",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return df


def trim_at_first_blank(df, column):
    df = df.copy()

    if column not in df.columns:
        return df

    null_rows = df[df[column].isna()].index

    if len(null_rows) > 0:
        first_position = df.index.get_loc(null_rows[0])
        df = df.iloc[:first_position]

    return df.reset_index(drop=True)


def numeric_series(series, default=0.0):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(default)


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_max(series):
    arr = numeric_series(series).to_numpy(dtype=float)

    if len(arr) == 0:
        return 0.0

    return float(np.nanmax(arr))


def align_length(df, n):
    df = df.copy()

    if len(df) >= n:
        return df.iloc[:n].reset_index(drop=True)

    missing = n - len(df)

    extra = pd.DataFrame(
        {
            col: [0] * missing
            for col in df.columns
        }
    )

    return pd.concat(
        [df, extra],
        ignore_index=True
    )


# ============================================================
# READ EXCEL
# ============================================================

@st.cache_data(show_spinner=False)
def read_excel_data(file_bytes):

    xls = pd.ExcelFile(file_bytes)

    required_sheets = [
        "Area & Efficiency",
        "Forecast Config",
        "Config Tilt Angle",
        "Result",
        "Fixed-C11",
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
        "Tracking",
    ]

    missing = [
        sheet
        for sheet in required_sheets
        if sheet not in xls.sheet_names
    ]

    if missing:
        raise ValueError(
            "Missing required sheets: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df = pd.read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = clean_columns(df)

    df = trim_at_first_blank(
        df,
        "S.No."
    )

    # --------------------------------------------------------
    # CLUSTER WEIGHTS
    # --------------------------------------------------------

    df_w = pd.read_excel(
        file_bytes,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w = clean_columns(df_w)

    if "Clusters" in df_w.columns:
        df_w = trim_at_first_blank(
            df_w,
            "Clusters"
        )

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    df_st = pd.read_excel(
        file_bytes,
        sheet_name="Forecast Config",
        header=8,
    )

    df_st = clean_columns(df_st)

    if "Lat" not in df_st.columns:
        raise ValueError(
            "Column 'Lat' not found in Forecast Config."
        )

    lat = safe_float(
        df_st.loc[0, "Lat"]
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        file_bytes,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt = clean_columns(df_tilt)

    if "Fixed" in df_tilt.columns:
        df_tilt = trim_at_first_blank(
            df_tilt,
            "Fixed"
        )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1
    )

    rename_map = {}

    if "Unnamed: 2" in df_tilt.columns:
        rename_map["Unnamed: 2"] = "Month_Num"

    if "Unnamed: 3" in df_tilt.columns:
        rename_map["Unnamed: 3"] = "Month"

    df_tilt = df_tilt.rename(
        columns=rename_map
    )

    if not {"Month", "Fixed"}.issubset(
        df_tilt.columns
    ):
        raise ValueError(
            "Could not find Month / Fixed columns "
            "in Config Tilt Angle."
        )

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        file_bytes,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = clean_columns(df_ghi)

    missing_ghi = [
        c for c in GHI_COLS
        if c not in df_ghi.columns
    ]

    if missing_ghi:
        raise ValueError(
            "Missing GHI columns: "
            + ", ".join(missing_ghi)
        )

    # --------------------------------------------------------
    # FIXED DATA
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        file_bytes,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix = clean_columns(df_fix)

    if "Date" in df_fix.columns:
        df_fix = trim_at_first_blank(
            df_fix,
            "Date"
        )

    if "Actual" not in df_fix.columns:
        raise ValueError(
            "Column 'Actual' not found in Fixed-C11."
        )

    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    backend_list = []

    for cluster in CLUSTERS:

        backend = pd.read_excel(
            file_bytes,
            sheet_name=f"Backend Cal {cluster}",
        )

        backend = clean_columns(backend)

        if "Block No." not in backend.columns:
            raise ValueError(
                f"'Block No.' missing in Backend Cal {cluster}"
            )

        backend_list.append(
            backend.reset_index(drop=True)
        )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    df_trac = pd.read_excel(
        file_bytes,
        sheet_name="Tracking",
        header=1,
    )

    df_trac = clean_columns(df_trac)

    return {
        "df": df,
        "df_w": df_w,
        "lat": lat,
        "month_lookup": month_lookup,
        "df_ghi": df_ghi,
        "df_fix": df_fix,
        "df_trac": df_trac,
        "backend_list": backend_list,
    }


# ============================================================
# PREPARE USER INPUT DATA
# ============================================================

def prepare_input_data(data):

    df_ghi = data["df_ghi"].copy()
    df_fix = data["df_fix"].copy()

    # --------------------------------------------------------
    # GHI
    # --------------------------------------------------------

    for col in GHI_COLS:
        df_ghi[col] = numeric_series(
            df_ghi[col]
        )

    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    df_fix["Actual"] = numeric_series(
        df_fix["Actual"]
    )

    n = min(
        len(df_ghi),
        len(df_fix),
    )

    if n <= 0:
        raise ValueError(
            "No valid rows found in GHI / Actual input."
        )

    df_ghi = df_ghi.iloc[:n].reset_index(drop=True)
    df_fix = df_fix.iloc[:n].reset_index(drop=True)

    return df_ghi, df_fix


# ============================================================
# EFFECTIVE AREA
# ============================================================

def calculate_cluster_areas(
    df,
    df_w,
    error_percent,
):

    work = df.copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Error % is applied ONLY HERE.
    #
    # Standard Efficiency
    #         -
    # Error %
    #         =
    # Net Efficiency
    #
    # Tracking optimization must never apply Error %
    # again.
    # --------------------------------------------------------

    work["Standard PV Efficiency (%)"] = numeric_series(
        work["Standard PV Efficiency (%)"]
    )

    work["No of Module"] = numeric_series(
        work["No of Module"]
    )

    work["Area of 1 Module (m2)"] = numeric_series(
        work["Area of 1 Module (m2)"]
    )

    work["Net Efficiency (%)"] = (
        work["Standard PV Efficiency (%)"]
        - float(error_percent)
    )

    work["Total area (m2)"] = (
        work["No of Module"]
        * work["Area of 1 Module (m2)"]
    )

    work["Eff Area"] = (
        work["Net Efficiency (%)"]
        * work["Total area (m2)"]
        / 100.0
    )

    cluster_sums = (
        work
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    weights = []

    cluster_table = df_w.copy()

    cluster_table["Clusters"] = (
        cluster_table["Clusters"]
        .astype(str)
        .str.strip()
    )

    for cluster in CLUSTERS:

        value = cluster_sums.get(
            cluster,
            0.0
        )

        weights.append(
            safe_float(value)
        )

    return (
        work,
        np.asarray(weights, dtype=float),
        cluster_table,
    )


# ============================================================
# FIXED SOLAR GEOMETRY
# ============================================================

def calculate_fixed_geometry(
    df_ghi,
    df_fix,
    lat,
    month_lookup,
):

    n = min(
        len(df_ghi),
        len(df_fix),
    )

    ghi = df_ghi.iloc[:n].copy()
    fix = df_fix.iloc[:n].copy()

    # --------------------------------------------------------
    # DO NOT use today's date for every row.
    #
    # Preserve Excel Date when available.
    # --------------------------------------------------------

    if "Date" in fix.columns:

        dates = pd.to_datetime(
            fix["Date"],
            errors="coerce"
        )

    else:

        dates = pd.Series(
            pd.Timestamp.today().normalize(),
            index=fix.index,
        )

    # If the user-input date is invalid, use today.
    dates = dates.fillna(
        pd.Timestamp.today().normalize()
    )

    first_date = pd.Timestamp(
        year=dates.dt.year.iloc[0],
        month=1,
        day=1,
    )

    day_of_year = (
        dates - first_date
    ).dt.days + 1

    # --------------------------------------------------------
    # DECLINATION
    # --------------------------------------------------------

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
        - float(lat)
        + declination
    )

    month_names = dates.dt.strftime("%B")

    tilt = month_names.map(
        month_lookup
    )

    tilt = pd.to_numeric(
        tilt,
        errors="coerce"
    ).fillna(0)

    a_plus_b = (
        elevation
        + tilt
    )

    sin_a_plus_b = np.sin(
        np.radians(a_plus_b)
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    # Avoid division by zero
    safe_sin_a = np.where(
        np.abs(sin_a) < 1e-8,
        np.nan,
        sin_a,
    )

    geometry = {
        "Date": dates,
        "Declination Angle ∆": declination,
        "Elevation angle a": elevation,
        "Tilt Angle b": tilt,
        "a+b": a_plus_b,
        "SIN(a+b)": sin_a_plus_b,
        "Sin(a)": sin_a,
    }

    poa_list = []

    for col in GHI_COLS:

        g = numeric_series(
            ghi[col]
        ).to_numpy(dtype=float)

        poa = (
            g
            * sin_a_plus_b
            / safe_sin_a
        )

        poa = np.nan_to_num(
            poa,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        poa_list.append(poa)

    return (
        geometry,
        np.column_stack(poa_list),
    )


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df_ghi,
    df_fix,
    df,
    df_w,
    lat,
    month_lookup,
    error_percent,
):

    work, weights, cluster_table = (
        calculate_cluster_areas(
            df,
            df_w,
            error_percent,
        )
    )

    geometry, poa_matrix = (
        calculate_fixed_geometry(
            df_ghi,
            df_fix,
            lat,
            month_lookup,
        )
    )

    n = min(
        len(poa_matrix),
        len(weights),
    )

    # --------------------------------------------------------
    # Power for each cluster
    # --------------------------------------------------------

    powers = (
        poa_matrix
        * weights[None, :]
        / 1_000_000
    )

    forecast = powers.sum(
        axis=1
    )

    return {
        "work": work,
        "weights": weights,
        "cluster_table": cluster_table,
        "geometry": geometry,
        "poa": poa_matrix,
        "powers": powers,
        "forecast": forecast,
    }


# ============================================================
# TRACKING GEOMETRY
# ============================================================

def calculate_tracking_geometry(
    blocks,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        < GHI_Max_Block
        < GHI_Ending_Block
    ):
        raise ValueError(
            "Tracking blocks must satisfy: "
            "Starting < Max < Ending."
        )

    if GHI_Max_Block <= (
        GHI_Starting_Block - 1
    ):
        raise ValueError(
            "Invalid GHI Starting Block."
        )

    if GHI_Ending_Block <= (
        GHI_Max_Block - 1
    ):
        raise ValueError(
            "Invalid GHI Ending Block."
        )

    # --------------------------------------------------------
    # EXACT ORIGINAL FORMULA
    # --------------------------------------------------------

    m1 = 90 / (
        GHI_Starting_Block
        - 1
        - GHI_Max_Block
    )

    m2 = 90 / (
        GHI_Ending_Block
        + 1
        - GHI_Max_Block
    )

    zenith = np.where(
        blocks <= GHI_Max_Block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - GHI_Max_Block
            ),
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - GHI_Max_Block
            ),
        ),
    )

    panel = np.where(
        blocks < GHI_Max_Block,

        np.minimum(
            zenith,
            abs(
                Tracking_angle_lim_E
            ),
        ),

        np.where(
            (
                (blocks > GHI_Max_Block)
                &
                (
                    zenith
                    > Tracking_angle_lim_W
                )
            ),

            Tracking_angle_lim_W,

            zenith,
        ),
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    return (
        zenith,
        panel,
        cos_alpha,
    )


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    df_ghi,
    weights,
    blocks,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
):

    ghi_matrix = np.column_stack(
        [
            numeric_series(
                df_ghi[col]
            ).to_numpy(dtype=float)
            for col in GHI_COLS
        ]
    )

    (
        zenith,
        panel,
        cos_alpha,
    ) = calculate_tracking_geometry(
        blocks,
        DHI,
        GHI_Starting_Block,
        GHI_Ending_Block,
        GHI_Max_Block,
        Tracking_angle_lim_E,
        Tracking_angle_lim_W,
    )

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * float(DHI)
        / 100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # Cluster power
    # --------------------------------------------------------

    powers = (
        dni
        * weights[None, :]
        / 1_000_000
    )

    forecast = powers.sum(
        axis=1
    )

    forecast = np.nan_to_num(
        forecast,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return {
        "zenith": zenith,
        "panel": panel,
        "cos_alpha": cos_alpha,
        "dhi": dhi,
        "dni": dni,
        "powers": powers,
        "forecast": forecast,
    }


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    df_ghi,
    actual,
    weights,
    backend_list,
    initial_error,
    bounds,
):

    # --------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------

    blocks = numeric_series(
        backend_list[0]["Block No."]
    ).to_numpy(dtype=float)

    n = min(
        len(df_ghi),
        len(actual),
        len(blocks),
    )

    df_ghi = df_ghi.iloc[:n].reset_index(
        drop=True
    )

    actual = (
        numeric_series(actual)
        .iloc[:n]
        .to_numpy(dtype=float)
    )

    blocks = blocks[:n]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # ERROR % IS CALCULATED ONCE HERE.
    #
    # We calculate effective areas using:
    #
    # Standard Efficiency - Error %
    #
    # The tracking objective NEVER subtracts Error %
    # again.
    #
    # This is the correction for the Jupyter vs Streamlit
    # discrepancy.
    # --------------------------------------------------------

    def get_weights(error):

        # This function should receive BASE df.
        #
        # It applies the candidate error only once.
        #
        # Since the user can change Error % later, this
        # function is intentionally recalculated.
        return weights_from_area(
            st.session_state["base_area_df"],
            st.session_state["base_df_w"],
            error,
        )

    # --------------------------------------------------------
    # ACTUAL MASK
    # --------------------------------------------------------

    mask = np.isfinite(
        actual
    ) & (
        actual != 0
    )

    if not np.any(mask):

        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_valid = actual[mask]

    actual_max = np.max(
        actual_valid
    )

    actual_sum = np.sum(
        actual_valid
    )

    if actual_max <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    def objective(x):

        (
            error_percent,
            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
        ) = x

        error_percent = float(
            error_percent
        )

        DHI = int(
            round(DHI)
        )

        start_block = int(
            round(start_block)
        )

        end_block = int(
            round(end_block)
        )

        max_block = int(
            round(max_block)
        )

        east_limit = int(
            round(east_limit)
        )

        west_limit = int(
            round(west_limit)
        )

        if not (
            start_block
            < max_block
            < end_block
        ):
            return 1e9

        try:

            # ------------------------------------------------
            # APPLY ERROR % ONCE
            # ------------------------------------------------

            _, candidate_weights = (
                calculate_cluster_weights_from_base(
                    st.session_state["base_area_df"],
                    st.session_state["base_df_w"],
                    error_percent,
                )
            )

            result = (
                calculate_tracking_forecast(
                    df_ghi=df_ghi,
                    weights=candidate_weights,
                    blocks=blocks,
                    DHI=DHI,
                    GHI_Starting_Block=start_block,
                    GHI_Ending_Block=end_block,
                    GHI_Max_Block=max_block,
                    Tracking_angle_lim_E=east_limit,
                    Tracking_angle_lim_W=west_limit,
                )
            )

            prediction = result[
                "forecast"
            ]

            if (
                np.isnan(prediction).any()
                or np.isinf(prediction).any()
            ):
                return 1e9

            prediction_valid = (
                prediction[mask]
            )

            if len(prediction_valid) == 0:
                return 1e9

            # ----------------------------------------------
            # BLOCK ERROR
            # ----------------------------------------------

            block_error = (
                np.mean(
                    np.abs(
                        actual_valid
                        - prediction_valid
                    )
                )
                / actual_max
            )

            # ----------------------------------------------
            # PEAK ERROR
            # ----------------------------------------------

            peak_error = (
                abs(
                    actual_max
                    - prediction_valid.max()
                )
                / actual_max
            )

            # ----------------------------------------------
            # ENERGY ERROR
            # ----------------------------------------------

            energy_error = (
                abs(
                    actual_sum
                    - prediction_valid.sum()
                )
                / actual_sum
                if actual_sum != 0
                else 0
            )

            # ----------------------------------------------
            # SAME SCORE AS JUPYTER
            # ----------------------------------------------

            score = (
                0.80 * block_error
                + 0.10 * peak_error
                + 0.10 * energy_error
            )

            return float(score)

        except Exception:
            return 1e9

    # --------------------------------------------------------
    # OPTIMIZE
    # --------------------------------------------------------

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

    best = result.x

    return {
        "Error %": float(
            best[0]
        ),
        "DHI": int(
            round(best[1])
        ),
        "GHI Starting Block": int(
            round(best[2])
        ),
        "GHI Ending Block": int(
            round(best[3])
        ),
        "GHI Max Block": int(
            round(best[4])
        ),
        "Tracking East Limit": int(
            round(best[5])
        ),
        "Tracking West Limit": int(
            round(best[6])
        ),
        "score": result.fun,
    }


# ============================================================
# WEIGHT HELPERS
# ============================================================

def calculate_cluster_weights_from_base(
    df,
    df_w,
    error_percent,
):

    work = df.copy()

    standard = numeric_series(
        work["Standard PV Efficiency (%)"]
    )

    modules = numeric_series(
        work["No of Module"]
    )

    module_area = numeric_series(
        work["Area of 1 Module (m2)"]
    )

    work["Net Efficiency (%)"] = (
        standard
        - float(error_percent)
    )

    work["Total area (m2)"] = (
        modules
        * module_area
    )

    work["Eff Area"] = (
        work["Net Efficiency (%)"]
        * work["Total area (m2)"]
        / 100
    )

    cluster_sums = (
        work
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    weights = np.array(
        [
            safe_float(
                cluster_sums.get(
                    cluster,
                    0
                )
            )
            for cluster in CLUSTERS
        ],
        dtype=float,
    )

    return work, weights


def weights_from_area(
    df,
    df_w,
    error_percent,
):

    _, weights = (
        calculate_cluster_weights_from_base(
            df,
            df_w,
            error_percent,
        )
    )

    return weights


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

def optimize_fixed_error(
    df_ghi,
    df_fix,
    df,
    df_w,
    lat,
    month_lookup,
):

    actual = numeric_series(
        df_fix["Actual"]
    ).to_numpy(dtype=float)

    mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    if not np.any(mask):

        raise ValueError(
            "No non-zero Actual values found for Fixed."
        )

    actual_valid = actual[mask]

    actual_peak = np.max(
        actual_valid
    )

    if actual_peak <= 0:
        raise ValueError(
            "Actual peak must be greater than zero."
        )

    results = []

    # --------------------------------------------------------
    # EXACT SAME 0 TO 10, STEP 0.1
    # --------------------------------------------------------

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        result = (
            calculate_fixed_forecast(
                df_ghi=df_ghi,
                df_fix=df_fix,
                df=df,
                df_w=df_w,
                lat=lat,
                month_lookup=month_lookup,
                error_percent=error,
            )
        )

        forecast = result[
            "forecast"
        ]

        forecast_valid = (
            forecast[mask]
        )

        calculated_peak = (
            np.max(
                forecast_valid
            )
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

        results.append(
            {
                "Error %": error,
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": peak_error_pct,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    best_row = results_df.loc[
        results_df["Peak Error"].idxmin()
    ]

    return (
        float(best_row["Error %"]),
        results_df,
    )


# ============================================================
# INPUT / UI
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload your plant workbook, edit GHI and Actual inputs, "
    "then select the plant type."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# 1. FILE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Excel File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Excel workbook",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to continue."
    )

    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:

    data = read_excel_data(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# PREPARE INPUTS
# ============================================================

try:

    df_ghi, df_fix = (
        prepare_input_data(
            data
        )
    )

except Exception as e:

    st.error(
        f"Input preparation failed: {e}"
    )

    st.stop()


# ============================================================
# 2. INPUT DATA
# ============================================================

st.markdown(
    '<div class="section-title">'
    "2. Input GHI & Actual Data"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Edit the GHI values and Actual values below. "
    "These edited values are used directly in the calculation."
)

input_left, input_right = st.columns(
    2
)


with input_left:

    st.markdown("**GHI Forecast**")

    ghi_display = df_ghi[
        GHI_COLS
    ].copy()

    edited_ghi = st.data_editor(
        ghi_display,
        use_container_width=True,
        num_rows="dynamic",
        height=350,
        key="ghi_editor",
    )


with input_right:

    st.markdown("**Actual Power**")

    actual_display = pd.DataFrame(
        {
            "Actual": df_fix[
                "Actual"
            ].copy()
        }
    )

    edited_actual = st.data_editor(
        actual_display,
        use_container_width=True,
        num_rows="dynamic",
        height=350,
        key="actual_editor",
    )


# ============================================================
# REBUILD INPUT DATA AFTER EDITING
# ============================================================

df_ghi = edited_ghi.copy()

df_fix = df_fix.copy()

df_fix["Actual"] = (
    pd.to_numeric(
        edited_actual["Actual"],
        errors="coerce",
    )
    .fillna(0)
)

n = min(
    len(df_ghi),
    len(df_fix),
)

df_ghi = (
    df_ghi.iloc[:n]
    .reset_index(drop=True)
)

df_fix = (
    df_fix.iloc[:n]
    .reset_index(drop=True)
)

for col in GHI_COLS:

    df_ghi[col] = numeric_series(
        df_ghi[col]
    )


# ============================================================
# 3. PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "3. Plant Type"
    "</div>",
    unsafe_allow_html=True,
)

plant_type = st.segmented_control(
    "Select plant type",
    options=[
        "Fixed",
        "Tracking",
    ],
    default="Fixed",
    key="plant_type",
)


# ============================================================
# SESSION BASE DATA
# ============================================================

st.session_state["base_area_df"] = (
    data["df"].copy()
)

st.session_state["base_df_w"] = (
    data["df_w"].copy()
)


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    "4. Automatic Calculation"
    "</div>",
    unsafe_allow_html=True,
)

run_auto = st.button(
    "⚙️ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
)


if run_auto:

    with st.spinner(
        "Running automatic calculation..."
    ):

        try:

            # ==================================================
            # FIXED
            # ==================================================

            if plant_type == "Fixed":

                best_error, optimization_table = (
                    optimize_fixed_error(
                        df_ghi=df_ghi,
                        df_fix=df_fix,
                        df=data["df"],
                        df_w=data["df_w"],
                        lat=data["lat"],
                        month_lookup=data[
                            "month_lookup"
                        ],
                    )
                )

                st.session_state[
                    "auto_params"
                ] = {
                    "Error %": best_error
                }

                st.session_state[
                    "optimization_table"
                ] = optimization_table

                st.session_state[
                    "calculated"
                ] = True

            # ==================================================
            # TRACKING
            # ==================================================

            else:

                bounds = [
                    (0, 10),       # Error %
                    (0, 10),       # DHI
                    (10, 30),      # Start
                    (65, 80),      # End
                    (47, 53),      # Max
                    (10, 70),      # East
                    (10, 70),      # West
                ]

                auto = optimize_tracking(
                    df_ghi=df_ghi,
                    actual=df_fix[
                        "Actual"
                    ],
                    weights=None,
                    backend_list=data[
                        "backend_list"
                    ],
                    initial_error=0,
                    bounds=bounds,
                )

                st.session_state[
                    "auto_params"
                ] = auto

                st.session_state[
                    "calculated"
                ] = True

        except Exception as e:

            st.session_state[
                "calculated"
            ] = False

            st.error(
                f"Automatic calculation failed: {e}"
            )


# ============================================================
# STOP UNTIL AUTO CALCULATION
# ============================================================

if not st.session_state.get(
    "calculated",
    False,
):

    st.info(
        "Run Automatic Calculation to generate the "
        "editable optimized parameters."
    )

    st.stop()


# ============================================================
# 5. EDITABLE PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "5. Optimized Parameters"
    "</div>",
    unsafe_allow_html=True,
)

auto = st.session_state[
    "auto_params"
]


# ============================================================
# FIXED PARAMETERS
# ============================================================

if plant_type == "Fixed":

    error_percent = st.number_input(
        "Error %",
        min_value=0.0,
        max_value=10.0,
        value=float(
            auto["Error %"]
        ),
        step=0.1,
        format="%.1f",
        key="fixed_error",
    )

    current_params = {
        "Error %": error_percent
    }


# ============================================================
# TRACKING PARAMETERS
# ============================================================

else:

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        error_percent = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=10.0,
            value=float(
                auto["Error %"]
            ),
            step=0.1,
            format="%.1f",
        )

    with c2:

        dhi = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(
                auto["DHI"]
            ),
            step=1,
        )

    with c3:

        start_block = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=95,
            value=int(
                auto[
                    "GHI Starting Block"
                ]
            ),
            step=1,
        )

    with c4:

        end_block = st.number_input(
            "GHI Ending Block",
            min_value=2,
            max_value=96,
            value=int(
                auto[
                    "GHI Ending Block"
                ]
            ),
            step=1,
        )

    c5, c6, c7 = st.columns(3)

    with c5:

        max_block = st.number_input(
            "GHI Max Block",
            min_value=2,
            max_value=95,
            value=int(
                auto[
                    "GHI Max Block"
                ]
            ),
            step=1,
        )

    with c6:

        east_limit = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                auto[
                    "Tracking East Limit"
                ]
            ),
            step=1,
        )

    with c7:

        west_limit = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                auto[
                    "Tracking West Limit"
                ]
            ),
            step=1,
        )

    current_params = {
        "Error %": error_percent,
        "DHI": dhi,
        "GHI Starting Block": start_block,
        "GHI Ending Block": end_block,
        "GHI Max Block": max_block,
        "Tracking East Limit": east_limit,
        "Tracking West Limit": west_limit,
    }


# ============================================================
# 6. RECALCULATE FINAL FORECAST
# ============================================================

st.markdown(
    '<div class="section-title">'
    "6. Final Forecast"
    "</div>",
    unsafe_allow_html=True,
)

try:

    # ========================================================
    # FIXED
    # ========================================================

    if plant_type == "Fixed":

        final = calculate_fixed_forecast(
            df_ghi=df_ghi,
            df_fix=df_fix,
            df=data["df"],
            df_w=data["df_w"],
            lat=data["lat"],
            month_lookup=data[
                "month_lookup"
            ],
            error_percent=current_params[
                "Error %"
            ],
        )

        forecast = final[
            "forecast"
        ]

        actual = (
            numeric_series(
                df_fix["Actual"]
            )
            .to_numpy(dtype=float)
        )

        weights = final[
            "weights"
        ]

        final_table = pd.DataFrame(
            {
                "GHI C11": df_ghi[
                    "GHI C11"
                ],
                "GHI C12": df_ghi[
                    "GHI C12"
                ],
                "GHI C13": df_ghi[
                    "GHI C13"
                ],
                "GHI C14": df_ghi[
                    "GHI C14"
                ],
                "GHI C15": df_ghi[
                    "GHI C15"
                ],
                "Forecast": forecast,
                "Actual": actual,
            }
        )

    # ========================================================
    # TRACKING
    # ========================================================

    else:

        _, weights = (
            calculate_cluster_weights_from_base(
                data["df"],
                data["df_w"],
                current_params[
                    "Error %"
                ],
            )
        )

        blocks = numeric_series(
            data["backend_list"][0][
                "Block No."
            ]
        ).to_numpy(dtype=float)

        n = min(
            len(df_ghi),
            len(df_fix),
            len(blocks),
        )

        df_ghi_final = (
            df_ghi.iloc[:n]
            .reset_index(drop=True)
        )

        actual = (
            numeric_series(
                df_fix["Actual"]
            )
            .iloc[:n]
            .to_numpy(dtype=float)
        )

        blocks = blocks[:n]

        final = (
            calculate_tracking_forecast(
                df_ghi=df_ghi_final,
                weights=weights,
                blocks=blocks,
                DHI=current_params[
                    "DHI"
                ],
                GHI_Starting_Block=current_params[
                    "GHI Starting Block"
                ],
                GHI_Ending_Block=current_params[
                    "GHI Ending Block"
                ],
                GHI_Max_Block=current_params[
                    "GHI Max Block"
                ],
                Tracking_angle_lim_E=current_params[
                    "Tracking East Limit"
                ],
                Tracking_angle_lim_W=current_params[
                    "Tracking West Limit"
                ],
            )
        )

        forecast = final[
            "forecast"
        ]

        final_table = pd.DataFrame(
            {
                "Block No.": blocks,
                "Zenith Angle": final[
                    "zenith"
                ],
                "Panel Angle": final[
                    "panel"
                ],
                "Forecast": forecast,
                "Actual": actual,
            }
        )


    # ========================================================
    # METRICS
    # ========================================================

    valid = (
        np.isfinite(actual)
        & np.isfinite(forecast)
        & (actual != 0)
    )

    if np.any(valid):

        actual_valid = actual[
            valid
        ]

        forecast_valid = forecast[
            valid
        ]

        actual_peak = np.max(
            actual_valid
        )

        forecast_peak = np.max(
            forecast_valid
        )

        peak_error = abs(
            forecast_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            / actual_peak
            * 100
            if actual_peak != 0
            else 0
        )

        energy_error_pct = (
            abs(
                forecast_valid.sum()
                - actual_valid.sum()
            )
            / actual_valid.sum()
            * 100
            if actual_valid.sum() != 0
            else 0
        )

    else:

        actual_peak = 0
        forecast_peak = 0
        peak_error = 0
        peak_error_pct = 0
        energy_error_pct = 0


    # ========================================================
    # METRIC CARDS
    # ========================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Actual Peak",
            f"{actual_peak:,.3f}",
        )

    with m2:

        st.metric(
            "Forecast Peak",
            f"{forecast_peak:,.3f}",
        )

    with m3:

        st.metric(
            "Peak Error",
            f"{peak_error_pct:.2f}%",
        )

    with m4:

        st.metric(
            "Energy Error",
            f"{energy_error_pct:.2f}%",
        )


    # ========================================================
    # GRAPH
    # ========================================================

    st.markdown(
        "### Forecast vs Actual"
    )

    x = np.arange(
        len(forecast)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2.5
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
                width=2.5
            ),
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="15-Minute Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(
            l=30,
            r=30,
            t=50,
            b=30,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # TRACKING ANGLE GRAPH
    # ========================================================

    if plant_type == "Tracking":

        st.markdown(
            "### Tracking Angles"
        )

        angle_fig = go.Figure()

        angle_fig.add_trace(
            go.Scatter(
                x=x,
                y=final["zenith"],
                mode="lines",
                name="Zenith Angle",
                line=dict(
                    width=2
                ),
            )
        )

        angle_fig.add_trace(
            go.Scatter(
                x=x,
                y=final["panel"],
                mode="lines",
                name="Panel Angle",
                line=dict(
                    width=2
                ),
            )
        )

        angle_fig.update_layout(
            height=400,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Block",
            yaxis_title="Angle (°)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(
            angle_fig,
            use_container_width=True,
        )


    # ========================================================
    # FINAL DATA
    # ========================================================

    with st.expander(
        "View Final Calculation Data"
    ):

        st.dataframe(
            final_table,
            use_container_width=True,
            height=400,
        )


    # ========================================================
    # TRACKING PARAMETERS SUMMARY
    # ========================================================

    if plant_type == "Tracking":

        with st.expander(
            "View Tracking Parameters"
        ):

            parameter_df = pd.DataFrame(
                {
                    "Parameter": list(
                        current_params.keys()
                    ),
                    "Value": list(
                        current_params.values()
                    ),
                }
            )

            st.dataframe(
                parameter_df,
                use_container_width=True,
                hide_index=True,
            )


except Exception as e:

    st.error(
        f"Final calculation failed: {e}"
    )
