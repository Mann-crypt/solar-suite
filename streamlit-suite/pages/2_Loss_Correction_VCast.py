# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED + TRACKING
# STREAMLIT + PLOTLY
# ============================================================

import io
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
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Fixed and Tracking solar power forecast correction, "
    "efficiency optimization and performance analysis."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    options=["Fixed", "Tracking"],
    default="Fixed",
)

if plant_type is None:
    plant_type = "Fixed"


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Input Workbook</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Excel workbook",
    type=["xlsx", "xls"],
)

if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to start."
    )

    st.stop()


# ============================================================
# EXCEL FILE
# ============================================================

file_bytes = uploaded_file.getvalue()

excel_file = io.BytesIO(file_bytes)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_until_first_null(
    df,
    column,
):

    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[
        df[column].isna()
    ].index

    if len(null_indices) == 0:
        return df

    first_null_pos = df.index.get_loc(
        null_indices[0]
    )

    return df.iloc[
        :first_null_pos
    ].copy()


def numeric_series(
    series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def numeric_array(
    series,
):

    return numeric_series(
        series
    ).to_numpy(
        dtype=float
    )


def check_columns(
    df,
    required,
    sheet_name,
):

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in '{sheet_name}': "
            + ", ".join(missing)
        )


# ============================================================
# LOAD COMMON DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_common_data(
    file_bytes,
):

    workbook = io.BytesIO(
        file_bytes
    )

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df_area = pd.read_excel(
        workbook,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df_area.columns = (
        df_area.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

    df_area = clean_until_first_null(
        df_area,
        "S.No.",
    )

    check_columns(
        df_area,
        [
            "Clusters",
            "Standard PV Efficiency (%)",
            "No of Module",
            "Area of 1 Module (m2)",
        ],
        "Area & Efficiency",
    )

    # --------------------------------------------------------
    # CLUSTER WEIGHTS
    # --------------------------------------------------------

    df_weights = pd.read_excel(
        workbook,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_weights.columns = (
        df_weights.columns
        .astype(str)
        .str.strip()
    )

    df_weights = clean_until_first_null(
        df_weights,
        "Clusters",
    )

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    df_config = pd.read_excel(
        workbook,
        sheet_name="Forecast Config",
        header=8,
    )

    check_columns(
        df_config,
        ["Lat"],
        "Forecast Config",
    )

    lat = float(
        pd.to_numeric(
            df_config.loc[0, "Lat"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    df_ghi = pd.read_excel(
        workbook,
        sheet_name="Result",
        usecols=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
    )

    df_ghi.columns = (
        df_ghi.columns
        .astype(str)
        .str.strip()
    )

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    check_columns(
        df_ghi,
        ghi_cols,
        "Result",
    )

    for col in ghi_cols:

        df_ghi[col] = numeric_series(
            df_ghi[col]
        )

    return (
        df_area,
        df_weights,
        df_config,
        df_ghi,
        lat,
    )


# ============================================================
# LOAD COMMON DATA
# ============================================================

try:

    (
        df_area,
        df_weights,
        df_config,
        df_ghi,
        lat,
    ) = load_common_data(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to load workbook: {e}"
    )

    st.stop()


# ============================================================
# COMMON UI
# ============================================================

st.markdown(
    '<div class="section-title">Workbook Information</div>',
    unsafe_allow_html=True,
)

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Plant Type",
        plant_type,
    )

with info2:

    st.metric(
        "Latitude",
        f"{lat:.4f}°",
    )

with info3:

    st.metric(
        "GHI Records",
        len(df_ghi),
    )


# ============================================================
# AREA CALCULATION
# ============================================================

def calculate_effective_area(
    df_area_input,
    df_weights_input,
    error_percent,
):

    df = df_area_input.copy()
    df_w = df_weights_input.copy()

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    df[
        "Standard PV Efficiency (%)"
    ] = numeric_series(
        df[
            "Standard PV Efficiency (%)"
        ]
    )

    df["No of Module"] = numeric_series(
        df["No of Module"]
    )

    df[
        "Area of 1 Module (m2)"
    ] = numeric_series(
        df[
            "Area of 1 Module (m2)"
        ]
    )

    # --------------------------------------------------------
    # ERROR %
    # --------------------------------------------------------

    df["Error %"] = error_percent

    # --------------------------------------------------------
    # NET EFFICIENCY
    # --------------------------------------------------------

    df[
        "Net Efficiency (%)"
    ] = (
        df[
            "Standard PV Efficiency (%)"
        ]
        - error_percent
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        * df["Total area (m2)"]
        / 100
    )

    # --------------------------------------------------------
    # CLUSTER AREA
    # --------------------------------------------------------

    cluster_sums = (
        df.groupby(
            "Clusters"
        )["Eff Area"]
        .sum()
    )

    df_w[
        "Eff Area(m2)"
    ] = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
    )

    return (
        df,
        df_w,
    )


# ============================================================
# FIXED DATA PREPARATION
# ============================================================

def prepare_fixed_data():

    workbook = io.BytesIO(
        file_bytes
    )

    # --------------------------------------------------------
    # TILT ANGLE
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        workbook,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" not in df_tilt.columns:

        raise ValueError(
            "'Fixed' column not found in "
            "Config Tilt Angle."
        )

    df_tilt = clean_until_first_null(
        df_tilt,
        "Fixed",
    )

    df_tilt = df_tilt.dropna(
        how="all",
        axis=1,
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    if "Month" not in df_tilt.columns:

        raise ValueError(
            "'Month' column not found in "
            "Config Tilt Angle."
        )

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    # --------------------------------------------------------
    # FIXED-C11
    # --------------------------------------------------------

    df_fix = pd.read_excel(
        workbook,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    df_fix = clean_until_first_null(
        df_fix,
        "Date",
    )

    check_columns(
        df_fix,
        [
            "Date",
            "Actual",
        ],
        "Fixed-C11",
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    dates = pd.to_datetime(
        df_fix["Date"],
        errors="coerce",
    )

    if dates.notna().any():

        dates = dates.ffill()

    else:

        dates = pd.Series(
            pd.Timestamp.today(),
            index=df_fix.index,
        )

    df_fix["Date"] = dates

    # --------------------------------------------------------
    # FIRST DATE
    # --------------------------------------------------------

    first_valid_date = (
        df_fix["Date"]
        .dropna()
        .iloc[0]
    )

    first_date = (
        first_valid_date
        .replace(
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    )

    day_number = (
        df_fix["Date"]
        - first_date
    ).dt.days

    # --------------------------------------------------------
    # DECLINATION
    # --------------------------------------------------------

    df_fix[
        "Declination Angle ∆"
    ] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_number
                    + 1
                )
                / 365
            )
        )
    )

    # --------------------------------------------------------
    # ELEVATION
    # --------------------------------------------------------

    df_fix[
        "Elevation angle a"
    ] = (
        90
        - lat
        + df_fix[
            "Declination Angle ∆"
        ]
    )

    # --------------------------------------------------------
    # MONTH TILT
    # --------------------------------------------------------

    month_names = (
        df_fix["Date"]
        .dt.strftime("%B")
    )

    df_fix[
        "Tilt Angle b"
    ] = (
        month_names
        .map(month_lookup)
    )

    df_fix[
        "Tilt Angle b"
    ] = numeric_series(
        df_fix[
            "Tilt Angle b"
        ]
    )

    # --------------------------------------------------------
    # ANGLE
    # --------------------------------------------------------

    df_fix["a+b"] = (
        df_fix[
            "Elevation angle a"
        ]
        + df_fix[
            "Tilt Angle b"
        ]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(
            df_fix[
                "Elevation angle a"
            ]
        )
    )

    # --------------------------------------------------------
    # SAFE DENOMINATOR
    # --------------------------------------------------------

    sin_a = df_fix[
        "Sin(a)"
    ].to_numpy(
        dtype=float
    )

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-6,
        1e-6,
        sin_a,
    )

    # --------------------------------------------------------
    # POA FOR ALL CLUSTERS
    # --------------------------------------------------------

    poa_names = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    for i, ghi_col in enumerate(
        [
            "GHI C11",
            "GHI C12",
            "GHI C13",
            "GHI C14",
            "GHI C15",
        ]
    ):

        ghi = numeric_array(
            df_ghi[ghi_col]
        )

        # Align length
        min_len = min(
            len(df_fix),
            len(ghi),
        )

        if i == 0:

            pass

        ghi = ghi[
            :len(df_fix)
        ]

        if len(ghi) < len(df_fix):

            ghi = np.pad(
                ghi,
                (
                    0,
                    len(df_fix) - len(ghi),
                ),
                mode="constant",
            )

        df_fix[
            f"GHI*sin(a)-CL{i+1}"
        ] = (
            ghi
            * df_fix[
                "Sin(a)"
            ].to_numpy()
        )

        df_fix[
            f"GHI*sin(a+b)-CL{i+1}"
        ] = (
            ghi
            * df_fix[
                "SIN(a+b)"
            ].to_numpy()
        )

        df_fix[
            poa_names[i]
        ] = (
            df_fix[
                f"GHI*sin(a+b)-CL{i+1}"
            ]
            / sin_a_safe
        )

    return df_fix


# ============================================================
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    df_fix_base,
    error_percent,
):

    df_fix = df_fix_base.copy()

    (
        df_area_result,
        df_w,
    ) = calculate_effective_area(
        df_area,
        df_weights,
        error_percent,
    )

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    power_cols = []

    for i in range(5):

        cluster_no = i + 1

        power_col = (
            f"CL{cluster_no}_Fixed Power=I*Ƞ*A"
        )

        df_fix[
            power_col
        ] = (
            df_fix[
                poa_cols[i]
            ]
            * float(
                df_w.iloc[i][
                    "Eff Area(m2)"
                ]
            )
            / 1_000_000
        )

        power_cols.append(
            power_col
        )

    df_fix[
        "Total Power (CL1+CL2+…)"
    ] = (
        df_fix[
            power_cols
        ].sum(axis=1)
    )

    return (
        df_fix,
        df_area_result,
        df_w,
    )


# ============================================================
# FIXED OPTIMIZATION
# ============================================================

def optimize_fixed(
    df_fix_base,
    minimum,
    maximum,
    step,
):

    actual = numeric_series(
        df_fix_base["Actual"]
    ).to_numpy(
        dtype=float
    )

    actual_peak = np.max(
        actual
    )

    if actual_peak <= 0:

        raise ValueError(
            "Actual power peak is zero."
        )

    errors = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    results = []

    for error in errors:

        (
            temp_df,
            _,
            temp_w,
        ) = calculate_fixed_power(
            df_fix_base,
            float(error),
        )

        forecast = numeric_array(
            temp_df[
                "Total Power (CL1+CL2+…)"
            ]
        )

        calculated_peak = np.max(
            forecast
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
                "Error %": float(error),
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": peak_error_pct,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    best_index = (
        result_df[
            "Peak Error"
        ].idxmin()
    )

    best_error = float(
        result_df.loc[
            best_index,
            "Error %",
        ]
    )

    (
        final_df,
        final_area,
        final_weights,
    ) = calculate_fixed_power(
        df_fix_base,
        best_error,
    )

    return (
        best_error,
        result_df,
        final_df,
        final_area,
        final_weights,
    )


# ============================================================
# TRACKING DATA PREPARATION
# ============================================================

def prepare_tracking_data():

    workbook = io.BytesIO(
        file_bytes
    )

    # --------------------------------------------------------
    # BACKEND SHEETS
    # --------------------------------------------------------

    backend_sheets = [
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
    ]

    backend_list = []

    for sheet in backend_sheets:

        temp = pd.read_excel(
            workbook,
            sheet_name=sheet,
        )

        temp.columns = (
            temp.columns
            .astype(str)
            .str.strip()
        )

        backend_list.append(
            temp
        )

    # --------------------------------------------------------
    # TRACKING SHEET
    # --------------------------------------------------------

    df_tracking = pd.read_excel(
        workbook,
        sheet_name="Tracking",
        header=1,
    )

    df_tracking.columns = (
        df_tracking.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------

    check_columns(
        backend_list[0],
        ["Block No."],
        "Backend Cal C11",
    )

    blocks = numeric_array(
        backend_list[0][
            "Block No."
        ]
    )

    # --------------------------------------------------------
    # GHI MATRIX
    # --------------------------------------------------------

    ghi_matrix = np.column_stack(
        [
            numeric_array(
                df_ghi[col]
            )
            for col in [
                "GHI C11",
                "GHI C12",
                "GHI C13",
                "GHI C14",
                "GHI C15",
            ]
        ]
    )

    # --------------------------------------------------------
    # ACTUAL
    #
    # IMPORTANT:
    # Tracking optimization uses Actual
    # from Fixed-C11.
    # --------------------------------------------------------

    df_actual = pd.read_excel(
        workbook,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_actual.columns = (
        df_actual.columns
        .astype(str)
        .str.strip()
    )

    df_actual = clean_until_first_null(
        df_actual,
        "Date",
    )

    check_columns(
        df_actual,
        ["Actual"],
        "Fixed-C11",
    )

    actual_full = numeric_array(
        df_actual["Actual"]
    )

    # --------------------------------------------------------
    # ALIGN LENGTHS
    # --------------------------------------------------------

    min_len = min(
        len(blocks),
        len(ghi_matrix),
        len(actual_full),
    )

    if min_len == 0:

        raise ValueError(
            "No usable Tracking data found."
        )

    blocks = blocks[
        :min_len
    ]

    ghi_matrix = ghi_matrix[
        :min_len
    ]

    actual_full = actual_full[
        :min_len
    ]

    # --------------------------------------------------------
    # VALIDATE ACTUAL
    # --------------------------------------------------------

    if not np.any(
        actual_full != 0
    ):

        raise ValueError(
            "No non-zero Actual values found "
            "in Fixed-C11."
        )

    return (
        backend_list,
        df_tracking,
        blocks,
        ghi_matrix,
        actual_full,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def build_tracking_objective(
    blocks,
    ghi_matrix,
    actual_full,
    cluster_weights,
    block_weight,
    peak_weight,
    energy_weight,
):

    # --------------------------------------------------------
    # DAYLIGHT MASK
    #
    # Actual must be non-zero.
    # GHI must also have some useful irradiance.
    # --------------------------------------------------------

    ghi_max = np.max(
        ghi_matrix,
        axis=1,
    )

    mask = (
        (actual_full > 0)
        &
        (ghi_max > 1)
    )

    if not np.any(mask):

        raise ValueError(
            "No valid daylight Actual/GHI points "
            "available for Tracking optimization."
        )

    actual = actual_full[
        mask
    ]

    actual_max = np.max(
        actual
    )

    actual_sum = np.sum(
        actual
    )

    if actual_max <= 0:

        raise ValueError(
            "Actual peak is zero."
        )

    if actual_sum <= 0:

        raise ValueError(
            "Actual energy is zero."
        )

    def objective(x):

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

        # ----------------------------------------------------
        # BLOCK VALIDATION
        # ----------------------------------------------------

        if not (
            start_block
            < max_block
            < end_block
        ):

            return 1e9

        den1 = (
            start_block
            - 1
            - max_block
        )

        den2 = (
            end_block
            + 1
            - max_block
        )

        if (
            den1 == 0
            or den2 == 0
        ):

            return 1e9

        # ----------------------------------------------------
        # SLOPE
        # ----------------------------------------------------

        m1 = 90 / den1

        m2 = 90 / den2

        # ----------------------------------------------------
        # ZENITH
        # ----------------------------------------------------

        zenith = np.where(

            blocks <= max_block,

            np.minimum(
                89,
                m1
                * (
                    blocks
                    - max_block
                ),
            ),

            np.minimum(
                89,
                m2
                * (
                    blocks
                    - max_block
                ),
            ),
        )

        zenith = np.abs(
            zenith
        )

        # ----------------------------------------------------
        # PANEL ANGLE
        # ----------------------------------------------------

        panel = np.where(

            blocks < max_block,

            np.minimum(
                zenith,
                abs(east_limit),
            ),

            np.where(

                (
                    (blocks > max_block)
                    &
                    (
                        zenith
                        > west_limit
                    )
                ),

                west_limit,

                zenith,
            ),
        )

        panel = np.abs(
            panel
        )

        # ----------------------------------------------------
        # COSINE
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(
                panel
            )
        )

        cos_alpha = np.clip(
            cos_alpha,
            1e-6,
            None,
        )

        # ----------------------------------------------------
        # DHI
        # ----------------------------------------------------

        dhi = (
            ghi_matrix
            * DHI
            / 100
        )

        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        prediction_full = (
            dni @ cluster_weights
        ) / 1_000_000

        if (
            np.isnan(
                prediction_full
            ).any()
            or np.isinf(
                prediction_full
            ).any()
        ):

            return 1e9

        prediction = (
            prediction_full[
                mask
            ]
        )

        if len(prediction) == 0:

            return 1e9

        # ----------------------------------------------------
        # BLOCK ERROR
        # ----------------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            / actual_max
        )

        # ----------------------------------------------------
        # PEAK ERROR
        # ----------------------------------------------------

        peak_error = (
            abs(
                actual_max
                - np.max(
                    prediction
                )
            )
            / actual_max
        )

        # ----------------------------------------------------
        # ENERGY ERROR
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_sum
                - np.sum(
                    prediction
                )
            )
            / actual_sum
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = (
            block_weight
            * block_error
            +
            peak_weight
            * peak_error
            +
            energy_weight
            * energy_error
        )

        return score

    return objective


# ============================================================
# TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    parameters,
    blocks,
    ghi_matrix,
    cluster_weights,
):

    DHI = int(
        round(parameters[0])
    )

    start_block = int(
        round(parameters[1])
    )

    end_block = int(
        round(parameters[2])
    )

    max_block = int(
        round(parameters[3])
    )

    east_limit = int(
        round(parameters[4])
    )

    west_limit = int(
        round(parameters[5])
    )

    den1 = (
        start_block
        - 1
        - max_block
    )

    den2 = (
        end_block
        + 1
        - max_block
    )

    if den1 == 0 or den2 == 0:

        raise ValueError(
            "Invalid Tracking block parameters."
        )

    m1 = 90 / den1

    m2 = 90 / den2

    # --------------------------------------------------------
    # ZENITH
    # --------------------------------------------------------

    zenith = np.where(

        blocks <= max_block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - max_block
            ),
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - max_block
            ),
        ),
    )

    zenith = np.abs(
        zenith
    )

    # --------------------------------------------------------
    # PANEL ANGLE
    # --------------------------------------------------------

    panel = np.where(

        blocks < max_block,

        np.minimum(
            zenith,
            abs(east_limit),
        ),

        np.where(

            (
                (blocks > max_block)
                &
                (
                    zenith
                    > west_limit
                )
            ),

            west_limit,

            zenith,
        ),
    )

    panel = np.abs(
        panel
    )

    # --------------------------------------------------------
    # COSINE
    # --------------------------------------------------------

    cos_alpha = np.cos(
        np.radians(
            panel
        )
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None,
    )

    # --------------------------------------------------------
    # DHI
    # --------------------------------------------------------

    dhi = (
        ghi_matrix
        * DHI
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
    # POWER
    # --------------------------------------------------------

    forecast = (
        dni @ cluster_weights
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# FIXED UI
# ============================================================

if plant_type == "Fixed":

    st.markdown(
        '<div class="section-title">'
        "Fixed Plant Parameters"
        "</div>",
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        error_min = st.number_input(
            "Error % Minimum",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=0.1,
        )

    with f2:

        error_max = st.number_input(
            "Error % Maximum",
            min_value=-50.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
        )

    with f3:

        error_step = st.number_input(
            "Error % Step",
            min_value=0.01,
            max_value=10.0,
            value=0.1,
            step=0.01,
        )

    if error_max <= error_min:

        st.error(
            "Error % Maximum must be greater than Minimum."
        )

        st.stop()

    # --------------------------------------------------------
    # PREPARE FIXED DATA
    # --------------------------------------------------------

    try:

        df_fix_base = prepare_fixed_data()

    except Exception as e:

        st.error(
            f"Fixed data preparation failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "Run Correction"
        "</div>",
        unsafe_allow_html=True,
    )

    run_fixed = st.button(
        "▶ Run Fixed Correction",
        type="primary",
        use_container_width=True,
    )

    if run_fixed:

        with st.spinner(
            "Searching for the best Error %..."
        ):

            try:

                (
                    best_error,
                    fixed_results,
                    final_fixed,
                    final_area,
                    final_weights,
                ) = optimize_fixed(
                    df_fix_base,
                    error_min,
                    error_max,
                    error_step,
                )

                st.session_state[
                    "fixed_best_error"
                ] = best_error

                st.session_state[
                    "fixed_results"
                ] = fixed_results

                st.session_state[
                    "fixed_final"
                ] = final_fixed

                st.session_state[
                    "fixed_area"
                ] = final_area

                st.session_state[
                    "fixed_weights"
                ] = final_weights

                st.success(
                    "Fixed correction completed successfully."
                )

            except Exception as e:

                st.error(
                    f"Fixed optimization failed: {e}"
                )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    if (
        "fixed_final"
        in st.session_state
    ):

        final_fixed = (
            st.session_state[
                "fixed_final"
            ]
        )

        fixed_results = (
            st.session_state[
                "fixed_results"
            ]
        )

        best_error = (
            st.session_state[
                "fixed_best_error"
            ]
        )

        forecast = numeric_array(
            final_fixed[
                "Total Power (CL1+CL2+…)"
            ]
        )

        actual = numeric_array(
            final_fixed[
                "Actual"
            ]
        )

        min_len = min(
            len(forecast),
            len(actual),
        )

        forecast = forecast[
            :min_len
        ]

        actual = actual[
            :min_len
        ]

        actual_peak = np.max(
            actual
        )

        forecast_peak = np.max(
            forecast
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
            else np.nan
        )

        actual_energy = np.sum(
            actual
        )

        forecast_energy = np.sum(
            forecast
        )

        energy_error_pct = (
            abs(
                forecast_energy
                - actual_energy
            )
            / actual_energy
            * 100
            if actual_energy != 0
            else np.nan
        )

        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Fixed Correction Result"
            "</div>",
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4, r5 = st.columns(5)

        with r1:

            st.metric(
                "Best Error %",
                f"{best_error:.2f}%",
            )

        with r2:

            st.metric(
                "Actual Peak",
                f"{actual_peak:.3f} MW",
            )

        with r3:

            st.metric(
                "Forecast Peak",
                f"{forecast_peak:.3f} MW",
            )

        with r4:

            st.metric(
                "Peak Error",
                f"{peak_error_pct:.2f}%",
            )

        with r5:

            st.metric(
                "Energy Error",
                f"{energy_error_pct:.2f}%",
            )

        # ----------------------------------------------------
        # FORECAST VS ACTUAL
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Forecast vs Actual"
            "</div>",
            unsafe_allow_html=True,
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
                    width=2,
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
                    width=2,
                ),
            )
        )

        fig.update_layout(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            xaxis_title="15-Minute Block",
            yaxis_title="Power (MW)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # ERROR OPTIMIZATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Error % Optimization"
            "</div>",
            unsafe_allow_html=True,
        )

        fig_error = go.Figure()

        fig_error.add_trace(
            go.Scatter(
                x=fixed_results[
                    "Error %"
                ],
                y=fixed_results[
                    "Peak Error %"
                ],
                mode="lines+markers",
                name="Peak Error %",
            )
        )

        fig_error.add_vline(
            x=best_error,
            line_dash="dash",
            annotation_text=(
                f"Best Error = "
                f"{best_error:.2f}%"
            ),
            annotation_position="top",
        )

        fig_error.update_layout(
            template="plotly_white",
            height=420,
            hovermode="x unified",
            xaxis_title="Error %",
            yaxis_title="Peak Error %",
        )

        st.plotly_chart(
            fig_error,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # EFFECTIVE AREA
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Cluster Effective Area"
            "</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            st.session_state[
                "fixed_weights"
            ],
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # OPTIMIZATION TABLE
        # ----------------------------------------------------

        with st.expander(
            "View Complete Error Optimization Results"
        ):

            st.dataframe(
                fixed_results,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# TRACKING UI
# ============================================================

else:

    st.markdown(
        '<div class="section-title">'
        "Tracking Optimization Parameters"
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PARAMETER BOUNDS
    # --------------------------------------------------------

    st.markdown(
        "**DHI Parameters**"
    )

    a1, a2 = st.columns(2)

    with a1:

        dhi_min = st.number_input(
            "DHI Minimum (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    with a2:

        dhi_max = st.number_input(
            "DHI Maximum (%)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        )

    st.markdown(
        "**GHI Block Parameters**"
    )

    b1, b2, b3, b4, b5, b6 = st.columns(6)

    with b1:

        start_min = st.number_input(
            "Start Min",
            min_value=1,
            max_value=95,
            value=10,
            step=1,
        )

    with b2:

        start_max = st.number_input(
            "Start Max",
            min_value=1,
            max_value=95,
            value=30,
            step=1,
        )

    with b3:

        max_min = st.number_input(
            "Max Block Min",
            min_value=1,
            max_value=95,
            value=47,
            step=1,
        )

    with b4:

        max_max = st.number_input(
            "Max Block Max",
            min_value=1,
            max_value=95,
            value=53,
            step=1,
        )

    with b5:

        end_min = st.number_input(
            "End Min",
            min_value=1,
            max_value=95,
            value=65,
            step=1,
        )

    with b6:

        end_max = st.number_input(
            "End Max",
            min_value=1,
            max_value=95,
            value=80,
            step=1,
        )

    st.markdown(
        "**Tracking Angle Parameters**"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        east_min = st.number_input(
            "East Limit Min",
            min_value=1,
            max_value=90,
            value=10,
            step=1,
        )

    with c2:

        east_max = st.number_input(
            "East Limit Max",
            min_value=1,
            max_value=90,
            value=70,
            step=1,
        )

    with c3:

        west_min = st.number_input(
            "West Limit Min",
            min_value=1,
            max_value=90,
            value=10,
            step=1,
        )

    with c4:

        west_max = st.number_input(
            "West Limit Max",
            min_value=1,
            max_value=90,
            value=70,
            step=1,
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    st.markdown(
        "**Differential Evolution Settings**"
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        maxiter = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=500,
            value=40,
            step=1,
        )

    with d2:

        popsize = st.number_input(
            "Population Size",
            min_value=1,
            max_value=100,
            value=15,
            step=1,
        )

    with d3:

        tolerance = st.number_input(
            "Tolerance",
            min_value=0.00001,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            format="%.5f",
        )

    with d4:

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    st.markdown(
        "**Optimization Score Weights**"
    )

    w1, w2, w3 = st.columns(3)

    with w1:

        block_weight = st.number_input(
            "Block Error Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.80,
            step=0.05,
        )

    with w2:

        peak_weight = st.number_input(
            "Peak Error Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
        )

    with w3:

        energy_weight = st.number_input(
            "Energy Error Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
        )

    total_weight = (
        block_weight
        + peak_weight
        + energy_weight
    )

    if abs(
        total_weight - 1.0
    ) > 0.0001:

        st.warning(
            f"Score weights total "
            f"{total_weight:.2f}. "
            "Recommended total is 1.00."
        )

    # --------------------------------------------------------
    # VALIDATE BOUNDS
    # --------------------------------------------------------

    if (
        dhi_min > dhi_max
        or start_min > start_max
        or max_min > max_max
        or end_min > end_max
        or east_min > east_max
        or west_min > west_max
    ):

        st.error(
            "Minimum value cannot be greater than "
            "Maximum value."
        )

        st.stop()

    # --------------------------------------------------------
    # PREPARE TRACKING DATA
    # --------------------------------------------------------

    try:

        (
            backend_list,
            df_tracking,
            blocks,
            ghi_matrix,
            actual_full,
        ) = prepare_tracking_data()

    except Exception as e:

        st.error(
            f"Tracking data preparation failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # CLUSTER WEIGHTS
    # --------------------------------------------------------

    try:

        cluster_weights = (
            pd.to_numeric(
                df_weights.iloc[
                    :5,
                    1,
                ],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )

    except Exception as e:

        st.error(
            f"Cluster effective area calculation failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # CHECK FIVE CLUSTERS
    # --------------------------------------------------------

    if len(cluster_weights) < 5:

        cluster_weights = np.pad(
            cluster_weights,
            (
                0,
                5 - len(
                    cluster_weights
                ),
            ),
            mode="constant",
        )

    cluster_weights = (
        cluster_weights[:5]
    )

    # --------------------------------------------------------
    # RUN BUTTON
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "Run Tracking Optimization"
        "</div>",
        unsafe_allow_html=True,
    )

    run_tracking = st.button(
        "▶ Run Tracking Optimization",
        type="primary",
        use_container_width=True,
    )

    if run_tracking:

        with st.spinner(
            "Running Tracking optimization..."
        ):

            try:

                objective = (
                    build_tracking_objective(
                        blocks=blocks,
                        ghi_matrix=ghi_matrix,
                        actual_full=actual_full,
                        cluster_weights=cluster_weights,
                        block_weight=block_weight,
                        peak_weight=peak_weight,
                        energy_weight=energy_weight,
                    )
                )

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

                # ------------------------------------------------
                # IMPORTANT VALIDATION
                # ------------------------------------------------

                if not (
                    start_max
                    < max_min
                    or max_max
                    < start_min
                ):

                    st.warning(
                        "GHI Start and GHI Max ranges overlap. "
                        "The optimizer will reject invalid combinations."
                    )

                if not (
                    max_max
                    < end_min
                    or end_max
                    < max_min
                ):

                    st.warning(
                        "GHI Max and GHI End ranges overlap. "
                        "The optimizer will reject invalid combinations."
                    )

                # ------------------------------------------------
                # OPTIMIZATION
                # ------------------------------------------------

                result = differential_evolution(
                    objective,
                    bounds=bounds,
                    strategy="best1bin",
                    maxiter=int(
                        maxiter
                    ),
                    popsize=int(
                        popsize
                    ),
                    tol=float(
                        tolerance
                    ),
                    mutation=(
                        0.5,
                        1.0,
                    ),
                    recombination=0.7,
                    seed=int(
                        seed
                    ),
                    polish=True,
                    workers=1,
                )

                if (
                    not result.success
                    and result.fun >= 1e8
                ):

                    raise ValueError(
                        "Optimizer could not find "
                        "a valid parameter combination. "
                        "Check the GHI block ranges."
                    )

                best = (
                    np.round(
                        result.x
                    ).astype(int)
                )

                (
                    forecast,
                    zenith,
                    panel,
                ) = calculate_tracking_forecast(
                    best,
                    blocks,
                    ghi_matrix,
                    cluster_weights,
                )

                st.session_state[
                    "tracking_result"
                ] = result

                st.session_state[
                    "tracking_best"
                ] = best

                st.session_state[
                    "tracking_forecast"
                ] = forecast

                st.session_state[
                    "tracking_zenith"
                ] = zenith

                st.session_state[
                    "tracking_panel"
                ] = panel

                st.session_state[
                    "tracking_blocks"
                ] = blocks

                st.session_state[
                    "tracking_actual"
                ] = actual_full

                st.success(
                    "Tracking optimization completed successfully."
                )

            except Exception as e:

                st.error(
                    f"Tracking optimization failed: {e}"
                )


    # ========================================================
    # TRACKING RESULTS
    # ========================================================

    if (
        "tracking_best"
        in st.session_state
    ):

        best = st.session_state[
            "tracking_best"
        ]

        result = st.session_state[
            "tracking_result"
        ]

        forecast = st.session_state[
            "tracking_forecast"
        ]

        zenith = st.session_state[
            "tracking_zenith"
        ]

        panel = st.session_state[
            "tracking_panel"
        ]

        blocks_result = st.session_state[
            "tracking_blocks"
        ]

        actual = st.session_state[
            "tracking_actual"
        ]

        # ----------------------------------------------------
        # ALIGN
        # ----------------------------------------------------

        min_len = min(
            len(forecast),
            len(actual),
            len(blocks_result),
        )

        forecast = forecast[
            :min_len
        ]

        actual = actual[
            :min_len
        ]

        blocks_result = blocks_result[
            :min_len
        ]

        zenith = zenith[
            :min_len
        ]

        panel = panel[
            :min_len
        ]

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        DHI = best[0]

        GHI_Starting_Block = best[1]

        GHI_Ending_Block = best[2]

        GHI_Max_Block = best[3]

        Tracking_angle_lim_E = best[4]

        Tracking_angle_lim_W = best[5]

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        valid_mask = (
            actual > 0
        )

        if not np.any(
            valid_mask
        ):

            st.error(
                "No non-zero Actual values available "
                "after optimization."
            )

            st.stop()

        actual_valid = actual[
            valid_mask
        ]

        forecast_valid = forecast[
            valid_mask
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
        )

        actual_energy = np.sum(
            actual_valid
        )

        forecast_energy = np.sum(
            forecast_valid
        )

        energy_error_pct = (
            abs(
                forecast_energy
                - actual_energy
            )
            / actual_energy
            * 100
        )

        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Tracking Optimization Result"
            "</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:

            st.metric(
                "Optimization Score",
                f"{result.fun:.5f}",
            )

        with m2:

            st.metric(
                "Actual Peak",
                f"{actual_peak:.3f} MW",
            )

        with m3:

            st.metric(
                "Forecast Peak",
                f"{forecast_peak:.3f} MW",
            )

        with m4:

            st.metric(
                "Peak Error",
                f"{peak_error_pct:.2f}%",
            )

        with m5:

            st.metric(
                "Energy Error",
                f"{energy_error_pct:.2f}%",
            )

        # ----------------------------------------------------
        # BEST PARAMETERS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Optimized Parameters"
            "</div>",
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4, p5, p6 = st.columns(6)

        with p1:

            st.metric(
                "DHI",
                f"{DHI}%",
            )

        with p2:

            st.metric(
                "GHI Start",
                GHI_Starting_Block,
            )

        with p3:

            st.metric(
                "GHI Max",
                GHI_Max_Block,
            )

        with p4:

            st.metric(
                "GHI End",
                GHI_Ending_Block,
            )

        with p5:

            st.metric(
                "East Limit",
                f"{Tracking_angle_lim_E}°",
            )

        with p6:

            st.metric(
                "West Limit",
                f"{Tracking_angle_lim_W}°",
            )

        # ----------------------------------------------------
        # FORECAST VS ACTUAL
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Tracking Forecast vs Actual"
            "</div>",
            unsafe_allow_html=True,
        )

        fig_tracking = go.Figure()

        fig_tracking.add_trace(
            go.Scatter(
                x=blocks_result,
                y=forecast,
                mode="lines",
                name="Tracking Forecast",
                line=dict(
                    width=2,
                ),
            )
        )

        fig_tracking.add_trace(
            go.Scatter(
                x=blocks_result,
                y=actual,
                mode="lines",
                name="Actual",
                line=dict(
                    width=2,
                ),
            )
        )

        fig_tracking.update_layout(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            xaxis_title="Block",
            yaxis_title="Power (MW)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(
            fig_tracking,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # TRACKING ANGLES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Tracking Angle Profile"
            "</div>",
            unsafe_allow_html=True,
        )

        fig_angles = go.Figure()

        fig_angles.add_trace(
            go.Scatter(
                x=blocks_result,
                y=zenith,
                mode="lines",
                name="Zenith Angle",
            )
        )

        fig_angles.add_trace(
            go.Scatter(
                x=blocks_result,
                y=panel,
                mode="lines",
                name="Panel Angle",
            )
        )

        fig_angles.update_layout(
            template="plotly_white",
            height=450,
            hovermode="x unified",
            xaxis_title="Block",
            yaxis_title="Angle (°)",
        )

        st.plotly_chart(
            fig_angles,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # TRACKING DATA TABLE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            "Tracking Calculation"
            "</div>",
            unsafe_allow_html=True,
        )

        tracking_output = (
            df_tracking
            .copy()
            .iloc[
                :min_len
            ]
        )

        tracking_output[
            "Zenith Angle"
        ] = zenith

        tracking_output[
            "Panel Angle"
        ] = panel

        tracking_output[
            "Tracking Forecast Power"
        ] = forecast

        tracking_output[
            "Actual Power"
        ] = actual

        with st.expander(
            "View Tracking Calculation Data"
        ):

            st.dataframe(
                tracking_output,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = (
            tracking_output
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(
            "Download Tracking Result",
            data=csv_data,
            file_name="tracking_forecast_result.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <hr>
    <div style="
        text-align:center;
        color:#6b7280;
        font-size:0.85rem;
    ">
        Solar Forecast Correction
    </div>
    """,
    unsafe_allow_html=True,
)
