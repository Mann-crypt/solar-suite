# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime, timedelta
from scipy.optimize import differential_evolution


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loss Correction — Solar Suite",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <h1 style='text-align:center;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-size:40px;
    font-weight:800;'>
    ⚡ Solar Suite
    </h1>

    <p style='text-align:center;color:gray;font-size:14px;'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    div[data-testid="stDataEditor"] {
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] {
        margin-bottom: 5px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
]

N_CLUSTERS = len(CLUSTERS)

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]


# ============================================================
# SESSION STATE
# ============================================================

if "lc_workbook_type" not in st.session_state:
    st.session_state.lc_workbook_type = None

if "lc_fixed_result" not in st.session_state:
    st.session_state.lc_fixed_result = None

if "lc_tracking_result" not in st.session_state:
    st.session_state.lc_tracking_result = None


# ============================================================
# TITLE
# ============================================================

st.title(
    "Guruji ne kaha tha Loss Correction kardo bhyii 🛐!!"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx", "xls"],
)


if uploaded_file is None:

    st.info(
        "Please upload the Excel workbook to start the "
        "Fixed / Tracking loss correction."
    )

    st.stop()


# ============================================================
# LOAD EXCEL WORKBOOK
# ============================================================

try:

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheet_names = excel_file.sheet_names

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# WORKBOOK TYPE IDENTIFICATION
# ============================================================

# Priority:
#
# Fixed-C11  -> VCast
# Fixed-CL1  -> Cluster
# Fixed      -> Non-cluster
#
# ============================================================

if "Fixed-C11" in sheet_names:

    workbook_type = "VCast"

    fixed_sheet = "Fixed-C11"

elif "Fixed-CL1" in sheet_names:

    workbook_type = "Cluster"

    fixed_sheet = "Fixed-CL1"

elif "Fixed" in sheet_names:

    workbook_type = "Non-cluster"

    fixed_sheet = "Fixed"

else:

    st.error(
        "Workbook type could not be identified.\n\n"
        "Expected one of:\n"
        "- Fixed\n"
        "- Fixed-CL1\n"
        "- Fixed-C11"
    )

    st.stop()


st.session_state.lc_workbook_type = workbook_type


# ============================================================
# WORKBOOK TYPE DISPLAY
# ============================================================

type_col1, type_col2, type_col3 = st.columns(3)

with type_col1:
    st.metric(
        "Workbook Type",
        workbook_type,
    )

with type_col2:
    st.metric(
        "Fixed Sheet",
        fixed_sheet,
    )

with type_col3:
    st.metric(
        "Tracking Available",
        "Yes" if "Tracking" in sheet_names else "No",
    )


st.divider()


# ============================================================
# HELPER: NUMERIC COLUMN
# ============================================================

def numeric_array(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)


# ============================================================
# 1. AREA & EFFICIENCY
# ============================================================

try:

    area_raw = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=None,
    )

except Exception as e:

    st.error(
        f"Unable to read Area & Efficiency sheet: {e}"
    )

    st.stop()


# ------------------------------------------------------------
# AREA & EFFICIENCY TABLE
# ------------------------------------------------------------

try:

    area_df = pd.read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    area_df.columns = (
        area_df.columns
        .astype(str)
        .str.replace(
            "*",
            "",
            regex=False,
        )
        .str.strip()
    )

except Exception as e:

    st.error(
        f"Unable to process Area & Efficiency: {e}"
    )

    st.stop()


# ============================================================
# 2. EFFECTIVE AREAS
# ============================================================

# The VCast workbook uses:
#
# Fixed:
# P3:P7
#
# Tracking:
# P29:P33
#
# ============================================================

try:

    fixed_weights = (
        pd.to_numeric(
            area_raw.iloc[
                2:7,
                15
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    tracking_weights = (
        pd.to_numeric(
            area_raw.iloc[
                28:33,
                15
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

except Exception as e:

    st.error(
        f"Unable to read effective areas: {e}"
    )

    st.stop()


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if len(fixed_weights) != N_CLUSTERS:

    st.error(
        "Could not identify 5 Fixed effective-area values."
    )

    st.stop()


if len(tracking_weights) != N_CLUSTERS:

    st.error(
        "Could not identify 5 Tracking effective-area values."
    )

    st.stop()


# ============================================================
# 3. STANDARD PV EFFICIENCY
# ============================================================

if "Standard PV Efficiency (%)" not in area_df.columns:

    st.error(
        "Column 'Standard PV Efficiency (%)' "
        "was not found in Area & Efficiency."
    )

    st.stop()


standard_efficiency = numeric_array(
    area_df[
        "Standard PV Efficiency (%)"
    ]
)


if len(standard_efficiency) < N_CLUSTERS:

    st.error(
        "Less than 5 Standard PV Efficiency values found."
    )

    st.stop()


standard_efficiency = (
    standard_efficiency[
        :N_CLUSTERS
    ]
)


# ============================================================
# 4. FORECAST CONFIG
# ============================================================

try:

    df_config = pd.read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = float(
        pd.to_numeric(
            df_config.loc[
                0,
                "Lat"
            ],
            errors="coerce",
        )
    )

except Exception as e:

    st.error(
        f"Unable to read latitude from Forecast Config: {e}"
    )

    st.stop()


# ============================================================
# 5. CONFIG TILT ANGLE
# ============================================================

try:

    df_tilt = pd.read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    df_tilt = df_tilt.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    df_tilt = df_tilt.dropna(
        subset=["Fixed"]
    ).copy()

    df_tilt["Month_Num"] = pd.to_numeric(
        df_tilt["Month_Num"],
        errors="coerce",
    )

    df_tilt["Fixed"] = pd.to_numeric(
        df_tilt["Fixed"],
        errors="coerce",
    )

    month_number_to_tilt = (
        df_tilt
        .dropna(
            subset=["Month_Num"]
        )
        .set_index("Month_Num")[
            "Fixed"
        ]
        .to_dict()
    )

except Exception as e:

    st.error(
        f"Unable to read Config Tilt Angle: {e}"
    )

    st.stop()


# ============================================================
# 6. RESULT / GHI
# ============================================================

try:

    df_ghi = pd.read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=range(6),
    )

    df_ghi.columns = [
        "Block",
        *GHI_COLS,
    ]

except Exception as e:

    st.error(
        f"Unable to read Result sheet: {e}"
    )

    st.stop()


df_ghi = df_ghi[
    pd.to_numeric(
        df_ghi["Block"],
        errors="coerce"
    ).notna()
].copy()


for col in GHI_COLS:

    df_ghi[col] = pd.to_numeric(
        df_ghi[col],
        errors="coerce"
    ).fillna(0)


blocks_result = pd.to_numeric(
    df_ghi["Block"],
    errors="coerce",
).to_numpy(dtype=float)


ghi_matrix = np.column_stack(
    [
        df_ghi[col].to_numpy(
            dtype=float
        )
        for col in GHI_COLS
    ]
)


# ============================================================
# 7. FIXED SHEET
# ============================================================

try:

    df_fix = pd.read_excel(
        uploaded_file,
        sheet_name=fixed_sheet,
        header=1,
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

except Exception as e:

    st.error(
        f"Unable to read {fixed_sheet}: {e}"
    )

    st.stop()


# ============================================================
# 8. DATE VALIDATION
# ============================================================

if "Date" not in df_fix.columns:

    st.error(
        f"'Date' column not found in {fixed_sheet}."
    )

    st.stop()


date_valid = (
    df_fix["Date"].notna()
)


if not date_valid.any():

    st.error(
        f"No valid dates found in {fixed_sheet}."
    )

    st.stop()


first_blank = np.where(
    ~date_valid.to_numpy()
)[0]


if len(first_blank) > 0:

    df_fix = df_fix.iloc[
        :first_blank[0]
    ].copy()

else:

    df_fix = df_fix.loc[
        date_valid
    ].copy()


df_fix.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# 9. ACTUAL POWER
# ============================================================

if "Actual" not in df_fix.columns:

    st.error(
        f"'Actual' column not found in {fixed_sheet}."
    )

    st.stop()


actual_full = numeric_array(
    df_fix["Actual"]
)


# ============================================================
# 10. ALIGN DATA
# ============================================================

n = min(
    len(df_fix),
    len(df_ghi),
)


if n == 0:

    st.error(
        "No valid rows available."
    )

    st.stop()


df_fix = df_fix.iloc[
    :n
].copy()

df_ghi = df_ghi.iloc[
    :n
].copy()


actual = actual_full[
    :n
]

ghi_matrix = ghi_matrix[
    :n
]

blocks = blocks_result[
    :n
]


# ============================================================
# 11. DATES
# ============================================================

dates = pd.to_datetime(
    df_fix["Date"],
    errors="coerce",
)


if dates.isna().any():

    st.error(
        "Invalid dates found in Fixed sheet."
    )

    st.stop()


# ============================================================
# 12. SOLAR GEOMETRY
# ============================================================

first_date = pd.Timestamp(
    year=2025,
    month=1,
    day=1,
)


day_offset = (
    dates
    - first_date
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
    - lat
    + declination
)


months = dates.dt.month.to_numpy()


tilt = np.array(
    [
        month_number_to_tilt.get(
            float(month),
            0,
        )
        for month in months
    ]
)


a_plus_b = (
    elevation
    + tilt
)


sin_a = np.sin(
    np.radians(
        elevation
    )
)

sin_ab = np.sin(
    np.radians(
        a_plus_b
    )
)


sin_a_safe = np.where(
    np.abs(sin_a) < 1e-8,
    1e-8,
    sin_a,
)


# ============================================================
# 13. FIXED POA
# ============================================================

fixed_poa = (
    ghi_matrix
    * sin_ab[:, None]
    / sin_a_safe[:, None]
)


# ============================================================
# 14. ACTUAL VALID MASK
# ============================================================

valid_mask = (
    np.isfinite(actual)
    &
    (actual != 0)
)


if not valid_mask.any():

    st.error(
        "Actual power contains no valid non-zero values."
    )

    st.stop()


actual_day = actual[
    valid_mask
]


actual_peak = np.max(
    actual_day
)


actual_energy = np.sum(
    actual_day
)


if actual_peak <= 0:

    st.error(
        "Actual peak must be greater than zero."
    )

    st.stop()


# ============================================================
# ============================================================
# FIXED MODEL
# ============================================================
# ============================================================

st.header(
    "Fixed Loss Correction"
)


# ============================================================
# FIXED EFFICIENCY LOSS OPTIMIZATION
# ============================================================

max_loss = np.min(
    standard_efficiency
)


loss_values = np.arange(
    0,
    max_loss + 0.0001,
    0.1,
)


fixed_results = []


for loss in loss_values:

    net_efficiency = (
        standard_efficiency
        - loss
    )

    net_efficiency = np.maximum(
        net_efficiency,
        0,
    )


    efficiency_factor = np.divide(
        net_efficiency,
        standard_efficiency,

        out=np.zeros_like(
            net_efficiency
        ),

        where=(
            standard_efficiency != 0
        ),
    )


    adjusted_fixed_weights = (
        fixed_weights
        * efficiency_factor
    )


    power_matrix = (
        fixed_poa
        * adjusted_fixed_weights[None, :]
        / 1_000_000
    )


    predicted = (
        power_matrix.sum(
            axis=1
        )
    )


    predicted_day = predicted[
        valid_mask
    ]


    if len(predicted_day) == 0:
        continue


    predicted_peak = (
        predicted_day.max()
    )


    peak_error = abs(
        actual_peak
        - predicted_peak
    )


    peak_error_percent = (
        peak_error
        / actual_peak
        * 100
    )


    block_error = (
        np.mean(
            np.abs(
                actual_day
                - predicted_day
            )
        )
        / actual_peak
    )


    predicted_energy = (
        predicted_day.sum()
    )


    energy_error = (
        abs(
            actual_energy
            - predicted_energy
        )
        / actual_energy
    )


    overall_score = (
        0.80 * block_error
        +
        0.10 * (
            peak_error
            / actual_peak
        )
        +
        0.10 * energy_error
    )


    fixed_results.append(
        {
            "Error %": loss,
            "Actual Peak": actual_peak,
            "Predicted Peak": predicted_peak,
            "Peak Error": peak_error,
            "Peak Error (%)": peak_error_percent,
            "Block Error": block_error,
            "Energy Error": energy_error,
            "Overall Score": overall_score,
        }
    )


fixed_results_df = pd.DataFrame(
    fixed_results
)


if fixed_results_df.empty:

    st.error(
        "Fixed optimization produced no results."
    )

    st.stop()


# ============================================================
# BEST FIXED LOSS
# ============================================================

best_fixed_row = fixed_results_df.loc[
    fixed_results_df[
        "Peak Error"
    ].idxmin()
]


best_loss = float(
    best_fixed_row["Error %"]
)


# ============================================================
# FINAL FIXED MODEL
# ============================================================

net_efficiency_fixed = (
    standard_efficiency
    - best_loss
)


net_efficiency_fixed = np.maximum(
    net_efficiency_fixed,
    0,
)


fixed_efficiency_factor = np.divide(
    net_efficiency_fixed,
    standard_efficiency,

    out=np.zeros_like(
        standard_efficiency
    ),

    where=(
        standard_efficiency != 0
    ),
)


final_fixed_weights = (
    fixed_weights
    * fixed_efficiency_factor
)


final_fixed_power_matrix = (
    fixed_poa
    * final_fixed_weights[None, :]
    / 1_000_000
)


fixed_forecast = (
    final_fixed_power_matrix.sum(
        axis=1
    )
)


fixed_day = fixed_forecast[
    valid_mask
]


fixed_block_error = (
    np.mean(
        np.abs(
            actual_day
            - fixed_day
        )
    )
    / actual_peak
)


fixed_peak_error = (
    abs(
        actual_peak
        - fixed_day.max()
    )
    / actual_peak
)


fixed_energy_error = (
    abs(
        actual_energy
        - fixed_day.sum()
    )
    / actual_energy
)


fixed_score = (
    0.80 * fixed_block_error
    +
    0.10 * fixed_peak_error
    +
    0.10 * fixed_energy_error
)


# ============================================================
# FIXED METRICS
# ============================================================

fc1, fc2, fc3, fc4 = st.columns(4)

with fc1:

    st.metric(
        "Efficiency Loss",
        f"{best_loss:.2f}%",
    )

with fc2:

    st.metric(
        "Actual Peak",
        f"{actual_peak:.4f}",
    )

with fc3:

    st.metric(
        "Fixed Peak",
        f"{fixed_day.max():.4f}",
    )

with fc4:

    st.metric(
        "Peak Error",
        f"{fixed_peak_error * 100:.3f}%",
    )


# ============================================================
# FIXED RESULTS
# ============================================================

with st.expander(
    "📊 Fixed Optimization Results"
):

    fixed_summary = pd.DataFrame(
        {
            "Metric": [
                "Efficiency Loss (%)",
                "Actual Peak",
                "Fixed Peak",
                "Peak Error (%)",
                "Block Error",
                "Energy Error",
                "Overall Score",
            ],
            "Value": [
                best_loss,
                actual_peak,
                fixed_day.max(),
                fixed_peak_error * 100,
                fixed_block_error,
                fixed_energy_error,
                fixed_score,
            ],
        }
    )

    st.dataframe(
        fixed_summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ============================================================
# TRACKING MODEL
# ============================================================
# ============================================================

tracking_available = (
    "Tracking" in sheet_names
)


if tracking_available:

    st.header(
        "Tracking Loss Correction"
    )


    # ========================================================
    # TRACKING SHEET
    # ========================================================

    try:

        df_trac = pd.read_excel(
            uploaded_file,
            sheet_name="Tracking",
            header=1,
        )

        df_trac.columns = (
            df_trac.columns
            .astype(str)
            .str.strip()
        )

        df_trac = df_trac.iloc[
            :n
        ].copy()

        df_trac.reset_index(
            drop=True,
            inplace=True,
        )

    except Exception as e:

        st.error(
            f"Unable to read Tracking sheet: {e}"
        )

        st.stop()


    # ========================================================
    # TRACKING CALCULATION
    # ========================================================

    def calculate_tracking(
        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit,
    ):

        if not (
            start_block
            < max_block
            < end_block
        ):

            return None


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


        if (
            denominator_1 == 0
            or denominator_2 == 0
        ):

            return None


        m1 = (
            90
            /
            denominator_1
        )


        m2 = (
            90
            /
            denominator_2
        )


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


        panel = np.where(

            blocks < max_block,

            np.where(

                zenith < abs(
                    east_limit
                ),

                zenith,

                abs(
                    east_limit
                ),
            ),

            np.where(

                (
                    (blocks > max_block)
                    &
                    (zenith > west_limit)
                ),

                west_limit,

                zenith,
            ),
        )


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


        dhi = (
            ghi_matrix
            * DHI
            / 100
        )


        dni = (
            ghi_matrix
            - dhi
        ) / cos_alpha[:, None]


        tracking_power_matrix = (
            dni
            * tracking_weights[None, :]
            / 1_000_000
        )


        tracking_forecast = (
            tracking_power_matrix.sum(
                axis=1
            )
        )


        return (
            tracking_forecast,
            tracking_power_matrix,
            zenith,
            panel,
            dni,
        )


    # ========================================================
    # TRACKING OBJECTIVE
    # ========================================================

    def tracking_objective(x):

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


        if not (
            start_block
            < max_block
            < end_block
        ):

            return 1e9


        result = calculate_tracking(

            DHI,
            start_block,
            end_block,
            max_block,
            east_limit,
            west_limit,
        )


        if result is None:

            return 1e9


        prediction = result[0]


        if not np.all(
            np.isfinite(
                prediction
            )
        ):

            return 1e9


        prediction_day = prediction[
            valid_mask
        ]


        if len(prediction_day) == 0:

            return 1e9


        # ----------------------------------------------------
        # TRACKING ERRORS
        # ----------------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    - prediction_day
                )
            )
            / actual_peak
        )


        peak_error = (
            abs(
                actual_peak
                - prediction_day.max()
            )
            / actual_peak
        )


        energy_error = (
            abs(
                actual_energy
                - prediction_day.sum()
            )
            / actual_energy
        )


        overall_score = (
            0.80 * block_error
            +
            0.10 * peak_error
            +
            0.10 * energy_error
        )


        return overall_score


    # ========================================================
    # TRACKING OPTIMIZATION
    # ========================================================

    tracking_bounds = [

        (0, 10),       # DHI %

        (10, 30),      # GHI Start

        (65, 80),      # GHI End

        (47, 53),      # GHI Max

        (10, 70),      # East Limit

        (10, 70),      # West Limit
    ]


    # --------------------------------------------------------
    # RUN OPTIMIZATION
    # --------------------------------------------------------

    with st.spinner(
        "Optimizing Tracking parameters..."
    ):

        tracking_result = differential_evolution(

            tracking_objective,

            bounds=tracking_bounds,

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


    best_tracking = np.rint(
        tracking_result.x
    ).astype(int)


    DHI = best_tracking[0]

    GHI_Starting_Block = (
        best_tracking[1]
    )

    GHI_Ending_Block = (
        best_tracking[2]
    )

    GHI_Max_Block = (
        best_tracking[3]
    )

    Tracking_angle_lim_E = (
        best_tracking[4]
    )

    Tracking_angle_lim_W = (
        best_tracking[5]
    )


    # ========================================================
    # FINAL TRACKING CALCULATION
    # ========================================================

    (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
    ) = calculate_tracking(

        DHI,

        GHI_Starting_Block,

        GHI_Ending_Block,

        GHI_Max_Block,

        Tracking_angle_lim_E,

        Tracking_angle_lim_W,
    )


    # ========================================================
    # TRACKING METRICS
    # ========================================================

    tracking_day = (
        tracking_forecast[
            valid_mask
        ]
    )


    tracking_block_error = (
        np.mean(
            np.abs(
                actual_day
                - tracking_day
            )
        )
        / actual_peak
    )


    tracking_peak_error = (
        abs(
            actual_peak
            - tracking_day.max()
        )
        / actual_peak
    )


    tracking_energy_error = (
        abs(
            actual_energy
            - tracking_day.sum()
        )
        / actual_energy
    )


    tracking_score = (
        0.80 * tracking_block_error
        +
        0.10 * tracking_peak_error
        +
        0.10 * tracking_energy_error
    )


    # ========================================================
    # TRACKING METRICS UI
    # ========================================================

    tc1, tc2, tc3, tc4 = st.columns(4)

    with tc1:

        st.metric(
            "DHI",
            f"{DHI}%",
        )

    with tc2:

        st.metric(
            "GHI Peak Block",
            GHI_Max_Block,
        )

    with tc3:

        st.metric(
            "Tracking Peak Error",
            f"{tracking_peak_error * 100:.3f}%",
        )

    with tc4:

        st.metric(
            "Tracking Score",
            f"{tracking_score:.5f}",
        )


    # ========================================================
    # TRACKING PARAMETERS
    # ========================================================

    with st.expander(
        "⚙️ Optimized Tracking Parameters",
        expanded=True,
    ):

        tracking_parameters_df = pd.DataFrame(
            {
                "Parameter": [
                    "DHI (%)",
                    "GHI Starting Block",
                    "GHI Ending Block",
                    "GHI Max Block",
                    "East Tracking Limit",
                    "West Tracking Limit",
                ],

                "Value": [
                    DHI,
                    GHI_Starting_Block,
                    GHI_Ending_Block,
                    GHI_Max_Block,
                    Tracking_angle_lim_E,
                    Tracking_angle_lim_W,
                ],
            }
        )


        st.dataframe(
            tracking_parameters_df,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # IMPORTANT TIME BLOCKS
    # ========================================================

    start_time = datetime.strptime(
        "00:00",
        "%H:%M",
    )


    time_blocks = [

        (
            start_time
            + timedelta(
                minutes=15 * i
            )
        ).strftime("%H:%M")
        +
        " - "
        +
        (
            start_time
            + timedelta(
                minutes=15 * (i + 1)
            )
        ).strftime("%H:%M")

        for i in range(96)
    ]


    tracking_lookup_blocks = [

        GHI_Starting_Block,

        GHI_Ending_Block,

        GHI_Max_Block,

    ]


    tracking_lookup_names = [

        "GHI Starting Block",

        "GHI Ending Block",

        "GHI Maximum Block",

    ]


    tracking_lookup_df = pd.DataFrame(
        {
            "Parameter":
                tracking_lookup_names,

            "Block":
                tracking_lookup_blocks,
        }
    )


    tracking_lookup_df[
        "Time Block"
    ] = tracking_lookup_df[
        "Block"
    ].apply(

        lambda x:
        time_blocks[
            int(x) - 1
        ]
        if 1 <= int(x) <= 96
        else "—"

    )


    with st.expander(
        "📅 Important Time Blocks"
    ):

        st.dataframe(
            tracking_lookup_df,
            use_container_width=True,
            hide_index=True,
        )


else:

    tracking_forecast = np.zeros(
        n
    )

    tracking_score = np.nan

    tracking_block_error = np.nan

    tracking_peak_error = np.nan

    tracking_energy_error = np.nan

    DHI = np.nan

    GHI_Starting_Block = np.nan

    GHI_Ending_Block = np.nan

    GHI_Max_Block = np.nan

    Tracking_angle_lim_E = np.nan

    Tracking_angle_lim_W = np.nan


    st.info(
        "Tracking sheet was not found in this workbook. "
        "Only Fixed correction is available."
    )


# ============================================================
# ============================================================
# FIXED VS TRACKING SUMMARY
# ============================================================
# ============================================================

st.header(
    "Fixed vs Tracking Summary"
)


summary = pd.DataFrame(
    {

        "Metric": [

            "Efficiency Loss (%)",

            "DHI (%)",

            "GHI Starting Block",

            "GHI Ending Block",

            "GHI Max Block",

            "East Tracking Limit",

            "West Tracking Limit",

            "Block Error",

            "Peak Error (%)",

            "Energy Error",

            "Overall Score",

            "Peak Power",

        ],

        "Fixed": [

            best_loss,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            np.nan,

            fixed_block_error,

            fixed_peak_error * 100,

            fixed_energy_error,

            fixed_score,

            fixed_day.max(),

        ],

        "Tracking": [

            np.nan,

            DHI,

            GHI_Starting_Block,

            GHI_Ending_Block,

            GHI_Max_Block,

            Tracking_angle_lim_E,

            Tracking_angle_lim_W,

            tracking_block_error,

            tracking_peak_error * 100,

            tracking_energy_error,

            tracking_score,

            tracking_day.max()
            if tracking_available
            else np.nan,

        ],
    }
)


st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ============================================================
# FINAL GRAPH
# ============================================================
# ============================================================

st.header(
    "Actual vs Fixed vs Tracking"
)


graph_blocks = np.arange(
    1,
    n + 1,
)


fig = go.Figure()


# ------------------------------------------------------------
# ACTUAL
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=graph_blocks,
        y=actual,
        name="Actual",
        mode="lines",
        line=dict(
            color="#ef4444",
            width=3,
        ),
        hovertemplate=(
            "Block: %{x}"
            "<br>Actual: %{y:.3f}"
            "<extra></extra>"
        ),
    )
)


# ------------------------------------------------------------
# FIXED
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=graph_blocks,
        y=fixed_forecast,
        name="Fixed Forecast",
        mode="lines",
        line=dict(
            color="#00c6ff",
            width=3,
        ),
        hovertemplate=(
            "Block: %{x}"
            "<br>Fixed: %{y:.3f}"
            "<extra></extra>"
        ),
    )
)


# ------------------------------------------------------------
# TRACKING
# ------------------------------------------------------------

if tracking_available:

    fig.add_trace(
        go.Scatter(
            x=graph_blocks,
            y=tracking_forecast,
            name="Tracking Forecast",
            mode="lines",
            line=dict(
                color="#22c55e",
                width=3,
            ),
            hovertemplate=(
                "Block: %{x}"
                "<br>Tracking: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )


# ============================================================
# GRAPH LAYOUT
# ============================================================

fig.update_layout(

    height=550,

    template="streamlit",

    hovermode="x unified",

    xaxis=dict(
        title="Block",
        dtick=4,
    ),

    yaxis=dict(
        title="Power (MW)",
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    ),

    margin=dict(
        l=20,
        r=20,
        t=70,
        b=20,
    ),
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
    },
)


# ============================================================
# FINAL OPTIMIZED PARAMETERS
# ============================================================

with st.expander(
    "📋 Final Optimized Parameters"
):

    final_parameters = pd.DataFrame(
        {

            "Parameter": [

                "Workbook Type",

                "Fixed Sheet",

                "Fixed Efficiency Loss (%)",

                "Fixed Actual Peak",

                "Fixed Predicted Peak",

                "Fixed Peak Error (%)",

                "Fixed Block Error",

                "Fixed Energy Error",

                "Fixed Overall Score",

                "Tracking DHI (%)",

                "Tracking GHI Starting Block",

                "Tracking GHI Ending Block",

                "Tracking GHI Max Block",

                "Tracking East Limit",

                "Tracking West Limit",

                "Tracking Actual Peak",

                "Tracking Predicted Peak",

                "Tracking Peak Error (%)",

                "Tracking Block Error",

                "Tracking Energy Error",

                "Tracking Overall Score",
            ],

            "Value": [

                workbook_type,

                fixed_sheet,

                best_loss,

                actual_peak,

                fixed_day.max(),

                fixed_peak_error * 100,

                fixed_block_error,

                fixed_energy_error,

                fixed_score,

                DHI,

                GHI_Starting_Block,

                GHI_Ending_Block,

                GHI_Max_Block,

                Tracking_angle_lim_E,

                Tracking_angle_lim_W,

                actual_peak,

                tracking_day.max()
                if tracking_available
                else np.nan,

                tracking_peak_error * 100,

                tracking_block_error,

                tracking_energy_error,

                tracking_score,
            ],
        }
    )


    st.dataframe(
        final_parameters,
        use_container_width=True,
        hide_index=True,
    )
