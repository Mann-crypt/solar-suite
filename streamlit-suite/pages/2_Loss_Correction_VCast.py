# ============================================================
# STREAMLIT PAGE
# VCAST LOSS CORRECTION
# FIXED / TRACKING
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
    page_title="VCast Loss Correction",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

CLUSTERS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15"
]

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15"
]


# ============================================================
# TITLE
# ============================================================

st.title("☀️ VCast Loss Correction")

st.caption(
    "Fixed / Tracking forecast correction using minimum peak error."
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload Workbook")

uploaded_file = st.file_uploader(
    "Upload VCast Excel workbook",
    type=["xlsx"],
    key="vcast_upload"
)


if uploaded_file is None:

    st.info(
        "Upload the VCast workbook to continue."
    )

    st.stop()


# ============================================================
# READ WORKBOOK SAFELY
#
# IMPORTANT:
# Do NOT cache openpyxl Workbook objects.
# ============================================================

file_bytes = uploaded_file.getvalue()


try:

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes),
        engine="openpyxl"
    )

    available_sheets = excel.sheet_names

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ============================================================
# WORKBOOK INFORMATION
# ============================================================

st.success(
    f"Workbook loaded: {uploaded_file.name}"
)


# ============================================================
# PLANT TYPE SELECTION
#
# IMPORTANT:
# This appears AFTER upload.
# ============================================================

st.subheader("2. Select Plant Type")

plant_type = st.segmented_control(
    "Plant Type",
    options=[
        "Fixed",
        "Tracking"
    ],
    default="Fixed",
    key="vcast_plant_type"
)


if plant_type is None:

    st.stop()


# ============================================================
# WORKBOOK TYPE IDENTIFICATION
# ============================================================

if "Fixed-C11" in available_sheets:

    workbook_type = "VCast"

elif "Fixed-CL1" in available_sheets:

    workbook_type = "Cluster"

elif "Fixed" in available_sheets:

    workbook_type = "Non-Cluster"

else:

    workbook_type = "Unknown"


st.info(
    f"Workbook detected as: **{workbook_type}**"
)


if workbook_type != "VCast":

    st.warning(
        "This VCast page is designed for workbooks containing "
        "the `Fixed-C11` sheet."
    )

    st.write(
        "Detected sheets:",
        ", ".join(available_sheets)
    )

    st.stop()


# ============================================================
# REQUIRED SHEETS
# ============================================================

required_sheets = [
    "Area & Efficiency",
    "Forecast Config",
    "Config Tilt Angle",
    "Result",
    "Fixed-C11"
]


if plant_type == "Tracking":

    required_sheets.extend([
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
        "Tracking"
    ])


missing_sheets = [
    s
    for s in required_sheets
    if s not in available_sheets
]


if missing_sheets:

    st.error(
        "Missing required sheets:\n\n"
        + "\n".join(
            f"- {s}"
            for s in missing_sheets
        )
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    return df


def first_blank_position(df, column):

    valid = df[column].notna()

    if not valid.any():

        return len(df)

    blank_indices = np.where(
        ~valid.to_numpy()
    )[0]

    if len(blank_indices) == 0:

        return len(df)

    return int(
        blank_indices[0]
    )


# ============================================================
# READ AREA & EFFICIENCY
# ============================================================

try:

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
        engine="openpyxl"
    )

except Exception as e:

    st.error(
        f"Unable to read Area & Efficiency sheet: {e}"
    )

    st.stop()


df = clean_columns(df)


if "S.No." not in df.columns:

    st.error(
        "Column `S.No.` was not found in Area & Efficiency."
    )

    st.stop()


df = df[
    df["S.No."].notna()
].copy()


df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

for col in [
    "Standard PV Efficiency (%)",
    "No of Module",
    "Area of 1 Module (m2)",
    "Total area (m2)",
    "Error %"
]:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# TOTAL AREA
# ============================================================

df["Total area (m2)"] = (
    df["No of Module"]
    *
    df["Area of 1 Module (m2)"]
)


# ============================================================
# READ CLUSTER EFFECTIVE AREA TABLE
# ============================================================

try:

    df_w = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
        engine="openpyxl"
    )

except Exception as e:

    st.error(
        f"Unable to read cluster area table: {e}"
    )

    st.stop()


df_w.columns = [
    str(c).strip()
    for c in df_w.columns
]


if "Clusters" not in df_w.columns:

    st.error(
        "Column `Clusters` was not found in Area & Efficiency."
    )

    st.stop()


df_w = df_w[
    df_w["Clusters"].notna()
].copy()


df_w.reset_index(
    drop=True,
    inplace=True
)


# Keep only five cluster rows when possible

df_w = df_w.iloc[
    :5
].copy()


# ============================================================
# FORECAST CONFIG
# ============================================================

try:

    df_config = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
        engine="openpyxl"
    )

    lat = float(
        pd.to_numeric(
            df_config.loc[0, "Lat"],
            errors="coerce"
        )
    )

except Exception as e:

    st.error(
        f"Unable to read latitude from Forecast Config: {e}"
    )

    st.stop()


# ============================================================
# CONFIG TILT ANGLE
# ============================================================

try:

    df_tilt = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Config Tilt Angle",
        header=7,
        engine="openpyxl"
    )

except Exception as e:

    st.error(
        f"Unable to read Config Tilt Angle: {e}"
    )

    st.stop()


df_tilt.columns = (
    df_tilt.columns
    .astype(str)
    .str.strip()
)


if "Fixed" not in df_tilt.columns:

    st.error(
        "Column `Fixed` not found in Config Tilt Angle."
    )

    st.stop()


tilt_valid = df_tilt["Fixed"].notna()


if tilt_valid.any():

    df_tilt = df_tilt.loc[
        :np.where(
            ~tilt_valid.to_numpy()
        )[0][0] - 1
    ].copy() if (
        (~tilt_valid.to_numpy()).any()
    ) else df_tilt.loc[
        tilt_valid
    ].copy()


df_tilt = df_tilt.dropna(
    how="all",
    axis=1
)


df_tilt = df_tilt.rename(
    columns={
        "Unnamed: 2": "Month_Num",
        "Unnamed: 3": "Month"
    }
)


if "Month" not in df_tilt.columns:

    st.error(
        "Month column could not be identified in Config Tilt Angle."
    )

    st.stop()


df_tilt["Fixed"] = pd.to_numeric(
    df_tilt["Fixed"],
    errors="coerce"
)


month_lookup = (
    df_tilt
    .dropna(subset=["Month"])
    .set_index("Month")["Fixed"]
    .to_dict()
)


# ============================================================
# RESULT / GHI
# ============================================================

try:

    df_ghi = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
        engine="openpyxl"
    )

except Exception as e:

    st.error(
        f"Unable to read Result sheet: {e}"
    )

    st.stop()


df_ghi = df_ghi.fillna(0)


if df_ghi.shape[1] < 6:

    st.error(
        "Result sheet does not contain five GHI columns."
    )

    st.stop()


df_ghi.columns = [
    "Block",
    *GHI_COLS
]


df_ghi["Block"] = pd.to_numeric(
    df_ghi["Block"],
    errors="coerce"
)


df_ghi = df_ghi[
    df_ghi["Block"].notna()
].copy()


for col in GHI_COLS:

    df_ghi[col] = pd.to_numeric(
        df_ghi[col],
        errors="coerce"
    ).fillna(0)


# ============================================================
# FIXED-C11
# ============================================================

try:

    df_fix = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Fixed-C11",
        header=1,
        engine="openpyxl"
    )

except Exception as e:

    st.error(
        f"Unable to read Fixed-C11 sheet: {e}"
    )

    st.stop()


df_fix = clean_columns(df_fix)


if "Date" not in df_fix.columns:

    st.error(
        "Column `Date` was not found in Fixed-C11."
    )

    st.stop()


date_valid = df_fix["Date"].notna()


if not date_valid.any():

    st.error(
        "No valid Date rows found in Fixed-C11."
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
    inplace=True
)


# ============================================================
# ACTUAL
# ============================================================

if "Actual" not in df_fix.columns:

    st.error(
        "`Actual` column was not found in Fixed-C11."
    )

    st.stop()


df_fix["Actual"] = pd.to_numeric(
    df_fix["Actual"],
    errors="coerce"
).fillna(0)


# ============================================================
# ALIGN LENGTH
# ============================================================

n = min(
    len(df_fix),
    len(df_ghi)
)


if n == 0:

    st.error(
        "No rows available for calculation."
    )

    st.stop()


df_fix = df_fix.iloc[
    :n
].copy()

df_ghi = df_ghi.iloc[
    :n
].copy()


# ============================================================
# DATES
# ============================================================

dates = pd.to_datetime(
    df_fix["Date"],
    errors="coerce"
)


if dates.isna().any():

    st.error(
        "Invalid Date values found in Fixed-C11."
    )

    st.stop()


# ============================================================
# CALCULATION DATE
#
# Keep workbook dates for calculation.
# ============================================================

first_date = pd.Timestamp(
    year=2025,
    month=1,
    day=1
)


day_offset = (
    dates
    - first_date
).dt.days.to_numpy(
    dtype=float
)


# ============================================================
# SOLAR ANGLES
# ============================================================

declination = (
    23.45
    *
    np.sin(
        np.radians(
            360
            *
            (
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


months = dates.dt.month


# ============================================================
# MONTH -> TILT
# ============================================================

tilt = np.array([

    month_lookup.get(
        pd.Timestamp(
            year=2025,
            month=int(month),
            day=1
        ).strftime("%B"),
        0
    )

    for month in months

], dtype=float)


# ============================================================
# FIXED POA
# ============================================================

a_plus_b = (
    elevation
    + tilt
)


sin_a = np.sin(
    np.radians(elevation)
)


sin_ab = np.sin(
    np.radians(a_plus_b)
)


sin_a_safe = np.where(
    np.abs(sin_a) < 1e-8,
    1e-8,
    sin_a
)


ghi_matrix = np.column_stack([

    df_ghi[col]
    .to_numpy(dtype=float)

    for col in GHI_COLS

])


fixed_poa = (
    ghi_matrix
    *
    sin_ab[:, None]
    /
    sin_a_safe[:, None]
)


# ============================================================
# ACTUAL MASK
# ============================================================

actual_full = (
    df_fix["Actual"]
    .to_numpy(dtype=float)
)


valid_mask = (
    np.isfinite(actual_full)
    &
    (actual_full != 0)
)


if not valid_mask.any():

    st.error(
        "Actual power contains no non-zero values."
    )

    st.stop()


actual_day = actual_full[
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
# AREA / EFFICIENCY PREPARATION
# ============================================================

standard_efficiency = pd.to_numeric(
    df["Standard PV Efficiency (%)"],
    errors="coerce"
).to_numpy(dtype=float)


if len(standard_efficiency) < 5:

    st.error(
        "Less than five Standard PV Efficiency values found."
    )

    st.stop()


standard_efficiency = (
    standard_efficiency[:5]
)


# ============================================================
# CALCULATION FUNCTIONS
# ============================================================

def calculate_fixed(
    error_percent
):

    work_df = df.copy()

    work_df["Error %"] = (
        error_percent
    )

    work_df["Net Efficiency (%)"] = (
        work_df["Standard PV Efficiency (%)"]
        -
        error_percent
    )

    work_df["Net Efficiency (%)"] = np.maximum(
        work_df["Net Efficiency (%)"],
        0
    )

    work_df["Total area (m2)"] = (
        pd.to_numeric(
            work_df["No of Module"],
            errors="coerce"
        )
        *
        pd.to_numeric(
            work_df["Area of 1 Module (m2)"],
            errors="coerce"
        )
    )

    work_df["Eff Area"] = (
        work_df["Net Efficiency (%)"]
        *
        work_df["Total area (m2)"]
        /
        100
    )

    cluster_sums = (
        work_df
        .groupby("Clusters")["Eff Area"]
        .sum()
    )

    local_weights = (
        df_w["Clusters"]
        .map(cluster_sums)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    power_matrix = (
        fixed_poa
        *
        local_weights[None, :]
        /
        1_000_000
    )

    forecast = (
        power_matrix.sum(axis=1)
    )

    return (
        work_df,
        local_weights,
        power_matrix,
        forecast
    )


# ============================================================
# FIXED ERROR OPTIMIZATION
#
# IMPORTANT:
# Best Error % = minimum peak error
# ============================================================

fixed_results = []


for error in np.arange(
    0,
    10.01,
    0.1
):

    (
        _,
        _,
        _,
        forecast
    ) = calculate_fixed(
        error
    )

    forecast_day = (
        forecast[
            valid_mask
        ]
    )

    calculated_peak = (
        forecast_day.max()
    )

    peak_error = abs(
        calculated_peak
        -
        actual_peak
    )

    peak_error_pct = (
        peak_error
        /
        actual_peak
        *
        100
    )

    fixed_results.append({

        "Error %": error,

        "Calculated Peak": calculated_peak,

        "Actual Peak": actual_peak,

        "Peak Error": peak_error,

        "Peak Error %": peak_error_pct

    })


fixed_results_df = pd.DataFrame(
    fixed_results
)


best_fixed_row = fixed_results_df.loc[
    fixed_results_df["Peak Error"].idxmin()
]


auto_fixed_error = float(
    best_fixed_row["Error %"]
)


# ============================================================
# TRACKING DATA
# ============================================================

if plant_type == "Tracking":

    backend_list = []

    for cl in CLUSTERS:

        try:

            backend_df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=f"Backend Cal {cl}",
                engine="openpyxl"
            )

            backend_list.append(
                backend_df
            )

        except Exception as e:

            st.error(
                f"Unable to read Backend Cal {cl}: {e}"
            )

            st.stop()


    try:

        df_trac = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Tracking",
            header=1,
            engine="openpyxl"
        )

    except Exception as e:

        st.error(
            f"Unable to read Tracking sheet: {e}"
        )

        st.stop()


    df_trac = clean_columns(
        df_trac
    )


    df_trac = df_trac.iloc[
        :n
    ].copy()


    df_trac.reset_index(
        drop=True,
        inplace=True
    )


    # --------------------------------------------------------
    # BLOCKS
    # --------------------------------------------------------

    if "Block No." in backend_list[0].columns:

        blocks = pd.to_numeric(
            backend_list[0]["Block No."],
            errors="coerce"
        ).to_numpy(dtype=float)

    elif "Block" in backend_list[0].columns:

        blocks = pd.to_numeric(
            backend_list[0]["Block"],
            errors="coerce"
        ).to_numpy(dtype=float)

    else:

        blocks = np.arange(
            1,
            n + 1,
            dtype=float
        )


    blocks = blocks[:n]


    # --------------------------------------------------------
    # TRACKING WEIGHTS
    #
    # IMPORTANT:
    # Tracking uses the tracking effective areas.
    # --------------------------------------------------------

    area_full = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=None,
        engine="openpyxl"
    )


    tracking_weights = (
        pd.to_numeric(
            area_full.iloc[
                28:33,
                15
            ],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )


    if len(tracking_weights) != 5:

        st.error(
            "Could not read five tracking effective areas."
        )

        st.stop()


    # ========================================================
    # TRACKING CALCULATION
    # ========================================================

    def calculate_tracking(
        error_percent,
        DHI,
        start_block,
        end_block,
        max_block,
        east_limit,
        west_limit
    ):

        # ----------------------------------------------------
        # Efficiency loss is applied to tracking areas
        # ----------------------------------------------------

        tracking_standard_eff = (
            standard_efficiency.copy()
        )

        tracking_net_eff = (
            tracking_standard_eff
            -
            error_percent
        )

        tracking_net_eff = np.maximum(
            tracking_net_eff,
            0
        )

        efficiency_factor = np.divide(
            tracking_net_eff,
            tracking_standard_eff,

            out=np.zeros_like(
                tracking_standard_eff
            ),

            where=(
                tracking_standard_eff != 0
            )
        )


        adjusted_weights = (
            tracking_weights
            *
            efficiency_factor
        )


        # ----------------------------------------------------
        # Validate blocks
        # ----------------------------------------------------

        if not (
            start_block
            <
            max_block
            <
            end_block
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


        # ----------------------------------------------------
        # Slopes
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Zenith
        # ----------------------------------------------------

        zenith = np.where(

            blocks <= max_block,

            np.minimum(
                89,
                m1
                *
                (
                    blocks
                    -
                    max_block
                )
            ),

            np.minimum(
                89,
                m2
                *
                (
                    blocks
                    -
                    max_block
                )
            )

        )


        # ----------------------------------------------------
        # Panel angle
        # ----------------------------------------------------

        panel = np.where(

            blocks < max_block,

            np.where(

                zenith
                <
                abs(east_limit),

                zenith,

                abs(east_limit)

            ),

            np.where(

                (
                    (blocks > max_block)
                    &
                    (zenith > west_limit)
                ),

                west_limit,

                zenith

            )

        )


        # ----------------------------------------------------
        # Cos alpha
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(panel)
        )


        cos_alpha = np.clip(
            cos_alpha,
            1e-6,
            None
        )


        # ----------------------------------------------------
        # DHI
        # ----------------------------------------------------

        dhi = (
            ghi_matrix
            *
            DHI
            /
            100
        )


        # ----------------------------------------------------
        # DNI
        # ----------------------------------------------------

        dni = (
            ghi_matrix
            -
            dhi
        ) / cos_alpha[:, None]


        # ----------------------------------------------------
        # Tracking power
        # ----------------------------------------------------

        power_matrix = (
            dni
            *
            adjusted_weights[None, :]
            /
            1_000_000
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
            adjusted_weights
        )


    # ========================================================
    # TRACKING OPTIMIZATION
    #
    # Error % is optimized FIRST.
    # Then other parameters are optimized.
    # ========================================================

    # --------------------------------------------------------
    # Initial tracking parameters
    # --------------------------------------------------------

    default_DHI = 5
    default_start = 20
    default_end = 72
    default_max = 48
    default_east = 45
    default_west = 45


    # --------------------------------------------------------
    # STEP 1
    #
    # Find Error % using current/default tracking parameters.
    # Minimum peak error.
    # --------------------------------------------------------

    tracking_error_results = []


    for error in np.arange(
        0,
        10.01,
        0.1
    ):

        result_tracking = calculate_tracking(

            error,

            default_DHI,

            default_start,

            default_end,

            default_max,

            default_east,

            default_west

        )


        if result_tracking is None:

            continue


        forecast = result_tracking[0]


        forecast_day = (
            forecast[
                valid_mask
            ]
        )


        calculated_peak = (
            forecast_day.max()
        )


        peak_error = abs(
            calculated_peak
            -
            actual_peak
        )


        peak_error_pct = (
            peak_error
            /
            actual_peak
            *
            100
        )


        tracking_error_results.append({

            "Error %": error,

            "Calculated Peak": calculated_peak,

            "Actual Peak": actual_peak,

            "Peak Error": peak_error,

            "Peak Error %": peak_error_pct

        })


    tracking_error_df = pd.DataFrame(
        tracking_error_results
    )


    if tracking_error_df.empty:

        st.error(
            "Tracking Error % optimization "
            "did not produce results."
        )

        st.stop()


    best_tracking_error_row = (
        tracking_error_df.loc[
            tracking_error_df["Peak Error"].idxmin()
        ]
    )


    auto_tracking_error = float(
        best_tracking_error_row["Error %"]
    )


    # ========================================================
    # STEP 2
    #
    # Optimize Tracking Parameters
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


        result_tracking = calculate_tracking(

            auto_tracking_error,

            DHI,

            start_block,

            end_block,

            max_block,

            east_limit,

            west_limit

        )


        if result_tracking is None:

            return 1e9


        forecast = (
            result_tracking[0]
        )


        if not np.all(
            np.isfinite(
                forecast
            )
        ):

            return 1e9


        forecast_day = (
            forecast[
                valid_mask
            ]
        )


        if len(forecast_day) == 0:

            return 1e9


        # ----------------------------------------------------
        # Block error
        # ----------------------------------------------------

        block_error = (
            np.mean(
                np.abs(
                    actual_day
                    -
                    forecast_day
                )
            )
            /
            actual_peak
        )


        # ----------------------------------------------------
        # Peak error
        # ----------------------------------------------------

        peak_error = (
            abs(
                actual_peak
                -
                forecast_day.max()
            )
            /
            actual_peak
        )


        # ----------------------------------------------------
        # Energy error
        # ----------------------------------------------------

        energy_error = (
            abs(
                actual_energy
                -
                forecast_day.sum()
            )
            /
            actual_energy
        )


        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            0.80 * block_error
            +
            0.10 * peak_error
            +
            0.10 * energy_error
        )


        return score


    tracking_bounds = [

        (0, 10),

        (10, 30),

        (65, 80),

        (47, 53),

        (10, 70),

        (10, 70)

    ]


    optimizer_result = differential_evolution(

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

        workers=1

    )


    best_tracking_parameters = (
        np.rint(
            optimizer_result.x
        ).astype(int)
    )


    auto_DHI = int(
        best_tracking_parameters[0]
    )

    auto_start = int(
        best_tracking_parameters[1]
    )

    auto_end = int(
        best_tracking_parameters[2]
    )

    auto_max = int(
        best_tracking_parameters[3]
    )

    auto_east = int(
        best_tracking_parameters[4]
    )

    auto_west = int(
        best_tracking_parameters[5]
    )


# ============================================================
# USER EDITABLE PARAMETERS
# ============================================================

st.divider()

st.subheader(
    "3. Optimized Parameters"
)


if plant_type == "Fixed":

    col1, col2 = st.columns(2)

    with col1:

        error_percent = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=10.0,
            value=float(auto_fixed_error),
            step=0.1,
            format="%.1f",
            key="fixed_error_input"
        )

    with col2:

        st.metric(
            "Automatically Calculated Error %",
            f"{auto_fixed_error:.1f}%"
        )


else:

    col1, col2, col3 = st.columns(3)

    with col1:

        error_percent = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=10.0,
            value=float(auto_tracking_error),
            step=0.1,
            format="%.1f",
            key="tracking_error_input"
        )

    with col2:

        DHI = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=10,
            value=auto_DHI,
            step=1,
            key="tracking_dhi_input"
        )

    with col3:

        GHI_Starting_Block = st.number_input(
            "GHI Starting Block",
            min_value=1,
            max_value=95,
            value=auto_start,
            step=1,
            key="tracking_start_input"
        )


    col4, col5, col6 = st.columns(3)

    with col4:

        GHI_Ending_Block = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=95,
            value=auto_end,
            step=1,
            key="tracking_end_input"
        )

    with col5:

        GHI_Max_Block = st.number_input(
            "GHI Max Block",
            min_value=1,
            max_value=95,
            value=auto_max,
            step=1,
            key="tracking_max_input"
        )

    with col6:

        Tracking_angle_lim_E = st.number_input(
            "Tracking East Limit (°)",
            min_value=0,
            max_value=70,
            value=auto_east,
            step=1,
            key="tracking_east_input"
        )


    Tracking_angle_lim_W = st.number_input(
        "Tracking West Limit (°)",
        min_value=0,
        max_value=70,
        value=auto_west,
        step=1,
        key="tracking_west_input"
    )


# ============================================================
# FINAL CALCULATION
# ============================================================

if plant_type == "Fixed":

    (
        final_df,
        final_weights,
        final_power_matrix,
        forecast
    ) = calculate_fixed(
        error_percent
    )


    # ========================================================
    # FINAL FIXED METRICS
    # ========================================================

    forecast_day = (
        forecast[
            valid_mask
        ]
    )


    calculated_peak = (
        forecast_day.max()
    )


    peak_error = abs(
        calculated_peak
        -
        actual_peak
    )


    peak_error_pct = (
        peak_error
        /
        actual_peak
        *
        100
    )


    block_error = (
        np.mean(
            np.abs(
                actual_day
                -
                forecast_day
            )
        )
        /
        actual_peak
    )


    energy_error = (
        abs(
            actual_energy
            -
            forecast_day.sum()
        )
        /
        actual_energy
    )


    overall_score = (
        0.80 * block_error
        +
        0.10 * (
            peak_error
            /
            actual_peak
        )
        +
        0.10 * energy_error
    )


    # ========================================================
    # OUTPUT DATAFRAME
    # ========================================================

    final_df["Error %"] = (
        error_percent
    )


    final_df["Net Efficiency (%)"] = (
        final_df[
            "Standard PV Efficiency (%)"
        ]
        -
        error_percent
    )


    final_df["Net Efficiency (%)"] = (
        final_df[
            "Net Efficiency (%)"
        ].clip(lower=0)
    )


    final_df["Total area (m2)"] = (
        pd.to_numeric(
            final_df["No of Module"],
            errors="coerce"
        )
        *
        pd.to_numeric(
            final_df["Area of 1 Module (m2)"],
            errors="coerce"
        )
    )


    final_df["Eff Area"] = (
        final_df["Net Efficiency (%)"]
        *
        final_df["Total area (m2)"]
        /
        100
    )


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    st.divider()

    st.subheader(
        "4. Fixed Results"
    )


    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Error %",
            f"{error_percent:.2f}%"
        )

    with m2:

        st.metric(
            "Actual Peak",
            f"{actual_peak:.4f}"
        )

    with m3:

        st.metric(
            "Forecast Peak",
            f"{calculated_peak:.4f}"
        )

    with m4:

        st.metric(
            "Peak Error",
            f"{peak_error_pct:.3f}%"
        )


    m5, m6, m7 = st.columns(3)

    with m5:

        st.metric(
            "Block Error",
            f"{block_error:.5f}"
        )

    with m6:

        st.metric(
            "Energy Error",
            f"{energy_error:.5f}"
        )

    with m7:

        st.metric(
            "Overall Score",
            f"{overall_score:.5f}"
        )


    # ========================================================
    # FIXED PLOT
    # ========================================================

    plot_df = pd.DataFrame({

        "Block": np.arange(
            n
        ),

        "Actual": actual_full,

        "Forecast": forecast

    })


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=plot_df["Block"],
            y=plot_df["Actual"],
            name="Actual",
            mode="lines"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=plot_df["Block"],
            y=plot_df["Forecast"],
            name="Fixed Forecast",
            mode="lines"
        )
    )


    fig.update_layout(
        title="Actual vs Fixed Forecast",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        height=500,
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # AREA TABLE
    # ========================================================

    st.subheader(
        "Fixed Area & Efficiency"
    )


    display_columns = [

        col

        for col in [

            "S.No.",
            "Clusters",
            "No of Module",
            "Area of 1 Module (m2)",
            "Total area (m2)",
            "Standard PV Efficiency (%)",
            "Error %",
            "Net Efficiency (%)",
            "Eff Area"

        ]

        if col in final_df.columns

    ]


    st.dataframe(
        final_df[
            display_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # CLUSTER EFFECTIVE AREA
    # ========================================================

    cluster_summary = (
        final_df
        .groupby("Clusters")["Eff Area"]
        .sum()
        .reset_index()
    )


    cluster_summary.columns = [
        "Cluster",
        "Effective Area (m²)"
    ]


    st.subheader(
        "Cluster Effective Area"
    )


    st.dataframe(
        cluster_summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # ERROR TEST RESULTS
    # ========================================================

    st.subheader(
        "Error % Optimization"
    )


    st.dataframe(
        fixed_results_df,
        use_container_width=True,
        hide_index=True
    )


else:

    # ========================================================
    # FINAL TRACKING CALCULATION
    # ========================================================

    final_tracking = calculate_tracking(

        error_percent,

        int(DHI),

        int(GHI_Starting_Block),

        int(GHI_Ending_Block),

        int(GHI_Max_Block),

        int(Tracking_angle_lim_E),

        int(Tracking_angle_lim_W)

    )


    if final_tracking is None:

        st.error(
            "Invalid tracking parameters. "
            "Required: Start Block < Max Block < End Block."
        )

        st.stop()


    (
        tracking_forecast,
        tracking_power_matrix,
        zenith,
        panel,
        dni,
        adjusted_tracking_weights
    ) = final_tracking


    tracking_day = (
        tracking_forecast[
            valid_mask
        ]
    )


    tracking_peak = (
        tracking_day.max()
    )


    tracking_peak_error = abs(
        actual_peak
        -
        tracking_peak
    )


    tracking_peak_error_pct = (
        tracking_peak_error
        /
        actual_peak
        *
        100
    )


    tracking_block_error = (
        np.mean(
            np.abs(
                actual_day
                -
                tracking_day
            )
        )
        /
        actual_peak
    )


    tracking_energy_error = (
        abs(
            actual_energy
            -
            tracking_day.sum()
        )
        /
        actual_energy
    )


    tracking_score = (
        0.80 * tracking_block_error
        +
        0.10 * (
            tracking_peak_error
            /
            actual_peak
        )
        +
        0.10 * tracking_energy_error
    )


    # ========================================================
    # OUTPUT TRACKING DATAFRAME
    # ========================================================

    df_trac["Zenith Angle"] = (
        zenith
    )


    df_trac["Panel Angle"] = (
        panel
    )


    for i, cluster in enumerate(
        CLUSTERS
    ):

        df_trac[
            f"{cluster}_Tracking Power=I*Ƞ*A"
        ] = (
            tracking_power_matrix[
                :, i
            ]
        )


    df_trac[
        "Tracking Power=I*Ƞ*A"
    ] = (
        tracking_forecast
    )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "4. Tracking Results"
    )


    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Error %",
            f"{error_percent:.2f}%"
        )

    with m2:

        st.metric(
            "DHI",
            f"{DHI}%"
        )

    with m3:

        st.metric(
            "Actual Peak",
            f"{actual_peak:.4f}"
        )

    with m4:

        st.metric(
            "Tracking Peak",
            f"{tracking_peak:.4f}"
        )


    m5, m6, m7 = st.columns(3)

    with m5:

        st.metric(
            "Peak Error",
            f"{tracking_peak_error_pct:.3f}%"
        )

    with m6:

        st.metric(
            "Block Error",
            f"{tracking_block_error:.5f}"
        )

    with m7:

        st.metric(
            "Energy Error",
            f"{tracking_energy_error:.5f}"
        )


    st.metric(
        "Overall Score",
        f"{tracking_score:.5f}"
    )


    # ========================================================
    # TRACKING PARAMETERS SUMMARY
    # ========================================================

    parameter_summary = pd.DataFrame({

        "Parameter": [

            "Error %",

            "DHI (%)",

            "GHI Starting Block",

            "GHI Ending Block",

            "GHI Max Block",

            "East Tracking Limit",

            "West Tracking Limit"

        ],

        "Value": [

            error_percent,

            DHI,

            GHI_Starting_Block,

            GHI_Ending_Block,

            GHI_Max_Block,

            Tracking_angle_lim_E,

            Tracking_angle_lim_W

        ]

    })


    st.subheader(
        "Tracking Parameters"
    )


    st.dataframe(
        parameter_summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PLOT
    # ========================================================

    plot_df = pd.DataFrame({

        "Block": np.arange(
            n
        ),

        "Actual": actual_full,

        "Tracking Forecast": tracking_forecast

    })


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=plot_df["Block"],
            y=plot_df["Actual"],
            name="Actual",
            mode="lines"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=plot_df["Block"],
            y=plot_df["Tracking Forecast"],
            name="Tracking Forecast",
            mode="lines"
        )
    )


    fig.update_layout(
        title="Actual vs Tracking Forecast",
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        height=500,
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # TRACKING DATA
    # ========================================================

    st.subheader(
        "Tracking Calculation Data"
    )


    st.dataframe(
        df_trac,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.subheader(
    "5. Download Results"
)


output = io.BytesIO()


with pd.ExcelWriter(
    output,
    engine="openpyxl"
) as writer:

    if plant_type == "Fixed":

        final_df.to_excel(
            writer,
            sheet_name="Area & Efficiency",
            index=False
        )

        df_fix_out = df_fix.copy()

        for i, cluster in enumerate(
            CLUSTERS
        ):

            df_fix_out[
                f"{cluster}_Fixed Power=I*Ƞ*A"
            ] = (
                final_power_matrix[
                    :, i
                ]
            )


        df_fix_out[
            "Total Power (CL1+CL2+…)"
        ] = forecast


        df_fix_out.to_excel(
            writer,
            sheet_name="Fixed-C11 Result",
            index=False
        )


        cluster_summary.to_excel(
            writer,
            sheet_name="Cluster Areas",
            index=False
        )


        fixed_results_df.to_excel(
            writer,
            sheet_name="Error Optimization",
            index=False
        )


        pd.DataFrame({

            "Metric": [

                "Error %",

                "Actual Peak",

                "Forecast Peak",

                "Peak Error %",

                "Block Error",

                "Energy Error",

                "Overall Score"

            ],

            "Value": [

                error_percent,

                actual_peak,

                calculated_peak,

                peak_error_pct,

                block_error,

                energy_error,

                overall_score

            ]

        }).to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )


    else:

        df_trac.to_excel(
            writer,
            sheet_name="Tracking Result",
            index=False
        )


        tracking_error_df.to_excel(
            writer,
            sheet_name="Error Optimization",
            index=False
        )


        parameter_summary.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )


        pd.DataFrame({

            "Metric": [

                "Error %",

                "DHI",

                "GHI Starting Block",

                "GHI Ending Block",

                "GHI Max Block",

                "East Tracking Limit",

                "West Tracking Limit",

                "Actual Peak",

                "Tracking Peak",

                "Peak Error %",

                "Block Error",

                "Energy Error",

                "Overall Score"

            ],

            "Value": [

                error_percent,

                DHI,

                GHI_Starting_Block,

                GHI_Ending_Block,

                GHI_Max_Block,

                Tracking_angle_lim_E,

                Tracking_angle_lim_W,

                actual_peak,

                tracking_peak,

                tracking_peak_error_pct,

                tracking_block_error,

                tracking_energy_error,

                tracking_score

            ]

        }).to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )


st.download_button(

    label="⬇️ Download Excel Results",

    data=output.getvalue(),

    file_name=(
        f"VCast_Loss_Correction_"
        f"{plant_type}.xlsx"
    ),

    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    )

)
