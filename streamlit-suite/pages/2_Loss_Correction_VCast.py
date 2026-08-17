# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# STREAMLIT APPLICATION
#
# IMPORTANT:
# Error % is applied ONLY ONCE.
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
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .app-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 8px;
        border: 1px solid #e5e7eb;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "calculated" not in st.session_state:
    st.session_state.calculated = False

if "calculation_data" not in st.session_state:
    st.session_state.calculation_data = None

if "plant_type" not in st.session_state:
    st.session_state.plant_type = "Fixed"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Automatic parameter optimization with editable final parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)


if uploaded_file is None:
    st.info("Upload the Excel file to start the calculation.")
    st.stop()


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.radio(
    "Plant Type",
    ["Fixed", "Tracking"],
    horizontal=True,
    index=0 if st.session_state.plant_type == "Fixed" else 1,
    label_visibility="collapsed",
)

st.session_state.plant_type = plant_type


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_excel(uploaded_file, **kwargs):
    """
    Read Excel file from Streamlit uploader.
    """
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, **kwargs)


# ============================================================
# CLEAN AREA & EFFICIENCY
# ============================================================

def load_area_efficiency(uploaded_file):

    df = read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=[1],
        usecols=range(12),
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:

        null_indices = df[df["S.No."].isna()].index

        if len(null_indices) > 0:
            first_null_pos = df.index.get_loc(null_indices[0])
            df = df.iloc[:first_null_pos].copy()

    # --------------------------------------------------------
    # Original values only
    # --------------------------------------------------------

    df["Standard PV Efficiency (%)"] = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    )

    df["No of Module"] = pd.to_numeric(
        df["No of Module"],
        errors="coerce",
    )

    df["Area of 1 Module (m2)"] = pd.to_numeric(
        df["Area of 1 Module (m2)"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Calculate total area once
    # --------------------------------------------------------

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df


# ============================================================
# CLUSTER AREA TABLE
# ============================================================

def load_cluster_table(uploaded_file):

    df_w = read_excel(
        uploaded_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w.columns = df_w.columns.astype(str).str.strip()

    if "Clusters" in df_w.columns:

        null_indices = df_w[
            df_w["Clusters"].isna()
        ].index

        if len(null_indices) > 0:
            first_null_pos = df_w.index.get_loc(
                null_indices[0]
            )
            df_w = df_w.iloc[:first_null_pos].copy()

    return df_w.reset_index(drop=True)


# ============================================================
# LOAD GHI
# ============================================================

def load_ghi(uploaded_file):

    df_ghi = read_excel(
        uploaded_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

    for col in [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]:

        if col in df_ghi.columns:
            df_ghi[col] = pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            ).fillna(0)

    return df_ghi


# ============================================================
# LOAD LATITUDE
# ============================================================

def load_latitude(uploaded_file):

    df_st = read_excel(
        uploaded_file,
        sheet_name="Forecast Config",
        header=[8],
    )

    lat = float(
        pd.to_numeric(
            df_st.loc[0, "Lat"],
            errors="coerce",
        )
    )

    return lat


# ============================================================
# LOAD TILT
# ============================================================

def load_tilt(uploaded_file):

    df_tilt = read_excel(
        uploaded_file,
        sheet_name="Config Tilt Angle",
        header=[7],
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    if "Fixed" in df_tilt.columns:

        null_indices = df_tilt[
            df_tilt["Fixed"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = df_tilt.index.get_loc(
                null_indices[0]
            )

            df_tilt = df_tilt.iloc[
                :first_null_pos
            ].copy()

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

    month_lookup = (
        df_tilt
        .set_index("Month")["Fixed"]
        .to_dict()
    )

    return month_lookup


# ============================================================
# LOAD FIXED DATA
# ============================================================

def load_fixed_data(uploaded_file):

    df_fix = read_excel(
        uploaded_file,
        sheet_name="Fixed-C11",
        header=[1],
    )

    df_fix.columns = (
        df_fix.columns
        .astype(str)
        .str.strip()
    )

    if "Date" in df_fix.columns:

        null_indices = df_fix[
            df_fix["Date"].isna()
        ].index

        if len(null_indices) > 0:

            first_null_pos = df_fix.index.get_loc(
                null_indices[0]
            )

            df_fix = df_fix.iloc[
                :first_null_pos
            ].copy()

    df_fix["Actual"] = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    return df_fix.reset_index(drop=True)


# ============================================================
# CALCULATE SOLAR GEOMETRY
# ============================================================

def prepare_fixed_geometry(
    df_fix,
    df_ghi,
    lat,
    month_lookup,
):

    df_fix = df_fix.copy()

    # --------------------------------------------------------
    # PRESERVE ORIGINAL LOGIC
    # --------------------------------------------------------

    today = pd.Timestamp.today()

    df_fix["Date"] = today

    first_date = (
        today
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    df_fix["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + (
                        df_fix["Date"]
                        - first_date
                    ).dt.days
                    + 1
                )
                / 365
            )
        )
    )

    df_fix["Elevation angle a"] = (
        90
        - lat
        + df_fix["Declination Angle ∆"]
    )

    df_fix["Tilt Angle b"] = (
        df_fix["Date"]
        .dt.strftime("%B")
        .map(month_lookup)
    )

    df_fix["a+b"] = (
        df_fix["Elevation angle a"]
        + df_fix["Tilt Angle b"]
    )

    df_fix["SIN(a+b)"] = np.sin(
        np.radians(
            df_fix["a+b"]
        )
    )

    df_fix["Sin(a)"] = np.sin(
        np.radians(
            df_fix["Elevation angle a"]
        )
    )

    # --------------------------------------------------------
    # C11
    # --------------------------------------------------------

    df_fix["GHI*sin(a)"] = (
        df_ghi["GHI C11"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)"] = (
        df_ghi["GHI C11"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA fixed"] = (
        df_fix["GHI*sin(a+b)"]
        / df_fix["Sin(a)"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # C12
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL2"] = (
        df_ghi["GHI C12"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL2"] = (
        df_ghi["GHI C12"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C12"] = (
        df_fix["GHI*sin(a+b)-CL2"]
        / df_fix["Sin(a)"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # C13
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL3"] = (
        df_ghi["GHI C13"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL3"] = (
        df_ghi["GHI C13"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C13"] = (
        df_fix["GHI*sin(a+b)-CL3"]
        / df_fix["Sin(a)"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # C14
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL4"] = (
        df_ghi["GHI C14"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL4"] = (
        df_ghi["GHI C14"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C14"] = (
        df_fix["GHI*sin(a+b)-CL4"]
        / df_fix["Sin(a)"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # C15
    # --------------------------------------------------------

    df_fix["GHI*sin(a)-CL5"] = (
        df_ghi["GHI C15"]
        * df_fix["Sin(a)"]
    )

    df_fix["GHI*sin(a+b)-CL5"] = (
        df_ghi["GHI C15"]
        * df_fix["SIN(a+b)"]
    )

    df_fix["POA Fixed-C15"] = (
        df_fix["GHI*sin(a+b)-CL5"]
        / df_fix["Sin(a)"].replace(0, np.nan)
    )

    return df_fix


# ============================================================
# EFFECTIVE AREA
#
# IMPORTANT:
# Error % enters HERE ONCE.
# ============================================================

def calculate_effective_area(
    df_original,
    df_w_original,
    error,
):

    df = df_original.copy()
    df_w = df_w_original.copy()

    df["Error %"] = error

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - error
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
# FIXED POWER CALCULATION
# ============================================================

def calculate_fixed_power(
    df_fix,
    df_w,
):

    df_result = df_fix.copy()

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    power_cols = []

    for i in range(5):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        df_result[power_col] = (
            df_result[poa_cols[i]]
            * df_w.iloc[i]["Eff Area(m2)"]
        ) / 1_000_000

        power_cols.append(power_col)

    df_result[
        "Total Power (CL1+CL2+…)"
    ] = df_result[power_cols].sum(
        axis=1
    )

    return df_result


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
# ============================================================

def optimize_error(
    df_original,
    df_w_original,
    df_fix,
):

    actual = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        df, df_w = calculate_effective_area(
            df_original,
            df_w_original,
            error,
        )

        calculated = calculate_fixed_power(
            df_fix,
            df_w,
        )

        calculated_peak = (
            calculated[
                "Total Power (CL1+CL2+…)"
            ]
            .max()
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

    result_df = pd.DataFrame(
        results
    )

    best_row = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    best_error = float(
        best_row["Error %"]
    )

    return (
        best_error,
        result_df,
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

        df_backend = read_excel(
            uploaded_file,
            sheet_name=f"Backend Cal {cluster}",
        )

        backend_list.append(
            df_backend
        )

    df_trac = read_excel(
        uploaded_file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac.columns = (
        df_trac.columns
        .astype(str)
        .str.strip()
    )

    return (
        backend_list,
        df_trac,
    )


# ============================================================
# TRACKING OBJECTIVE
#
# Error % IS NOT USED HERE.
#
# It has already been applied to df_w.
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

    # --------------------------------------------------------
    # Cluster effective areas
    #
    # These already contain the Error % adjustment.
    # DO NOT subtract Error % again.
    # --------------------------------------------------------

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            for col in ghi_cols
        ]
    )

    blocks = pd.to_numeric(
        backend_list[0]["Block No."],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    if len(actual_full) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI data "
            "have different lengths."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. "
            "have different lengths."
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(x):

        DHI = int(round(x[0]))
        GHI_Starting_Block = int(round(x[1]))
        GHI_Ending_Block = int(round(x[2]))
        GHI_Max_Block = int(round(x[3]))
        Tracking_angle_lim_E = int(round(x[4]))
        Tracking_angle_lim_W = int(round(x[5]))

        # ----------------------------------------------------
        # EXACT ORIGINAL VALIDATION
        # ----------------------------------------------------

        if not (
            GHI_Starting_Block
            < GHI_Max_Block
            < GHI_Ending_Block
        ):
            return 1e9

        # ----------------------------------------------------
        # Slopes
        # ----------------------------------------------------

        denominator_1 = (
            GHI_Starting_Block
            - 1
            - GHI_Max_Block
        )

        denominator_2 = (
            GHI_Ending_Block
            + 1
            - GHI_Max_Block
        )

        if denominator_1 == 0:
            return 1e9

        if denominator_2 == 0:
            return 1e9

        m1 = 90 / denominator_1

        m2 = 90 / denominator_2

        # ----------------------------------------------------
        # Zenith angle
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Panel angle
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Cosine
        # ----------------------------------------------------

        cos_alpha = np.cos(
            np.radians(panel)
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
        # POWER
        #
        # IMPORTANT:
        # cl_weights already include Error %.
        # No additional efficiency correction here.
        # ----------------------------------------------------

        prediction_full = (
            dni @ cl_weights
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
            prediction_full[mask]
        )

        if len(prediction) == 0:
            return 1e9

        # ----------------------------------------------------
        # Errors
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

        score = (
            0.80 * block_error
            + 0.10 * peak_error
            + 0.10 * energy_error
        )

        return score

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
        (0, 10),       # DHI
        (10, 30),      # GHI Starting Block
        (65, 80),      # GHI Ending Block
        (47, 53),      # GHI Max Block
        (10, 70),      # Tracking East
        (10, 70),      # Tracking West
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
# FINAL TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    blocks,
    ghi_matrix,
    cl_weights,
    DHI,
    GHI_Starting_Block,
    GHI_Ending_Block,
    GHI_Max_Block,
    Tracking_angle_lim_E,
    Tracking_angle_lim_W,
):

    # --------------------------------------------------------
    # Slopes
    # --------------------------------------------------------

    denominator_1 = (
        GHI_Starting_Block
        - 1
        - GHI_Max_Block
    )

    denominator_2 = (
        GHI_Ending_Block
        + 1
        - GHI_Max_Block
    )

    if denominator_1 == 0:
        raise ValueError(
            "Invalid Tracking parameters: "
            "East slope denominator is zero."
        )

    if denominator_2 == 0:
        raise ValueError(
            "Invalid Tracking parameters: "
            "West slope denominator is zero."
        )

    m1 = 90 / denominator_1

    m2 = 90 / denominator_2

    # --------------------------------------------------------
    # Zenith
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Panel
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Cosine
    # --------------------------------------------------------

    cos_alpha = np.cos(
        np.radians(panel)
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
    # FINAL FORECAST
    #
    # Error % is NOT applied here.
    #
    # cl_weights already contain:
    #
    # Standard Efficiency - Error %
    # --------------------------------------------------------

    forecast = (
        dni @ cl_weights
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# FINAL ERROR METRICS
# ============================================================

def calculate_metrics(
    actual,
    forecast,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

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

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
        "Peak Error": peak_error,
        "Peak Error %": peak_error_pct,
        "Actual Energy": actual_energy,
        "Forecast Energy": forecast_energy,
        "Energy Error %": energy_error_pct,
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    x = np.arange(
        len(actual)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                width=2.5,
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
                width=2.5,
            ),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
        },
        height=480,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30,
        ),
        xaxis=dict(
            title="Block",
            dtick=5,
        ),
        yaxis=dict(
            title="Power",
        ),
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
# AUTOMATIC CALCULATION
# ============================================================

if st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Running automatic calculation..."
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

            lat = load_latitude(
                uploaded_file
            )

            month_lookup = load_tilt(
                uploaded_file
            )

            df_fix_raw = load_fixed_data(
                uploaded_file
            )

            df_fix = prepare_fixed_geometry(
                df_fix_raw,
                df_ghi,
                lat,
                month_lookup,
            )

            # ------------------------------------------------
            # AUTOMATIC ERROR %
            #
            # ALWAYS calculated once.
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
            # APPLY ERROR %
            #
            # THIS IS THE ONLY PLACE WHERE TRACKING
            # EFFECTIVE AREA RECEIVES ERROR %.
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
            # FIXED
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
            # FINAL TRACKING FORECAST
            # ------------------------------------------------

            (
                tracking_forecast,
                zenith,
                panel,
            ) = calculate_tracking_forecast(
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

            # ------------------------------------------------
            # STORE EVERYTHING
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

                "zenith":
                    zenith,

                "panel":
                    panel,
            }

            st.session_state.calculated = True

        st.success(
            "Automatic calculation completed."
        )

    except Exception as e:

        st.error(
            f"Calculation failed: {e}"
        )

        st.stop()


# ============================================================
# STOP UNTIL CALCULATION IS DONE
# ============================================================

if not st.session_state.calculated:
    st.stop()


# ============================================================
# LOAD STORED DATA
# ============================================================

data = st.session_state.calculation_data


# ============================================================
# EDITABLE PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ERROR %
# ============================================================

error_col1, error_col2, error_col3 = st.columns(
    [1, 1, 1]
)

with error_col1:

    error_value = st.number_input(
        "Error %",
        min_value=0.0,
        max_value=20.0,
        value=float(
            data["best_error"]
        ),
        step=0.1,
        format="%.1f",
    )


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    params = data[
        "tracking_parameters"
    ]

    st.markdown(
        "#### Tracking Parameters"
    )

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
        )

else:

    st.markdown(
        "#### Fixed Plant"
    )

    st.caption(
        "Error % controls the efficiency correction."
    )


# ============================================================
# APPLY EDITED PARAMETERS
# ============================================================

st.markdown("")

if st.button(
    "🔄 Apply Parameters",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Recalculating forecast..."
        ):

            # =================================================
            # ERROR % APPLIED ONCE
            # =================================================

            (
                df_final,
                df_w_final,
            ) = calculate_effective_area(
                data["df_original"],
                data["df_w_original"],
                error_value,
            )

            # =================================================
            # FIXED
            # =================================================

            fixed_final = calculate_fixed_power(
                data["df_fix"],
                df_w_final,
            )

            # =================================================
            # TRACKING
            # =================================================

            if plant_type == "Tracking":

                (
                    tracking_forecast,
                    zenith,
                    panel,
                ) = calculate_tracking_forecast(
                    data["blocks"],
                    data["ghi_matrix"],
                    (
                        pd.to_numeric(
                            df_w_final.iloc[:5, 1],
                            errors="coerce",
                        )
                        .fillna(0)
                        .to_numpy(
                            dtype=float
                        )
                    ),
                    int(dhi_value),
                    int(start_value),
                    int(end_value),
                    int(max_value),
                    int(east_value),
                    int(west_value),
                )

                data[
                    "tracking_forecast"
                ] = tracking_forecast

                data["zenith"] = zenith
                data["panel"] = panel

                data[
                    "tracking_parameters"
                ] = {
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

            # =================================================
            # SAVE
            # =================================================

            data["best_error"] = (
                float(error_value)
            )

            data["df_final"] = df_final
            data["df_w_final"] = df_w_final
            data["fixed_final"] = fixed_final

            st.session_state.calculation_data = data

        st.success(
            "Parameters applied successfully."
        )

    except Exception as e:

        st.error(
            f"Recalculation failed: {e}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

data = st.session_state.calculation_data


# ============================================================
# FORECAST / ACTUAL
# ============================================================

if plant_type == "Fixed":

    actual = (
        pd.to_numeric(
            data["df_fix"]["Actual"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy()
    )

    forecast = (
        data["fixed_final"][
            "Total Power (CL1+CL2+…)"
        ]
        .fillna(0)
        .to_numpy()
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    graph_title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    actual = (
        data["actual_tracking"]
    )

    forecast = (
        data["tracking_forecast"]
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    graph_title = (
        "Tracking Plant | Actual vs Forecast"
    )


# ============================================================
# METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown(
        f"""
        <div class="info-card">
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
        <div class="info-card">
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

with m3:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="metric-label">
                Peak Error %
            </div>
            <div class="metric-value">
                {metrics["Peak Error %"]:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="metric-label">
                Energy Error %
            </div>
            <div class="metric-value">
                {metrics["Energy Error %"]:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

fig = build_graph(
    actual,
    forecast,
    graph_title,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TRACKING ANGLE GRAPH
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section-title">🎯 Tracking Angles</div>',
        unsafe_allow_html=True,
    )

    angle_fig = go.Figure()

    x = np.arange(
        len(data["zenith"])
    )

    angle_fig.add_trace(
        go.Scatter(
            x=x,
            y=data["zenith"],
            mode="lines",
            name="Zenith Angle",
            line=dict(
                width=2,
            ),
        )
    )

    angle_fig.add_trace(
        go.Scatter(
            x=x,
            y=data["panel"],
            mode="lines",
            name="Panel Angle",
            line=dict(
                width=2,
            ),
        )
    )

    angle_fig.update_layout(
        height=360,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Block",
        yaxis_title="Angle (°)",
        margin=dict(
            l=30,
            r=30,
            t=20,
            b=30,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        angle_fig,
        use_container_width=True,
    )
