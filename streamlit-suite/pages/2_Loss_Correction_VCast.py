# ============================================================
# SOLAR FORECAST CORRECTION
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
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        padding: 12px;
        border-radius: 10px;
    }

    .result-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 15px 18px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Forecast correction and plant power calculation for Fixed and Tracking plants"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_PARAMS = {
    "DHI (%)": 1,
    "GHI Starting Block": 30,
    "GHI Ending Block": 79,
    "GHI Max Block": 53,
    "Tracking East Limit": 11,
    "Tracking West Limit": 23,
    "Error %": 4.9,
}


def initialize_state():

    if "auto_results" not in st.session_state:
        st.session_state.auto_results = None

    if "input_df" not in st.session_state:
        st.session_state.input_df = None

    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    if "plant_type" not in st.session_state:
        st.session_state.plant_type = "Fixed"

    if "params" not in st.session_state:
        st.session_state.params = DEFAULT_PARAMS.copy()


initialize_state()


# ============================================================
# FILE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">1. Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Excel workbook",
    type=["xlsx", "xls"],
)

if uploaded_file is None:

    st.info(
        "Upload the Excel workbook containing the required configuration "
        "and forecast sheets."
    )

    st.stop()


# ============================================================
# READ WORKBOOK
# ============================================================

@st.cache_data(show_spinner=False)
def load_workbook(file_bytes):

    return pd.ExcelFile(io.BytesIO(file_bytes))


file_bytes = uploaded_file.getvalue()

try:
    excel_file = load_workbook(file_bytes)
    sheet_names = excel_file.sheet_names

except Exception as e:

    st.error(f"Unable to read workbook: {e}")
    st.stop()


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section-title">2. Plant Type</div>',
    unsafe_allow_html=True,
)

plant_type = st.segmented_control(
    "Select plant type",
    ["Fixed", "Tracking"],
    default=st.session_state.plant_type,
)

if plant_type is None:
    plant_type = "Fixed"

st.session_state.plant_type = plant_type


# ============================================================
# COMMON DATA LOADER
# ============================================================

def first_valid_part(df, key_column):

    if key_column not in df.columns:
        return df.copy()

    null_indices = df[df[key_column].isna()].index

    if len(null_indices) == 0:
        return df.copy()

    first_null_pos = df.index.get_loc(null_indices[0])

    return df.iloc[:first_null_pos].copy()


def load_common_data(file_bytes):

    bio = io.BytesIO(file_bytes)

    # --------------------------------------------------------
    # AREA & EFFICIENCY
    # --------------------------------------------------------

    df = pd.read_excel(
        bio,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df = first_valid_part(df, "S.No.")

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    if (
        "No of Module" in df.columns
        and "Area of 1 Module (m2)" in df.columns
    ):

        df["Total area (m2)"] = (
            pd.to_numeric(df["No of Module"], errors="coerce")
            *
            pd.to_numeric(
                df["Area of 1 Module (m2)"],
                errors="coerce",
            )
        )

    else:

        df["Total area (m2)"] = pd.to_numeric(
            df.get("Total area (m2)", 0),
            errors="coerce",
        )

    # --------------------------------------------------------
    # CLUSTER TABLE
    # --------------------------------------------------------

    df_w = pd.read_excel(
        bio,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_w = first_valid_part(df_w, "Clusters")

    df_w.columns = (
        df_w.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # FORECAST CONFIG
    # --------------------------------------------------------

    df_st = pd.read_excel(
        bio,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = float(
        pd.to_numeric(
            df_st.loc[0, "Lat"],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # TILT
    # --------------------------------------------------------

    df_tilt = pd.read_excel(
        bio,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df_tilt.columns = (
        df_tilt.columns
        .astype(str)
        .str.strip()
    )

    df_tilt = first_valid_part(df_tilt, "Fixed")

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

    return df, df_w, lat, month_lookup


# ============================================================
# LOAD COMMON DATA
# ============================================================

try:

    (
        df_area,
        df_cluster,
        latitude,
        month_lookup,
    ) = load_common_data(file_bytes)

except Exception as e:

    st.error(
        "Error while reading Area & Efficiency / Forecast Config "
        f"data:\n\n{e}"
    )

    st.stop()


# ============================================================
# LOAD GHI
# ============================================================

def load_ghi_data(file_bytes):

    bio = io.BytesIO(file_bytes)

    df_ghi = pd.read_excel(
        bio,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

    df_ghi.columns = (
        df_ghi.columns
        .astype(str)
        .str.strip()
    )

    return df_ghi


# ============================================================
# LOAD ACTUAL
# ============================================================

def load_actual_data(file_bytes):

    bio = io.BytesIO(file_bytes)

    df_actual = pd.read_excel(
        bio,
        sheet_name="Fixed-C11",
        header=1,
    )

    df_actual.columns = (
        df_actual.columns
        .astype(str)
        .str.strip()
    )

    df_actual = first_valid_part(
        df_actual,
        "Date",
    )

    if "Actual" not in df_actual.columns:

        raise ValueError(
            "Column 'Actual' was not found in Fixed-C11 sheet."
        )

    actual = pd.to_numeric(
        df_actual["Actual"],
        errors="coerce",
    ).fillna(0)

    return actual.reset_index(drop=True)


# ============================================================
# INPUT DATA
# ============================================================

try:

    df_ghi = load_ghi_data(file_bytes)
    actual_series = load_actual_data(file_bytes)

except Exception as e:

    st.error(f"Unable to load GHI / Actual data: {e}")
    st.stop()


# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================

ghi_columns = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

missing_ghi = [
    col
    for col in ghi_columns
    if col not in df_ghi.columns
]

if missing_ghi:

    st.error(
        "Missing GHI columns: "
        + ", ".join(missing_ghi)
    )

    st.stop()


n = min(
    len(df_ghi),
    len(actual_series),
)

input_df = pd.DataFrame()

for col in ghi_columns:

    input_df[col] = pd.to_numeric(
        df_ghi[col].iloc[:n],
        errors="coerce",
    ).fillna(0).reset_index(drop=True)

input_df["Actual"] = (
    pd.to_numeric(
        actual_series.iloc[:n],
        errors="coerce",
    )
    .fillna(0)
    .reset_index(drop=True)
)


# ============================================================
# INPUT DATA EDITOR
# ============================================================

st.markdown(
    '<div class="section-title">3. Input Forecast & Actual Power</div>',
    unsafe_allow_html=True,
)

st.caption(
    "You can directly edit the GHI forecast and Actual Power values below. "
    "The calculation will use the edited values."
)

edited_input_df = st.data_editor(
    input_df,
    use_container_width=True,
    height=250,
    num_rows="fixed",
    key="input_data_editor",
)

# Ensure numeric
for col in edited_input_df.columns:

    edited_input_df[col] = pd.to_numeric(
        edited_input_df[col],
        errors="coerce",
    ).fillna(0)

st.session_state.input_df = edited_input_df.copy()


# ============================================================
# VALIDATION
# ============================================================

actual_array = (
    edited_input_df["Actual"]
    .to_numpy(dtype=float)
)

if np.count_nonzero(actual_array) == 0:

    st.error(
        "No non-zero Actual values found. "
        "Please enter Actual Power values in the Input DataFrame."
    )

    st.stop()


# ============================================================
# COMMON CALCULATION
# ============================================================

def calculate_solar_geometry(
    df_ghi,
    latitude,
    month_lookup,
):

    n = len(df_ghi)

    # --------------------------------------------------------
    # Original calculation intentionally preserved
    # --------------------------------------------------------

    dates = pd.Series(
        pd.Timestamp.today(),
        index=np.arange(n),
    )

    first_date = (
        pd.Timestamp.today()
        .replace(
            month=1,
            day=1,
        )
        .normalize()
    )

    declination = 23.45 * (
        np.sin(
            np.radians(
                360
                * (
                    284
                    + (dates - first_date).dt.days
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

    months = dates.dt.strftime("%B")

    tilt = months.map(
        month_lookup
    )

    a_plus_b = (
        elevation
        + pd.to_numeric(
            tilt,
            errors="coerce",
        )
    )

    sin_ab = np.sin(
        np.radians(a_plus_b)
    )

    sin_a = np.sin(
        np.radians(elevation)
    )

    return {
        "Declination Angle ∆": declination.to_numpy(),
        "Elevation angle a": elevation.to_numpy(),
        "Tilt Angle b": pd.to_numeric(
            tilt,
            errors="coerce",
        ).fillna(0).to_numpy(),
        "a+b": a_plus_b.to_numpy(),
        "SIN(a+b)": sin_ab.to_numpy(),
        "Sin(a)": sin_a.to_numpy(),
    }


# ============================================================
# ERROR % CALCULATION
# ============================================================

def calculate_cluster_areas(
    df_area_original,
    df_cluster_original,
    error_percent,
):

    df = df_area_original.copy()
    df_w = df_cluster_original.copy()

    # --------------------------------------------------------
    # ERROR IS APPLIED EXACTLY ONCE HERE
    # --------------------------------------------------------

    standard_eff = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    ).fillna(0)

    df["Error %"] = error_percent

    df["Net Efficiency (%)"] = (
        standard_eff
        - error_percent
    )

    # --------------------------------------------------------
    # TOTAL AREA
    # --------------------------------------------------------

    if (
        "No of Module" in df.columns
        and "Area of 1 Module (m2)" in df.columns
    ):

        df["Total area (m2)"] = (
            pd.to_numeric(
                df["No of Module"],
                errors="coerce",
            ).fillna(0)
            *
            pd.to_numeric(
                df["Area of 1 Module (m2)"],
                errors="coerce",
            ).fillna(0)
        )

    # --------------------------------------------------------
    # EFFECTIVE AREA
    # --------------------------------------------------------

    df["Eff Area"] = (
        df["Net Efficiency (%)"]
        *
        df["Total area (m2)"]
        /
        100
    )

    # --------------------------------------------------------
    # CLUSTER EFFECTIVE AREA
    # --------------------------------------------------------

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
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    input_df,
    cluster_df,
    latitude,
    month_lookup,
):

    ghi_matrix = np.column_stack(
        [
            input_df[col]
            .to_numpy(dtype=float)
            for col in ghi_columns
        ]
    )

    geometry = calculate_solar_geometry(
        input_df,
        latitude,
        month_lookup,
    )

    sin_a = geometry["Sin(a)"]
    sin_ab = geometry["SIN(a+b)"]

    sin_a_safe = np.where(
        np.abs(sin_a) < 1e-9,
        np.nan,
        sin_a,
    )

    poa = (
        ghi_matrix
        * sin_ab[:, None]
        /
        sin_a_safe[:, None]
    )

    # --------------------------------------------------------
    # Cluster effective areas
    # --------------------------------------------------------

    areas = pd.to_numeric(
        cluster_df["Eff Area(m2)"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    if len(areas) < 5:

        areas = np.pad(
            areas,
            (0, 5 - len(areas)),
        )

    areas = areas[:5]

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    cluster_power = (
        poa
        * areas[None, :]
        /
        1_000_000
    )

    forecast = np.sum(
        cluster_power,
        axis=1,
    )

    return forecast


# ============================================================
# FIND BEST ERROR %
# ============================================================

def find_best_error_fixed(
    input_df,
    df_area,
    df_cluster,
    latitude,
    month_lookup,
):

    actual = input_df["Actual"].to_numpy(
        dtype=float
    )

    valid = np.isfinite(actual)

    if not valid.any():
        return 0.0

    actual_peak = np.max(
        actual[valid]
    )

    if actual_peak <= 0:
        return 0.0

    results = []

    for error in np.arange(
        0,
        10.01,
        0.1,
    ):

        _, cluster_df = calculate_cluster_areas(
            df_area,
            df_cluster,
            error,
        )

        forecast = calculate_fixed_forecast(
            input_df,
            cluster_df,
            latitude,
            month_lookup,
        )

        calculated_peak = np.nanmax(
            forecast
        )

        peak_error = abs(
            calculated_peak
            - actual_peak
        )

        peak_error_pct = (
            peak_error
            /
            actual_peak
            *
            100
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

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["Peak Error"].idxmin()
    ]

    return float(
        best_row["Error %"]
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def build_tracking_forecast(
    x,
    ghi_matrix,
    blocks,
    cluster_areas,
):

    DHI = int(round(x[0]))

    GHI_Starting_Block = int(
        round(x[1])
    )

    GHI_Ending_Block = int(
        round(x[2])
    )

    GHI_Max_Block = int(
        round(x[3])
    )

    Tracking_angle_lim_E = int(
        round(x[4])
    )

    Tracking_angle_lim_W = int(
        round(x[5])
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not (
        GHI_Starting_Block
        <
        GHI_Max_Block
        <
        GHI_Ending_Block
    ):

        return None

    # --------------------------------------------------------
    # ORIGINAL SLOPE CALCULATION
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

    if (
        denominator_1 == 0
        or denominator_2 == 0
    ):

        return None

    m1 = 90 / denominator_1

    m2 = 90 / denominator_2

    # --------------------------------------------------------
    # ZENITH
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
    # PANEL ANGLE
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
                    >
                    Tracking_angle_lim_W
                )
            ),

            Tracking_angle_lim_W,

            zenith,
        ),
    )

    # --------------------------------------------------------
    # COSINE
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
        /
        100
    )

    # --------------------------------------------------------
    # DNI
    # --------------------------------------------------------

    dni = (
        ghi_matrix
        - dhi
    ) / cos_alpha[:, None]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # cluster_areas ALREADY contain Error %
    #
    # DO NOT APPLY Error % AGAIN.
    # --------------------------------------------------------

    forecast = (
        dni
        @
        cluster_areas
    ) / 1_000_000

    return (
        forecast,
        zenith,
        panel,
    )


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    input_df,
    cluster_df,
):

    actual_full = (
        input_df["Actual"]
        .to_numpy(dtype=float)
    )

    mask = (
        np.isfinite(actual_full)
        &
        (actual_full != 0)
    )

    actual = actual_full[mask]

    if len(actual) == 0:

        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual_max = np.max(actual)

    actual_sum = np.sum(actual)

    if actual_max <= 0:

        raise ValueError(
            "Actual Power maximum is zero."
        )

    ghi_matrix = np.column_stack(
        [
            input_df[col]
            .to_numpy(dtype=float)
            for col in ghi_columns
        ]
    )

    # --------------------------------------------------------
    # CLUSTER AREAS
    #
    # These already include Error % exactly once.
    # --------------------------------------------------------

    cluster_areas = pd.to_numeric(
        cluster_df["Eff Area(m2)"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    if len(cluster_areas) < 5:

        cluster_areas = np.pad(
            cluster_areas,
            (0, 5 - len(cluster_areas)),
        )

    cluster_areas = cluster_areas[:5]

    blocks = np.arange(
        len(input_df),
        dtype=float,
    )

    def objective(x):

        result = build_tracking_forecast(
            x,
            ghi_matrix,
            blocks,
            cluster_areas,
        )

        if result is None:
            return 1e9

        forecast_full = result[0]

        if (
            np.isnan(
                forecast_full
            ).any()
            or
            np.isinf(
                forecast_full
            ).any()
        ):

            return 1e9

        prediction = forecast_full[
            mask
        ]

        if len(prediction) == 0:
            return 1e9

        block_error = (
            np.mean(
                np.abs(
                    actual
                    - prediction
                )
            )
            /
            actual_max
        )

        peak_error = (
            abs(
                actual_max
                -
                np.max(
                    prediction
                )
            )
            /
            actual_max
        )

        energy_error = (
            abs(
                actual_sum
                -
                np.sum(
                    prediction
                )
            )
            /
            actual_sum
        )

        score = (
            0.80 * block_error
            +
            0.10 * peak_error
            +
            0.10 * energy_error
        )

        return score

    # --------------------------------------------------------
    # SAME BOUNDS
    # --------------------------------------------------------

    bounds = [
        (0, 10),       # DHI
        (10, 30),      # Starting
        (65, 80),      # Ending
        (47, 53),      # Max
        (10, 70),      # East
        (10, 70),      # West
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

    return {
        "DHI (%)": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
        "Optimization Score": float(
            result.fun
        ),
    }


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

st.markdown(
    '<div class="section-title">4. Automatic Calculation</div>',
    unsafe_allow_html=True,
)

if st.button(
    "⚙️ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Calculating solar forecast correction..."
    ):

        try:

            # ------------------------------------------------
            # STEP 1
            # Find Error %
            #
            # Error is determined only once.
            # ------------------------------------------------

            best_error = find_best_error_fixed(
                edited_input_df,
                df_area,
                df_cluster,
                latitude,
                month_lookup,
            )

            # ------------------------------------------------
            # STEP 2
            # Apply Error %
            #
            # THIS IS THE ONLY PLACE WHERE ERROR IS APPLIED.
            # ------------------------------------------------

            final_area_df, final_cluster_df = (
                calculate_cluster_areas(
                    df_area,
                    df_cluster,
                    best_error,
                )
            )

            # ------------------------------------------------
            # STEP 3
            # Plant calculation
            # ------------------------------------------------

            if plant_type == "Fixed":

                forecast = calculate_fixed_forecast(
                    edited_input_df,
                    final_cluster_df,
                    latitude,
                    month_lookup,
                )

                auto_results = {
                    "plant_type": "Fixed",
                    "forecast": forecast,
                    "error_percent": best_error,
                    "area_df": final_area_df,
                    "cluster_df": final_cluster_df,
                }

            else:

                # --------------------------------------------
                # TRACKING OPTIMIZATION
                #
                # Uses final_cluster_df.
                #
                # Error is NOT applied here.
                # --------------------------------------------

                tracking_params = optimize_tracking(
                    edited_input_df,
                    final_cluster_df,
                )

                x = np.array(
                    [
                        tracking_params["DHI (%)"],
                        tracking_params[
                            "GHI Starting Block"
                        ],
                        tracking_params[
                            "GHI Ending Block"
                        ],
                        tracking_params[
                            "GHI Max Block"
                        ],
                        tracking_params[
                            "Tracking East Limit"
                        ],
                        tracking_params[
                            "Tracking West Limit"
                        ],
                    ],
                    dtype=float,
                )

                ghi_matrix = np.column_stack(
                    [
                        edited_input_df[col]
                        .to_numpy(
                            dtype=float
                        )
                        for col in ghi_columns
                    ]
                )

                blocks = np.arange(
                    len(edited_input_df),
                    dtype=float,
                )

                cluster_areas = pd.to_numeric(
                    final_cluster_df[
                        "Eff Area(m2)"
                    ],
                    errors="coerce",
                ).fillna(0).to_numpy(
                    dtype=float
                )

                if len(cluster_areas) < 5:

                    cluster_areas = np.pad(
                        cluster_areas,
                        (
                            0,
                            5
                            -
                            len(cluster_areas),
                        ),
                    )

                cluster_areas = (
                    cluster_areas[:5]
                )

                result = build_tracking_forecast(
                    x,
                    ghi_matrix,
                    blocks,
                    cluster_areas,
                )

                forecast = result[0]

                auto_results = {
                    "plant_type": "Tracking",
                    "forecast": forecast,
                    "error_percent": best_error,
                    "area_df": final_area_df,
                    "cluster_df": final_cluster_df,
                    "tracking_params": tracking_params,
                    "zenith": result[1],
                    "panel": result[2],
                }

            st.session_state.auto_results = (
                auto_results
            )

            st.session_state.params[
                "Error %"
            ] = round(
                best_error,
                1,
            )

            if plant_type == "Tracking":

                for key in [
                    "DHI (%)",
                    "GHI Starting Block",
                    "GHI Ending Block",
                    "GHI Max Block",
                    "Tracking East Limit",
                    "Tracking West Limit",
                ]:

                    st.session_state.params[
                        key
                    ] = auto_results[
                        "tracking_params"
                    ][key]

            st.success(
                "Automatic calculation completed."
            )

        except Exception as e:

            st.error(
                f"Calculation failed: {e}"
            )


# ============================================================
# MANUAL PARAMETERS
# ============================================================

if st.session_state.auto_results is not None:

    results = st.session_state.auto_results

    st.markdown(
        '<div class="section-title">'
        "5. Parameters"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "These values are automatically calculated first. "
        "You can edit them and recalculate the final forecast."
    )

    # ========================================================
    # ERROR %
    # ========================================================

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        error_percent = st.number_input(
            "Error %",
            min_value=0.0,
            max_value=20.0,
            value=float(
                st.session_state.params[
                    "Error %"
                ]
            ),
            step=0.1,
            format="%.1f",
            key="manual_error",
        )

    st.session_state.params[
        "Error %"
    ] = error_percent

    # ========================================================
    # TRACKING PARAMETERS
    # ========================================================

    if plant_type == "Tracking":

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            dhi = st.number_input(
                "DHI (%)",
                min_value=0,
                max_value=10,
                value=int(
                    st.session_state.params[
                        "DHI (%)"
                    ]
                ),
                step=1,
            )

            starting_block = st.number_input(
                "GHI Starting Block",
                min_value=0,
                max_value=95,
                value=int(
                    st.session_state.params[
                        "GHI Starting Block"
                    ]
                ),
                step=1,
            )

        with col2:

            ending_block = st.number_input(
                "GHI Ending Block",
                min_value=1,
                max_value=96,
                value=int(
                    st.session_state.params[
                        "GHI Ending Block"
                    ]
                ),
                step=1,
            )

            max_block = st.number_input(
                "GHI Max Block",
                min_value=1,
                max_value=95,
                value=int(
                    st.session_state.params[
                        "GHI Max Block"
                    ]
                ),
                step=1,
            )

        with col3:

            east_limit = st.number_input(
                "Tracking East Limit",
                min_value=0,
                max_value=70,
                value=int(
                    st.session_state.params[
                        "Tracking East Limit"
                    ]
                ),
                step=1,
            )

            west_limit = st.number_input(
                "Tracking West Limit",
                min_value=0,
                max_value=70,
                value=int(
                    st.session_state.params[
                        "Tracking West Limit"
                    ]
                ),
                step=1,
            )

        st.session_state.params[
            "DHI (%)"
        ] = dhi

        st.session_state.params[
            "GHI Starting Block"
        ] = starting_block

        st.session_state.params[
            "GHI Ending Block"
        ] = ending_block

        st.session_state.params[
            "GHI Max Block"
        ] = max_block

        st.session_state.params[
            "Tracking East Limit"
        ] = east_limit

        st.session_state.params[
            "Tracking West Limit"
        ] = west_limit

        # ====================================================
        # RECALCULATE BUTTON
        # ====================================================

        if st.button(
            "🔄 Recalculate Forecast",
            type="primary",
            use_container_width=True,
        ):

            try:

                # --------------------------------------------
                # IMPORTANT:
                #
                # Rebuild cluster areas with Error %
                # exactly once.
                # --------------------------------------------

                final_area_df, final_cluster_df = (
                    calculate_cluster_areas(
                        df_area,
                        df_cluster,
                        error_percent,
                    )
                )

                ghi_matrix = np.column_stack(
                    [
                        edited_input_df[col]
                        .to_numpy(
                            dtype=float
                        )
                        for col in ghi_columns
                    ]
                )

                blocks = np.arange(
                    len(edited_input_df),
                    dtype=float,
                )

                cluster_areas = pd.to_numeric(
                    final_cluster_df[
                        "Eff Area(m2)"
                    ],
                    errors="coerce",
                ).fillna(0).to_numpy(
                    dtype=float
                )

                if len(cluster_areas) < 5:

                    cluster_areas = np.pad(
                        cluster_areas,
                        (
                            0,
                            5
                            -
                            len(cluster_areas),
                        ),
                    )

                cluster_areas = (
                    cluster_areas[:5]
                )

                x = np.array(
                    [
                        dhi,
                        starting_block,
                        ending_block,
                        max_block,
                        east_limit,
                        west_limit,
                    ],
                    dtype=float,
                )

                result = build_tracking_forecast(
                    x,
                    ghi_matrix,
                    blocks,
                    cluster_areas,
                )

                if result is None:

                    st.error(
                        "Invalid block parameters. "
                        "Required condition: "
                        "Starting < Max < Ending."
                    )

                    st.stop()

                forecast = result[0]

                st.session_state.auto_results = {
                    "plant_type": "Tracking",
                    "forecast": forecast,
                    "error_percent": error_percent,
                    "area_df": final_area_df,
                    "cluster_df": final_cluster_df,
                    "tracking_params": {
                        "DHI (%)": dhi,
                        "GHI Starting Block": starting_block,
                        "GHI Ending Block": ending_block,
                        "GHI Max Block": max_block,
                        "Tracking East Limit": east_limit,
                        "Tracking West Limit": west_limit,
                    },
                    "zenith": result[1],
                    "panel": result[2],
                }

                st.success(
                    "Forecast recalculated."
                )

            except Exception as e:

                st.error(
                    f"Recalculation failed: {e}"
                )

    else:

        # ====================================================
        # FIXED RECALCULATION
        # ====================================================

        if st.button(
            "🔄 Recalculate Forecast",
            type="primary",
            use_container_width=True,
        ):

            try:

                final_area_df, final_cluster_df = (
                    calculate_cluster_areas(
                        df_area,
                        df_cluster,
                        error_percent,
                    )
                )

                forecast = calculate_fixed_forecast(
                    edited_input_df,
                    final_cluster_df,
                    latitude,
                    month_lookup,
                )

                st.session_state.auto_results = {
                    "plant_type": "Fixed",
                    "forecast": forecast,
                    "error_percent": error_percent,
                    "area_df": final_area_df,
                    "cluster_df": final_cluster_df,
                }

                st.success(
                    "Forecast recalculated."
                )

            except Exception as e:

                st.error(
                    f"Recalculation failed: {e}"
                )


# ============================================================
# FINAL RESULTS
# ============================================================

if st.session_state.auto_results is not None:

    results = st.session_state.auto_results

    forecast = np.asarray(
        results["forecast"],
        dtype=float,
    )

    actual = (
        edited_input_df["Actual"]
        .to_numpy(dtype=float)
    )

    # ========================================================
    # METRICS
    # ========================================================

    calculated_peak = np.nanmax(
        forecast
    )

    actual_peak = np.nanmax(
        actual
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
        if actual_peak != 0
        else np.nan
    )

    st.markdown(
        '<div class="section-title">'
        "6. Results"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "Plant Type",
            plant_type,
        )

    with c2:

        st.metric(
            "Error %",
            f"{results['error_percent']:.1f}%",
        )

    with c3:

        st.metric(
            "Actual Peak",
            f"{actual_peak:.3f}",
        )

    with c4:

        st.metric(
            "Peak Error",
            f"{peak_error_pct:.2f}%",
        )

    # ========================================================
    # TRACKING PARAMETERS SUMMARY
    # ========================================================

    if (
        plant_type == "Tracking"
        and "tracking_params" in results
    ):

        params = results[
            "tracking_params"
        ]

        st.markdown(
            '<div class="section-title">'
            "Tracking Parameters"
            "</div>",
            unsafe_allow_html=True,
        )

        p1, p2, p3 = st.columns(
            3
        )

        with p1:

            st.metric(
                "DHI",
                params["DHI (%)"],
            )

            st.metric(
                "GHI Starting",
                params[
                    "GHI Starting Block"
                ],
            )

        with p2:

            st.metric(
                "GHI Ending",
                params[
                    "GHI Ending Block"
                ],
            )

            st.metric(
                "GHI Max",
                params[
                    "GHI Max Block"
                ],
            )

        with p3:

            st.metric(
                "East Limit",
                params[
                    "Tracking East Limit"
                ],
            )

            st.metric(
                "West Limit",
                params[
                    "Tracking West Limit"
                ],
            )


    # ========================================================
    # GRAPH
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        "Forecast vs Actual"
        "</div>",
        unsafe_allow_html=True,
    )

    x_axis = np.arange(
        len(forecast)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_axis,
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
            x=x_axis,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                width=2.5,
            ),
        )
    )

    if plant_type == "Tracking":

        zenith = results.get(
            "zenith"
        )

        panel = results.get(
            "panel"
        )

        if (
            zenith is not None
            and panel is not None
        ):

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=panel,
                    mode="lines",
                    name="Panel Angle",
                    visible="legendonly",
                    yaxis="y2",
                    line=dict(
                        dash="dot",
                    ),
                )
            )

            fig.update_layout(
                yaxis2=dict(
                    title="Panel Angle",
                    overlaying="y",
                    side="right",
                )
            )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        xaxis_title="Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# IMPORTANT CALCULATION NOTE
# ============================================================

st.caption(
    "Tracking calculation uses the error-adjusted cluster effective areas "
    "directly. Error % is not applied again during Tracking optimization "
    "or final Tracking forecast calculation."
)
