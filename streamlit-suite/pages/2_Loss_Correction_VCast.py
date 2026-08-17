# ============================================================
# SOLAR FORECAST / POWER CORRECTION
# FIXED + TRACKING PLANT
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
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1rem;
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
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Fixed and Tracking plant forecast correction with efficiency '
    'optimization and Plotly visualization.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    options=["Fixed", "Tracking"],
    default="Fixed",
    key="plant_type",
)

if plant_type is None:
    plant_type = "Fixed"


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Excel workbook",
    type=["xlsx", "xls"],
    help="Upload the Excel workbook containing the required solar sheets.",
)


if uploaded_file is None:

    st.info(
        "Upload the Excel workbook to start the calculation."
    )

    st.stop()


# ============================================================
# EXCEL HELPER
# ============================================================

@st.cache_data(show_spinner=False)
def load_excel(file_bytes):

    return io.BytesIO(file_bytes)


excel_file = load_excel(uploaded_file.getvalue())


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_until_first_null(df, column):

    if column not in df.columns:
        return df.copy()

    null_indices = df[df[column].isna()].index

    if len(null_indices) == 0:
        return df.copy()

    first_null_pos = df.index.get_loc(null_indices[0])

    return df.iloc[:first_null_pos].copy()


def numeric_array(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)


def safe_float(value, default=0.0):

    try:
        value = float(value)

        if np.isnan(value):
            return default

        return value

    except Exception:
        return default


# ============================================================
# READ COMMON DATA
# ============================================================

try:

    df_area = pd.read_excel(
        excel_file,
        sheet_name="Area & Efficiency",
        header=[1],
        usecols=range(12),
    )

    df_area = clean_until_first_null(
        df_area,
        "S.No."
    )

    df_area.columns = (
        df_area.columns
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    df_weights = pd.read_excel(
        excel_file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df_weights = clean_until_first_null(
        df_weights,
        "Clusters"
    )

    df_config = pd.read_excel(
        excel_file,
        sheet_name="Forecast Config",
        header=8,
    )

    lat = safe_float(
        df_config.loc[0, "Lat"]
    )

    df_ghi = pd.read_excel(
        excel_file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    )

    df_ghi = df_ghi.fillna(0)

except Exception as e:

    st.error(
        f"Error while reading common workbook data: {e}"
    )

    st.stop()


# ============================================================
# CHECK GHI COLUMNS
# ============================================================

ghi_cols = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]

missing_ghi = [
    col
    for col in ghi_cols
    if col not in df_ghi.columns
]

if missing_ghi:

    st.error(
        "Missing GHI columns: "
        + ", ".join(missing_ghi)
    )

    st.stop()


# ============================================================
# AREA / EFFICIENCY PREPARATION
# ============================================================

required_area_columns = [
    "Clusters",
    "Standard PV Efficiency (%)",
    "No of Module",
    "Area of 1 Module (m2)",
]

missing_area = [
    col
    for col in required_area_columns
    if col not in df_area.columns
]

if missing_area:

    st.error(
        "Missing columns in Area & Efficiency sheet: "
        + ", ".join(missing_area)
    )

    st.stop()


# ============================================================
# MAIN PARAMETERS
# ============================================================

st.markdown(
    '<div class="section-title">Correction Parameters</div>',
    unsafe_allow_html=True,
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    error_min = st.number_input(
        "Error % Minimum",
        min_value=-50.0,
        max_value=50.0,
        value=0.0,
        step=0.1,
    )

with p2:

    error_max = st.number_input(
        "Error % Maximum",
        min_value=-50.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
    )

with p3:

    error_step = st.number_input(
        "Error % Step",
        min_value=0.01,
        max_value=10.0,
        value=0.1,
        step=0.01,
    )

with p4:

    st.metric(
        "Latitude",
        f"{lat:.4f}°",
    )


if error_max < error_min:

    st.error(
        "Error % Maximum must be greater than Minimum."
    )

    st.stop()


# ============================================================
# AREA EFFICIENCY CALCULATION
# ============================================================

def calculate_cluster_area(
    df_area_input,
    df_weights_input,
    error_percent,
):

    df = df_area_input.copy()
    df_w = df_weights_input.copy()

    df["Standard PV Efficiency (%)"] = pd.to_numeric(
        df["Standard PV Efficiency (%)"],
        errors="coerce",
    ).fillna(0)

    df["No of Module"] = pd.to_numeric(
        df["No of Module"],
        errors="coerce",
    ).fillna(0)

    df["Area of 1 Module (m2)"] = pd.to_numeric(
        df["Area of 1 Module (m2)"],
        errors="coerce",
    ).fillna(0)

    df["Error %"] = error_percent

    df["Net Efficiency (%)"] = (
        df["Standard PV Efficiency (%)"]
        - error_percent
    )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
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
# FIXED PLANT FUNCTIONS
# ============================================================

def prepare_fixed_data():

    try:

        df_tilt = pd.read_excel(
            excel_file,
            sheet_name="Config Tilt Angle",
            header=[7],
        )

        df_tilt.columns = (
            df_tilt.columns
            .astype(str)
            .str.strip()
        )

        if "Fixed" not in df_tilt.columns:

            raise ValueError(
                "Column 'Fixed' not found in Config Tilt Angle."
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
                "Month column not found in Config Tilt Angle."
            )

        month_lookup = (
            df_tilt
            .set_index("Month")["Fixed"]
            .to_dict()
        )

        df_fix = pd.read_excel(
            excel_file,
            sheet_name="Fixed-C11",
            header=[1],
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

        if "Actual" not in df_fix.columns:

            raise ValueError(
                "Actual column not found in Fixed-C11."
            )

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        original_date = pd.to_datetime(
            df_fix["Date"],
            errors="coerce",
        )

        if original_date.notna().any():

            df_fix["Date"] = (
                original_date
                .ffill()
            )

        else:

            df_fix["Date"] = pd.Timestamp.today()

        # ----------------------------------------------------
        # SOLAR CALCULATIONS
        # ----------------------------------------------------

        first_date = (
            df_fix["Date"]
            .iloc[0]
            .replace(
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
        )

        days = (
            df_fix["Date"] - first_date
        ).dt.days

        df_fix["Declination Angle ∆"] = (
            23.45
            * np.sin(
                np.radians(
                    360
                    * (
                        284
                        + days
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

        df_fix["Tilt Angle b"] = pd.to_numeric(
            df_fix["Tilt Angle b"],
            errors="coerce",
        ).fillna(0)

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

        # Prevent division by zero
        sin_a = np.where(
            np.abs(df_fix["Sin(a)"]) < 1e-6,
            1e-6,
            df_fix["Sin(a)"],
        )

        # ----------------------------------------------------
        # POA FOR FIVE CLUSTERS
        # ----------------------------------------------------

        for i, ghi_col in enumerate(ghi_cols, start=1):

            ghi = numeric_array(
                df_ghi[ghi_col]
            )

            ghi_sin_a = (
                ghi
                * df_fix["Sin(a)"].to_numpy()
            )

            ghi_sin_ab = (
                ghi
                * df_fix["SIN(a+b)"].to_numpy()
            )

            if i == 1:

                poa_name = "POA fixed"

            else:

                poa_name = f"POA Fixed-C{i}"

            df_fix[f"GHI*sin(a)-CL{i}"] = (
                ghi_sin_a
            )

            df_fix[f"GHI*sin(a+b)-CL{i}"] = (
                ghi_sin_ab
            )

            df_fix[poa_name] = (
                ghi_sin_ab
                / sin_a
            )

        return df_fix

    except Exception as e:

        raise ValueError(
            f"Fixed data preparation failed: {e}"
        )


def calculate_fixed_power(
    df_fix_input,
    df_weights_input,
    error_percent,
):

    df_fix = df_fix_input.copy()

    _, df_w = calculate_cluster_area(
        df_area,
        df_weights_input,
        error_percent,
    )

    poa_cols = [
        "POA fixed",
        "POA Fixed-C2",
        "POA Fixed-C3",
        "POA Fixed-C4",
        "POA Fixed-C5",
    ]

    power_cols = []

    for i in range(5):

        cluster_number = i + 1

        power_col = (
            f"CL{cluster_number}_Fixed Power=I*Ƞ*A"
        )

        df_fix[power_col] = (
            df_fix[poa_cols[i]]
            * df_w.iloc[i]["Eff Area(m2)"]
            / 1_000_000
        )

        power_cols.append(power_col)

    df_fix["Total Power (CL1+CL2+…)"] = (
        df_fix[power_cols]
        .sum(axis=1)
    )

    return df_fix, df_w


# ============================================================
# FIND BEST FIXED ERROR
# ============================================================

def optimize_fixed_error(
    df_fix,
    df_weights_input,
    minimum,
    maximum,
    step,
):

    actual = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0)

    actual_peak = actual.max()

    if actual_peak <= 0:

        raise ValueError(
            "Actual power peak is zero or invalid."
        )

    errors = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    results = []

    for error in errors:

        temp_df, temp_w = calculate_fixed_power(
            df_fix,
            df_weights_input,
            error,
        )

        calculated_peak = (
            temp_df[
                "Total Power (CL1+CL2+…)"
            ].max()
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

    result_df = pd.DataFrame(results)

    best_row = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    best_error = float(
        best_row["Error %"]
    )

    final_df, final_weights = calculate_fixed_power(
        df_fix,
        df_weights_input,
        best_error,
    )

    return (
        best_error,
        result_df,
        final_df,
        final_weights,
    )


# ============================================================
# TRACKING DATA PREPARATION
# ============================================================

def prepare_tracking_data():

    backend_sheets = [
        "Backend Cal C11",
        "Backend Cal C12",
        "Backend Cal C13",
        "Backend Cal C14",
        "Backend Cal C15",
    ]

    backend_list = []

    for sheet in backend_sheets:

        backend_list.append(
            pd.read_excel(
                excel_file,
                sheet_name=sheet,
            )
        )

    df_trac = pd.read_excel(
        excel_file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac.columns = (
        df_trac.columns
        .astype(str)
        .str.strip()
    )

    if "Block No." not in backend_list[0].columns:

        raise ValueError(
            "Block No. column not found in Backend Cal C11."
        )

    blocks = numeric_array(
        backend_list[0]["Block No."]
    )

    ghi_matrix = np.column_stack(
        [
            numeric_array(df_ghi[col])
            for col in ghi_cols
        ]
    )

    actual_full = numeric_array(
        df_trac["Actual"]
        if "Actual" in df_trac.columns
        else pd.Series(
            np.zeros(len(blocks))
        )
    )

    # If Tracking sheet has no usable Actual,
    # use Fixed-C11 Actual.
    if len(actual_full) != len(blocks):

        df_fixed = pd.read_excel(
            excel_file,
            sheet_name="Fixed-C11",
            header=1,
        )

        df_fixed = clean_until_first_null(
            df_fixed,
            "Date",
        )

        actual_full = numeric_array(
            df_fixed["Actual"]
        )

    min_len = min(
        len(blocks),
        len(ghi_matrix),
        len(actual_full),
    )

    blocks = blocks[:min_len]

    ghi_matrix = ghi_matrix[:min_len]

    actual_full = actual_full[:min_len]

    return (
        backend_list,
        df_trac,
        blocks,
        ghi_matrix,
        actual_full,
    )


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    blocks,
    ghi_matrix,
    actual_full,
    cl_weights,
    score_block,
    score_peak,
    score_energy,
):

    mask = actual_full != 0

    if not mask.any():

        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    if actual_max <= 0:

        raise ValueError(
            "Actual peak is zero or invalid."
        )

    if actual_sum <= 0:

        raise ValueError(
            "Actual energy is zero or invalid."
        )

    def objective(x):

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

        if not (
            GHI_Starting_Block
            < GHI_Max_Block
            < GHI_Ending_Block
        ):

            return 1e9

        # ----------------------------------------------------
        # SLOPES
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

        if denominator_1 == 0 or denominator_2 == 0:

            return 1e9

        m1 = 90 / denominator_1

        m2 = 90 / denominator_2

        # ----------------------------------------------------
        # ZENITH
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

        zenith = np.abs(zenith)

        # ----------------------------------------------------
        # PANEL ANGLE
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

        panel = np.abs(panel)

        # ----------------------------------------------------
        # COSINE
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
        # PREDICTION
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
            score_block * block_error
            + score_peak * peak_error
            + score_energy * energy_error
        )

        return score

    return objective


# ============================================================
# FINAL TRACKING FORECAST
# ============================================================

def calculate_tracking_forecast(
    params,
    blocks,
    ghi_matrix,
    cl_weights,
):

    DHI = int(round(params[0]))

    GHI_Starting_Block = int(
        round(params[1])
    )

    GHI_Ending_Block = int(
        round(params[2])
    )

    GHI_Max_Block = int(
        round(params[3])
    )

    Tracking_angle_lim_E = int(
        round(params[4])
    )

    Tracking_angle_lim_W = int(
        round(params[5])
    )

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

    m1 = 90 / denominator_1
    m2 = 90 / denominator_2

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

    zenith = np.abs(zenith)

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

    panel = np.abs(panel)

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

    forecast = (
        dni @ cl_weights
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
        'Fixed Plant Correction'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Plant Type",
            "Fixed",
        )

    with c2:

        st.metric(
            "Latitude",
            f"{lat:.4f}°",
        )

    with c3:

        st.metric(
            "Error Search",
            f"{error_min:.1f}% → {error_max:.1f}%",
        )

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    try:

        df_fixed_base = prepare_fixed_data()

    except Exception as e:

        st.error(str(e))

        st.stop()

    # --------------------------------------------------------
    # RUN BUTTON
    # --------------------------------------------------------

    run_fixed = st.button(
        "▶ Run Fixed Correction",
        type="primary",
        use_container_width=True,
    )

    if run_fixed:

        with st.spinner(
            "Calculating Fixed plant correction..."
        ):

            try:

                (
                    best_error,
                    fixed_results,
                    final_fixed,
                    final_weights,
                ) = optimize_fixed_error(
                    df_fixed_base,
                    df_weights,
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
                    "fixed_weights"
                ] = final_weights

                st.success(
                    "Fixed correction completed."
                )

            except Exception as e:

                st.error(
                    f"Fixed calculation failed: {e}"
                )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "fixed_final" in st.session_state:

        final_fixed = st.session_state[
            "fixed_final"
        ]

        fixed_results = st.session_state[
            "fixed_results"
        ]

        best_error = st.session_state[
            "fixed_best_error"
        ]

        actual_peak = pd.to_numeric(
            final_fixed["Actual"],
            errors="coerce",
        ).max()

        forecast_peak = pd.to_numeric(
            final_fixed[
                "Total Power (CL1+CL2+…)"
            ],
            errors="coerce",
        ).max()

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

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Optimization Result'
            '</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Best Error %",
                f"{best_error:.2f}%",
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

        # ----------------------------------------------------
        # FORECAST VS ACTUAL
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Forecast vs Actual'
            '</div>',
            unsafe_allow_html=True,
        )

        forecast = pd.to_numeric(
            final_fixed[
                "Total Power (CL1+CL2+…)"
            ],
            errors="coerce",
        ).fillna(0)

        actual = pd.to_numeric(
            final_fixed["Actual"],
            errors="coerce",
        ).fillna(0)

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
            height=500,
            xaxis_title="15-Minute Block",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            template="plotly_white",
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
        # ERROR OPTIMIZATION GRAPH
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Error % Optimization'
            '</div>',
            unsafe_allow_html=True,
        )

        fig_error = go.Figure()

        fig_error.add_trace(
            go.Scatter(
                x=fixed_results["Error %"],
                y=fixed_results["Peak Error %"],
                mode="lines+markers",
                name="Peak Error %",
            )
        )

        fig_error.add_vline(
            x=best_error,
            line_dash="dash",
            annotation_text=(
                f"Best = {best_error:.2f}%"
            ),
            annotation_position="top",
        )

        fig_error.update_layout(
            height=420,
            xaxis_title="Error %",
            yaxis_title="Peak Error %",
            template="plotly_white",
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_error,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CLUSTER EFFECTIVE AREA
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Cluster Effective Area'
            '</div>',
            unsafe_allow_html=True,
        )

        weights_display = (
            st.session_state[
                "fixed_weights"
            ][
                [
                    "Clusters",
                    "Eff Area(m2)",
                ]
            ]
            .copy()
        )

        st.dataframe(
            weights_display,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # OPTIMIZATION TABLE
        # ----------------------------------------------------

        with st.expander(
            "View Error Optimization Results"
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
        'Tracking Plant Optimization'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TRACKING PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        "**Optimization Parameters**"
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        dhi_min = st.number_input(
            "DHI Minimum (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    with t2:

        dhi_max = st.number_input(
            "DHI Maximum (%)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
        )

    with t3:

        start_min = st.number_input(
            "GHI Start Min",
            min_value=1,
            max_value=95,
            value=10,
            step=1,
        )

    with t4:

        start_max = st.number_input(
            "GHI Start Max",
            min_value=1,
            max_value=95,
            value=30,
            step=1,
        )

    t5, t6, t7, t8 = st.columns(4)

    with t5:

        end_min = st.number_input(
            "GHI End Min",
            min_value=1,
            max_value=95,
            value=65,
            step=1,
        )

    with t6:

        end_max = st.number_input(
            "GHI End Max",
            min_value=1,
            max_value=95,
            value=80,
            step=1,
        )

    with t7:

        max_min = st.number_input(
            "GHI Max Block Min",
            min_value=1,
            max_value=95,
            value=47,
            step=1,
        )

    with t8:

        max_max = st.number_input(
            "GHI Max Block Max",
            min_value=1,
            max_value=95,
            value=53,
            step=1,
        )

    t9, t10, t11, t12 = st.columns(4)

    with t9:

        east_min = st.number_input(
            "East Angle Min",
            min_value=1,
            max_value=90,
            value=10,
            step=1,
        )

    with t10:

        east_max = st.number_input(
            "East Angle Max",
            min_value=1,
            max_value=90,
            value=70,
            step=1,
        )

    with t11:

        west_min = st.number_input(
            "West Angle Min",
            min_value=1,
            max_value=90,
            value=10,
            step=1,
        )

    with t12:

        west_max = st.number_input(
            "West Angle Max",
            min_value=1,
            max_value=90,
            value=70,
            step=1,
        )

    # --------------------------------------------------------
    # OPTIMIZER SETTINGS
    # --------------------------------------------------------

    st.markdown(
        "**Optimizer Settings**"
    )

    o1, o2, o3, o4 = st.columns(4)

    with o1:

        maxiter = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=500,
            value=40,
            step=1,
        )

    with o2:

        popsize = st.number_input(
            "Population Size",
            min_value=1,
            max_value=100,
            value=15,
            step=1,
        )

    with o3:

        tolerance = st.number_input(
            "Tolerance",
            min_value=0.00001,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            format="%.5f",
        )

    with o4:

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

    # --------------------------------------------------------
    # SCORE WEIGHTS
    # --------------------------------------------------------

    st.markdown(
        "**Optimization Score Weights**"
    )

    s1, s2, s3 = st.columns(3)

    with s1:

        block_weight = st.number_input(
            "Block Error Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.80,
            step=0.05,
        )

    with s2:

        peak_weight = st.number_input(
            "Peak Error Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
        )

    with s3:

        energy_weight = st.number_input(
            "Energy Error Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.05,
        )

    weight_total = (
        block_weight
        + peak_weight
        + energy_weight
    )

    if abs(weight_total - 1.0) > 0.0001:

        st.warning(
            f"Score weights currently total "
            f"{weight_total:.2f}. "
            "They are normally expected to total 1.00."
        )

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
            str(e)
        )

        st.stop()

    # --------------------------------------------------------
    # CLUSTER WEIGHTS
    # --------------------------------------------------------

    try:

        cl_weights = pd.to_numeric(
            df_weights.iloc[:5, 1],
            errors="coerce",
        ).fillna(0).to_numpy(
            dtype=float
        )

    except Exception as e:

        st.error(
            f"Could not calculate cluster weights: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # VALIDATE BOUNDS
    # --------------------------------------------------------

    valid_bounds = (
        dhi_min <= dhi_max
        and start_min <= start_max
        and end_min <= end_max
        and max_min <= max_max
        and east_min <= east_max
        and west_min <= west_max
    )

    if not valid_bounds:

        st.error(
            "One or more parameter minimum values "
            "are greater than their maximum values."
        )

        st.stop()

    # --------------------------------------------------------
    # RUN OPTIMIZATION
    # --------------------------------------------------------

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

                objective = create_tracking_objective(
                    blocks=blocks,
                    ghi_matrix=ghi_matrix,
                    actual_full=actual_full,
                    cl_weights=cl_weights,
                    score_block=block_weight,
                    score_peak=peak_weight,
                    score_energy=energy_weight,
                )

                bounds_tracking = [
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

                result = differential_evolution(
                    objective,
                    bounds=bounds_tracking,
                    strategy="best1bin",
                    maxiter=int(maxiter),
                    popsize=int(popsize),
                    tol=float(tolerance),
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=int(seed),
                    polish=True,
                    workers=1,
                )

                best_params = (
                    np.round(
                        result.x
                    ).astype(int)
                )

                (
                    forecast,
                    zenith,
                    panel,
                ) = calculate_tracking_forecast(
                    best_params,
                    blocks,
                    ghi_matrix,
                    cl_weights,
                )

                st.session_state[
                    "tracking_result"
                ] = result

                st.session_state[
                    "tracking_best"
                ] = best_params

                st.session_state[
                    "tracking_forecast"
                ] = forecast

                st.session_state[
                    "tracking_zenith"
                ] = zenith

                st.session_state[
                    "tracking_panel"
                ] = panel

                st.success(
                    "Tracking optimization completed."
                )

            except Exception as e:

                st.error(
                    f"Tracking optimization failed: {e}"
                )


    # --------------------------------------------------------
    # TRACKING RESULTS
    # --------------------------------------------------------

    if "tracking_best" in st.session_state:

        best = st.session_state[
            "tracking_best"
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

        result = st.session_state[
            "tracking_result"
        ]

        actual = actual_full[
            :len(forecast)
        ]

        # ----------------------------------------------------
        # PARAMETER VALUES
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

        actual_peak = (
            np.max(actual)
            if len(actual)
            else 0
        )

        forecast_peak = (
            np.max(forecast)
            if len(forecast)
            else 0
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
            'Tracking Optimization Result'
            '</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

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

        # ----------------------------------------------------
        # BEST PARAMETERS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Best Tracking Parameters'
            '</div>',
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
                "GHI End",
                GHI_Ending_Block,
            )

        with p4:

            st.metric(
                "GHI Max",
                GHI_Max_Block,
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
            'Tracking Forecast vs Actual'
            '</div>',
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
                name="Tracking Forecast",
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
            height=500,
            xaxis_title="15-Minute Block",
            yaxis_title="Power (MW)",
            hovermode="x unified",
            template="plotly_white",
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
        # TRACKING ANGLE GRAPH
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Tracking Angle Profile'
            '</div>',
            unsafe_allow_html=True,
        )

        fig_angle = go.Figure()

        fig_angle.add_trace(
            go.Scatter(
                x=x,
                y=zenith,
                mode="lines",
                name="Zenith Angle",
            )
        )

        fig_angle.add_trace(
            go.Scatter(
                x=x,
                y=panel,
                mode="lines",
                name="Panel Angle",
            )
        )

        fig_angle.update_layout(
            height=420,
            xaxis_title="15-Minute Block",
            yaxis_title="Angle (°)",
            hovermode="x unified",
            template="plotly_white",
        )

        st.plotly_chart(
            fig_angle,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # ENERGY RESULT
        # ----------------------------------------------------

        e1, e2, e3 = st.columns(3)

        with e1:

            st.metric(
                "Actual Energy",
                f"{actual_energy:.3f}",
            )

        with e2:

            st.metric(
                "Forecast Energy",
                f"{forecast_energy:.3f}",
            )

        with e3:

            st.metric(
                "Energy Error",
                f"{energy_error_pct:.2f}%",
            )

        # ----------------------------------------------------
        # DOWNLOAD TRACKING RESULT
        # ----------------------------------------------------

        output_tracking = df_tracking.copy()

        output_tracking = output_tracking.iloc[
            :len(forecast)
        ].copy()

        output_tracking[
            "Zenith Angle"
        ] = zenith

        output_tracking[
            "Panel Angle"
        ] = panel

        output_tracking[
            "Tracking Forecast Power"
        ] = forecast

        output_tracking[
            "Actual"
        ] = actual

        csv_tracking = (
            output_tracking
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Tracking Result CSV",
            data=csv_tracking,
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
    <div style="text-align:center;color:#6b7280;">
        Solar Forecast Correction
    </div>
    """,
    unsafe_allow_html=True,
)
