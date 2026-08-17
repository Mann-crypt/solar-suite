# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# COMPACT STREAMLIT APP
#
# CALCULATION LOGIC PRESERVED
# ERROR % IS APPLIED ONLY ONCE
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.optimize import differential_evolution


# ============================================================
# PAGE
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
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
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
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 700;
        margin-top: 2px;
    }

    div[data-testid="stFileUploader"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 5px;
        background: white;
    }

    .stButton > button {
        min-height: 42px;
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 7px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "calculated": False,
    "calculation_data": None,
    "plant_type": "Fixed",
    "file_key": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Automatic optimization with editable final parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Input Data</div>',
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "Input mode",
    ["Excel File", "Input DataFrame"],
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# EXCEL INPUT
# ============================================================

uploaded_file = None

if input_mode == "Excel File":

    uploaded_file = st.file_uploader(
        "Upload Solar Excel File",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload the Excel file to start.")
        st.stop()

    file_key = (
        uploaded_file.name,
        uploaded_file.size,
    )

else:

    # --------------------------------------------------------
    # Allows another page / parent application to provide:
    #
    # st.session_state["input_df"]
    #
    # The existing calculation functions still require the
    # Excel workbook for the other sheets, so this option is
    # intended for integration with a prepared dataframe.
    # --------------------------------------------------------

    input_df = st.session_state.get("input_df")

    if input_df is None:
        st.warning(
            "No Input DataFrame found. "
            "Set st.session_state['input_df'] before using "
            "this option."
        )
        st.stop()

    if not isinstance(input_df, pd.DataFrame):
        st.error("Input DataFrame must be a pandas DataFrame.")
        st.stop()

    st.success(
        f"Input DataFrame loaded: "
        f"{input_df.shape[0]:,} rows × {input_df.shape[1]:,} columns"
    )

    file_key = ("dataframe", id(input_df))


# ============================================================
# RESET CALCULATION WHEN INPUT CHANGES
# ============================================================

if st.session_state.file_key != file_key:

    st.session_state.file_key = file_key
    st.session_state.calculated = False
    st.session_state.calculation_data = None


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
    index=(
        0
        if st.session_state.plant_type == "Fixed"
        else 1
    ),
    label_visibility="collapsed",
)

if plant_type != st.session_state.plant_type:

    st.session_state.plant_type = plant_type
    st.session_state.calculated = False
    st.session_state.calculation_data = None

    st.rerun()


# ============================================================
# HELPERS
# ============================================================

def read_excel(file, **kwargs):
    file.seek(0)
    return pd.read_excel(file, **kwargs)


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def load_area_efficiency(file):

    df = read_excel(
        file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(12),
    )

    df.columns = (
        df.columns.astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )

    if "S.No." in df.columns:
        idx = df[df["S.No."].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])]

    for col in [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]:
        df[col] = numeric(df[col])

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


# ============================================================
# CLUSTER AREA
# ============================================================

def load_cluster_table(file):

    df = read_excel(
        file,
        sheet_name="Area & Efficiency",
        header=1,
        usecols=[14, 15],
    )

    df.columns = df.columns.astype(str).str.strip()

    if "Clusters" in df.columns:
        idx = df[df["Clusters"].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])]

    return df.reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

GHI_COLS = [
    "GHI C11",
    "GHI C12",
    "GHI C13",
    "GHI C14",
    "GHI C15",
]


def load_ghi(file):

    df = read_excel(
        file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    ).fillna(0)

    for col in GHI_COLS:
        if col in df.columns:
            df[col] = numeric(df[col])

    return df


# ============================================================
# LATITUDE
# ============================================================

def load_latitude(file):

    df = read_excel(
        file,
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
# TILT
# ============================================================

def load_tilt(file):

    df = read_excel(
        file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Fixed" in df.columns:
        idx = df[df["Fixed"].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])]

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

    return (
        df.set_index("Month")["Fixed"]
        .to_dict()
    )


# ============================================================
# FIXED DATA
# ============================================================

def load_fixed_data(file):

    df = read_excel(
        file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    if "Date" in df.columns:
        idx = df[df["Date"].isna()].index
        if len(idx):
            df = df.iloc[:df.index.get_loc(idx[0])]

    df["Actual"] = numeric(df["Actual"])

    return df.reset_index(drop=True)


# ============================================================
# SOLAR GEOMETRY
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

    day_number = (
        df["Date"] - first_date
    ).dt.days + 1

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    df["Elevation angle a"] = (
        90 - lat
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
        np.radians(df["a+b"])
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

    for i, cluster in enumerate(
        ["", "-CL2", "-CL3", "-CL4", "-CL5"]
    ):

        ghi = df_ghi[GHI_COLS[i]]

        df[f"GHI*sin(a){cluster}"] = (
            ghi * df["Sin(a)"]
        )

        df[f"GHI*sin(a+b){cluster}"] = (
            ghi * df["SIN(a+b)"]
        )

        poa_name = (
            "POA fixed"
            if i == 0
            else f"POA Fixed-C{i + 1}"
        )

        df[poa_name] = (
            df[f"GHI*sin(a+b){cluster}"]
            / sin_a
        )

    return df


# ============================================================
# EFFECTIVE AREA
# ERROR % APPLIED ONLY HERE
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
# FIXED POWER
# ============================================================

def calculate_fixed_power(
    df_fix,
    df_w,
):

    df = df_fix.copy()

    poa_cols = [
        "POA fixed",
        "POA Fixed-C12",
        "POA Fixed-C13",
        "POA Fixed-C14",
        "POA Fixed-C15",
    ]

    power_cols = []

    for i, poa in enumerate(poa_cols):

        col = f"CL{i + 1}_Fixed Power=I*Ƞ*A"

        df[col] = (
            df[poa]
            * df_w.iloc[i]["Eff Area(m2)"]
            / 1_000_000
        )

        power_cols.append(col)

    df["Total Power (CL1+CL2+…)"] = (
        df[power_cols]
        .sum(axis=1)
    )

    return df


# ============================================================
# AUTOMATIC ERROR OPTIMIZATION
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

    best_error = 0
    best_peak_error = np.inf

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

        forecast_peak = (
            calculated[
                "Total Power (CL1+CL2+…)"
            ]
            .max()
        )

        peak_error = abs(
            forecast_peak
            - actual_peak
        )

        if peak_error < best_peak_error:

            best_peak_error = peak_error
            best_error = error

    return round(
        float(best_error),
        1,
    )


# ============================================================
# TRACKING DATA
# ============================================================

def load_tracking_data(file):

    backend = []

    for cluster in [
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
    ]:

        backend.append(
            read_excel(
                file,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    tracking = read_excel(
        file,
        sheet_name="Tracking",
        header=1,
    )

    tracking.columns = (
        tracking.columns.astype(str)
        .str.strip()
    )

    return backend, tracking


# ============================================================
# TRACKING OBJECTIVE
# ============================================================

def create_tracking_objective(
    backend_list,
    df_ghi,
    df_fix,
    df_w,
):

    cl_weights = (
        numeric(df_w.iloc[:5, 1])
        .to_numpy(float)
    )

    ghi_matrix = np.column_stack(
        [
            numeric(df_ghi[col]).to_numpy(float)
            for col in GHI_COLS
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    ).to_numpy(float)

    actual_full = numeric(
        df_fix["Actual"]
    ).to_numpy(float)

    if len(actual_full) == 0:
        raise ValueError(
            "Actual data is empty."
        )

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found."
        )

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI lengths differ."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. lengths differ."
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
            start
            < maximum
            < end
        ):
            return 1e9

        d1 = (
            start
            - 1
            - maximum
        )

        d2 = (
            end
            + 1
            - maximum
        )

        if d1 == 0 or d2 == 0:
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(
            blocks <= maximum,
            np.minimum(
                89,
                m1 * (
                    blocks
                    - maximum
                ),
            ),
            np.minimum(
                89,
                m2 * (
                    blocks
                    - maximum
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

        prediction = prediction_full[mask]

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

    params = {
        "DHI": int(best[0]),
        "GHI Starting Block": int(best[1]),
        "GHI Ending Block": int(best[2]),
        "GHI Max Block": int(best[3]),
        "Tracking East Limit": int(best[4]),
        "Tracking West Limit": int(best[5]),
    }

    return (
        params,
        blocks,
        ghi_matrix,
        actual_full,
        cl_weights,
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

    d1 = (
        start
        - 1
        - maximum
    )

    d2 = (
        end
        + 1
        - maximum
    )

    if d1 == 0 or d2 == 0:
        raise ValueError(
            "Invalid tracking parameters."
        )

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1 * (
                blocks
                - maximum
            ),
        ),
        np.minimum(
            89,
            m2 * (
                blocks
                - maximum
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
        ghi_matrix
        - dhi
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

    actual = np.asarray(
        actual,
        dtype=float,
    )

    forecast = np.asarray(
        forecast,
        dtype=float,
    )

    return {
        "Actual Peak": np.max(actual),
        "Forecast Peak": np.max(forecast),
    }


# ============================================================
# GRAPH
# ============================================================

def build_graph(
    actual,
    forecast,
    title,
):

    n = min(
        len(actual),
        len(forecast),
    )

    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual[:n],
            mode="lines",
            name="Actual",
            line=dict(width=2.2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast[:n],
            mode="lines",
            name="Forecast",
            line=dict(width=2.2),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
        },
        height=450,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=25,
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
# RUN AUTOMATIC CALCULATION
# ============================================================

st.markdown("")

if st.button(
    "⚡ Run Automatic Calculation",
    type="primary",
    use_container_width=True,
):

    if input_mode != "Excel File":
        st.error(
            "The automatic solar calculation currently "
            "requires the Excel workbook because it uses "
            "multiple calculation sheets."
        )
        st.stop()

    try:

        with st.spinner(
            "Optimizing parameters and calculating forecast..."
        ):

            # ------------------------------------------------
            # COMMON DATA
            # ------------------------------------------------

            df_original = load_area_efficiency(
                uploaded_file
            )

            df_w_original = load_cluster_table(
                uploaded_file
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
            # OPTIMIZE ERROR
            # ------------------------------------------------

            best_error = optimize_error(
                df_original,
                df_w_original,
                df_fix,
            )

            # ------------------------------------------------
            # APPLY ERROR ONCE
            # ------------------------------------------------

            df_final, df_w_final = (
                calculate_effective_area(
                    df_original,
                    df_w_original,
                    best_error,
                )
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
            ) = optimize_tracking(
                backend_list,
                df_ghi,
                df_fix,
                df_w_final,
            )

            # ------------------------------------------------
            # INITIAL TRACKING FORECAST
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
            # STORE
            # ------------------------------------------------

            st.session_state.calculation_data = {
                "df_original": df_original,
                "df_w_original": df_w_original,
                "df_final": df_final,
                "df_w_final": df_w_final,
                "df_ghi": df_ghi,
                "df_fix": df_fix,
                "fixed_final": fixed_final,
                "backend_list": backend_list,
                "df_trac": df_trac,
                "blocks": blocks,
                "ghi_matrix": ghi_matrix,
                "actual_tracking": actual_tracking,
                "cl_weights": cl_weights,
                "best_error": best_error,
                "tracking_parameters":
                    tracking_parameters,
                "tracking_forecast":
                    tracking_forecast,
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
# WAIT FOR CALCULATION
# ============================================================

if not st.session_state.calculated:
    st.info(
        "Edit the plant type if required, then click "
        "**Run Automatic Calculation**."
    )
    st.stop()


# ============================================================
# DATA
# ============================================================

data = st.session_state.calculation_data


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

if plant_type == "Fixed":

    error_value = st.number_input(
        "Efficiency Error (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(
            data["best_error"]
        ),
        step=0.1,
        format="%.1f",
        key="fixed_error",
    )

else:

    error_value = st.number_input(
        "Efficiency Error (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(
            data["best_error"]
        ),
        step=0.1,
        format="%.1f",
        key="tracking_error",
    )


# ============================================================
# TRACKING EDITABLE PARAMETERS
# ============================================================

if plant_type == "Tracking":

    params = data["tracking_parameters"]

    st.markdown(
        "#### Tracking Parameters"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(params["DHI"]),
            step=1,
            key="dhi_input",
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
            key="start_input",
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
            key="end_input",
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
            key="max_input",
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
            key="east_input",
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
            key="west_input",
        )


# ============================================================
# LIVE FORECAST CALCULATION
#
# No Apply button.
#
# Automatic optimization does NOT run again.
# Only the forecast is recalculated using edited values.
# ============================================================

try:

    # --------------------------------------------------------
    # ERROR % APPLIED ONCE
    # --------------------------------------------------------

    df_final, df_w_final = (
        calculate_effective_area(
            data["df_original"],
            data["df_w_original"],
            error_value,
        )
    )

    # --------------------------------------------------------
    # FIXED FORECAST
    # --------------------------------------------------------

    fixed_final = calculate_fixed_power(
        data["df_fix"],
        df_w_final,
    )

    data["df_final"] = df_final
    data["df_w_final"] = df_w_final
    data["fixed_final"] = fixed_final
    data["best_error"] = float(
        error_value
    )

    # --------------------------------------------------------
    # TRACKING FORECAST
    # --------------------------------------------------------

    if plant_type == "Tracking":

        if not (
            start_value
            < max_value
            < end_value
        ):
            st.warning(
                "Tracking parameters must satisfy: "
                "GHI Starting Block < GHI Max Block "
                "< GHI Ending Block."
            )
            st.stop()

        tracking_weights = (
            numeric(
                df_w_final.iloc[:5, 1]
            )
            .to_numpy(float)
        )

        tracking_forecast = (
            calculate_tracking_forecast(
                data["blocks"],
                data["ghi_matrix"],
                tracking_weights,
                int(dhi_value),
                int(start_value),
                int(end_value),
                int(max_value),
                int(east_value),
                int(west_value),
            )
        )

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

except Exception as e:

    st.error(
        f"Parameter calculation failed: {e}"
    )
    st.stop()


# ============================================================
# RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Results</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FIXED / TRACKING FORECAST
# ============================================================

if plant_type == "Fixed":

    actual = numeric(
        data["df_fix"]["Actual"]
    ).to_numpy()

    forecast = numeric(
        data["fixed_final"][
            "Total Power (CL1+CL2+…)"
        ]
    ).to_numpy()

    title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    actual = np.asarray(
        data["actual_tracking"],
        dtype=float,
    )

    forecast = np.asarray(
        data["tracking_forecast"],
        dtype=float,
    )

    title = (
        "Tracking Plant | Actual vs Forecast"
    )


# ============================================================
# ONLY TWO METRICS
# ============================================================

metrics = calculate_metrics(
    actual,
    forecast,
)

m1, m2, m3, m4 = st.columns(4)

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
        "displaylogo": False,
        "responsive": True,
    },
)
