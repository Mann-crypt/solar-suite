# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# CLEAN + COMPACT STREAMLIT APP
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
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .section {
        font-size: 19px;
        font-weight: 650;
        margin: 20px 0 10px 0;
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
        margin-top: 3px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background: white;
    }

    .stButton > button {
        min-height: 42px;
        border-radius: 8px;
        font-weight: 600;
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
    "data": None,
    "plant_type": "Fixed",
    "input_df": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">☀️ Solar Forecast Correction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Automatic solar forecast correction with editable GHI, Actual Power and optimization parameters"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def read_excel(file, **kwargs):
    file.seek(0)
    return pd.read_excel(file, **kwargs)


def numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ============================================================
# LOAD AREA + EFFICIENCY
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
        idx = df["S.No."].isna()
        if idx.any():
            df = df.iloc[:df.index.get_loc(idx[idx].index[0])]

    df["Standard PV Efficiency (%)"] = numeric(
        df["Standard PV Efficiency (%)"]
    )

    df["No of Module"] = numeric(df["No of Module"])
    df["Area of 1 Module (m2)"] = numeric(
        df["Area of 1 Module (m2)"]
    )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df.reset_index(drop=True)


# ============================================================
# LOAD CLUSTER AREA
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
        idx = df["Clusters"].isna()
        if idx.any():
            df = df.iloc[:df.index.get_loc(idx[idx].index[0])]

    return df.reset_index(drop=True)


# ============================================================
# LOAD GHI
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
    )

    df = df.fillna(0)

    for col in GHI_COLS:
        if col in df.columns:
            df[col] = numeric(df[col])

    return df


# ============================================================
# LOAD LATITUDE
# ============================================================

def load_latitude(file):

    df = read_excel(
        file,
        sheet_name="Forecast Config",
        header=8,
    )

    return float(
        pd.to_numeric(
            df.loc[0, "Lat"],
            errors="coerce",
        )
    )


# ============================================================
# LOAD TILT
# ============================================================

def load_tilt(file):

    df = read_excel(
        file,
        sheet_name="Config Tilt Angle",
        header=7,
    )

    df.columns = df.columns.astype(str).str.strip()

    if "Fixed" in df.columns:
        idx = df["Fixed"].isna()
        if idx.any():
            df = df.iloc[:df.index.get_loc(idx[idx].index[0])]

    df = df.dropna(axis=1, how="all")

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    return df.set_index("Month")["Fixed"].to_dict()


# ============================================================
# LOAD FIXED DATA
# ============================================================

def load_fixed_data(file):

    df = read_excel(
        file,
        sheet_name="Fixed-C11",
        header=1,
    )

    df.columns = df.columns.astype(str).str.strip()

    if "Date" in df.columns:
        idx = df["Date"].isna()
        if idx.any():
            df = df.iloc[:df.index.get_loc(idx[idx].index[0])]

    df["Actual"] = numeric(df["Actual"])

    return df.reset_index(drop=True)


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
        .replace(month=1, day=1)
        .normalize()
    )

    days = (
        df["Date"] - first_date
    ).dt.days + 1

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360 * (284 + days) / 365
            )
        )
    )

    df["Elevation angle a"] = (
        90 - lat + df["Declination Angle ∆"]
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
        np.radians(df["Elevation angle a"])
    )

    for i, col in enumerate(GHI_COLS):

        suffix = "" if i == 0 else f"-CL{i + 1}"

        df[f"GHI*sin(a){suffix}"] = (
            df_ghi[col].to_numpy()
            * df["Sin(a)"].to_numpy()
        )

        df[f"GHI*sin(a+b){suffix}"] = (
            df_ghi[col].to_numpy()
            * df["SIN(a+b)"].to_numpy()
        )

        poa_name = (
            "POA fixed"
            if i == 0
            else f"POA Fixed-C{i + 1}"
        )

        df[poa_name] = (
            df[f"GHI*sin(a+b){suffix}"]
            / df["Sin(a)"].replace(0, np.nan)
        )

    return df


# ============================================================
# EFFECTIVE AREA
# ERROR % APPLIED ONLY ONCE
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
        "POA Fixed-C2",
        "POA Fixed-C3",
        "POA Fixed-C4",
        "POA Fixed-C5",
    ]

    power_cols = []

    for i, poa in enumerate(poa_cols):

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        area = pd.to_numeric(
            df_w.iloc[i]["Eff Area(m2)"],
            errors="coerce",
        )

        area = 0 if pd.isna(area) else area

        df[power_col] = (
            df[poa] * area / 1_000_000
        )

        power_cols.append(power_col)

    df["Total Power (CL1+CL2+…)"] = (
        df[power_cols].sum(axis=1)
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

    actual = numeric(df_fix["Actual"])

    actual_peak = actual.max()

    if actual_peak <= 0:
        raise ValueError(
            "No non-zero Actual values found."
        )

    best_error = 0
    best_peak_error = np.inf

    for error in np.arange(0, 10.01, 0.1):

        _, df_w = calculate_effective_area(
            df_original,
            df_w_original,
            error,
        )

        calculated = calculate_fixed_power(
            df_fix,
            df_w,
        )

        calculated_peak = calculated[
            "Total Power (CL1+CL2+…)"
        ].max()

        peak_error = abs(
            calculated_peak - actual_peak
        )

        if peak_error < best_peak_error:
            best_peak_error = peak_error
            best_error = error

    return round(best_error, 1)


# ============================================================
# LOAD TRACKING DATA
# ============================================================

def load_tracking_data(file):

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
                file,
                sheet_name=f"Backend Cal {cluster}",
            )
        )

    df_trac = read_excel(
        file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac.columns = (
        df_trac.columns
        .astype(str)
        .str.strip()
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

    ghi_matrix = np.column_stack(
        [
            numeric(df_ghi[col]).to_numpy()
            for col in GHI_COLS
        ]
    )

    blocks = numeric(
        backend_list[0]["Block No."]
    ).to_numpy()

    actual_full = numeric(
        df_fix["Actual"]
    ).to_numpy()

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI data have different lengths."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. have different lengths."
        )

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

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

        d1 = start - 1 - maximum
        d2 = end + 1 - maximum

        if d1 == 0 or d2 == 0:
            return 1e9

        m1 = 90 / d1
        m2 = 90 / d2

        zenith = np.where(
            blocks <= maximum,
            np.minimum(
                89,
                m1 * (blocks - maximum),
            ),
            np.minimum(
                89,
                m2 * (blocks - maximum),
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
            np.cos(np.radians(panel)),
            1e-6,
            None,
        )

        dhi = ghi_matrix * DHI / 100

        dni = (
            ghi_matrix - dhi
        ) / cos_alpha[:, None]

        prediction_full = (
            dni @ cl_weights
        ) / 1_000_000

        if not np.isfinite(
            prediction_full
        ).all():
            return 1e9

        prediction = prediction_full[mask]

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
        actual,
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
        actual,
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

    d1 = start - 1 - maximum
    d2 = end + 1 - maximum

    if d1 == 0 or d2 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    m1 = 90 / d1
    m2 = 90 / d2

    zenith = np.where(
        blocks <= maximum,
        np.minimum(
            89,
            m1 * (blocks - maximum),
        ),
        np.minimum(
            89,
            m2 * (blocks - maximum),
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
        np.cos(np.radians(panel)),
        1e-6,
        None,
    )

    dhi = ghi_matrix * DHI / 100

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

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    actual_peak = np.max(actual)
    forecast_peak = np.max(forecast)

    return {
        "Actual Peak": actual_peak,
        "Forecast Peak": forecast_peak,
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

    actual = actual[:n]
    forecast = forecast[:n]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(width=2.3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(width=2.3),
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
        },
        height=450,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
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
# INPUT FILE
# ============================================================

st.markdown(
    '<div class="section">📁 Input File</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info(
        "Upload the Excel file to start the calculation."
    )
    st.stop()


# ============================================================
# PLANT TYPE
# ============================================================

st.markdown(
    '<div class="section">🏭 Plant Type</div>',
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

st.session_state.plant_type = plant_type


# ============================================================
# LOAD USER INPUT DATA
# ============================================================

try:

    if st.session_state.input_df is None:

        ghi = load_ghi(uploaded_file)
        fixed = load_fixed_data(uploaded_file)

        n = min(
            len(ghi),
            len(fixed),
        )

        input_df = pd.DataFrame()

        input_df["Block"] = np.arange(1, n + 1)

        for col in GHI_COLS:
            input_df[col] = numeric(
                ghi[col].iloc[:n]
            ).to_numpy()

        input_df["Actual"] = numeric(
            fixed["Actual"].iloc[:n]
        ).to_numpy()

        st.session_state.input_df = input_df

except Exception as e:

    st.error(
        f"Unable to load input data: {e}"
    )

    st.stop()


# ============================================================
# EDITABLE INPUT DATA
# ============================================================

st.markdown(
    '<div class="section">📝 Input Data</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Edit GHI forecast and Actual Power values below. "
    "The edited values are used when you run the calculation."
)

edited_df = st.data_editor(
    st.session_state.input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    height=360,
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

st.session_state.input_df = edited_df.copy()


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
# CALCULATION
# ============================================================

if run_calculation:

    try:

        with st.spinner(
            "Running calculation..."
        ):

            # ------------------------------------------------
            # LOAD STATIC EXCEL DATA
            # ------------------------------------------------

            df_original = load_area_efficiency(
                uploaded_file
            )

            df_w_original = load_cluster_table(
                uploaded_file
            )

            lat = load_latitude(
                uploaded_file
            )

            month_lookup = load_tilt(
                uploaded_file
            )

            # ------------------------------------------------
            # USER-EDITED INPUT DATA
            # ------------------------------------------------

            user_df = st.session_state.input_df.copy()

            df_ghi = user_df[
                GHI_COLS
            ].copy()

            df_fix_raw = load_fixed_data(
                uploaded_file
            )

            n = min(
                len(df_fix_raw),
                len(df_ghi),
            )

            df_fix_raw = df_fix_raw.iloc[
                :n
            ].copy()

            df_ghi = df_ghi.iloc[
                :n
            ].copy()

            # Replace Actual with user's edited Actual
            df_fix_raw["Actual"] = numeric(
                user_df["Actual"].iloc[:n]
            ).to_numpy()

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
            # AUTOMATIC ERROR %
            # ------------------------------------------------

            best_error = optimize_error(
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

            st.session_state.data = {

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
# RESULTS
# ============================================================

if not st.session_state.calculated:
    st.info(
        "Edit the GHI / Actual values if required, "
        "then click Run Automatic Calculation."
    )
    st.stop()


data = st.session_state.data


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ERROR
# ============================================================

error_value = st.number_input(
    "Error %",
    min_value=0.0,
    max_value=20.0,
    value=float(data["best_error"]),
    step=0.1,
    format="%.1f",
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

if plant_type == "Tracking":

    params = data["tracking_parameters"]

    c1, c2, c3 = st.columns(3)

    with c1:

        dhi_value = st.number_input(
            "DHI (%)",
            min_value=0,
            max_value=100,
            value=int(params["DHI"]),
            step=1,
        )

        start_value = st.number_input(
            "GHI Starting Block",
            min_value=0,
            max_value=95,
            value=int(
                params["GHI Starting Block"]
            ),
            step=1,
        )

    with c2:

        end_value = st.number_input(
            "GHI Ending Block",
            min_value=1,
            max_value=96,
            value=int(
                params["GHI Ending Block"]
            ),
            step=1,
        )

        max_value = st.number_input(
            "GHI Max Block",
            min_value=0,
            max_value=95,
            value=int(
                params["GHI Max Block"]
            ),
            step=1,
        )

    with c3:

        east_value = st.number_input(
            "Tracking East Limit",
            min_value=0,
            max_value=90,
            value=int(
                params["Tracking East Limit"]
            ),
            step=1,
        )

        west_value = st.number_input(
            "Tracking West Limit",
            min_value=0,
            max_value=90,
            value=int(
                params["Tracking West Limit"]
            ),
            step=1,
        )


# ============================================================
# RECALCULATE PARAMETERS AUTOMATICALLY
# ============================================================

# Parameter widgets trigger Streamlit reruns.
# No Apply Parameters button is required.

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

    if plant_type == "Tracking":

        tracking_forecast = (
            calculate_tracking_forecast(
                data["blocks"],
                data["ghi_matrix"],
                pd.to_numeric(
                    df_w_final.iloc[:5, 1],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=float),
                int(dhi_value),
                int(start_value),
                int(end_value),
                int(max_value),
                int(east_value),
                int(west_value),
            )
        )

    else:

        tracking_forecast = data[
            "tracking_forecast"
        ]

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
    ).to_numpy()

    forecast = numeric(
        fixed_final[
            "Total Power (CL1+CL2+…)"
        ]
    ).to_numpy()

    title = (
        "Fixed Plant | Actual vs Forecast"
    )

else:

    actual = numeric(
        data["actual_tracking"]
    ).to_numpy()

    forecast = np.asarray(
        tracking_forecast,
        dtype=float,
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

st.markdown(
    '<div class="section">📊 Results</div>',
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
# GRAPH
# ============================================================

st.markdown(
    '<div class="section">📈 Forecast Comparison</div>',
    unsafe_allow_html=True,
)

st.plotly_chart(
    build_graph(
        actual,
        forecast,
        title,
    ),
    use_container_width=True,
)
