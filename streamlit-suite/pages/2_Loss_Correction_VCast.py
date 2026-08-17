# ============================================================
# SOLAR FORECAST CORRECTION
# FIXED / TRACKING
# COMPACT STREAMLIT UI
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
        padding-top: 1rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .section {
        font-size: 18px;
        font-weight: 650;
        margin: 18px 0 8px;
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 14px;
    }

    .metric-label {
        font-size: 12px;
        color: #6b7280;
    }

    .metric-value {
        font-size: 21px;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        padding: 0;
        border: none;
        background: transparent;
    }

    .stButton button {
        min-height: 40px;
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
# FILE
# ============================================================

st.markdown(
    '<div class="section">Input</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Solar Excel File",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("Upload the solar Excel file to start.")
    st.stop()


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.segmented_control(
    "Plant Type",
    ["Fixed", "Tracking"],
    default=st.session_state.plant_type,
)

if plant_type is None:
    plant_type = st.session_state.plant_type

if plant_type != st.session_state.plant_type:
    st.session_state.plant_type = plant_type
    st.session_state.calculated = False
    st.session_state.data = None

plant_type = st.session_state.plant_type


# ============================================================
# EXCEL HELPER
# ============================================================

def read_excel(file, **kwargs):
    file.seek(0)
    return pd.read_excel(file, **kwargs)


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
        idx = df["S.No."].isna()
        if idx.any():
            df = df.iloc[:df.index.get_loc(idx.idxmax())]

    for col in [
        "Standard PV Efficiency (%)",
        "No of Module",
        "Area of 1 Module (m2)",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["Total area (m2)"] = (
        df["No of Module"]
        * df["Area of 1 Module (m2)"]
    )

    return df


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
        idx = df["Clusters"].isna()
        if idx.any():
            df = df.iloc[:df.index.get_loc(idx.idxmax())]

    return df.reset_index(drop=True)


# ============================================================
# GHI
# ============================================================

def load_ghi(file):

    df = read_excel(
        file,
        sheet_name="Result",
        usecols=[0, 1, 2, 3, 4, 5],
    ).fillna(0)

    for col in [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

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

    return float(
        pd.to_numeric(
            df.loc[0, "Lat"],
            errors="coerce",
        )
    )


# ============================================================
# TILT
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
            df = df.iloc[:df.index.get_loc(idx.idxmax())]

    df = df.dropna(axis=1, how="all")

    df = df.rename(
        columns={
            "Unnamed: 2": "Month_Num",
            "Unnamed: 3": "Month",
        }
    )

    return df.set_index("Month")["Fixed"].to_dict()


# ============================================================
# FIXED DATA
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
            df = df.iloc[:df.index.get_loc(idx.idxmax())]

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    ).fillna(0)

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

    df["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + (
                        df["Date"]
                        - first_date
                    ).dt.days
                    + 1
                )
                / 365
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

    clusters = [
        ("C11", "GHI C11", "POA fixed"),
        ("C12", "GHI C12", "POA Fixed-C12"),
        ("C13", "GHI C13", "POA Fixed-C13"),
        ("C14", "GHI C14", "POA Fixed-C14"),
        ("C15", "GHI C15", "POA Fixed-C15"),
    ]

    for cluster, ghi_col, poa_col in clusters:

        ghi = pd.to_numeric(
            df_ghi[ghi_col],
            errors="coerce",
        ).fillna(0)

        df[f"GHI*sin(a)-{cluster}"] = (
            ghi * df["Sin(a)"]
        )

        df[f"GHI*sin(a+b)-{cluster}"] = (
            ghi * df["SIN(a+b)"]
        )

        df[poa_col] = (
            df[f"GHI*sin(a+b)-{cluster}"]
            / df["Sin(a)"].replace(0, np.nan)
        )

    return df


# ============================================================
# EFFECTIVE AREA
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

        power_col = (
            f"CL{i + 1}_Fixed Power=I*Ƞ*A"
        )

        df[power_col] = (
            df[poa]
            * df_w.iloc[i]["Eff Area(m2)"]
            / 1_000_000
        )

        power_cols.append(power_col)

    df["Total Power (CL1+CL2+…)"] = (
        df[power_cols].sum(axis=1)
    )

    return df


# ============================================================
# ERROR OPTIMIZATION
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

        results.append(
            {
                "Error %": round(error, 1),
                "Calculated Peak": calculated_peak,
                "Actual Peak": actual_peak,
                "Peak Error": peak_error,
                "Peak Error %": (
                    peak_error / actual_peak * 100
                ),
            }
        )

    result_df = pd.DataFrame(results)

    best = result_df.loc[
        result_df["Peak Error"].idxmin()
    ]

    return float(best["Error %"]), result_df


# ============================================================
# TRACKING DATA
# ============================================================

def load_tracking_data(file):

    backend_list = [
        read_excel(
            file,
            sheet_name=f"Backend Cal {cluster}",
        )
        for cluster in [
            "C11",
            "C12",
            "C13",
            "C14",
            "C15",
        ]
    ]

    df_trac = read_excel(
        file,
        sheet_name="Tracking",
        header=1,
    )

    df_trac.columns = (
        df_trac.columns.astype(str).str.strip()
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

    ghi_cols = [
        "GHI C11",
        "GHI C12",
        "GHI C13",
        "GHI C14",
        "GHI C15",
    ]

    cl_weights = (
        pd.to_numeric(
            df_w.iloc[:5, 1],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=float)
    )

    ghi_matrix = np.column_stack(
        [
            pd.to_numeric(
                df_ghi[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(dtype=float)
            for col in ghi_cols
        ]
    )

    blocks = pd.to_numeric(
        backend_list[0]["Block No."],
        errors="coerce",
    ).to_numpy(dtype=float)

    actual_full = pd.to_numeric(
        df_fix["Actual"],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=float)

    if len(actual_full) == 0:
        raise ValueError("Actual data is empty.")

    mask = actual_full != 0

    if not mask.any():
        raise ValueError(
            "No non-zero Actual values found for Tracking."
        )

    actual = actual_full[mask]

    actual_max = actual.max()
    actual_sum = actual.sum()

    if len(blocks) != len(ghi_matrix):
        raise ValueError(
            "Tracking Block No. and GHI data have different lengths."
        )

    if len(actual_full) != len(blocks):
        raise ValueError(
            "Tracking Actual and Block No. have different lengths."
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

        den1 = start - 1 - maximum
        den2 = end + 1 - maximum

        if den1 == 0 or den2 == 0:
            return 1e9

        m1 = 90 / den1
        m2 = 90 / den2

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
                (blocks > maximum)
                & (zenith > west),
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

        if (
            np.isnan(prediction_full).any()
            or np.isinf(prediction_full).any()
        ):
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

    best = np.round(result.x).astype(int)

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

    den1 = start - 1 - maximum
    den2 = end + 1 - maximum

    if den1 == 0 or den2 == 0:
        raise ValueError(
            "Invalid Tracking parameters."
        )

    m1 = 90 / den1
    m2 = 90 / den2

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
            (blocks > maximum)
            & (zenith > west),
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

    return forecast, zenith, panel


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, forecast):

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    actual_peak = np.max(actual)
    forecast_peak = np.max(forecast)

    peak_error = abs(
        forecast_peak - actual_peak
    )

    peak_error_pct = (
        peak_error / actual_peak * 100
        if actual_peak != 0
        else np.nan
    )

    actual_energy = np.sum(actual)
    forecast_energy = np.sum(forecast)

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
# GRAPH
# ============================================================

def build_graph(actual, forecast, title):

    x = np.arange(len(actual))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(
            l=30,
            r=20,
            t=55,
            b=30,
        ),
        xaxis_title="Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            y=1.02,
            x=1,
            xanchor="right",
        ),
    )

    return fig


# ============================================================
# AUTOMATIC CALCULATION
# ============================================================

if not st.session_state.calculated:

    if st.button(
        "⚡ Run Automatic Calculation",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Optimizing solar forecast..."
            ):

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

                best_error, error_results = (
                    optimize_error(
                        df_original,
                        df_w_original,
                        df_fix,
                    )
                )

                df_final, df_w_final = (
                    calculate_effective_area(
                        df_original,
                        df_w_original,
                        best_error,
                    )
                )

                fixed_final = calculate_fixed_power(
                    df_fix,
                    df_w_final,
                )

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

                st.session_state.data = {
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
                    "error_results": error_results,
                    "tracking_parameters": tracking_parameters,
                    "tracking_score": tracking_score,
                    "tracking_forecast": tracking_forecast,
                    "zenith": zenith,
                    "panel": panel,
                }

                st.session_state.calculated = True

            st.rerun()

        except Exception as e:
            st.error(
                f"Calculation failed: {e}"
            )

    st.stop()


# ============================================================
# DATA
# ============================================================

data = st.session_state.data


# ============================================================
# PARAMETERS
# ============================================================

st.markdown(
    '<div class="section">⚙️ Parameters</div>',
    unsafe_allow_html=True,
)

params = data["tracking_parameters"]


with st.form("parameter_form"):

    if plant_type == "Fixed":

        c1, c2 = st.columns([1, 3])

        with c1:
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

    else:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
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

        with c2:
            dhi_value = st.number_input(
                "DHI (%)",
                min_value=0,
                max_value=100,
                value=int(params["DHI"]),
                step=1,
            )

        with c3:
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

        with c4:
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

        c5, c6, c7 = st.columns(3)

        with c5:
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

        with c6:
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

        with c7:
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

    apply_parameters = st.form_submit_button(
        "🔄 Apply Parameters",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# APPLY PARAMETERS
# ============================================================

if apply_parameters:

    try:

        with st.spinner(
            "Recalculating..."
        ):

            df_final, df_w_final = (
                calculate_effective_area(
                    data["df_original"],
                    data["df_w_original"],
                    error_value,
                )
            )

            fixed_final = calculate_fixed_power(
                data["df_fix"],
                df_w_final,
            )

            if plant_type == "Tracking":

                weights = (
                    pd.to_numeric(
                        df_w_final.iloc[:5, 1],
                        errors="coerce",
                    )
                    .fillna(0)
                    .to_numpy(dtype=float)
                )

                (
                    tracking_forecast,
                    zenith,
                    panel,
                ) = calculate_tracking_forecast(
                    data["blocks"],
                    data["ghi_matrix"],
                    weights,
                    int(dhi_value),
                    int(start_value),
                    int(end_value),
                    int(max_value),
                    int(east_value),
                    int(west_value),
                )

                data["tracking_forecast"] = (
                    tracking_forecast
                )

                data["zenith"] = zenith
                data["panel"] = panel

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

            data["best_error"] = float(
                error_value
            )

            data["df_final"] = df_final
            data["df_w_final"] = df_w_final
            data["fixed_final"] = fixed_final

            st.session_state.data = data

        st.rerun()

    except Exception as e:
        st.error(
            f"Recalculation failed: {e}"
        )


# ============================================================
# FINAL FORECAST
# ============================================================

data = st.session_state.data

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

    title = "Fixed Plant | Actual vs Forecast"

else:

    actual = np.asarray(
        data["actual_tracking"],
        dtype=float,
    )

    forecast = np.asarray(
        data["tracking_forecast"],
        dtype=float,
    )

    title = "Tracking Plant | Actual vs Forecast"


metrics = calculate_metrics(
    actual,
    forecast,
)


# ============================================================
# RESULTS
# ============================================================

st.markdown(
    '<div class="section">📊 Results</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

metric_data = [
    (
        "Actual Peak",
        f'{metrics["Actual Peak"]:.3f}',
    ),
    (
        "Forecast Peak",
        f'{metrics["Forecast Peak"]:.3f}',
    ),
    (
        "Peak Error",
        f'{metrics["Peak Error %"]:.2f}%',
    ),
    (
        "Energy Error",
        f'{metrics["Energy Error %"]:.2f}%',
    ),
]

for col, (label, value) in zip(
    [m1, m2, m3, m4],
    metric_data,
):

    with col:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">
                    {label}
                </div>
                <div class="metric-value">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FORECAST GRAPH
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
    config={
        "displayModeBar": False,
        "responsive": True,
    },
)


# ============================================================
# TRACKING GRAPH
# ============================================================

if plant_type == "Tracking":

    st.markdown(
        '<div class="section">🎯 Tracking Angles</div>',
        unsafe_allow_html=True,
    )

    x = np.arange(
        len(data["zenith"])
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["zenith"],
            mode="lines",
            name="Zenith",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["panel"],
            mode="lines",
            name="Panel",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        height=330,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(
            l=30,
            r=20,
            t=20,
            b=30,
        ),
        xaxis_title="Block",
        yaxis_title="Angle (°)",
        legend=dict(
            orientation="h",
            y=1.02,
            x=1,
            xanchor="right",
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )
