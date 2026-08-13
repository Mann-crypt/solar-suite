import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
from io import BytesIO
from datetime import datetime
import random


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Solar Loss Correction",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #777;
    font-size: 15px;
}

.metric-card {
    padding: 12px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">⚡ Solar Loss Correction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Forecast correction and efficiency-loss optimization platform</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "calculated": False,
    "optimization_result": None,
    "best_loss": None,
    "optimized_params": None,
    "current_file": None,
    "editor_version": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CACHE HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes):
    """
    Read workbook sheet names only.
    Cached so Streamlit does not repeatedly open the workbook.
    """
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_cached(
    file_bytes,
    sheet_name,
    header=0,
    usecols=None
):
    """
    Cached Excel reader.
    """
    return pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=header,
        usecols=usecols
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_until_empty(df, column):
    """
    Keep rows until the first null value in a selected column.
    """
    df = df.copy()

    if column not in df.columns:
        return df

    null_indices = df[df[column].isna()].index

    if len(null_indices) > 0:
        first_null_position = df.index.get_loc(null_indices[0])
        df = df.iloc[:first_null_position]

    return df.reset_index(drop=True)


def calculate_metrics(actual, forecast):
    """
    Calculate forecast performance metrics.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(forecast)

    actual = actual[valid]
    forecast = forecast[valid]

    if len(actual) == 0:
        return {
            "MAE": np.nan,
            "MAPE": np.nan,
            "Peak Error": np.nan,
            "Energy Error": np.nan,
            "R2": np.nan,
        }

    daylight = actual != 0

    if daylight.any():
        a = actual[daylight]
        f = forecast[daylight]
    else:
        a = actual
        f = forecast

    mae = np.mean(np.abs(a - f))

    non_zero = a != 0

    if non_zero.any():
        mape = np.mean(
            np.abs((a[non_zero] - f[non_zero]) / a[non_zero])
        ) * 100
    else:
        mape = np.nan

    actual_peak = np.max(a)
    forecast_peak = np.max(f)

    if actual_peak != 0:
        peak_error = (
            abs(actual_peak - forecast_peak)
            / abs(actual_peak)
        ) * 100
    else:
        peak_error = np.nan

    actual_energy = np.sum(a)
    forecast_energy = np.sum(f)

    if actual_energy != 0:
        energy_error = (
            abs(actual_energy - forecast_energy)
            / abs(actual_energy)
        ) * 100
    else:
        energy_error = np.nan

    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)

    if ss_tot != 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = np.nan

    return {
        "MAE": mae,
        "MAPE": mape,
        "Peak Error": peak_error,
        "Energy Error": energy_error,
        "R2": r2,
    }


def create_chart(actual, forecast, title="Forecast vs Actual Power"):

    x = np.arange(1, len(actual) + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                color="#2563EB",
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(
                color="#DC2626",
                width=3
            )
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        hovermode="x unified",

        xaxis=dict(
            title="15 Minute Block",
            dtick=4
        ),

        yaxis=dict(
            title="Power (MW)"
        ),

        legend=dict(
            orientation="h",
            y=1.08,
            x=0
        ),

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return fig


# ============================================================
# EFFICIENCY LOSS OPTIMIZATION
# ============================================================

def optimize_efficiency_loss(
    area_df,
    base_forecast_factor,
    actual
):
    """
    Find efficiency loss producing minimum peak error.

    base_forecast_factor represents forecast power when
    efficiency loss = 0%.

    Since power is linearly dependent on net efficiency,
    efficiency-loss testing can be vectorized instead of
    rebuilding the complete dataframe for every loss.
    """

    standard_eff = pd.to_numeric(
        area_df["Standard PV Efficiency (%)"],
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)

    total_area = pd.to_numeric(
        area_df["Total area(m2)"],
        errors="coerce"
    ).fillna(0).to_numpy(dtype=float)

    # Maximum allowed loss
    max_loss = float(np.nanmin(standard_eff))

    if not np.isfinite(max_loss) or max_loss < 0:
        max_loss = 0

    losses = np.round(
        np.arange(
            0,
            max_loss + 0.0001,
            0.1
        ),
        1
    )

    # Total effective area for each loss
    #
    # Effective area:
    # Area * (Efficiency - Loss) / 100
    #
    # Shape:
    # losses x modules
    net_eff = (
        standard_eff[None, :]
        - losses[:, None]
    )

    eff_area = (
        total_area[None, :]
        * net_eff
        / 100
    )

    total_eff_area = np.sum(
        eff_area,
        axis=1
    )

    # base_forecast_factor is POA / 1e6
    #
    # Each candidate forecast:
    #
    # forecast = base_forecast_factor * total_eff_area

    forecast_matrix = (
        base_forecast_factor[None, :]
        * total_eff_area[:, None]
    )

    actual_peak = np.max(actual)

    predicted_peaks = np.max(
        forecast_matrix,
        axis=1
    )

    peak_errors = np.abs(
        actual_peak - predicted_peaks
    )

    best_index = np.argmin(
        peak_errors
    )

    best_loss = float(
        losses[best_index]
    )

    return best_loss, pd.DataFrame({
        "Efficiency Loss (%)": losses,
        "Actual Peak": actual_peak,
        "Predicted Peak": predicted_peaks,
        "Peak Error": peak_errors
    })


# ============================================================
# AREA & EFFICIENCY
# ============================================================

def prepare_area_efficiency(
    file_bytes,
    is_cluster
):

    if is_cluster:

        df = read_excel_cached(
            file_bytes,
            "Area & Efficiency",
            header=1,
            usecols=range(8)
        )

    else:

        df = read_excel_cached(
            file_bytes,
            "Area & Efficiency",
            header=1
        )

    df.columns = df.columns.astype(str).str.strip()

    df = clean_until_empty(
        df,
        "Module Type"
    )

    return df


# ============================================================
# FIXED PLANT PREPARATION
# ============================================================

def prepare_fixed_data(
    file_bytes,
    edited_input,
    is_cluster
):

    if is_cluster:

        df_fix = read_excel_cached(
            file_bytes,
            "Fixed-CL1",
            header=1
        )

        df_fix.columns = (
            df_fix.columns
            .astype(str)
            .str.strip()
        )

        df_fix = clean_until_empty(
            df_fix,
            "Date"
        )

        df_fix = df_fix.iloc[:96].copy()

        ghi_cols = [
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI"
        ]

        for col in ghi_cols:
            if col in edited_input.columns:
                df_fix[col] = pd.to_numeric(
                    edited_input[col],
                    errors="coerce"
                ).fillna(0).to_numpy()

        df_fix["Actual"] = pd.to_numeric(
            edited_input["Actual"],
            errors="coerce"
        ).fillna(0).to_numpy()

        return df_fix, ghi_cols

    else:

        df_fix = read_excel_cached(
            file_bytes,
            "Fixed",
            header=1
        )

        df_fix.columns = (
            df_fix.columns
            .astype(str)
            .str.strip()
        )

        df_fix = clean_until_empty(
            df_fix,
            "Date"
        )

        df_fix = df_fix.iloc[:96].copy()

        df_fix["GHI_Forecast"] = pd.to_numeric(
            edited_input["GHI_Forecast"],
            errors="coerce"
        ).fillna(0).to_numpy()

        df_fix["Actual"] = pd.to_numeric(
            edited_input["Actual"],
            errors="coerce"
        ).fillna(0).to_numpy()

        return df_fix, ["GHI_Forecast"]


# ============================================================
# SOLAR GEOMETRY
# ============================================================

def add_solar_geometry(
    df_fix,
    file_bytes,
    is_tracking
):

    config = read_excel_cached(
        file_bytes,
        "Forecast Config",
        header=8
    )

    config.columns = (
        config.columns
        .astype(str)
        .str.strip()
    )

    lat = float(
        config.loc[0, "Lat"]
    )

    df_fix = df_fix.copy()

    today = pd.Timestamp.today().normalize()

    df_fix["Date"] = today

    first_date = today.replace(
        month=1,
        day=1
    )

    day_number = (
        df_fix["Date"] - first_date
    ).dt.days + 1

    df_fix["Declination Angle ∆"] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (
                    284
                    + day_number
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

    if is_tracking:

        df_fix["Tilt Angle b"] = 0

    else:

        tilt = read_excel_cached(
            file_bytes,
            "Config Tilt Angle",
            header=7
        )

        tilt.columns = (
            tilt.columns
            .astype(str)
            .str.strip()
        )

        tilt = clean_until_empty(
            tilt,
            "Fixed"
        )

        tilt = tilt.dropna(
            how="all",
            axis=1
        )

        tilt = tilt.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month"
            }
        )

        if "Month" in tilt.columns:

            month_lookup = (
                tilt
                .set_index("Month")["Fixed"]
                .to_dict()
            )

            df_fix["Tilt Angle b"] = (
                df_fix["Date"]
                .dt.strftime("%B")
                .map(month_lookup)
                .fillna(0)
            )

        else:

            df_fix["Tilt Angle b"] = 0

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

    return df_fix, lat


# ============================================================
# FIXED FORECAST
# ============================================================

def calculate_fixed_forecast(
    df_fix,
    area_df,
    ghi_cols,
    efficiency_loss
):

    standard_eff = pd.to_numeric(
        area_df["Standard PV Efficiency (%)"],
        errors="coerce"
    ).fillna(0)

    total_area = pd.to_numeric(
        area_df["Total area(m2)"],
        errors="coerce"
    ).fillna(0)

    net_eff = (
        standard_eff
        - efficiency_loss
    )

    effective_area = (
        total_area
        * net_eff
        / 100
    )

    total_effective_area = (
        effective_area.sum()
    )

    df_fix = df_fix.copy()

    if len(ghi_cols) == 1:

        ghi = pd.to_numeric(
            df_fix[ghi_cols[0]],
            errors="coerce"
        ).fillna(0).to_numpy()

        df_fix["GHI*sin(a)"] = (
            ghi
            * df_fix["Sin(a)"].to_numpy()
        )

        df_fix["GHI*sin(a+b)"] = (
            ghi
            * df_fix["SIN(a+b)"].to_numpy()
        )

        sin_a = df_fix[
            "Sin(a)"
        ].to_numpy()

        sin_a = np.where(
            np.abs(sin_a) < 1e-8,
            1e-8,
            sin_a
        )

        poa = (
            df_fix[
                "GHI*sin(a+b)"
            ].to_numpy()
            / sin_a
        )

        forecast = (
            poa
            * total_effective_area
            / 1_000_000
        )

        df_fix[
            "Fixed Power=I*Ƞ*A"
        ] = forecast

    else:

        # Cluster weights
        weight_df = read_cluster_weights_from_area(
            area_df,
            file_bytes=None
        )

        # If no explicit cluster weights exist,
        # distribute effective area equally.
        cluster_weights = np.repeat(
            total_effective_area / len(ghi_cols),
            len(ghi_cols)
        )

        forecast = np.zeros(
            len(df_fix),
            dtype=float
        )

        for i, col in enumerate(ghi_cols):

            ghi = pd.to_numeric(
                df_fix[col],
                errors="coerce"
            ).fillna(0).to_numpy()

            sin_a = df_fix[
                "Sin(a)"
            ].to_numpy()

            sin_a = np.where(
                np.abs(sin_a) < 1e-8,
                1e-8,
                sin_a
            )

            poa = (
                ghi
                * df_fix[
                    "SIN(a+b)"
                ].to_numpy()
                / sin_a
            )

            forecast += (
                poa
                * cluster_weights[i]
                / 1_000_000
            )

        df_fix[
            "Total Power (CL1+CL2+…)"
        ] = forecast

    return df_fix


def read_cluster_weights_from_area(
    area_df,
    file_bytes=None
):

    # This helper is intentionally conservative.
    #
    # If the Area & Efficiency sheet does not expose
    # cluster weighting columns, the main calculation
    # uses equal distribution.

    possible = [
        "CL-1",
        "CL-2",
        "CL-3",
        "CL-4",
        "CL-5"
    ]

    existing = [
        col for col in possible
        if col in area_df.columns
    ]

    if len(existing) == 5:

        result = []

        for col in existing:

            values = pd.to_numeric(
                area_df[col],
                errors="coerce"
            ).fillna(0)

            result.append(
                values.sum()
            )

        return np.array(
            result,
            dtype=float
        )

    return np.array(
        [],
        dtype=float
    )


# ============================================================
# NON-CLUSTER FIXED BASE FACTOR
# ============================================================

def calculate_fixed_base_factor(
    df_fix,
    area_df
):

    total_area = pd.to_numeric(
        area_df["Total area(m2)"],
        errors="coerce"
    ).fillna(0).sum()

    sin_a = df_fix[
        "Sin(a)"
    ].to_numpy()

    sin_a = np.where(
        np.abs(sin_a) < 1e-8,
        1e-8,
        sin_a
    )

    poa = (
        df_fix[
            "GHI_Forecast"
        ].to_numpy()
        * df_fix[
            "SIN(a+b)"
        ].to_numpy()
        / sin_a
    )

    return (
        poa / 1_000_000
    )


# ============================================================
# TRACKING PREPARATION
# ============================================================

def prepare_tracking_backend(
    file_bytes,
    is_cluster
):

    if is_cluster:

        backend_list = []

        for i in range(1, 6):

            df = read_excel_cached(
                file_bytes,
                f"Backend Cal CL{i}"
            )

            backend_list.append(df)

        tracking = read_excel_cached(
            file_bytes,
            "Tracking",
            header=1
        )

        return backend_list, tracking

    else:

        backend = read_excel_cached(
            file_bytes,
            "Backend Cal"
        )

        tracking = read_excel_cached(
            file_bytes,
            "Tracking",
            header=1
        )

        return [backend], tracking


# ============================================================
# TRACKING FORECAST
# ============================================================

def tracking_forecast(
    params,
    backend_list,
    ghi_arrays,
    effective_weights
):

    DHI = int(round(params[0]))
    start = int(round(params[1]))
    end = int(round(params[2]))
    max_block = int(round(params[3]))
    east = int(round(params[4]))
    west = int(round(params[5]))

    if not (
        start < max_block < end
    ):
        return None

    blocks = backend_list[0][
        "Block No."
    ].to_numpy(dtype=float)

    m1 = 90 / (
        start - 1 - max_block
    )

    m2 = 90 / (
        end + 1 - max_block
    )

    zenith = np.where(
        blocks <= max_block,

        np.minimum(
            89,
            m1
            * (
                blocks
                - max_block
            )
        ),

        np.minimum(
            89,
            m2
            * (
                blocks
                - max_block
            )
        )
    )

    panel = np.where(
        blocks < max_block,

        np.minimum(
            zenith,
            abs(east)
        ),

        np.where(
            (
                (blocks > max_block)
                & (zenith > west)
            ),

            west,

            zenith
        )
    )

    cos_alpha = np.cos(
        np.radians(panel)
    )

    cos_alpha = np.clip(
        cos_alpha,
        1e-6,
        None
    )

    forecast = np.zeros(
        len(blocks),
        dtype=float
    )

    for i, ghi in enumerate(
        ghi_arrays
    ):

        dhi = (
            ghi
            * DHI
            / 100
        )

        dni = (
            ghi
            - dhi
        ) / cos_alpha

        forecast += (
            dni
            * effective_weights[i]
            / 1_000_000
        )

    return forecast


# ============================================================
# TRACKING OPTIMIZATION
# ============================================================

def optimize_tracking(
    backend_list,
    ghi_arrays,
    effective_weights,
    actual,
    progress_callback=None
):

    # Use only non-zero actual values
    mask = (
        np.isfinite(actual)
        & (actual != 0)
    )

    actual_daylight = actual[mask]

    if len(actual_daylight) == 0:
        return None

    def objective(x):

        try:

            prediction = tracking_forecast(
                x,
                backend_list,
                ghi_arrays,
                effective_weights
            )

            if prediction is None:
                return 1e9

            prediction = prediction[mask]

            if len(prediction) != len(
                actual_daylight
            ):
                return 1e9

            if (
                not np.all(
                    np.isfinite(prediction)
                )
            ):
                return 1e9

            actual_max = np.max(
                actual_daylight
            )

            actual_sum = np.sum(
                actual_daylight
            )

            if (
                actual_max == 0
                or actual_sum == 0
            ):
                return 1e9

            block_error = (
                np.mean(
                    np.abs(
                        actual_daylight
                        - prediction
                    )
                )
                / actual_max
            )

            peak_error = (
                abs(
                    actual_max
                    - np.max(prediction)
                )
                / actual_max
            )

            energy_error = (
                abs(
                    actual_sum
                    - np.sum(prediction)
                )
                / actual_sum
            )

            score = (
                0.80 * block_error
                + 0.10 * peak_error
                + 0.10 * energy_error
            )

            return score

        except Exception:
            return 1e9

    bounds = [
        (0, 10),       # DHI
        (0, 30),       # Start
        (65, 80),      # End
        (44, 60),      # Max
        (0, 70),       # East
        (0, 70)        # West
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=40,
        popsize=10,
        tol=0.002,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
        updating="immediate",
        callback=progress_callback
    )

    return result


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📁 Upload Solar Forecast Excel File",
    type=["xlsx"],
    key="excel_uploader"
)

if uploaded_file is None:

    st.info(
        "Upload your Excel workbook to start Loss Correction."
    )

    st.stop()


# ============================================================
# FILE CHANGE DETECTION
# ============================================================

file_bytes = uploaded_file.getvalue()

file_identifier = (
    uploaded_file.name,
    len(file_bytes)
)

if (
    st.session_state.current_file
    != file_identifier
):

    st.session_state.current_file = (
        file_identifier
    )

    st.session_state.calculated = False
    st.session_state.optimization_result = None
    st.session_state.best_loss = None
    st.session_state.optimized_params = None

    # Clear widget states
    for key in [
        "efficiency_loss",
        "plant_type"
    ]:
        if key in st.session_state:
            del st.session_state[key]


# ============================================================
# WORKBOOK TYPE
# ============================================================

sheet_names = get_sheet_names(
    file_bytes
)

is_cluster = (
    "Fixed-CL1"
    in sheet_names
)


# ============================================================
# INPUT SHEET
# ============================================================

if is_cluster:

    input_sheet = "Fixed-CL1"

    ghi_cols = [
        "CL1-GHI",
        "CL2-GHI",
        "CL3-GHI",
        "CL4-GHI",
        "CL5-GHI"
    ]

else:

    input_sheet = "Fixed"

    ghi_cols = [
        "GHI_Forecast"
    ]


df_input = read_excel_cached(
    file_bytes,
    input_sheet,
    header=1
)

df_input.columns = (
    df_input.columns
    .astype(str)
    .str.strip()
)

df_input = clean_until_empty(
    df_input,
    "Date"
)

df_input = df_input.iloc[:96].copy()


# ============================================================
# INPUT DATA EDITOR
# ============================================================

available_ghi = [
    col for col in ghi_cols
    if col in df_input.columns
]

if not available_ghi:

    st.error(
        "Required GHI column(s) were not found in the workbook."
    )

    st.stop()


input_columns = (
    available_ghi
    + ["Actual"]
)

input_df = df_input[
    input_columns
].copy()

input_df = input_df.fillna(0)


st.subheader("📊 Input Data")

edited_df = st.data_editor(
    input_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="loss_input_editor"
)


# ============================================================
# PLANT TYPE
# ============================================================

plant_type = st.radio(
    "Select Plant Type",
    [
        "🏗️ Fixed",
        "🔄 Tracking"
    ],
    horizontal=True,
    key="plant_type"
)


# ============================================================
# CALCULATE BUTTON
# ============================================================

st.divider()

calculate_clicked = st.button(
    "🚀 Calculate Loss Correction",
    use_container_width=True,
    type="primary"
)


# ============================================================
# CALCULATION
# ============================================================

if calculate_clicked:

    # --------------------------------------------------------
    # Convert edited input
    # --------------------------------------------------------

    edited_clean = edited_df.copy()

    for col in available_ghi:

        edited_clean[col] = pd.to_numeric(
            edited_clean[col],
            errors="coerce"
        ).fillna(0)

    edited_clean["Actual"] = pd.to_numeric(
        edited_clean["Actual"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # Read Area & Efficiency
    # --------------------------------------------------------

    area_df = prepare_area_efficiency(
        file_bytes,
        is_cluster
    )

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    if plant_type == "🏗️ Fixed":

        df_fix, ghi_columns = (
            prepare_fixed_data(
                file_bytes,
                edited_clean,
                is_cluster
            )
        )

        df_fix, latitude = (
            add_solar_geometry(
                df_fix,
                file_bytes,
                is_tracking=False
            )
        )

        actual = df_fix[
            "Actual"
        ].to_numpy(dtype=float)

        if is_cluster:

            # ------------------------------------------------
            # Cluster fixed forecast
            # ------------------------------------------------

            standard_eff = pd.to_numeric(
                area_df[
                    "Standard PV Efficiency (%)"
                ],
                errors="coerce"
            ).fillna(0).to_numpy()

            total_area = pd.to_numeric(
                area_df[
                    "Total area(m2)"
                ],
                errors="coerce"
            ).fillna(0).to_numpy()

            # Load cluster weights if available
            cluster_weight_cols = [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5"
            ]

            cluster_weights = []

            for col in cluster_weight_cols:

                if col in area_df.columns:

                    value = pd.to_numeric(
                        area_df[col],
                        errors="coerce"
                    ).fillna(0).sum()

                else:

                    value = 1.0

                cluster_weights.append(
                    value
                )

            cluster_weights = np.asarray(
                cluster_weights,
                dtype=float
            )

            if np.sum(
                cluster_weights
            ) == 0:

                cluster_weights = (
                    np.ones(5)
                )

            # ------------------------------------------------
            # Base forecast factor
            # ------------------------------------------------

            sin_a = df_fix[
                "Sin(a)"
            ].to_numpy()

            sin_a = np.where(
                np.abs(sin_a) < 1e-8,
                1e-8,
                sin_a
            )

            tilt_factor = (
                df_fix[
                    "SIN(a+b)"
                ].to_numpy()
                / sin_a
            )

            base_forecast_factor = (
                np.zeros(
                    len(df_fix)
                )
            )

            for i, col in enumerate(
                ghi_columns
            ):

                ghi = pd.to_numeric(
                    df_fix[col],
                    errors="coerce"
                ).fillna(0).to_numpy()

                poa = (
                    ghi
                    * tilt_factor
                )

                base_forecast_factor += (
                    poa
                    * cluster_weights[i]
                    / 1_000_000
                )

            # The factor must multiply effective area.
            # Normalize cluster weights to avoid
            # accidental scale changes.

            weight_sum = (
                np.sum(cluster_weights)
            )

            if weight_sum != 0:

                base_forecast_factor /= (
                    weight_sum
                )

                effective_area_scale = (
                    total_area
                )

            else:

                effective_area_scale = (
                    total_area
                )

            # Temporarily use total area inside
            # the optimization.
            #
            # base factor x effective area

            base_factor = (
                base_forecast_factor
                * 1
            )

            # Vectorized loss optimization
            temp_area_df = area_df.copy()

            standard_eff = pd.to_numeric(
                temp_area_df[
                    "Standard PV Efficiency (%)"
                ],
                errors="coerce"
            ).fillna(0).to_numpy()

            total_area_values = pd.to_numeric(
                temp_area_df[
                    "Total area(m2)"
                ],
                errors="coerce"
            ).fillna(0).to_numpy()

            max_loss = float(
                np.nanmin(
                    standard_eff
                )
            )

            losses = np.round(
                np.arange(
                    0,
                    max_loss + 0.0001,
                    0.1
                ),
                1
            )

            net_eff = (
                standard_eff[None, :]
                - losses[:, None]
            )

            eff_area = (
                total_area_values[None, :]
                * net_eff
                / 100
            )

            total_eff_area = (
                eff_area.sum(axis=1)
            )

            forecast_matrix = (
                base_factor[None, :]
                * total_eff_area[:, None]
            )

            actual_peak = (
                np.max(actual)
            )

            predicted_peaks = (
                np.max(
                    forecast_matrix,
                    axis=1
                )
            )

            peak_errors = np.abs(
                actual_peak
                - predicted_peaks
            )

            best_index = np.argmin(
                peak_errors
            )

            best_loss = float(
                losses[best_index]
            )

        else:

            # ------------------------------------------------
            # Non-cluster fixed
            # ------------------------------------------------

            base_factor = (
                calculate_fixed_base_factor(
                    df_fix,
                    area_df
                )
            )

            best_loss, loss_results = (
                optimize_efficiency_loss(
                    area_df,
                    base_factor,
                    actual
                )
            )

        # Store state
        st.session_state.best_loss = (
            best_loss
        )

        st.session_state.optimized_params = {
            "plant_type": plant_type
        }

        st.session_state.calculated = True

        # Important:
        # We deliberately DO NOT rerun here.
        # Streamlit will naturally render the
        # calculated section below.

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    else:

        df_fix, ghi_columns = (
            prepare_fixed_data(
                file_bytes,
                edited_clean,
                is_cluster
            )
        )

        df_fix, latitude = (
            add_solar_geometry(
                df_fix,
                file_bytes,
                is_tracking=True
            )
        )

        actual = df_fix[
            "Actual"
        ].to_numpy(dtype=float)

        # --------------------------------------------
        # Calculate maximum-loss correction first
        # --------------------------------------------

        base_factor = (
            calculate_fixed_base_factor(
                df_fix,
                area_df
            )
        )

        best_loss, _ = (
            optimize_efficiency_loss(
                area_df,
                base_factor,
                actual
            )
        )

        # --------------------------------------------
        # Effective area after optimized loss
        # --------------------------------------------

        standard_eff = pd.to_numeric(
            area_df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce"
        ).fillna(0).to_numpy()

        total_area = pd.to_numeric(
            area_df[
                "Total area(m2)"
            ],
            errors="coerce"
        ).fillna(0).to_numpy()

        net_eff = (
            standard_eff
            - best_loss
        )

        effective_area = (
            total_area
            * net_eff
            / 100
        )

        eff_area_sum = (
            np.sum(effective_area)
        )

        # --------------------------------------------
        # Backend calculation
        # --------------------------------------------

        backend_list, tracking_df = (
            prepare_tracking_backend(
                file_bytes,
                is_cluster
            )
        )

        ghi_arrays = [
            pd.to_numeric(
                df_fix[col],
                errors="coerce"
            ).fillna(0).to_numpy(
                dtype=float
            )
            for col in ghi_columns
        ]

        if is_cluster:

            weight_columns = [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5"
            ]

            weight_values = []

            for col in weight_columns:

                if col in area_df.columns:

                    value = pd.to_numeric(
                        area_df[col],
                        errors="coerce"
                    ).fillna(0).sum()

                else:

                    value = 1.0

                weight_values.append(
                    value
                )

            effective_weights = (
                effective_area.sum()
                * (
                    np.asarray(
                        weight_values,
                        dtype=float
                    )
                    / max(
                        np.sum(
                            weight_values
                        ),
                        1e-8
                    )
                )
            )

        else:

            effective_weights = np.array([
                eff_area_sum
            ])

        # --------------------------------------------
        # Tracking optimization
        # --------------------------------------------

        progress = st.progress(0)

        quotes = [
            "☕ Optimization chal raha hai...",
            "🌦 Solar parameters ko samajh rahe hain...",
            "😎 Forecast ko Actual ke paas laa rahe hain...",
            "⚡ Thoda patience, solar power ban rahi hai...",
            "🔬 Parameters ka best combination dhoond rahe hain...",
            "📊 Curve ko fit kar rahe hain...",
            "🚀 Almost there..."
        ]

        status = st.empty()

        generation = {
            "count": 0
        }

        MAX_ITER = 40

        def callback(xk, convergence):

            generation["count"] += 1

            progress.progress(
                min(
                    generation["count"]
                    / MAX_ITER,
                    1.0
                )
            )

            if (
                generation["count"]
                % 5 == 0
            ):

                quote = random.choice(
                    quotes
                )

                status.info(
                    f"{quote}\n\n"
                    f"Generation "
                    f"{generation['count']} / "
                    f"{MAX_ITER}"
                )

            return False

        result = optimize_tracking(
            backend_list,
            ghi_arrays,
            effective_weights,
            actual,
            progress_callback=callback
        )

        progress.empty()
        status.empty()

        if result is None:

            st.error(
                "Tracking optimization could not be completed."
            )

            st.stop()

        best_params = (
            np.round(
                result.x
            ).astype(int)
        )

        st.session_state.best_loss = (
            best_loss
        )

        st.session_state.optimized_params = {
            "plant_type": plant_type,
            "DHI": int(best_params[0]),
            "start": int(best_params[1]),
            "end": int(best_params[2]),
            "max": int(best_params[3]),
            "east": int(best_params[4]),
            "west": int(best_params[5])
        }

        st.session_state.calculated = True


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.calculated:

    st.divider()

    st.subheader(
        "⚙️ Loss Correction Parameters"
    )

    # --------------------------------------------------------
    # Efficiency loss
    # --------------------------------------------------------

    best_loss = st.session_state.best_loss

    efficiency_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(best_loss),
        step=0.1,
        format="%.1f",
        key="efficiency_loss"
    )

    st.caption(
        "You can change the efficiency loss manually. "
        "The forecast updates automatically without running "
        "the optimization again."
    )


    # ========================================================
    # REBUILD AREA EFFICIENCY
    # ========================================================

    area_df = prepare_area_efficiency(
        file_bytes,
        is_cluster
    )

    area_display = area_df.copy()

    if (
        "Standard PV Efficiency (%)"
        in area_display.columns
    ):

        area_display[
            "Efficiency Losses(%)"
        ] = efficiency_loss

        area_display[
            "Net Efficiency (%)"
        ] = (
            pd.to_numeric(
                area_display[
                    "Standard PV Efficiency (%)"
                ],
                errors="coerce"
            ).fillna(0)
            - efficiency_loss
        )

        area_display[
            "Total area(m2)"
        ] = pd.to_numeric(
            area_display[
                "Total area(m2)"
            ],
            errors="coerce"
        ).fillna(0)

        area_display[
            "Eff Area"
        ] = (
            area_display[
                "Total area(m2)"
            ]
            * area_display[
                "Net Efficiency (%)"
            ]
            / 100
        )


    # ========================================================
    # EFFICIENCY TABLE
    # ========================================================

    display_columns = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)"
    ]

    display_columns = [
        col for col in display_columns
        if col in area_display.columns
    ]

    display_df = area_display[
        display_columns
    ].copy()

    numeric_cols = (
        display_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    display_df[numeric_cols] = (
        display_df[numeric_cols]
        .round(2)
    )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # FINAL FORECAST
    # ========================================================

    edited_clean = edited_df.copy()

    for col in available_ghi:

        edited_clean[col] = pd.to_numeric(
            edited_clean[col],
            errors="coerce"
        ).fillna(0)

    edited_clean["Actual"] = pd.to_numeric(
        edited_clean["Actual"],
        errors="coerce"
    ).fillna(0)


    # ========================================================
    # FIXED
    # ========================================================

    if plant_type == "🏗️ Fixed":

        df_fix, ghi_columns = (
            prepare_fixed_data(
                file_bytes,
                edited_clean,
                is_cluster
            )
        )

        df_fix, latitude = (
            add_solar_geometry(
                df_fix,
                file_bytes,
                is_tracking=False
            )
        )

        standard_eff = pd.to_numeric(
            area_df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce"
        ).fillna(0).to_numpy()

        total_area = pd.to_numeric(
            area_df[
                "Total area(m2)"
            ],
            errors="coerce"
        ).fillna(0).to_numpy()

        effective_area = (
            total_area
            * (
                standard_eff
                - efficiency_loss
            )
            / 100
        )

        total_effective_area = (
            np.sum(effective_area)
        )

        actual = df_fix[
            "Actual"
        ].to_numpy(dtype=float)

        sin_a = df_fix[
            "Sin(a)"
        ].to_numpy()

        sin_a = np.where(
            np.abs(sin_a) < 1e-8,
            1e-8,
            sin_a
        )

        tilt_factor = (
            df_fix[
                "SIN(a+b)"
            ].to_numpy()
            / sin_a
        )

        forecast = np.zeros(
            len(df_fix),
            dtype=float
        )

        if is_cluster:

            # --------------------------------------------
            # Cluster fixed
            # --------------------------------------------

            cluster_weight_cols = [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5"
            ]

            cluster_weights = []

            for col in cluster_weight_cols:

                if col in area_df.columns:

                    value = pd.to_numeric(
                        area_df[col],
                        errors="coerce"
                    ).fillna(0).sum()

                else:

                    value = 1.0

                cluster_weights.append(
                    value
                )

            cluster_weights = np.asarray(
                cluster_weights,
                dtype=float
            )

            if np.sum(
                cluster_weights
            ) == 0:

                cluster_weights = np.ones(
                    len(ghi_columns)
                )

            # Normalize so total effective
            # area remains physically consistent.

            normalized_weights = (
                cluster_weights
                / np.sum(
                    cluster_weights
                )
            )

            for i, col in enumerate(
                ghi_columns
            ):

                ghi = pd.to_numeric(
                    df_fix[col],
                    errors="coerce"
                ).fillna(0).to_numpy()

                poa = (
                    ghi
                    * tilt_factor
                )

                forecast += (
                    poa
                    * total_effective_area
                    * normalized_weights[i]
                    / 1_000_000
                )

            df_fix[
                "Total Power (CL1+CL2+…)"
            ] = forecast

        else:

            # --------------------------------------------
            # Non-cluster fixed
            # --------------------------------------------

            ghi = pd.to_numeric(
                df_fix[
                    "GHI_Forecast"
                ],
                errors="coerce"
            ).fillna(0).to_numpy()

            poa = (
                ghi
                * tilt_factor
            )

            forecast = (
                poa
                * total_effective_area
                / 1_000_000
            )

            df_fix[
                "Fixed Power=I*Ƞ*A"
            ] = forecast


    # ========================================================
    # TRACKING
    # ========================================================

    else:

        df_fix, ghi_columns = (
            prepare_fixed_data(
                file_bytes,
                edited_clean,
                is_cluster
            )
        )

        backend_list, tracking_df = (
            prepare_tracking_backend(
                file_bytes,
                is_cluster
            )
        )

        standard_eff = pd.to_numeric(
            area_df[
                "Standard PV Efficiency (%)"
            ],
            errors="coerce"
        ).fillna(0).to_numpy()

        total_area = pd.to_numeric(
            area_df[
                "Total area(m2)"
            ],
            errors="coerce"
        ).fillna(0).to_numpy()

        effective_area = (
            total_area
            * (
                standard_eff
                - efficiency_loss
            )
            / 100
        )

        total_effective_area = (
            effective_area.sum()
        )

        if is_cluster:

            weight_cols = [
                "CL-1",
                "CL-2",
                "CL-3",
                "CL-4",
                "CL-5"
            ]

            cluster_weights = []

            for col in weight_cols:

                if col in area_df.columns:

                    value = pd.to_numeric(
                        area_df[col],
                        errors="coerce"
                    ).fillna(0).sum()

                else:

                    value = 1.0

                cluster_weights.append(
                    value
                )

            cluster_weights = np.asarray(
                cluster_weights,
                dtype=float
            )

            if np.sum(
                cluster_weights
            ) == 0:

                cluster_weights = np.ones(
                    5
                )

            normalized = (
                cluster_weights
                / np.sum(
                    cluster_weights
                )
            )

            effective_weights = (
                total_effective_area
                * normalized
            )

        else:

            effective_weights = np.array([
                total_effective_area
            ])

        ghi_arrays = [
            pd.to_numeric(
                df_fix[col],
                errors="coerce"
            ).fillna(0).to_numpy(
                dtype=float
            )
            for col in ghi_columns
        ]

        params = st.session_state.optimized_params

        tracking_parameters = [
            params["DHI"],
            params["start"],
            params["end"],
            params["max"],
            params["east"],
            params["west"]
        ]

        forecast = tracking_forecast(
            tracking_parameters,
            backend_list,
            ghi_arrays,
            effective_weights
        )

        if forecast is None:

            st.error(
                "Invalid tracking parameters."
            )

            st.stop()

        df_fix[
            "Tracking Forecast"
        ] = forecast


    # ========================================================
    # METRICS
    # ========================================================

    actual = df_fix[
        "Actual"
    ].to_numpy(dtype=float)

    if plant_type == "🏗️ Fixed":

        if is_cluster:

            forecast = df_fix[
                "Total Power (CL1+CL2+…)"
            ].to_numpy(dtype=float)

        else:

            forecast = df_fix[
                "Fixed Power=I*Ƞ*A"
            ].to_numpy(dtype=float)

    else:

        forecast = df_fix[
            "Tracking Forecast"
        ].to_numpy(dtype=float)


    metrics = calculate_metrics(
        actual,
        forecast
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Efficiency Loss",
        f"{efficiency_loss:.1f}%"
    )

    c2.metric(
        "MAE",
        f"{metrics['MAE']:.3f} MW"
        if np.isfinite(metrics["MAE"])
        else "N/A"
    )

    c3.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%"
        if np.isfinite(metrics["MAPE"])
        else "N/A"
    )

    c4.metric(
        "Peak Error",
        f"{metrics['Peak Error']:.2f}%"
        if np.isfinite(metrics["Peak Error"])
        else "N/A"
    )

    c5.metric(
        "R²",
        f"{metrics['R2']:.4f}"
        if np.isfinite(metrics["R2"])
        else "N/A"
    )


    # ========================================================
    # TRACKING PARAMETERS
    # ========================================================

    if plant_type == "🔄 Tracking":

        st.subheader(
            "🔄 Optimized Tracking Parameters"
        )

        params = (
            st.session_state
            .optimized_params
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "DHI",
            params["DHI"]
        )

        p2.metric(
            "Starting Block",
            params["start"]
        )

        p3.metric(
            "Ending Block",
            params["end"]
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Max Block",
            params["max"]
        )

        p2.metric(
            "East Limit",
            params["east"]
        )

        p3.metric(
            "West Limit",
            params["west"]
        )


    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "📈 Forecast vs Actual Power"
    )

    fig = create_chart(
        actual,
        forecast
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader(
        "📥 Download Corrected Forecast"
    )

    download_df = df_fix.copy()

    csv_data = download_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Forecast CSV",
        data=csv_data,
        file_name=(
            "loss_corrected_forecast.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )
