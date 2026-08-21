# ============================================================
# LOSS CORRECTION MODEL
# FIXED PLANT ONLY
# Optimized Streamlit Execution
#
# Calculation logic preserved
# No "Update Input Data" button
# Calculations run only on RUN LOSS CORRECTION
# ============================================================

import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loss Correction Model",
    page_icon="☀️",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

FIXED_PLANT = "🏗️ Fixed"

CLUSTER_GHI_COLUMNS = [
    "CL1-GHI",
    "CL2-GHI",
    "CL3-GHI",
    "CL4-GHI",
    "CL5-GHI",
]

CLUSTER_WEIGHT_COLUMNS = [
    "CL-1",
    "CL-2",
    "CL-3",
    "CL-4",
    "CL-5",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #8b949e;
        margin-bottom: 22px;
    }

    .section {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 18px 0 8px;
    }

    div.stButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "uploaded_signature": None,
    "input_df": None,
    "editor_version": 0,
    "model_result": None,
    "fixed_loss": None,
    "cluster_fixed_loss": None,
    "run_requested": False,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GENERIC HELPERS
# ============================================================

def numeric(values):
    """
    Safely convert values to numeric.
    """

    if isinstance(values, pd.Series):

        return pd.to_numeric(
            values,
            errors="coerce",
        ).fillna(0)

    return pd.Series(
        pd.to_numeric(
            np.asarray(values),
            errors="coerce",
        )
    ).fillna(0)


def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


def safe_array(values):
    """
    Convert values to a clean float numpy array.
    """

    if isinstance(values, pd.Series):

        values = pd.to_numeric(
            values,
            errors="coerce",
        ).fillna(0)

    else:

        values = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        ).fillna(0)

    return np.asarray(
        values,
        dtype=float,
    )


def validate_columns(
    df,
    required,
    name="Data",
):
    """
    Validate required columns.
    """

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def clean_data_rows(
    df,
    date_column="Date",
):
    """
    Remove rows after the first blank Date.
    """

    result = df.copy()

    if date_column not in result.columns:
        return result.reset_index(drop=True)

    null_rows = result[
        result[date_column].isna()
    ].index

    if len(null_rows):

        first_null = null_rows[0]

        position = result.index.get_loc(
            first_null
        )

        result = result.iloc[:position]

    return result.reset_index(drop=True)


def workbook_signature(file_bytes):
    """
    Stable signature for uploaded workbook.
    """

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# ============================================================
# CACHED EXCEL FUNCTIONS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def workbook_sheets(file_bytes):
    """
    Return workbook sheet names.
    """

    excel = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    return tuple(
        excel.sheet_names
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_area_efficiency_cached(
    file_bytes,
    cluster,
):
    """
    Read Area & Efficiency sheet.
    """

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=1,
        usecols=range(8) if cluster else None,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        [
            "Module Type",
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        "Area & Efficiency",
    )

    null_rows = df[
        df["Module Type"].isna()
    ].index

    if len(null_rows):

        first_null = null_rows[0]

        position = df.index.get_loc(
            first_null
        )

        df = df.iloc[:position]

    df = df.dropna(
        subset=[
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        how="all",
    )

    df[
        "Standard PV Efficiency (%)"
    ] = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).fillna(0)

    df["Total area(m2)"] = pd.to_numeric(
        df["Total area(m2)"],
        errors="coerce",
    ).fillna(0)

    return df.reset_index(drop=True)


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_cluster_weights_cached(
    file_bytes,
):
    """
    Read cluster area weights.
    """

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Area & Efficiency",
        header=2,
        usecols=[12, 13, 14, 15, 16],
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        CLUSTER_WEIGHT_COLUMNS,
        "Cluster Weights",
    )

    weights = {}

    for col in CLUSTER_WEIGHT_COLUMNS:

        series = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna()

        weights[col] = (
            float(series.iloc[0])
            if len(series)
            else 0.0
        )

    return weights


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_latitude_cached(
    file_bytes,
):
    """
    Read latitude from Forecast Config.
    """

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Forecast Config",
        header=8,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    validate_columns(
        df,
        ["Lat"],
        "Forecast Config",
    )

    values = pd.to_numeric(
        df["Lat"],
        errors="coerce",
    ).dropna()

    if values.empty:

        raise ValueError(
            "No valid latitude found in Forecast Config."
        )

    return float(
        values.iloc[0]
    )


@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def read_tilt_lookup_cached(
    file_bytes,
):
    """
    Read monthly fixed tilt values.
    """

    try:

        df = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name="Config Tilt Angle",
            header=7,
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        if "Fixed" not in df.columns:
            return {}

        null_rows = df[
            df["Fixed"].isna()
        ].index

        if len(null_rows):

            first_null = null_rows[0]

            position = df.index.get_loc(
                first_null
            )

            df = df.iloc[:position]

        df = df.dropna(
            axis=1,
            how="all",
        )

        df = df.rename(
            columns={
                "Unnamed: 2": "Month_Num",
                "Unnamed: 3": "Month",
            }
        )

        if "Month" not in df.columns:
            return {}

        df["Fixed"] = pd.to_numeric(
            df["Fixed"],
            errors="coerce",
        )

        df["Month"] = (
            df["Month"]
            .astype(str)
            .str.strip()
        )

        result = (
            df.dropna(
                subset=["Month"]
            )
            .set_index("Month")["Fixed"]
            .dropna()
            .to_dict()
        )

        return result

    except Exception:

        return {}


# ============================================================
# INPUT DATA
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=5,
)
def load_input_data_cached(
    file_bytes,
    cluster,
):
    """
    Load Fixed forecast sheet.

    Standard workbook:
        Fixed

    Cluster workbook:
        Fixed-CL1

    Cluster GHI values are recovered
    from Result when required.
    """

    sheet = (
        "Fixed-CL1"
        if cluster
        else "Fixed"
    )

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet,
        header=1,
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = clean_data_rows(df)

    validate_columns(
        df,
        ["Actual"],
        "Forecast Sheet",
    )

    # --------------------------------------------------------
    # CLUSTER
    # --------------------------------------------------------

    if cluster:

        try:

            result = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name="Result",
                usecols=range(6),
            )

            result = result.fillna(0)

            for i, col in enumerate(
                CLUSTER_GHI_COLUMNS
            ):

                if col in df.columns:
                    continue

                if i >= len(
                    result.columns
                ):
                    continue

                values = pd.to_numeric(
                    result.iloc[
                        :len(df),
                        i,
                    ],
                    errors="coerce",
                ).fillna(0).to_numpy()

                if len(values) < len(df):

                    values = np.pad(
                        values,
                        (
                            0,
                            len(df) - len(values),
                        ),
                    )

                df[col] = values

        except Exception:
            pass

        validate_columns(
            df,
            CLUSTER_GHI_COLUMNS,
            "Cluster Forecast",
        )

    # --------------------------------------------------------
    # NORMAL FIXED
    # --------------------------------------------------------

    else:

        validate_columns(
            df,
            ["GHI_Forecast"],
            "Fixed Forecast",
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    df["Actual"] = numeric(
        df["Actual"]
    ).to_numpy()

    if cluster:

        for col in CLUSTER_GHI_COLUMNS:

            df[col] = numeric(
                df[col]
            ).to_numpy()

    else:

        df["GHI_Forecast"] = numeric(
            df["GHI_Forecast"]
        ).to_numpy()

    return df


# ============================================================
# SOLAR ANGLES
# ============================================================

def prepare_solar_angles(
    df,
    lat,
    tilt_lookup=None,
):
    """
    Calculate solar-angle components.

    Existing calculation retained.
    """

    result = df.copy()

    if "Date" in result.columns:

        dates = pd.to_datetime(
            result["Date"],
            errors="coerce",
        )

        if dates.notna().any():

            valid_dates = dates.dropna()

            fallback_date = (
                valid_dates.iloc[0]
                if len(valid_dates)
                else pd.Timestamp.today()
            )

            dates = dates.fillna(
                fallback_date
            )

        else:

            dates = pd.Series(
                pd.Timestamp.today(),
                index=result.index,
            )

    else:

        dates = pd.Series(
            pd.Timestamp.today(),
            index=result.index,
        )

    dates = pd.to_datetime(
        dates
    ).dt.normalize()

    result["Date"] = dates

    first_date = pd.to_datetime(
        dates.dt.year.astype(str)
        + "-01-01"
    )

    day_number = (
        dates - first_date
    ).dt.days + 1

    result[
        "Declination Angle ∆"
    ] = (
        23.45
        * np.sin(
            np.radians(
                360
                * (284 + day_number)
                / 365
            )
        )
    )

    result[
        "Elevation angle a"
    ] = (
        90
        - float(lat)
        + result[
            "Declination Angle ∆"
        ]
    )

    if tilt_lookup:

        result["Tilt Angle b"] = (
            result["Date"]
            .dt.strftime("%B")
            .map(tilt_lookup)
            .fillna(0)
        )

    else:

        result["Tilt Angle b"] = 0.0

    result["a+b"] = (
        result["Elevation angle a"]
        + result["Tilt Angle b"]
    )

    result["SIN(a+b)"] = np.sin(
        np.radians(
            result["a+b"]
        )
    )

    result["Sin(a)"] = (
        np.sin(
            np.radians(
                result["Elevation angle a"]
            )
        )
        .clip(lower=1e-6)
    )

    return result


# ============================================================
# EFFICIENCY LOSS
# ============================================================

def calculate_efficiency_loss(
    df,
    poa,
    actual,
):
    """
    Automatically calculate efficiency
    loss from peak power.

    Calculation preserved.
    """

    validate_columns(
        df,
        [
            "Standard PV Efficiency (%)",
            "Total area(m2)",
        ],
        "Efficiency Data",
    )

    standard = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    area = pd.to_numeric(
        df["Total area(m2)"],
        errors="coerce",
    ).fillna(0).to_numpy(
        dtype=float
    )

    actual = safe_array(
        actual
    )

    poa = safe_array(
        poa
    )

    n = min(
        len(actual),
        len(poa),
    )

    if n == 0:
        return 0.0

    actual = actual[:n]
    poa = poa[:n]

    valid = (
        np.isfinite(actual)
        & np.isfinite(poa)
    )

    actual = actual[valid]
    poa = poa[valid]

    if len(actual) == 0:
        return 0.0

    actual_peak = np.nanmax(
        actual
    )

    poa_peak = np.nanmax(
        poa
    )

    if (
        not np.isfinite(actual_peak)
        or not np.isfinite(poa_peak)
        or actual_peak <= 0
        or poa_peak <= 0
    ):
        return 0.0

    base_area = np.sum(
        area
        * standard
        / 100
    )

    loss_coeff = np.sum(
        area / 100
    )

    if (
        not np.isfinite(loss_coeff)
        or loss_coeff <= 0
    ):
        return 0.0

    target_area = (
        actual_peak
        * 1_000_000
        / poa_peak
    )

    loss = (
        base_area
        - target_area
    ) / loss_coeff

    max_loss = (
        np.nanmin(standard)
        if len(standard)
        else 0.0
    )

    if not np.isfinite(max_loss):
        max_loss = 0.0

    return float(
        np.clip(
            loss,
            0,
            max_loss,
        )
    )


def apply_efficiency_loss(
    df,
    loss,
):
    """
    Apply one scalar efficiency loss
    to all module rows.

    Calculation preserved.
    """

    result = df.copy()

    loss = safe_float(
        loss,
        default=0.0,
    )

    standard = pd.to_numeric(
        result[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).fillna(0)

    area = pd.to_numeric(
        result["Total area(m2)"],
        errors="coerce",
    ).fillna(0)

    result[
        "Efficiency Losses(%)"
    ] = loss

    result[
        "Net Efficiency (%)"
    ] = (
        standard - loss
    ).clip(
        lower=0
    )

    result["Eff Area"] = (
        area
        * result[
            "Net Efficiency (%)"
        ]
        / 100
    )

    return result


# ============================================================
# EFFICIENCY CONTROL
# ============================================================

def efficiency_control(
    df,
    auto_loss,
):
    """
    Efficiency loss control.

    The loss is calculated automatically.
    User can edit it before running again.
    """

    efficiency = pd.to_numeric(
        df[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).dropna()

    if efficiency.empty:

        max_loss = 0.0

    else:

        max_loss = float(
            efficiency.min()
        )

        if not np.isfinite(
            max_loss
        ):
            max_loss = 0.0

    default_loss = safe_float(
        auto_loss,
        default=0.0,
    )

    default_loss = float(
        np.clip(
            default_loss,
            0,
            max_loss,
        )
    )

    # --------------------------------------------------------
    # IMPORTANT
    # Do not overwrite the widget value after it has
    # already been created by the user.
    # --------------------------------------------------------

    if (
        st.session_state.get(
            "fixed_loss_input"
        )
        is None
    ):

        st.session_state[
            "fixed_loss_input"
        ] = default_loss

    loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        step=0.1,
        format="%.2f",
        key="fixed_loss_input",
    )

    return (
        apply_efficiency_loss(
            df,
            loss,
        ),
        float(loss),
    )


# ============================================================
# INPUT EDITOR
# ============================================================

def input_data_editor(
    original_df,
    cluster,
):
    """
    Editable GHI + Actual table.

    There is intentionally NO separate
    Update/Recalculate button.

    Editing this table only updates the editor state.
    Heavy calculations happen when RUN LOSS CORRECTION
    is pressed.
    """

    st.markdown(
        '<div class="section">'
        "📊 Input GHI & Actual Power"
        "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Edit the values if required. "
        "Changes are used when you click "
        "**RUN LOSS CORRECTION**."
    )

    if cluster:

        columns = [
            "Actual",
            "CL1-GHI",
            "CL2-GHI",
            "CL3-GHI",
            "CL4-GHI",
            "CL5-GHI",
        ]

    else:

        columns = [
            "GHI_Forecast",
            "Actual",
        ]

    columns = [
        col
        for col in columns
        if col in original_df.columns
    ]

    display = original_df[
        columns
    ].copy()

    for col in columns:

        display[col] = numeric(
            display[col]
        ).to_numpy()

    edited = st.data_editor(
        display,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key=(
            "loss_input_editor_"
            + str(
                st.session_state.editor_version
            )
        ),
        column_config={
            col: st.column_config.NumberColumn(
                col,
                step=0.01,
                format="%.2f",
            )
            for col in columns
        },
    )

    return edited


# ============================================================
# BUILD CURRENT INPUT DATA
# ============================================================

def build_current_input_df(
    original_df,
    edited_df,
):
    """
    Merge edited input values back into the
    original dataframe.

    No calculations are performed here.
    """

    result = original_df.copy()

    for col in edited_df.columns:

        values = numeric(
            edited_df[col]
        ).to_numpy()

        if len(values) != len(result):

            raise ValueError(
                f"Length mismatch in column '{col}'."
            )

        result[col] = values

    return result


# ============================================================
# EFFICIENCY TABLE
# ============================================================

def show_efficiency_table(
    df,
):
    """
    Display efficiency calculations.
    """

    cols = [
        "Module Type",
        "Standard PV Efficiency (%)",
        "Efficiency Losses(%)",
        "Net Efficiency (%)",
        "Total area(m2)",
        "Eff Area",
    ]

    cols = [
        col
        for col in cols
        if col in df.columns
    ]

    display = df[
        cols
    ].copy()

    numeric_cols = (
        display.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_cols):

        display[numeric_cols] = (
            display[numeric_cols]
            .round(2)
        )

    with st.expander(
        "🔍 View Efficiency Calculations"
    ):

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    forecast,
    actual,
):
    """
    Calculate forecast metrics.
    """

    forecast = safe_array(
        forecast
    )

    actual = safe_array(
        actual
    )

    n = min(
        len(forecast),
        len(actual),
    )

    if n == 0:

        return {
            "MAPE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Peak Error": np.nan,
            "Energy Error": np.nan,
        }

    forecast = forecast[:n]
    actual = actual[:n]

    valid = (
        np.isfinite(forecast)
        & np.isfinite(actual)
    )

    forecast = forecast[valid]
    actual = actual[valid]

    if len(actual) == 0:

        return {
            "MAPE": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Peak Error": np.nan,
            "Energy Error": np.nan,
        }

    error = (
        forecast
        - actual
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    nonzero = (
        np.abs(actual) > 1e-9
    )

    if np.any(nonzero):

        mape = (
            np.mean(
                np.abs(
                    error[nonzero]
                    / actual[nonzero]
                )
            )
            * 100
        )

    else:

        mape = np.nan

    actual_peak = np.max(
        np.abs(actual)
    )

    if actual_peak > 0:

        peak_error = (
            abs(
                np.max(forecast)
                - np.max(actual)
            )
            / actual_peak
            * 100
        )

        energy_error = (
            abs(
                np.sum(forecast)
                - np.sum(actual)
            )
            / np.sum(
                np.abs(actual)
            )
            * 100
        )

    else:

        peak_error = np.nan
        energy_error = np.nan

    return {
        "MAPE": float(mape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Peak Error": float(peak_error),
        "Energy Error": float(energy_error),
    }


def show_metrics(
    forecast,
    actual,
):
    """
    Display metrics.
    """

    metrics = calculate_metrics(
        forecast,
        actual,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    values = [
        (
            c1,
            "MAPE",
            metrics["MAPE"],
            "%",
        ),
        (
            c2,
            "MAE",
            metrics["MAE"],
            " MW",
        ),
        (
            c3,
            "RMSE",
            metrics["RMSE"],
            " MW",
        ),
        (
            c4,
            "Peak Error",
            metrics["Peak Error"],
            "%",
        ),
        (
            c5,
            "Energy Error",
            metrics["Energy Error"],
            "%",
        ),
    ]

    for (
        column,
        label,
        value,
        suffix,
    ) in values:

        if np.isfinite(value):

            column.metric(
                label,
                f"{value:.2f}{suffix}",
            )

        else:

            column.metric(
                label,
                "N/A",
            )


# ============================================================
# FORECAST CHART
# ============================================================

def show_forecast_chart(
    forecast,
    actual,
    title,
):
    """
    Forecast vs Actual chart.
    """

    forecast = safe_array(
        forecast
    )

    actual = safe_array(
        actual
    )

    n = min(
        len(forecast),
        len(actual),
    )

    if n == 0:

        st.warning(
            "No forecast data available."
        )

        return

    forecast = forecast[:n]
    actual = actual[:n]

    x = np.arange(
        1,
        n + 1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast,
            mode="lines",
            name="Forecast",
            line=dict(
                color="#2563EB",
                width=2.5,
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
                color="#EF4444",
                width=2.5,
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=430,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        xaxis_title="15 Minute Block",
        yaxis_title="Power (MW)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# ============================================================
# FIXED FORECAST CALCULATION
# ============================================================

def calculate_fixed_forecast(
    efficiency_df,
    input_df,
    lat,
    tilt_lookup,
    cluster,
    weights,
):
    """
    Core Fixed calculation.

    IMPORTANT:
    The mathematical calculation is preserved
    from the original page.

    No Streamlit UI operations are performed here.
    """

    solar = prepare_solar_angles(
        input_df,
        lat,
        tilt_lookup,
    )

    actual = safe_array(
        input_df["Actual"]
    )

    # ========================================================
    # CLUSTER FIXED
    # ========================================================

    if cluster:

        poa_for_loss = (
            solar["CL1-GHI"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = (
            calculate_efficiency_loss(
                efficiency_df,
                poa_for_loss,
                actual,
            )
        )

        loss = st.session_state.get(
            "cluster_fixed_loss"
        )

        if loss is None:

            loss = auto_loss

        efficiency_df = (
            apply_efficiency_loss(
                efficiency_df,
                loss,
            )
        )

        forecast = np.zeros(
            len(input_df),
            dtype=float,
        )

        for (
            ghi_col,
            weight_col,
        ) in zip(
            CLUSTER_GHI_COLUMNS,
            CLUSTER_WEIGHT_COLUMNS,
        ):

            cluster_weight = safe_float(
                weights.get(
                    weight_col,
                    0.0,
                )
            )

            eff_area = (
                efficiency_df[
                    "Total area(m2)"
                ]
                * efficiency_df[
                    "Net Efficiency (%)"
                ]
                / 100
                * cluster_weight
            ).sum()

            poa = (
                solar[ghi_col]
                * solar["SIN(a+b)"]
                / solar["Sin(a)"]
            )

            forecast += (
                poa.to_numpy(
                    dtype=float
                )
                * float(eff_area)
                / 1_000_000
            )

        title = (
            "🏗️ Fixed Cluster Forecast vs Actual"
        )

    # ========================================================
    # NORMAL FIXED
    # ========================================================

    else:

        poa = (
            solar["GHI_Forecast"]
            * solar["SIN(a+b)"]
            / solar["Sin(a)"]
        )

        auto_loss = (
            calculate_efficiency_loss(
                efficiency_df,
                poa,
                actual,
            )
        )

        loss = st.session_state.get(
            "fixed_loss"
        )

        if loss is None:

            loss = auto_loss

        efficiency_df = (
            apply_efficiency_loss(
                efficiency_df,
                loss,
            )
        )

        eff_area_total = (
            efficiency_df[
                "Eff Area"
            ].sum()
        )

        forecast = (
            poa.to_numpy(
                dtype=float
            )
            * float(eff_area_total)
            / 1_000_000
        )

        title = (
            "🏗️ Fixed Forecast vs Actual"
        )

    return {
        "forecast": forecast,
        "actual": actual,
        "efficiency": efficiency_df,
        "auto_loss": float(loss),
        "title": title,
    }


# ============================================================
# RESET
# ============================================================

def reset_model_state():
    """
    Reset calculation state.
    """

    st.session_state.model_result = None

    st.session_state.fixed_loss = None

    st.session_state.cluster_fixed_loss = None

    st.session_state.run_requested = False


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="title">'
        "☀️ Loss Correction Model"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Fixed plant loss correction. "
        "Edit GHI/Actual data if required and run the model."
        "</div>",
        unsafe_allow_html=True,
    )

    # ========================================================
    # FILE
    # ========================================================

    st.markdown(
        '<div class="section">'
        "📁 Input Excel"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Excel Workbook",
        type=[
            "xlsx",
            "xls",
        ],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload the Excel workbook to begin."
        )

        return

    file_bytes = (
        uploaded_file.getvalue()
    )

    if not file_bytes:

        st.error(
            "The uploaded workbook is empty."
        )

        return

    # ========================================================
    # WORKBOOK SIGNATURE
    # ========================================================

    current_signature = (
        workbook_signature(
            file_bytes
        )
    )

    if (
        st.session_state.uploaded_signature
        != current_signature
    ):

        st.session_state.uploaded_signature = (
            current_signature
        )

        st.session_state.input_df = None

        st.session_state.editor_version += 1

        reset_model_state()

        # Remove old editor state if possible.
        for key in list(
            st.session_state.keys()
        ):

            if key.startswith(
                "loss_input_editor_"
            ):

                del st.session_state[key]

    # ========================================================
    # WORKBOOK SHEETS
    # ========================================================

    try:

        sheets = workbook_sheets(
            file_bytes
        )

    except Exception as exc:

        st.error(
            "Unable to read the uploaded workbook."
        )

        st.exception(exc)

        return

    # ========================================================
    # WORKBOOK TYPE
    # ========================================================

    cluster = (
        "Fixed" not in sheets
    )

    # ========================================================
    # LOAD PARAMETERS
    # ========================================================

    try:

        base_df = (
            read_area_efficiency_cached(
                file_bytes,
                cluster,
            )
        )

        lat = (
            read_latitude_cached(
                file_bytes
            )
        )

        tilt_lookup = (
            read_tilt_lookup_cached(
                file_bytes
            )
        )

        weights = (
            read_cluster_weights_cached(
                file_bytes
            )
            if cluster
            else None
        )

        original_input = (
            load_input_data_cached(
                file_bytes,
                cluster,
            )
        )

    except Exception as exc:

        st.error(
            "Unable to load workbook parameters."
        )

        st.exception(exc)

        return

    # ========================================================
    # INPUT DATA EDITOR
    # ========================================================

    edited_input = input_data_editor(
        original_input,
        cluster,
    )

    # ========================================================
    # FIXED PLANT
    # ========================================================

    st.markdown(
        '<div class="section">'
        "🏭 Plant Type"
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "🏗️ Fixed Plant"
    )

    # ========================================================
    # EFFICIENCY LOSS CONTROL
    #
    # Only display after first calculation.
    # This prevents unnecessary widget creation
    # before the model is run.
    # ========================================================

    # ========================================================
    # RUN BUTTON
    # ========================================================

    st.markdown("")

    run_clicked = st.button(
        "🚀 RUN LOSS CORRECTION",
        type="primary",
        use_container_width=True,
        key="run_loss_correction",
    )

    if run_clicked:

        # ----------------------------------------------------
        # Build the current edited dataframe.
        # This is the ONLY point where edited values
        # enter the calculation.
        # ----------------------------------------------------

        try:

            current_input = (
                build_current_input_df(
                    original_input,
                    edited_input,
                )
            )

            st.session_state.input_df = (
                current_input
            )

            # ------------------------------------------------
            # Reset only calculated outputs.
            # ------------------------------------------------

            st.session_state.model_result = None

            # Keep the user-selected efficiency loss
            # if it already exists.

            if cluster:

                existing_loss = (
                    st.session_state
                    .get(
                        "cluster_fixed_loss"
                    )
                )

            else:

                existing_loss = (
                    st.session_state
                    .get(
                        "fixed_loss"
                    )
                )

            # ------------------------------------------------
            # Calculate
            # ------------------------------------------------

            with st.spinner(
                "☀️ Running Fixed Loss Correction..."
            ):

                result = (
                    calculate_fixed_forecast(
                        base_df,
                        current_input,
                        lat,
                        tilt_lookup,
                        cluster,
                        weights,
                    )
                )

            # ------------------------------------------------
            # Save automatic loss
            # ------------------------------------------------

            if cluster:

                if existing_loss is None:

                    st.session_state[
                        "cluster_fixed_loss"
                    ] = result[
                        "auto_loss"
                    ]

            else:

                if existing_loss is None:

                    st.session_state[
                        "fixed_loss"
                    ] = result[
                        "auto_loss"
                    ]

            st.session_state.model_result = (
                result
            )

            st.session_state.run_requested = (
                False
            )

        except Exception as exc:

            st.session_state.model_result = None

            st.error(
                "❌ Loss correction failed."
            )

            st.exception(exc)

    # ========================================================
    # RESULT
    # ========================================================

    result = (
        st.session_state.model_result
    )

    if result is None:

        st.info(
            "Edit the input data if required, "
            "then click **RUN LOSS CORRECTION**."
        )

        return

    # ========================================================
    # EFFICIENCY LOSS
    # ========================================================

    st.markdown(
        '<div class="section">'
        "⚙️ Efficiency Loss"
        "</div>",
        unsafe_allow_html=True,
    )

    efficiency = result[
        "efficiency"
    ]

    if cluster:

        current_loss = (
            st.session_state
            .get(
                "cluster_fixed_loss",
                result["auto_loss"],
            )
        )

        efficiency_loss_key = (
            "cluster_fixed_loss_result"
        )

    else:

        current_loss = (
            st.session_state
            .get(
                "fixed_loss",
                result["auto_loss"],
            )
        )

        efficiency_loss_key = (
            "fixed_loss_result"
        )

    efficiency_values = pd.to_numeric(
        efficiency[
            "Standard PV Efficiency (%)"
        ],
        errors="coerce",
    ).dropna()

    if efficiency_values.empty:

        max_loss = 0.0

    else:

        max_loss = float(
            efficiency_values.min()
        )

        if not np.isfinite(
            max_loss
        ):

            max_loss = 0.0

    current_loss = float(
        np.clip(
            safe_float(
                current_loss,
                0.0,
            ),
            0,
            max_loss,
        )
    )

    new_loss = st.number_input(
        "Efficiency Loss (%)",
        min_value=0.0,
        max_value=max_loss,
        value=current_loss,
        step=0.1,
        format="%.2f",
        key=efficiency_loss_key,
    )

    # ========================================================
    # EFFICIENCY LOSS CHANGED
    # ========================================================

    if (
        abs(
            float(new_loss)
            - float(current_loss)
        )
        > 1e-12
    ):

        if cluster:

            st.session_state[
                "cluster_fixed_loss"
            ] = float(new_loss)

        else:

            st.session_state[
                "fixed_loss"
            ] = float(new_loss)

        # ----------------------------------------------------
        # Recalculate only the lightweight fixed forecast.
        #
        # This does NOT require another workbook read.
        # ----------------------------------------------------

        current_input = (
            st.session_state.input_df
        )

        try:

            updated_result = (
                calculate_fixed_forecast(
                    base_df,
                    current_input,
                    lat,
                    tilt_lookup,
                    cluster,
                    weights,
                )
            )

            st.session_state.model_result = (
                updated_result
            )

            result = updated_result

        except Exception as exc:

            st.error(
                "Unable to apply efficiency loss."
            )

            st.exception(exc)

            return

    # ========================================================
    # FINAL EFFICIENCY DATA
    # ========================================================

    show_efficiency_table(
        result["efficiency"]
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section">'
        "📈 Forecast Performance"
        "</div>",
        unsafe_allow_html=True,
    )

    show_metrics(
        result["forecast"],
        result["actual"],
    )

    # ========================================================
    # CHART
    # ========================================================

    show_forecast_chart(
        result["forecast"],
        result["actual"],
        result["title"],
    )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()
