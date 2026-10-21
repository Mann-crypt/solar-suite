import io
import csv
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.signal import savgol_filter
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Aeromal — Solar Suite",
    page_icon="⚡",
    layout="wide",
)

AEROMAL_PASSWORD = "asdfghjkl;'"

VALID_COLUMNS = [
    "Green Gen-Meter (Developer)",
    "Green Gen-SCADA (Developer)",
    "Green Gen-Meter (Generator)",
    "SEMS",
    "SEMS (SEMS)",
    "Green Gen-SCADA (OPC)",
    "Green Gen-Meter (OPC)",
    "Green Gen-Meter (API)",
    "Green Gen-SCADA (API)",
    "Power (MW)",
    "Green Gen-SCADA (Generator)"
]


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    """
    <h1 style='text-align:center;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-size:40px;font-weight:800;'>
    ⚡ Solar Suite
    </h1>

    <p style='text-align:center;color:gray;font-size:14px;'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ==========================================================
# AUTHENTICATION
# ==========================================================

if "aeromal_auth" not in st.session_state:
    st.session_state.aeromal_auth = False


if st.session_state.aeromal_auth:

    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True,
    ):
        st.session_state.aeromal_auth = False
        st.rerun()

else:

    st.title("🔒 Access bas bade logo ke paas hai")

    password = st.text_input(
        "Enter Password",
        type="password",
    )

    if st.button(
        "Login",
        type="primary",
        use_container_width=True,
    ):

        if password == AEROMAL_PASSWORD:

            st.session_state.aeromal_auth = True
            st.rerun()

        else:
            st.error("Incorrect Password")

    st.stop()


# ==========================================================
# FILE READERS
# ==========================================================

@st.cache_data(show_spinner=False)
def read_excel_sheets(file_bytes):

    xls = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_sheet(
    file_bytes,
    sheet_name,
):

    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
    )


@st.cache_data(show_spinner=False)
def read_csv(file_bytes):

    return pd.read_csv(
        io.BytesIO(file_bytes)
    )


# ==========================================================
# PERCENTILE
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_percentile(
    power_tuple,
    days,
):

    power = np.asarray(
        power_tuple,
        dtype=float,
    )

    profile = power.reshape(
        days,
        96,
    )

    percentile = np.percentile(
        profile,
        95,
        axis=0,
    )

    return tuple(
        percentile.tolist()
    )


# ==========================================================
# BEST SYMMETRY SHIFT
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_best_shift(
    profile_tuple,
):

    profile = np.asarray(
        profile_tuple,
        dtype=float,
    )

    best_error = np.inf
    best_shift = 0

    for shift in range(96):

        shifted = np.roll(
            profile,
            -shift,
        )

        symmetric = (
            profile + shifted[::-1]
        ) / 2

        error = np.sqrt(
            np.mean(
                (profile - symmetric) ** 2
            )
        )

        if error < best_error:

            best_error = error
            best_shift = shift

    return int(best_shift)


# ==========================================================
# NO CURTAILMENT
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_no_curtailment(
    power_tuple,
    days,
    window,
    power_availability,
):

    percentile = np.asarray(
        cached_percentile(
            power_tuple,
            days,
        )
    )

    # ------------------------------------------------------
    # Initial smoothing
    # ------------------------------------------------------

    smooth = savgol_filter(
        percentile,
        window_length=window,
        polyorder=3,
    )

    # ------------------------------------------------------
    # Best symmetry shift
    # ------------------------------------------------------

    best_shift = cached_best_shift(
        tuple(smooth.tolist())
    )

    shifted = np.roll(
        smooth,
        -best_shift,
    )

    symmetric = (
        0.50 * smooth
        + 0.50 * shifted[::-1]
    )

    # ------------------------------------------------------
    # Morning / Evening blending
    # ------------------------------------------------------

    active = np.where(
        percentile > 0.1
    )[0]

    if len(active) > 0:

        start = active[0]
        end = active[-1]

        blend = 8

        # Morning
        morning_end = min(
            start + 1 + blend,
            96,
        )

        if morning_end > start + 1:

            idx = np.arange(
                start + 1,
                morning_end,
            )

            weights = np.linspace(
                1,
                0,
                len(idx),
            )

            symmetric[idx] = (
                weights * percentile[idx]
                + (1 - weights) * symmetric[idx]
            )

        # Evening
        evening_start = max(
            end - blend,
            0,
        )

        if end > evening_start:

            idx = np.arange(
                evening_start,
                end,
            )

            weights = np.linspace(
                0,
                1,
                len(idx),
            )

            symmetric[idx] = (
                weights * percentile[idx]
                + (1 - weights) * symmetric[idx]
            )

    # ------------------------------------------------------
    # Final smoothing
    # ------------------------------------------------------

    smooth = savgol_filter(
        percentile,
        window_length=window,
        polyorder=3,
    )

    symmetric = savgol_filter(
        symmetric,
        window_length=window,
        polyorder=3,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    symmetric = np.clip(
        symmetric,
        0,
        None,
    )

    smooth[
        smooth < 0.1
    ] = 0

    symmetric[
        symmetric < 0.1
    ] = 0

    # ------------------------------------------------------
    # Power availability
    # ------------------------------------------------------

    factor = (
        power_availability / 100
    )

    smooth *= factor
    symmetric *= factor

    return (
        tuple(percentile.tolist()),
        tuple(smooth.tolist()),
        tuple(symmetric.tolist()),
        best_shift,
    )


# ==========================================================
# MANUAL SYMMETRY SHIFT
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_sym_shift_nc(
    smooth_tuple,
    shift,
):

    smooth = np.asarray(
        smooth_tuple,
        dtype=float,
    )

    shifted = np.roll(
        smooth,
        -int(shift),
    )

    symmetric = (
        0.50 * smooth
        + 0.50 * shifted[::-1]
    )

    symmetric = savgol_filter(
        symmetric,
        window_length=window,
        polyorder=3,
    )

    symmetric = np.clip(
        symmetric,
        0,
        None,
    )

    symmetric[
        symmetric < 0.1
    ] = 0

    return tuple(
        symmetric.tolist()
    )


# ==========================================================
# CURTAILMENT
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_curtailment(
    power_tuple,
    days,
    peak_cap,
    target_width,
    window,
    power_availability,
):

    percentile = (
        np.asarray(
            cached_percentile(
                power_tuple,
                days,
            )
        )
        * 1.03
    )

    y = percentile.copy()
    n = len(y)

    result = np.zeros(n)

    # ------------------------------------------------------
    # Smooth and gradient
    # ------------------------------------------------------

    smooth = savgol_filter(
        y,
        7,
        2,
    )

    gradient = np.gradient(
        smooth
    )

    # ------------------------------------------------------
    # LEFT SIDE
    # ------------------------------------------------------

    left_peak = np.argmax(
        smooth[: n // 2]
    )

    left_peak = max(
        left_peak,
        2,
    )

    left_start = np.argmax(
        gradient[:left_peak]
    )

    if left_start >= left_peak:
        left_start = 0

    x_left = np.arange(
        left_start,
        left_peak,
    )

    y_left = smooth[
        left_start:left_peak
    ]

    if len(x_left) >= 2:

        m1, c1 = np.polyfit(
            x_left,
            y_left,
            1,
        )

    else:

        m1 = 1
        c1 = 0

    # ------------------------------------------------------
    # RIGHT SIDE
    # ------------------------------------------------------

    right_peak = (
        np.argmax(
            smooth[n // 2:]
        )
        + n // 2
    )

    threshold = (
        0.02 * np.max(smooth)
    )

    active = np.where(
        smooth > threshold
    )[0]

    if len(active) > 0:
        right_end = active[-1]
    else:
        right_end = n - 1

    if right_end <= right_peak:

        right_end = min(
            n - 1,
            right_peak + 2,
        )

    x_right = np.arange(
        right_peak,
        right_end,
    )

    y_right = smooth[
        right_peak:right_end
    ]

    if len(x_right) >= 2:

        m2, c2 = np.polyfit(
            x_right,
            y_right,
            1,
        )

    else:

        m2 = -1
        c2 = smooth[right_peak]

    # ------------------------------------------------------
    # Avoid division by zero
    # ------------------------------------------------------

    if abs(m1) < 1e-9:
        m1 = 1e-9

    if abs(m2) < 1e-9:
        m2 = -1e-9

    # ------------------------------------------------------
    # Trip point
    # ------------------------------------------------------

    A = (
        1 / m2
        - 1 / m1
    )

    B = (
        c1 / m1
        - c2 / m2
    )

    if abs(A) < 1e-9:

        trip = float(
            target_width
        )

    else:

        trip = max(
            0,
            (target_width - B) / A,
        )

    # ------------------------------------------------------
    # Peak indices
    # ------------------------------------------------------

    left_idx = int(
        np.clip(
            round(
                (trip - c1) / m1
            ),
            0,
            n - 1,
        )
    )

    right_idx = int(
        np.clip(
            round(
                (trip - c2) / m2
            ),
            0,
            n - 1,
        )
    )

    if right_idx <= left_idx:

        right_idx = min(
            n - 1,
            left_idx + 1,
        )

    # ======================================================
    # DOME MODE
    # ======================================================

    if peak_cap >= trip:

        left_curve = np.zeros(n)

        for i in range(n):

            value = (
                m1 * i + c1
            )

            left_curve[i] = min(
                value,
                trip,
            )

            if value >= trip:

                left_idx = i
                break

        right_curve = np.zeros(n)

        for i in range(
            n - 1,
            -1,
            -1,
        ):

            value = (
                m2 * i + c2
            )

            right_curve[i] = min(
                value,
                trip,
            )

            if value >= trip:

                right_idx = i
                break

        result = np.maximum(
            left_curve,
            right_curve,
        )

        width = max(
            right_idx - left_idx,
            2,
        )

        dome_height = max(
            20,
            0.12 * trip,
        )

        xc = np.linspace(
            -1,
            1,
            width,
        )

        shape = np.sqrt(
            np.maximum(
                0,
                1 - xc ** 2,
            )
        )

        dome = (
            trip
            + dome_height * shape
        )

        dome[0] = trip
        dome[-1] = trip

        end_idx = min(
            left_idx + width,
            n,
        )

        result[
            left_idx:end_idx
        ] = dome[
            :end_idx - left_idx
        ]

        result = savgol_filter(
            result,
            window,
            3,
        )

        result = np.clip(
            result,
            0,
            None,
        )

        result[
            :left_start
        ] = smooth[
            :left_start
        ]

        result[
            right_end:
        ] = smooth[
            right_end:
        ]

        result = savgol_filter(
            result,
            7,
            3,
        )

        result *= (
            power_availability / 100
        )

        result = np.clip(
            result,
            0,
            None,
        )

        result[
            result < 0.2
        ] = 0

    # ======================================================
    # FLAT PEAK MODE
    # ======================================================

    else:

        for i in range(n):

            value = (
                m1 * i + c1
            )

            result[i] = value

            if value >= peak_cap:

                left_idx = i
                break

        right_curve = np.zeros(n)

        for i in range(
            n - 1,
            -1,
            -1,
        ):

            value = (
                m2 * i + c2
            )

            right_curve[i] = value

            if value >= peak_cap:

                right_idx = i
                break

        result = np.maximum(
            result,
            right_curve,
        )

        if right_idx > left_idx:

            result[
                left_idx:right_idx
            ] = peak_cap

        result = np.clip(
            result,
            0,
            peak_cap,
        )

        result = savgol_filter(
            result,
            window,
            3,
        )

        result[
            :left_start
        ] = smooth[
            :left_start
        ]

        result[
            right_end:
        ] = smooth[
            right_end:
        ]

        result = savgol_filter(
            result,
            7,
            3,
        )

        result *= (
            power_availability / 100
        )

        result = np.clip(
            result,
            0,
            peak_cap,
        )

        result[
            result < 1
        ] = 0

    # ------------------------------------------------------
    # Best shift
    # ------------------------------------------------------

    best_shift = cached_best_shift(
        tuple(result.tolist())
    )

    return (
        tuple(percentile.tolist()),
        tuple(result.tolist()),
        best_shift,
    )


# ==========================================================
# CURTAILMENT SYMMETRY
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_sym_shift_c(
    final_profile_tuple,
    shift,
):

    profile = np.asarray(
        final_profile_tuple,
        dtype=float,
    )

    shifted = np.roll(
        profile,
        -int(shift),
    )

    symmetric = (
        profile + shifted[::-1]
    ) / 2

    return tuple(
        symmetric.tolist()
    )


# ==========================================================
# REPORT STYLING
# ==========================================================

def style_report_sheet(ws):

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="0072FF",
    )

    header_font = Font(
        color="FFFFFF",
        bold=True,
    )

    # Header
    for cell in ws[1]:

        cell.fill = header_fill
        cell.font = header_font

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    # Column widths
    for column in ws.columns:

        max_length = 0

        column_letter = get_column_letter(
            column[0].column
        )

        for cell in column:

            if cell.value is not None:

                max_length = max(
                    max_length,
                    len(str(cell.value)),
                )

        ws.column_dimensions[
            column_letter
        ].width = min(
            max_length + 2,
            25,
        )

    ws.freeze_panes = "A2"

    ws.auto_filter.ref = ws.dimensions


# ==========================================================
# BUILD EXCEL REPORT
# ==========================================================

def build_excel_report(
    original_bytes,
    file_type,
    source_df,
    result_df,
):

    output = io.BytesIO()

    # ======================================================
    # EXCEL INPUT
    # ======================================================

    if file_type == "excel":

        output.write(
            original_bytes
        )

        output.seek(0)

        wb = load_workbook(
            output
        )

        # Remove old report
        if "Final Report" in wb.sheetnames:

            del wb[
                "Final Report"
            ]

        # Create clean report
        ws = wb.create_sheet(
            "Final Report"
        )

        # ONLY these columns
        for col_idx, column in enumerate(
            result_df.columns,
            start=1,
        ):

            ws.cell(
                row=1,
                column=col_idx,
                value=column,
            )

        for row_idx, values in enumerate(
            result_df.itertuples(
                index=False,
                name=None,
            ),
            start=2,
        ):

            for col_idx, value in enumerate(
                values,
                start=1,
            ):

                if isinstance(
                    value,
                    (np.integer, np.floating),
                ):

                    value = float(value)

                ws.cell(
                    row=row_idx,
                    column=col_idx,
                    value=value,
                )

        style_report_sheet(
            ws
        )

        wb.save(
            output
        )

        output.seek(0)

        return output.getvalue()

    # ======================================================
    # CSV INPUT
    # ======================================================

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        source_df.to_excel(
            writer,
            sheet_name="Source Data",
            index=False,
        )

        result_df.to_excel(
            writer,
            sheet_name="Final Report",
            index=False,
        )

    output.seek(0)

    wb = load_workbook(
        output
    )

    ws = wb[
        "Final Report"
    ]

    style_report_sheet(
        ws
    )

    wb.save(
        output
    )

    output.seek(0)

    return output.getvalue()


# ==========================================================
# CHART
# ==========================================================

def make_chart(
    x,
    curves,
    title,
):

    fig = go.Figure()

    for name, values, color in curves:

        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                name=name,
                line=dict(
                    color=color,
                    width=4,
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=550,
        hovermode="x unified",
        template="streamlit",
        xaxis_title="Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ==========================================================
# CSV TEMPLATE HELPERS
# ==========================================================

POOLING_STATION_SUFFIXES = (
    "_ALL12.5_F1",
    "_MICO_F1",
    "_EN1",
    "_AM1",
    "_T1",
)


def extract_pooling_station_name(source_column):
    """Convert a source column name into its pooling station name."""
    column_name = str(source_column).strip()

    for suffix in POOLING_STATION_SUFFIXES:
        if column_name.endswith(suffix):
            return column_name[:-len(suffix)].strip()

    return column_name


def build_profile_template_csv(station_column, curve_values):
    """Build a two-column CSV template containing 96 time blocks."""
    values = np.asarray(curve_values, dtype=float).flatten()

    if len(values) != 96:
        raise ValueError(
            f"Expected 96 curve values, but received {len(values)}."
        )

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")

    # Row 1: pooling station name
    writer.writerow([
        "Pooling Station Name",
        extract_pooling_station_name(station_column),
    ])

    # Row 2: column headers
    writer.writerow([
        "Time Block",
        "Power(MW)",
    ])

    # Rows 3-98: time blocks 1 through 96
    for block_number, value in enumerate(values, start=1):
        writer.writerow([block_number, value])

    return output.getvalue().encode("utf-8-sig")


# ==========================================================
# PAGE
# ==========================================================

st.title(
    "Kaha hai Aeromal ka khauf?!!"
)


# ==========================================================
# FILE UPLOADER
# ==========================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)


if uploaded_file is None:

    st.info(
        "Pehle CSV ya Excel file upload karo!!!"
    )

    st.stop()


file_bytes = uploaded_file.getvalue()

file_name = uploaded_file.name.lower()


# ==========================================================
# LOAD FILE
# ==========================================================

if file_name.endswith(".csv"):

    file_type = "csv"

    try:

        source_df = read_csv(
            file_bytes
        )

    except Exception as e:

        st.error(
            f"CSV read nahi ho payi: {e}"
        )

        st.stop()

else:

    file_type = "excel"

    try:

        sheet_names = read_excel_sheets(
            file_bytes
        )

    except Exception as e:

        st.error(
            f"Excel read nahi ho payi: {e}"
        )

        st.stop()

    sheet_name = st.selectbox(
        "📑 Select Excel Sheet",
        sheet_names,
    )

    source_df = read_excel_sheet(
        file_bytes,
        sheet_name,
    )


# ==========================================================
# CLEAN COLUMN NAMES
# ==========================================================

source_df.columns = [
    str(column).strip()
    for column in source_df.columns
]


# ==========================================================
# FIND VALID COLUMNS
# ==========================================================

available_columns = [
    column
    for column in VALID_COLUMNS
    if column in source_df.columns
]


if not available_columns:

    st.error(
        "Required generation column nahi mili."
    )

    st.write(
        "Expected columns:"
    )

    st.write(
        VALID_COLUMNS
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(source_df.columns)
    )

    st.stop()


# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.subheader(
    "⚙️ Calculation Input"
)

selected_column = st.selectbox(
    "Select Generation / Power Column",
    available_columns,
)


# ==========================================================
# POWER DATA
# ==========================================================

power_series = pd.to_numeric(
    source_df[
        selected_column
    ],
    errors="coerce",
).fillna(0)


power_series = power_series[
    np.isfinite(
        power_series
    )
]


power_list = power_series.tolist()


if len(power_list) == 0:

    st.warning(
        "Selected column mein valid numerical data nahi hai."
    )

    st.stop()


# ==========================================================
# 96 BLOCK VALIDATION
# ==========================================================

n_rows = len(power_list)


if n_rows % 96 != 0:

    st.error(
        f"""
        Number of rows must be divisible by 96.

        Current rows: {n_rows}

        Required:
        96 rows = 1 day
        192 rows = 2 days
        288 rows = 3 days
        """
    )

    st.stop()


days = n_rows // 96


st.success(
    f"✅ {n_rows} rows detected = {days} day(s)"
)


power_tuple = tuple(
    float(value)
    for value in power_list
)


# ==========================================================
# MODE TOGGLE
# ==========================================================

st.markdown(
    """
    <style>

    div[data-testid="stToggle"] {
        background:#1f2937;
        border:2px solid #0072ff;
        border-radius:14px;
        padding:14px 18px;
        margin-bottom:15px;
    }

    div[data-testid="stToggle"] label {
        font-size:18px !important;
        font-weight:700 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


curtailment = st.toggle(
    "⚡ Curtailment Mode",
    value=False,
)


# ==========================================================
# X AXIS
# ==========================================================

x = np.arange(
    1,
    97,
)


# ==========================================================
# NO CURTAILMENT
# ==========================================================

if not curtailment:

    st.subheader(
        "☀️ No Curtailment Mode"
    )

    col1, col2, col3 = st.columns(
        3
    )

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------

    window = col1.number_input(
        "Window Length",
        min_value=5,
        max_value=31,
        step=2,
        value=11,
    )

    power_availability = col2.number_input(
        "Power Availability (%)",
        min_value=0,
        max_value=1000,
        value=100,
    )

    # ------------------------------------------------------
    # Calculation
    # ------------------------------------------------------

    (
        percentile_t,
        profile_t,
        symmetric_t,
        best_shift,
    ) = cached_no_curtailment(
        power_tuple,
        days,
        window,
        power_availability,
    )

    # ------------------------------------------------------
    # Shift
    # ------------------------------------------------------

    shift = col3.number_input(
        "Shift",
        min_value=0,
        max_value=95,
        value=best_shift,
        step=1,
    )

    if shift != best_shift:

        symmetric_t = cached_sym_shift_nc(
            profile_t,
            shift,
        )

    percentile = np.asarray(
        percentile_t
    )

    profile = np.asarray(
        profile_t
    )

    symmetric = np.asarray(
        symmetric_t
    )

    # ------------------------------------------------------
    # Chart
    # ------------------------------------------------------

    fig = make_chart(
        x,
        [
            (
                "Sym Profile",
                symmetric,
                "#00c6ff",
            ),
            (
                "Profile",
                profile,
                "#22c55e",
            ),
            (
                "95th Percentile",
                percentile,
                "#ef4444",
            ),
        ],
        "Aeromal - No Curtailment",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ------------------------------------------------------
    # DISPLAY TABLE
    # ------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "Block": x,
            "Percentile": percentile,
            "Profile": profile,
            "Sym Profile": symmetric,
        }
    )

    st.subheader(
        "📊 Generated Curve"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# CURTAILMENT
# ==========================================================

else:

    st.subheader(
        "⚡ Curtailment Mode"
    )

    if not np.any(
        np.asarray(power_list) > 0
    ):

        st.warning(
            "Please enter Power values to continue."
        )

        st.stop()

    col1, col2, col3 = st.columns(
        3
    )

    # ------------------------------------------------------
    # Parameters
    # ------------------------------------------------------

    power_availability = col1.number_input(
        "Power Availability (%)",
        min_value=0,
        max_value=1000,
        value=100,
        step=1,
    )

    peak_cap = col1.number_input(
        "Peak Cap",
        min_value=1,
        value=max(
            1,
            int(max(power_list)),
        ),
        step=1,
    )

    target_width = col2.number_input(
        "Target Width",
        min_value=1,
        value=25,
        step=1,
    )

    window = col2.slider(
        "Window Length",
        min_value=5,
        max_value=31,
        value=11,
        step=2,
    )

    # ------------------------------------------------------
    # Calculation
    # ------------------------------------------------------

    (
        percentile_t,
        profile_t,
        best_shift,
    ) = cached_curtailment(
        power_tuple,
        days,
        peak_cap,
        target_width,
        window,
        power_availability,
    )

    # ------------------------------------------------------
    # Shift
    # ------------------------------------------------------

    shift = col3.number_input(
        "Shift",
        min_value=0,
        max_value=95,
        value=best_shift,
        step=1,
    )

    symmetric_t = cached_sym_shift_c(
        profile_t,
        shift,
    )

    percentile = np.asarray(
        percentile_t
    )

    profile = np.asarray(
        profile_t
    )

    symmetric = np.asarray(
        symmetric_t
    )

    # ------------------------------------------------------
    # Chart
    # ------------------------------------------------------

    fig = make_chart(
        x,
        [
            (
                "Generation",
                percentile,
                "#ef4444",
            ),
            (
                "Profile",
                profile,
                "#00c6ff",
            ),
            (
                "Sym Profile",
                symmetric,
                "#0072ff",
            ),
        ],
        "Aeromal - Curtailment",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ------------------------------------------------------
    # DISPLAY TABLE
    # ------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "Block": x,
            "Percentile": percentile,
            "Profile": profile,
            "Sym Profile": symmetric,
        }
    )

    st.subheader(
        "📊 Generated Curve"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# DOWNLOAD TEMPLATE CSV
# ==========================================================

st.divider()

st.subheader("📄 Download Template CSV")
st.caption(
    "Choose a curve and pooling station to download a CSV with "
    "the station name, Time Block, and Power(MW) for blocks 1-96."
)

# Find source columns that look like pooling-station columns.
station_columns = [
    str(column)
    for column in source_df.columns
    if any(str(column).strip().endswith(suffix)
           for suffix in POOLING_STATION_SUFFIXES)
]

# If the uploaded file does not contain those suffixes, allow the user
# to select from its available columns rather than failing.
if not station_columns:
    station_columns = [str(column) for column in source_df.columns]

# Keep the current selected calculation column available as a fallback.
if selected_column not in station_columns and not any(
    any(str(column).strip().endswith(suffix) for suffix in POOLING_STATION_SUFFIXES)
    for column in source_df.columns
):
    station_columns = list(dict.fromkeys(station_columns + [selected_column]))

template_col1, template_col2 = st.columns(2)

with template_col1:
    selected_station = st.selectbox(
        "Select Pooling Station",
        options=station_columns,
        key="template_pooling_station",
    )

with template_col2:
    selected_curve = st.selectbox(
        "Select Curve",
        options=["Profile", "Percentile", "Sym Profile"],
        key="template_curve",
    )

apply_one_block_shift = st.checkbox(
    "Apply one-time-block shift",
    value=False,
    key="template_one_block_shift",
    help=(
        "Shifts all values forward by one block. This is a circular shift, "
        "so the last value moves to block 1."
    ),
)

try:
    curve_map = {
        "Profile": profile,
        "Percentile": percentile,
        "Sym Profile": symmetric,
    }

    curve_values = np.asarray(
        curve_map[selected_curve],
        dtype=float,
    ).flatten()

    if len(curve_values) != 96:
        st.error(
            f"Selected curve must contain 96 values. Found {len(curve_values)}."
        )
    else:
        if apply_one_block_shift:
            curve_values = np.roll(curve_values, 1)

        template_bytes = build_profile_template_csv(
            station_column=selected_station,
            curve_values=curve_values,
        )

        safe_station_name = (
            extract_pooling_station_name(selected_station)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        safe_curve_name = selected_curve.replace(" ", "_")
        shift_suffix = "_Shifted" if apply_one_block_shift else ""
        template_file_name = (
            f"{safe_station_name}_{safe_curve_name}"
            f"{shift_suffix}_Template.csv"
        )

        st.download_button(
            label="⬇️ Download Template CSV",
            data=template_bytes,
            file_name=template_file_name,
            mime="text/csv",
            key="download_profile_template_csv",
            use_container_width=True,
        )

except Exception as e:
    st.error(f"Template CSV could not be generated: {e}")

# ==========================================================
# FINAL REPORT
# ==========================================================

st.divider()

st.subheader(
    "📥 Final Report"
)

st.caption(
    "Final Report contains only Block, Percentile, "
    "Profile and Sym Profile."
)


# ==========================================================
# BUILD REPORT
# ==========================================================

try:

    report_bytes = build_excel_report(
        original_bytes=file_bytes,
        file_type=file_type,
        source_df=source_df,
        result_df=result_df,
    )

    base_name = uploaded_file.name.rsplit(
        ".",
        1,
    )[0]

    output_name = (
        f"{base_name}_Aeromal_Report.xlsx"
    )

    st.download_button(
        label="📥 Download Final Excel Report",
        data=report_bytes,
        file_name=output_name,
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
        type="primary",
    )

except Exception as e:

    st.error(
        f"Report generate nahi ho payi: {e}"
    )
