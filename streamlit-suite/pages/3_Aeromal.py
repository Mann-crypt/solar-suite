
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.signal import savgol_filter
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Aeromal — Solar Suite",
    page_icon="⚡",
    layout="wide",
)

APP_TITLE = "Kaha hai Aeromal ka khauf?!!"
AEROMAL_PASSWORD = "asdfghjkl;'"
BLOCKS_PER_DAY = 96

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
    "Green Gen-SCADA (Generator)",
]


# ==========================================================
# 2. SIDEBAR & STYLING
# ==========================================================

def add_page_styling():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        div[data-testid="stMetric"] {
            padding: 14px;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.2);
        }

        div[data-testid="stFileUploader"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <h1 style="
            text-align:center;
            background:linear-gradient(90deg,#00c6ff,#0072ff);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            font-size:36px;
            font-weight:800;">
            ⚡ Solar Suite
        </h1>

        <p style="text-align:center;color:gray;font-size:14px;">
            Forecast Correction Platform
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()


# ==========================================================
# 3. AUTHENTICATION
# ==========================================================

def authenticate():
    if "aeromal_auth" not in st.session_state:
        st.session_state.aeromal_auth = False

    if st.session_state.aeromal_auth:
        if st.sidebar.button(
            "🚪 Logout",
            use_container_width=True,
        ):
            st.session_state.aeromal_auth = False
            st.rerun()

        return True

    st.title("🔒 Access bas bade logo ke paas hai")

    with st.form("aeromal_login_form"):
        password = st.text_input(
            "Enter Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if password == AEROMAL_PASSWORD:
            st.session_state.aeromal_auth = True
            st.rerun()
        else:
            st.error("Incorrect Password")

    return False


# ==========================================================
# 4. FILE READERS
# ==========================================================

@st.cache_data(show_spinner=False)
def read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_excel_sheets(file_bytes):
    return pd.ExcelFile(
        io.BytesIO(file_bytes)
    ).sheet_names


@st.cache_data(show_spinner=False)
def read_excel_sheet(file_bytes, sheet_name):
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
    )


def clean_dataframe(df):
    df = df.copy()
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]
    return df


def load_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        source_df = read_csv(file_bytes)
        file_type = "csv"

    elif filename.endswith(".xlsx"):
        sheet_names = read_excel_sheets(file_bytes)

        sheet_name = st.selectbox(
            "📑 Select Excel Sheet",
            sheet_names,
        )

        source_df = read_excel_sheet(
            file_bytes,
            sheet_name,
        )
        file_type = "xlsx"

    elif filename.endswith(".xls"):
        sheet_names = read_excel_sheets(file_bytes)

        sheet_name = st.selectbox(
            "📑 Select Excel Sheet",
            sheet_names,
        )

        source_df = read_excel_sheet(
            file_bytes,
            sheet_name,
        )
        file_type = "xls"

    else:
        raise ValueError("Unsupported file format.")

    return (
        clean_dataframe(source_df),
        file_bytes,
        file_type,
    )


def get_generation_columns(df):
    return [
        column
        for column in VALID_COLUMNS
        if column in df.columns
    ]


def prepare_power_data(df, selected_column):
    """
    Convert the selected generation column to numeric values.
    Missing/non-numeric values become zero.
    Original row count is preserved for 96-block validation.
    """
    values = pd.to_numeric(
        df[selected_column],
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    power = values.to_numpy(dtype=float)

    if len(power) == 0:
        raise ValueError(
            "Selected generation column contains no rows."
        )

    if len(power) % BLOCKS_PER_DAY != 0:
        raise ValueError(
            f"Row count must be divisible by {BLOCKS_PER_DAY}. "
            f"Current row count: {len(power)}."
        )

    return power


# ==========================================================
# 5. PROFILE HELPERS
# ==========================================================

def validate_window(window, profile_length=96):
    window = int(window)

    if window % 2 == 0:
        window += 1

    maximum = (
        profile_length
        if profile_length % 2 == 1
        else profile_length - 1
    )

    window = min(window, maximum)

    if window < 5:
        raise ValueError(
            "Profile must contain at least 5 points."
        )

    return window


def smooth_profile(profile, window, polyorder=3):
    profile = np.asarray(
        profile,
        dtype=float,
    )

    if len(profile) < 5:
        return profile.copy()

    window = validate_window(
        window,
        len(profile),
    )

    polyorder = min(
        int(polyorder),
        window - 2,
    )

    return savgol_filter(
        profile,
        window_length=window,
        polyorder=polyorder,
    )


# ==========================================================
# 6. PERCENTILE CALCULATION
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_percentile(power_tuple, days):
    power = np.asarray(
        power_tuple,
        dtype=float,
    )

    daily_matrix = power.reshape(
        int(days),
        BLOCKS_PER_DAY,
    )

    percentile = np.percentile(
        daily_matrix,
        95,
        axis=0,
    )

    return tuple(percentile.tolist())


# ==========================================================
# 7. BEST SHIFT CALCULATION
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_best_shift(profile_tuple):
    profile = np.asarray(
        profile_tuple,
        dtype=float,
    )

    best_error = np.inf
    best_shift = 0

    for shift in range(len(profile)):
        shifted = np.roll(
            profile,
            -shift,
        )

        symmetric = (
            profile + shifted[::-1]
        ) / 2.0

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
# 8. SYMMETRIC PROFILE
# ==========================================================

def create_symmetric_profile(sym):
    """
    Requested symmetry logic:

    1. Replace negative values with zero.
    2. Enforce ascending order on the first half.
    3. If a value falls below the previous value,
       replace it with max_val.
    4. Reverse the corrected first half into the second half.
    """
    sym = np.asarray(
        sym,
        dtype=float,
    ).copy()

    sym = np.where(
        sym < 0,
        0,
        sym,
    )

    mid = len(sym) // 2
    max_val = np.max(sym) if len(sym) else 0

    # Enforce ascending order on the first half
    for i in range(1, mid):
        if sym[i] < sym[i - 1]:
            sym[i] = max_val

    # Copy the corrected first half
    first_half = sym[:mid].copy()

    # Mirror the first half into the second half
    sym[mid:mid + len(first_half)] = first_half[::-1]

    return sym


# ==========================================================
# 9. NON-CIRCULAR MANUAL SHIFT
# ==========================================================

def shift_profile_blockwise(profile, shift):
    """
    Positive shift: move right.
    Negative shift: move left.
    Vacated blocks are filled with zeros.
    No circular wrapping.
    """
    profile = np.asarray(
        profile,
        dtype=float,
    )

    shift = int(shift)
    length = len(profile)

    result = np.zeros_like(profile)

    if shift == 0:
        return profile.copy()

    if abs(shift) >= length:
        return result

    if shift > 0:
        result[shift:] = profile[:-shift]
    else:
        result[:shift] = profile[-shift:]

    return result


@st.cache_data(show_spinner=False)
def cached_sym_shift_nc(symmetric_tuple, shift):
    """
    Shift the already-created symmetric profile.
    This function does not recreate symmetry.
    """
    symmetric = np.asarray(
        symmetric_tuple,
        dtype=float,
    )

    shifted = shift_profile_blockwise(
        symmetric,
        int(shift),
    )

    return tuple(shifted.tolist())


# ==========================================================
# 10. NO CURTAILMENT
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
        ),
        dtype=float,
    )

    # Initial smoothing
    smooth = smooth_profile(
        percentile,
        window,
    )

    # Calculate recommended shift for the UI
    best_shift = cached_best_shift(
        tuple(smooth.tolist())
    )

    # Create symmetry using the requested algorithm
    symmetric = create_symmetric_profile(
        smooth
    )

    # Smooth the symmetric curve while retaining its symmetry
    symmetric = smooth_profile(
        symmetric,
        window,
    )

    # Keep the original profile non-negative
    smooth = np.maximum(
        smooth,
        0,
    )

    symmetric = np.maximum(
        symmetric,
        0,
    )

    # Remove very small values
    smooth[smooth < 0.1] = 0
    symmetric[symmetric < 0.1] = 0

    # Apply availability equally to both curves
    factor = float(power_availability) / 100.0

    smooth *= factor
    symmetric *= factor

    return (
        tuple(percentile.tolist()),
        tuple(smooth.tolist()),
        tuple(symmetric.tolist()),
        int(best_shift),
    )


# ==========================================================
# 11. CURTAILMENT CALCULATION
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
            ),
            dtype=float,
        )
        * 1.03
    )

    y = percentile.copy()
    n = len(y)

    result = np.zeros(n)

    # Smooth and gradient
    smooth = savgol_filter(
        y,
        7,
        2,
    )

    gradient = np.gradient(smooth)

    # LEFT SIDE
    left_peak = int(
        np.argmax(smooth[:n // 2])
    )

    left_peak = max(
        left_peak,
        2,
    )

    left_start = int(
        np.argmax(gradient[:left_peak])
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
        m1, c1 = 1.0, 0.0

    # RIGHT SIDE
    right_peak = (
        int(np.argmax(smooth[n // 2:]))
        + n // 2
    )

    threshold = 0.02 * np.max(smooth)

    active = np.where(
        smooth > threshold
    )[0]

    if len(active) > 0:
        right_end = int(active[-1])
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
        m2 = -1.0
        c2 = float(smooth[right_peak])

    # Avoid division by zero
    if abs(m1) < 1e-9:
        m1 = 1e-9

    if abs(m2) < 1e-9:
        m2 = -1e-9

    # Trip point
    A = (
        1 / m2
        - 1 / m1
    )

    B = (
        c1 / m1
        - c2 / m2
    )

    if abs(A) < 1e-9:
        trip = float(target_width)
    else:
        trip = max(
            0,
            (target_width - B) / A,
        )

    # Peak indices
    left_idx = int(
        np.clip(
            round((trip - c1) / m1),
            0,
            n - 1,
        )
    )

    right_idx = int(
        np.clip(
            round((trip - c2) / m2),
            0,
            n - 1,
        )
    )

    if right_idx <= left_idx:
        right_idx = min(
            n - 1,
            left_idx + 1,
        )

    # DOME MODE
    if peak_cap >= trip:
        left_curve = np.zeros(n)

        for i in range(n):
            value = m1 * i + c1

            left_curve[i] = min(
                value,
                trip,
            )

            if value >= trip:
                left_idx = i
                break

        right_curve = np.zeros(n)

        for i in range(n - 1, -1, -1):
            value = m2 * i + c2

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

        result[left_idx:end_idx] = dome[
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

        result[:left_start] = smooth[:left_start]
        result[right_end:] = smooth[right_end:]

        result = savgol_filter(
            result,
            7,
            3,
        )

        result *= (
            float(power_availability) / 100.0
        )

        result = np.clip(
            result,
            0,
            None,
        )

        result[result < 0.2] = 0

    # FLAT-PEAK MODE
    else:
        for i in range(n):
            value = m1 * i + c1
            result[i] = value

            if value >= peak_cap:
                left_idx = i
                break

        right_curve = np.zeros(n)

        for i in range(n - 1, -1, -1):
            value = m2 * i + c2
            right_curve[i] = value

            if value >= peak_cap:
                right_idx = i
                break

        result = np.maximum(
            result,
            right_curve,
        )

        if right_idx > left_idx:
            result[left_idx:right_idx] = peak_cap

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

        result[:left_start] = smooth[:left_start]
        result[right_end:] = smooth[right_end:]

        result = savgol_filter(
            result,
            7,
            3,
        )

        result *= (
            float(power_availability) / 100.0
        )

        result = np.clip(
            result,
            0,
            peak_cap,
        )

        result[result < 1] = 0

    best_shift = cached_best_shift(
        tuple(result.tolist())
    )

    return (
        tuple(percentile.tolist()),
        tuple(result.tolist()),
        int(best_shift),
    )


# ==========================================================
# 12. CURTAILMENT SYMMETRY
# ==========================================================

@st.cache_data(show_spinner=False)
def cached_sym_shift_c(profile_tuple, shift):
    profile = np.asarray(
        profile_tuple,
        dtype=float,
    )

    shifted = np.roll(
        profile,
        -int(shift),
    )

    symmetric = (
        profile + shifted[::-1]
    ) / 2.0

    return tuple(
        symmetric.tolist()
    )


# ==========================================================
# 13. EXCEL REPORT STYLING
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

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    for column in ws.columns:
        column_letter = get_column_letter(
            column[0].column
        )

        max_length = max(
            (
                len(str(cell.value))
                for cell in column
                if cell.value is not None
            ),
            default=0,
        )

        ws.column_dimensions[column_letter].width = min(
            max_length + 2,
            35,
        )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def write_result_sheet(ws, result_df):
    ws.append(
        result_df.columns.tolist()
    )

    for row in result_df.itertuples(
        index=False,
        name=None,
    ):
        cleaned_row = []

        for value in row:
            if isinstance(value, np.generic):
                value = value.item()

            if pd.isna(value):
                value = None

            cleaned_row.append(value)

        ws.append(cleaned_row)

    style_report_sheet(ws)


def build_excel_report(
    original_bytes,
    file_type,
    source_df,
    result_df,
):
    output = io.BytesIO()

    # Preserve original workbook sheets for XLSX
    if file_type == "xlsx":
        wb = load_workbook(
            io.BytesIO(original_bytes)
        )

        if "Final Report" in wb.sheetnames:
            del wb["Final Report"]

        ws = wb.create_sheet("Final Report")
        write_result_sheet(ws, result_df)

        wb.save(output)

    else:
        # CSV and XLS inputs are exported as a new XLSX workbook
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

        wb = load_workbook(output)

        for sheet in wb.worksheets:
            style_report_sheet(sheet)

        output = io.BytesIO()
        wb.save(output)

    return output.getvalue()


# ==========================================================
# 14. CHART
# ==========================================================

def make_chart(x, curves, title):
    fig = go.Figure()

    for name, values, color in curves:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                name=name,
                line=dict(
                    color=color,
                    width=3,
                ),
            )
        )

    fig.update_layout(
        title=title,
        height=540,
        hovermode="x unified",
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
            t=70,
            b=20,
        ),
    )

    return fig


# ==========================================================
# 15. MAIN PAGE
# ==========================================================

def main():
    add_page_styling()

    if not authenticate():
        st.stop()

    st.title(APP_TITLE)

    st.caption(
        "Solar Generation Profile Analysis & Correction"
    )

    st.divider()

    # ------------------------------------------------------
    # File upload
    # ------------------------------------------------------

    uploaded_file = st.file_uploader(
        "📂 Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is None:
        st.info(
            "Pehle CSV ya Excel file upload karo."
        )
        st.stop()

    # ------------------------------------------------------
    # Read uploaded file
    # ------------------------------------------------------

    try:
        source_df, file_bytes, file_type = (
            load_uploaded_file(uploaded_file)
        )
    except Exception as exc:
        st.error(
            f"File read nahi ho payi: {exc}"
        )
        st.stop()

    # ------------------------------------------------------
    # Generation column selection
    # ------------------------------------------------------

    available_columns = get_generation_columns(
        source_df
    )

    if not available_columns:
        st.error(
            "Required generation column nahi mili."
        )

        st.write(
            "Expected columns:",
            VALID_COLUMNS,
        )

        st.write(
            "Available columns:",
            list(source_df.columns),
        )

        st.stop()

    st.subheader("⚙️ Calculation Input")

    selected_column = st.selectbox(
        "Select Generation / Power Column",
        available_columns,
    )

    # ------------------------------------------------------
    # Prepare power data
    # ------------------------------------------------------

    try:
        power = prepare_power_data(
            source_df,
            selected_column,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    days = len(power) // BLOCKS_PER_DAY

    power_tuple = tuple(
        float(value)
        for value in power
    )

    x = np.arange(
        1,
        BLOCKS_PER_DAY + 1,
    )

    # ------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total Rows",
        f"{len(power):,}",
    )

    metric2.metric(
        "Complete Days",
        f"{days:,}",
    )

    metric3.metric(
        "Blocks Per Day",
        BLOCKS_PER_DAY,
    )

    st.divider()

    # ------------------------------------------------------
    # Mode selector
    # ------------------------------------------------------

    curtailment = st.toggle(
        "⚡ Curtailment Mode",
        value=False,
    )

    # ------------------------------------------------------
    # NO CURTAILMENT MODE
    # ------------------------------------------------------

    if not curtailment:
        st.subheader("☀️ No Curtailment Mode")

        col1, col2, col3 = st.columns(3)

        window = col1.number_input(
            "Window Length",
            min_value=5,
            max_value=31,
            value=11,
            step=2,
        )

        power_availability = col2.number_input(
            "Power Availability (%)",
            min_value=0,
            max_value=1000,
            value=100,
            step=1,
        )

        (
            percentile_t,
            profile_t,
            symmetric_t,
            best_shift,
        ) = cached_no_curtailment(
            power_tuple,
            days,
            int(window),
            float(power_availability),
        )

        # The suggested shift is the default control value.
        # The actual shift is applied AFTER symmetry creation.
        shift = col3.number_input(
            "Shift (Blocks)",
            min_value=-95,
            max_value=95,
            value=int(best_shift),
            step=1,
            help=(
                "Positive = move right. Negative = move left. "
                "Zero = no shift. No circular wrapping."
            ),
        )

        symmetric = np.asarray(
            cached_sym_shift_nc(
                symmetric_t,
                int(shift),
            ),
            dtype=float,
        )

        percentile = np.asarray(
            percentile_t,
            dtype=float,
        )

        profile = np.asarray(
            profile_t,
            dtype=float,
        )

        st.caption(
            f"Recommended shift: {best_shift} blocks. "
            "The selected shift is applied to the symmetric curve."
        )

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
            "Aeromal — No Curtailment",
        )

    # ------------------------------------------------------
    # CURTAILMENT MODE
    # ------------------------------------------------------

    else:
        st.subheader("⚡ Curtailment Mode")

        if not np.any(power > 0):
            st.warning(
                "Please enter Power values to continue."
            )
            st.stop()

        col1, col2, col3 = st.columns(3)

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
            value=max(1, int(np.max(power))),
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

        (
            percentile_t,
            profile_t,
            best_shift,
        ) = cached_curtailment(
            power_tuple,
            days,
            float(peak_cap),
            float(target_width),
            int(window),
            float(power_availability),
        )

        shift = col3.number_input(
            "Shift",
            min_value=0,
            max_value=95,
            value=int(best_shift),
            step=1,
        )

        symmetric_t = cached_sym_shift_c(
            profile_t,
            int(shift),
        )

        percentile = np.asarray(
            percentile_t,
            dtype=float,
        )

        profile = np.asarray(
            profile_t,
            dtype=float,
        )

        symmetric = np.asarray(
            symmetric_t,
            dtype=float,
        )

        st.caption(
            f"Recommended symmetry shift: {best_shift} blocks."
        )

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
            "Aeromal — Curtailment",
        )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ------------------------------------------------------
    # Results table
    # ------------------------------------------------------

    result_df = pd.DataFrame(
        {
            "Block": x,
            "Percentile": percentile,
            "Profile": profile,
            "Sym Profile": symmetric,
        }
    )

    st.subheader("📊 Generated Curve")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )

    # ------------------------------------------------------
    # Excel report download
    # ------------------------------------------------------

    st.divider()

    st.subheader("📥 Final Report")

    st.caption(
        "The Final Report sheet contains only Block, "
        "Percentile, Profile and Sym Profile."
    )

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

        st.download_button(
            label="📥 Download Final Excel Report",
            data=report_bytes,
            file_name=f"{base_name}_Aeromal_Report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
            type="primary",
        )

    except Exception as exc:
        st.error(
            f"Report generate nahi ho payi: {exc}"
        )


# ==========================================================
# 16. RUN APPLICATION
# ==========================================================

if __name__ == "__main__":
    main()
