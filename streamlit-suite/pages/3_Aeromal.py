
# ============================================================
# AEROMAL SOLAR SUITE
# Clean application structure
# ============================================================

# -------------------------
# 1. IMPORTS & CONFIGURATION
# -------------------------
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.signal import savgol_filter
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


st.set_page_config(
    page_title="Aeromal — Solar Suite",
    page_icon="⚡",
    layout="wide",
)

APP_TITLE = "Kaha hai Aeromal ka khauf?!!"
AEROMAL_PASSWORD = "asdfghjkl;'"
BLOCKS_PER_DAY = 96

GENERATION_COLUMNS = [
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


# -------------------------
# 2. PAGE STYLING
# -------------------------
def add_page_styling():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            letter-spacing: -0.3px;
        }

        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.07);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.15);
        }

        div[data-testid="stFileUploader"] {
            border-radius: 12px;
        }

        .section-divider {
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# 3. AUTHENTICATION
# -------------------------
def authenticate():
    if st.session_state.get("authenticated", False):
        return True

    st.title("⚡ Aeromal Solar Suite")
    st.subheader("Login")

    with st.form("login_form"):
        password = st.text_input(
            "Enter password",
            type="password",
        )
        submitted = st.form_submit_button(
            "Login",
            use_container_width=True,
        )

    if submitted:
        if password == AEROMAL_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False


# -------------------------
# 4. INPUT FILE READING
# -------------------------
def read_uploaded_file(uploaded_file):
    """Read CSV or Excel input into a DataFrame."""
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)

        if filename.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)

        raise ValueError("Unsupported file format.")

    except Exception as exc:
        raise ValueError(
            f"Unable to read {uploaded_file.name}: {exc}"
        ) from exc


def clean_dataframe(df):
    """Clean column names and normalize missing/non-numeric values."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    return df


def get_generation_columns(df):
    """Return supported generation columns present in the uploaded file."""
    return [
        column
        for column in GENERATION_COLUMNS
        if column in df.columns
    ]


def prepare_generation_data(df, generation_column):
    """Convert the selected generation column to a clean numeric array."""
    if generation_column not in df.columns:
        raise ValueError(
            f"Column '{generation_column}' was not found."
        )

    values = pd.to_numeric(
        df[generation_column],
        errors="coerce",
    )

    values = (
        values.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=float)
    )

    if len(values) == 0:
        raise ValueError("The selected generation column is empty.")

    if len(values) % BLOCKS_PER_DAY != 0:
        raise ValueError(
            f"Expected the number of rows to be divisible by "
            f"{BLOCKS_PER_DAY}. Received {len(values)} rows."
        )

    return values


# -------------------------
# 5. GENERAL PROFILE HELPERS
# -------------------------
def validate_smoothing_window(window, profile_length=BLOCKS_PER_DAY):
    """Ensure the Savitzky–Golay window is valid."""
    window = int(window)

    if window % 2 == 0:
        window += 1

    maximum = profile_length if profile_length % 2 else profile_length - 1
    window = min(window, maximum)

    if window < 5:
        window = 5

    return window


def smooth_profile(profile, window, polyorder=3):
    """Smooth a profile using a valid Savitzky–Golay window."""
    profile = np.asarray(profile, dtype=float)

    if len(profile) < 5:
        return profile.copy()

    window = validate_smoothing_window(
        window,
        profile_length=len(profile),
    )

    polyorder = min(polyorder, window - 2)

    return savgol_filter(
        profile,
        window_length=window,
        polyorder=polyorder,
    )


@st.cache_data(show_spinner=False)
def cached_percentile(power_tuple, days):
    """Calculate the 95th percentile for each of the 96 daily blocks."""
    power = np.asarray(power_tuple, dtype=float)
    matrix = power.reshape(int(days), BLOCKS_PER_DAY)

    return tuple(
        np.percentile(matrix, 95, axis=0).tolist()
    )


@st.cache_data(show_spinner=False)
def cached_best_shift(profile_tuple):
    """Find the shift with the lowest RMSE against the reversed profile."""
    profile = np.asarray(profile_tuple, dtype=float)

    best_shift = 0
    best_error = float("inf")

    for shift in range(len(profile)):
        shifted = np.roll(profile, -shift)
        symmetric = (profile + shifted[::-1]) / 2.0

        error = np.sqrt(
            np.mean((symmetric - profile) ** 2)
        )

        if error < best_error:
            best_error = error
            best_shift = shift

    return int(best_shift)


def create_symmetric_profile(profile):
    """
    Create a profile with a single rise and a single set.

    This retains the existing approach of enforcing a non-decreasing
    first half and mirroring it into the second half.
    """
    profile = np.asarray(profile, dtype=float).copy()
    profile = np.maximum(profile, 0)

    length = len(profile)
    midpoint = length // 2

    if length == 0:
        return profile

    max_value = float(np.max(profile))

    for index in range(1, midpoint):
        if profile[index] < profile[index - 1]:
            profile[index] = max_value

    profile[midpoint:] = profile[: length - midpoint][::-1]

    return profile


def shift_profile_blockwise(profile, shift):
    """
    Shift profile blocks without circular wrapping.

    Positive shift: move right and fill the beginning with zeros.
    Negative shift: move left and fill the end with zeros.
    """
    profile = np.asarray(profile, dtype=float)
    shift = int(shift)
    result = np.zeros_like(profile)

    if len(profile) == 0:
        return result

    if shift == 0:
        return profile.copy()

    if shift >= len(profile) or shift <= -len(profile):
        return result

    if shift > 0:
        result[shift:] = profile[:-shift]
    else:
        result[:shift] = profile[-shift:]

    return result


# -------------------------
# 6. CHART HELPER
# -------------------------
def make_chart(x, curves, title):
    """Build a reusable Plotly line chart."""
    figure = go.Figure()

    for curve_name, curve_values in curves.items():
        figure.add_trace(
            go.Scatter(
                x=x,
                y=curve_values,
                mode="lines",
                name=curve_name,
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Block",
        yaxis_title="Power (MW)",
        height=550,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=80, b=30),
    )

    return figure


# -------------------------
# 7. EXCEL REPORT STYLING
# -------------------------
def style_report_sheet(worksheet):
    """Apply consistent formatting to an Excel worksheet."""
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )
    header_font = Font(
        bold=True,
        color="FFFFFF",
    )

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    for column_cells in worksheet.columns:
        column_letter = get_column_letter(
            column_cells[0].column
        )

        maximum_length = max(
            len(str(cell.value or ""))
            for cell in column_cells
        )

        worksheet.column_dimensions[column_letter].width = min(
            maximum_length + 3,
            35,
        )


# -------------------------
# 8. APPLICATION ENTRY POINT
# -------------------------
def main():
    add_page_styling()

    if not authenticate():
        st.stop()

    st.title(APP_TITLE)
    st.caption(
        "Solar generation profile analysis and correction"
    )

    st.divider()

    st.info(
        "Upload your generation data to begin. "
        "The file must contain complete days of 96 blocks each."
    )

    uploaded_file = st.file_uploader(
        "Upload generation data",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX and XLS.",
    )

    if uploaded_file is None:
        st.stop()

    try:
        source_df = clean_dataframe(
            read_uploaded_file(uploaded_file)
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    available_columns = get_generation_columns(source_df)

    if not available_columns:
        st.error(
            "No supported generation column was found in this file."
        )
        st.write("Available columns:", source_df.columns.tolist())
        st.stop()

    generation_column = st.selectbox(
        "Select generation column",
        options=available_columns,
    )

    try:
        power = prepare_generation_data(
            source_df,
            generation_column,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    days = len(power) // BLOCKS_PER_DAY

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric("Total rows", f"{len(power):,}")
    metric_2.metric("Complete days", f"{days:,}")
    metric_3.metric("Blocks per day", BLOCKS_PER_DAY)

    st.divider()

    # The two mode renderers will be added in Part 2.
    mode = st.radio(
        "Select correction mode",
        options=["No Curtailment", "Curtailment"],
        horizontal=True,
    )

    st.session_state["source_df"] = source_df
    st.session_state["power"] = power
    st.session_state["days"] = days
    st.session_state["generation_column"] = generation_column
    st.session_state["uploaded_filename"] = uploaded_file.name

    st.warning(
        f"The UI foundation is ready. Selected mode: {mode}. "
        "The calculation and report sections are added in the next part."
    )


if __name__ == "__main__":
    main()
