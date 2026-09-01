````python
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="96-Block Percentile Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

BLOCKS_PER_DAY = 96
MIN_VALUES_REQUIRED = 96

TIME_LABELS = pd.date_range(
    start="00:00",
    periods=BLOCKS_PER_DAY,
    freq="15min",
).strftime("%H:%M").tolist()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .info-box {
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 96-Block Percentile Calculator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload 15-minute time-series data, select one or more columns, "
    "reshape the data into Days × 96 blocks, and calculate a "
    "percentile profile for every time block."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


@st.cache_data(show_spinner=False)
def read_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Read CSV or Excel file from uploaded bytes.
    """

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    extension = file_name.lower().split(".")[-1]

    try:

        if extension == "csv":

            # Try normal CSV first
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes)
                )

            except Exception:

                # Fallback for common Indian/Excel CSV formats
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding="latin1",
                )

        elif extension in ("xlsx", "xls"):

            df = pd.read_excel(
                io.BytesIO(file_bytes)
            )

        else:

            raise ValueError(
                "Unsupported file type. "
                "Please upload CSV, XLSX, or XLS."
            )

    except Exception as exc:

        raise ValueError(
            f"Could not read the uploaded file: {exc}"
        ) from exc

    if df is None:
        raise ValueError("The file could not be loaded.")

    if df.empty:
        raise ValueError(
            "The uploaded file contains no rows."
        )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all",
    )

    if df.empty or len(df.columns) == 0:
        raise ValueError(
            "The uploaded file contains no usable columns."
        )

    # Make column names strings
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Handle duplicate column names safely
    df = make_unique_columns(df)

    return df


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make duplicate column names unique.
    Example:
        Power, Power
    becomes:
        Power, Power_2
    """

    columns = []
    counts = {}

    for column in df.columns:

        if column not in counts:

            counts[column] = 1
            columns.append(column)

        else:

            counts[column] += 1

            columns.append(
                f"{column}_{counts[column]}"
            )

    result = df.copy()
    result.columns = columns

    return result


def get_numeric_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Detect columns containing at least one numeric value.
    """

    numeric_columns = []

    for column in df.columns:

        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric_series.notna().any():

            numeric_columns.append(column)

    return numeric_columns


def convert_to_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a series to numeric safely.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def process_column(
    series: pd.Series,
    percentile: float,
) -> Optional[Dict]:

    """
    Convert one column into Days × 96 and calculate
    percentile across the days.

    Returns None if fewer than 96 valid values exist.
    """

    numeric = convert_to_numeric(series)

    valid_values = (
        numeric
        .dropna()
        .reset_index(drop=True)
    )

    valid_count = len(valid_values)

    complete_days = (
        valid_count // BLOCKS_PER_DAY
    )

    remainder = (
        valid_count % BLOCKS_PER_DAY
    )

    usable_values = (
        complete_days * BLOCKS_PER_DAY
    )

    if complete_days < 1:

        return None

    trimmed = valid_values.iloc[
        :usable_values
    ].to_numpy(
        dtype=float
    )

    try:

        daily_matrix = trimmed.reshape(
            complete_days,
            BLOCKS_PER_DAY,
        )

    except ValueError:

        return None

    percentile_values = np.percentile(
        daily_matrix,
        percentile,
        axis=0,
    )

    return {
        "valid_count": valid_count,
        "complete_days": complete_days,
        "remainder": remainder,
        "usable_values": usable_values,
        "daily_matrix": daily_matrix,
        "percentile_values": percentile_values,
    }


def create_result_dataframe(
    processed: Dict[str, Dict],
    percentile: float,
) -> pd.DataFrame:

    """
    Create combined 96-block result.
    """

    result = pd.DataFrame(
        {
            "Block": np.arange(
                1,
                BLOCKS_PER_DAY + 1,
            ),

            "Time": TIME_LABELS,
        }
    )

    for column, data in processed.items():

        result[
            f"{column} P{percentile:g}"
        ] = data["percentile_values"]

    return result


def create_daily_matrix(
    daily_matrix: np.ndarray,
) -> pd.DataFrame:

    """
    Convert numpy Days × 96 matrix to dataframe.
    """

    block_columns = [
        f"Block_{i:02d}"
        for i in range(
            1,
            BLOCKS_PER_DAY + 1,
        )
    ]

    result = pd.DataFrame(
        daily_matrix,
        columns=block_columns,
    )

    result.insert(
        0,
        "Day",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


def dataframe_to_excel(
    df: pd.DataFrame,
) -> bytes:

    """
    Convert dataframe to an Excel file in memory.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Percentile_Profile",
        )

    return output.getvalue()


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Data</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help=(
        "Upload a file containing your 15-minute "
        "time-series data."
    ),
)

if uploaded_file is None:

    st.info(
        "Please upload a CSV or Excel file to begin."
    )

    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

    df = read_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as exc:

    st.error(
        f"❌ File reading failed: {exc}"
    )

    st.stop()


# ============================================================
# FILE SUCCESS
# ============================================================

st.success(
    f"✅ File loaded successfully: "
    f"**{uploaded_file.name}**"
)


# ============================================================
# FILE INFORMATION
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with info2:

    st.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

numeric_columns = get_numeric_columns(df)

with info3:

    st.metric(
        "Numeric columns",
        f"{len(numeric_columns):,}",
    )


# ============================================================
# PREVIEW
# ============================================================

with st.expander(
    "Preview uploaded data",
    expanded=False,
):

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CHECK NUMERIC COLUMNS
# ============================================================

if not numeric_columns:

    st.error(
        "❌ No numeric columns were detected."
    )

    st.info(
        "At least one column must contain numeric "
        "time-series values."
    )

    st.stop()


# ============================================================
# COLUMN SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">2. Select Data Columns</div>',
    unsafe_allow_html=True,
)

selected_columns = st.multiselect(
    "Select one or more columns",
    options=numeric_columns,
    default=[],
    help=(
        "Select multiple columns if you want to calculate "
        "a separate percentile profile for each column."
    ),
)

if not selected_columns:

    st.info(
        "Select at least one column to continue."
    )

    st.stop()


# ============================================================
# SELECTED COLUMN SUMMARY
# ============================================================

st.write(
    f"**{len(selected_columns)} column(s) selected:** "
    + ", ".join(selected_columns)
)


# ============================================================
# PERCENTILE SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">3. Select Percentile</div>',
    unsafe_allow_html=True,
)

percentile_col1, percentile_col2 = st.columns(
    [3, 1]
)

with percentile_col1:

    percentile = st.slider(
        "Percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
        help=(
            "Choose the percentile to calculate for "
            "each of the 96 time blocks."
        ),
    )

with percentile_col2:

    st.metric(
        "Selected",
        f"P{percentile:g}",
    )


# ============================================================
# PROCESS SELECTED COLUMNS
# ============================================================

st.markdown(
    '<div class="section-title">4. Data Validation</div>',
    unsafe_allow_html=True,
)

processed_data = {}

validation_rows = []

for column in selected_columns:

    result = process_column(
        df[column],
        percentile,
    )

    numeric_series = convert_to_numeric(
        df[column]
    )

    valid_count = int(
        numeric_series.notna().sum()
    )

    complete_days = (
        valid_count // BLOCKS_PER_DAY
    )

    remainder = (
        valid_count % BLOCKS_PER_DAY
    )

    usable_values = (
        complete_days * BLOCKS_PER_DAY
    )

    if result is not None:

        status = "✅ Valid"

        processed_data[column] = result

    else:

        status = "❌ Insufficient data"

    validation_rows.append(
        {
            "Column": column,
            "Valid Values": valid_count,
            "Complete Days": complete_days,
            "Usable Values": usable_values,
            "Incomplete Values": remainder,
            "Status": status,
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


# ============================================================
# VALIDATION TABLE
# ============================================================

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INVALID COLUMNS
# ============================================================

invalid_columns = [
    column
    for column in selected_columns
    if column not in processed_data
]


if invalid_columns:

    st.warning(
        "The following columns do not contain enough "
        "valid values for one complete 96-block day "
        "and will be excluded:"
    )

    for column in invalid_columns:

        st.write(
            f"- `{column}`"
        )


# ============================================================
# NO VALID COLUMNS
# ============================================================

if not processed_data:

    st.error(
        "None of the selected columns contains at least "
        "96 valid values."
    )

    st.stop()


# ============================================================
# INCOMPLETE DAY WARNINGS
# ============================================================

columns_with_incomplete_days = []

for column, data in processed_data.items():

    if data["remainder"] > 0:

        columns_with_incomplete_days.append(
            column
        )


if columns_with_incomplete_days:

    st.warning(
        "Incomplete values detected. "
        "The incomplete final day is excluded from "
        "the percentile calculation."
    )


# ============================================================
# CALCULATE FINAL RESULT
# ============================================================

result_df = create_result_dataframe(
    processed_data,
    percentile,
)


# ============================================================
# RESULT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">5. Percentile Result</div>',
    unsafe_allow_html=True,
)

summary1, summary2, summary3, summary4 = st.columns(4)

with summary1:

    st.metric(
        "Columns processed",
        len(processed_data),
    )

with summary2:

    st.metric(
        "96-block profiles",
        len(processed_data),
    )

with summary3:

    st.metric(
        "Blocks/day",
        BLOCKS_PER_DAY,
    )

with summary4:

    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# RESULT TABLE
# ============================================================

st.markdown(
    f"### P{percentile:g} Profile"
)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=550,
)


# ============================================================
# PROFILE GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">6. 96-Block Percentile Profile</div>',
    unsafe_allow_html=True,
)

plot_df = result_df.set_index(
    "Time"
)

plot_columns = [
    column
    for column in result_df.columns
    if column not in [
        "Block",
        "Time",
    ]
]

if plot_columns:

    st.line_chart(
        plot_df[plot_columns],
        use_container_width=True,
    )


# ============================================================
# INDIVIDUAL GRAPHS
# ============================================================

with st.expander(
    "View individual column profiles",
    expanded=False,
):

    for column in processed_data:

        result_column = (
            f"{column} P{percentile:g}"
        )

        st.markdown(
            f"#### {column}"
        )

        individual_df = result_df.set_index(
            "Time"
        )[[result_column]]

        st.line_chart(
            individual_df,
            use_container_width=True,
        )


# ============================================================
# DAILY MATRIX VIEWER
# ============================================================

st.markdown(
    '<div class="section-title">7. Days × 96 Matrix</div>',
    unsafe_allow_html=True,
)

matrix_column = st.selectbox(
    "Select a column to view its reshaped matrix",
    options=list(processed_data.keys()),
    key="matrix_column",
)

matrix_data = processed_data[
    matrix_column
]["daily_matrix"]

st.caption(
    f"Shape: "
    f"{matrix_data.shape[0]:,} days × "
    f"{matrix_data.shape[1]} blocks"
)

daily_matrix_df = create_daily_matrix(
    matrix_data
)

st.dataframe(
    daily_matrix_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)


# ============================================================
# DOWNLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">8. Download Result</div>',
    unsafe_allow_html=True,
)

download_col1, download_col2 = st.columns(2)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

csv_data = result_df.to_csv(
    index=False
).encode("utf-8")

with download_col1:

    st.download_button(
        label=(
            f"⬇️ Download P{percentile:g} CSV"
        ),
        data=csv_data,
        file_name=(
            f"96_block_percentile_P"
            f"{percentile:g}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ------------------------------------------------------------
# EXCEL
# ------------------------------------------------------------

try:

    excel_data = dataframe_to_excel(
        result_df
    )

    with download_col2:

        st.download_button(
            label=(
                f"⬇️ Download P{percentile:g} Excel"
            ),
            data=excel_data,
            file_name=(
                f"96_block_percentile_P"
                f"{percentile:g}.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

except Exception as exc:

    st.warning(
        f"Excel download is unavailable: {exc}"
    )


# ============================================================
# CALCULATION DETAILS
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.markdown(
        f"""
        ### How the calculation works

        Each selected column is processed independently.

        **1. Numeric conversion**

        Non-numeric values are treated as missing values.

        **2. Valid value extraction**

        Only valid numeric values are used.

        **3. Complete-day calculation**

        Each day requires exactly **96 values**.

        ```
        Complete Days = Valid Values // 96
        ```

        **4. Reshaping**

        The usable data is reshaped into:

        ```
        Days × 96
        ```

        **5. Percentile calculation**

        For each of the 96 blocks:

        ```
        np.percentile(
            daily_matrix,
            {percentile:g},
            axis=0
        )
        ```

        This means Block 1 is calculated from Block 1
        across all days, Block 2 from Block 2 across all
        days, and so on.

        **6. Time mapping**

        The 96 blocks represent:

        ```
        00:00
        00:15
        00:30
        ...
        23:30
        23:45
        ```

        If the number of valid values is not divisible by
        96, the incomplete final day is excluded.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "96-Block Percentile Calculator • "
    "15-minute resolution • "
    "Multiple columns supported"
)
````
