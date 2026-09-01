import io

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
)


# ============================================================
# CONSTANTS
# ============================================================

BLOCKS_PER_DAY = 96

TIME_LABELS = pd.date_range(
    start="00:00",
    periods=BLOCKS_PER_DAY,
    freq="15min",
).strftime("%H:%M").tolist()

# Only this many daily rows are shown in the UI.
# The actual calculation still uses ALL complete days.
MAX_MATRIX_PREVIEW_ROWS = 100

# Maximum number of rows displayed in the uploaded-data preview.
MAX_DATA_PREVIEW_ROWS = 20


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e5e5e5;
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    "📊 96-Block Percentile Calculator"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Upload time-series data, select multiple columns, "
    "reshape the data into Days × 96 blocks, and calculate "
    "a selectable percentile for every 15-minute block."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FILE READER
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def read_uploaded_file(
    file_bytes,
    file_name,
):
    """
    Read CSV/XLSX/XLS file from bytes.

    Returns:
        pandas DataFrame
    """

    file_name = str(file_name).lower()

    try:

        if file_name.endswith(".csv"):

            # First attempt
            try:

                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    low_memory=False,
                )

            except UnicodeDecodeError:

                # Fallback for non-UTF8 CSV files
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding="latin1",
                    low_memory=False,
                )

        elif file_name.endswith(
            (".xlsx", ".xls")
        ):

            df = pd.read_excel(
                io.BytesIO(file_bytes)
            )

        else:

            raise ValueError(
                "Unsupported file format. "
                "Please upload CSV, XLSX, or XLS."
            )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to read the uploaded file: {exc}"
        ) from exc

    if df is None:

        raise ValueError(
            "The file could not be read."
        )

    if df.empty:

        raise ValueError(
            "The uploaded file contains no rows."
        )

    return df


# ============================================================
# COLUMN CLEANING
# ============================================================

def make_unique_columns(df):
    """
    Convert column names to strings and make duplicates unique.
    """

    result = df.copy()

    new_columns = []
    counts = {}

    for column in result.columns:

        name = str(column).strip()

        if not name:
            name = "Unnamed"

        if name not in counts:

            counts[name] = 1
            new_columns.append(name)

        else:

            counts[name] += 1

            new_columns.append(
                f"{name}_{counts[name]}"
            )

    result.columns = new_columns

    return result


# ============================================================
# NUMERIC COLUMN DETECTION
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def detect_numeric_columns(df):
    """
    Detect columns containing numeric values.

    The original dataframe is not modified.
    """

    numeric_columns = []

    for column in df.columns:

        try:

            numeric = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if numeric.notna().any():

                numeric_columns.append(
                    column
                )

        except Exception:

            continue

    return numeric_columns


# ============================================================
# COLUMN PROCESSING
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=20,
)
def prepare_column(
    series,
):
    """
    Convert one column to numeric.

    Blank / NaN / non-numeric values become zero.

    Returns a NumPy float array.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.fillna(0)

    return numeric.to_numpy(
        dtype=np.float64
    )


# ============================================================
# CALCULATE 96-BLOCK PROFILE
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=50,
)
def calculate_profile(
    values,
    percentile,
):
    """
    Calculate:

        values
           ↓
        Days × 96
           ↓
        percentile across days

    Returns:

        reshaped matrix
        percentile profile
        complete days
        incomplete values
    """

    if values is None:

        return None

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    total_values = values.size

    complete_days = (
        total_values // BLOCKS_PER_DAY
    )

    incomplete_values = (
        total_values % BLOCKS_PER_DAY
    )

    if complete_days < 1:

        return None

    usable_values = (
        complete_days
        * BLOCKS_PER_DAY
    )

    # Only complete days are used.
    trimmed = values[
        :usable_values
    ]

    # Days × 96
    reshaped = trimmed.reshape(
        complete_days,
        BLOCKS_PER_DAY,
    )

    # Percentile for each 15-minute block.
    percentile_profile = np.percentile(
        reshaped,
        percentile,
        axis=0,
    )

    return (
        reshaped,
        percentile_profile,
        complete_days,
        incomplete_values,
    )


# ============================================================
# RESULT DATAFRAME
# ============================================================

def create_result_dataframe(
    profiles,
    percentile,
):
    """
    Create a 96-row result dataframe.
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

    for column, profile in profiles.items():

        result[
            f"{column} P{percentile:g}"
        ] = profile

    return result


# ============================================================
# EXCEL EXPORT
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def create_excel_file(
    result_df,
):
    """
    Convert result dataframe to Excel bytes.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        result_df.to_excel(
            writer,
            index=False,
            sheet_name="Percentile",
        )

    return output.getvalue()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload Data")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help=(
        "Upload your 15-minute time-series data."
    ),
)


if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file to start."
    )

    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:

        st.error(
            "The uploaded file is empty."
        )

        st.stop()

    df = read_uploaded_file(
        file_bytes,
        uploaded_file.name,
    )

except Exception as exc:

    st.error(
        f"❌ {exc}"
    )

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

if df is None or df.empty:

    st.error(
        "The uploaded file contains no data."
    )

    st.stop()


# Make column names safe and unique.
df = make_unique_columns(df)


# Count completely empty columns BEFORE filling nulls.
empty_columns = [
    column
    for column in df.columns
    if df[column].isna().all()
]


# Remove completely empty columns.
if empty_columns:

    df = df.drop(
        columns=empty_columns
    )


if df.empty or len(df.columns) == 0:

    st.error(
        "The file contains no usable columns."
    )

    st.stop()


# ============================================================
# NULL HANDLING
# ============================================================

null_count = int(
    df.isna().sum().sum()
)


# IMPORTANT:
# Null values become zero.
#
# Existing zeroes remain zero.
#
# Zero is NOT removed from the calculation.

df = df.fillna(0)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"✅ File loaded successfully: "
    f"**{uploaded_file.name}**"
)


if null_count > 0:

    st.info(
        f"{null_count:,} blank/null value(s) "
        "were replaced with 0."
    )


# ============================================================
# FILE INFORMATION
# ============================================================

numeric_columns = detect_numeric_columns(
    df
)


info_col1, info_col2, info_col3, info_col4 = (
    st.columns(4)
)


with info_col1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )


with info_col2:

    st.metric(
        "Columns",
        f"{len(df.columns):,}",
    )


with info_col3:

    st.metric(
        "Numeric Columns",
        f"{len(numeric_columns):,}",
    )


with info_col4:

    st.metric(
        "Complete 96-Block Days",
        f"{len(df) // BLOCKS_PER_DAY:,}",
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "Preview uploaded data",
    expanded=False,
):

    st.dataframe(
        df.head(
            MAX_DATA_PREVIEW_ROWS
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# NUMERIC COLUMN CHECK
# ============================================================

if not numeric_columns:

    st.error(
        "❌ No numeric columns were detected."
    )

    st.write(
        "Please upload a file containing "
        "numeric time-series data."
    )

    st.stop()


# ============================================================
# COLUMN SELECTION
# ============================================================

st.subheader(
    "2. Select Data Columns"
)


selected_columns = st.multiselect(
    "Select one or more columns",
    options=numeric_columns,
    default=[],
    help=(
        "Multiple columns can be selected. "
        "Each column gets its own 96-block percentile profile."
    ),
)


if not selected_columns:

    st.info(
        "Select at least one column to continue."
    )

    st.stop()


st.write(
    f"**{len(selected_columns)} column(s) selected**"
)


st.caption(
    ", ".join(selected_columns)
)


# ============================================================
# PERCENTILE
# ============================================================

st.subheader(
    "3. Percentile Selection"
)


percentile_col1, percentile_col2 = (
    st.columns([4, 1])
)


with percentile_col1:

    percentile = st.slider(
        "Select percentile",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=0.5,
        help=(
            "The percentile is calculated independently "
            "for each 15-minute block."
        ),
    )


with percentile_col2:

    st.metric(
        "Selected Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# PREPARE SELECTED COLUMNS
# ============================================================

st.subheader(
    "4. Data Validation"
)


column_data = {}
validation_rows = []


for column in selected_columns:

    try:

        values = prepare_column(
            df[column]
        )

        result = calculate_profile(
            values,
            percentile,
        )

    except Exception as exc:

        validation_rows.append(
            {
                "Column": column,
                "Rows": len(df),
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": len(df),
                "Status": f"Error: {exc}",
            }
        )

        continue


    if result is None:

        validation_rows.append(
            {
                "Column": column,
                "Rows": len(values),
                "Complete Days": 0,
                "Usable Values": 0,
                "Incomplete Values": len(values),
                "Status": "Insufficient data",
            }
        )

        continue


    reshaped, profile, days, remainder = result


    column_data[column] = {
        "values": values,
        "reshaped": reshaped,
        "profile": profile,
        "days": days,
        "remainder": remainder,
    }


    validation_rows.append(
        {
            "Column": column,
            "Rows": len(values),
            "Complete Days": days,
            "Usable Values": days * BLOCKS_PER_DAY,
            "Incomplete Values": remainder,
            "Status": "Valid",
        }
    )


# ============================================================
# VALIDATION TABLE
# ============================================================

validation_df = pd.DataFrame(
    validation_rows
)


st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INCOMPLETE DAY WARNING
# ============================================================

incomplete_columns = [
    column
    for column, data in column_data.items()
    if data["remainder"] > 0
]


if incomplete_columns:

    for column in incomplete_columns:

        remainder = column_data[
            column
        ]["remainder"]

        st.warning(
            f"**{column}:** {remainder} row(s) "
            "after the last complete 96-block day "
            "will be excluded."
        )

else:

    st.success(
        "All selected columns contain complete 96-block days."
    )


# ============================================================
# STOP IF NOTHING CAN BE PROCESSED
# ============================================================

if not column_data:

    st.error(
        "None of the selected columns contains "
        "at least 96 rows."
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)


with summary_col1:

    st.metric(
        "Columns Processed",
        len(column_data),
    )


with summary_col2:

    st.metric(
        "Blocks / Day",
        BLOCKS_PER_DAY,
    )


with summary_col3:

    min_days = min(
        data["days"]
        for data in column_data.values()
    )

    st.metric(
        "Complete Days",
        f"{min_days:,}",
    )


with summary_col4:

    st.metric(
        "Percentile",
        f"P{percentile:g}",
    )


# ============================================================
# CREATE RESULT
# ============================================================

profiles = {
    column: data["profile"]
    for column, data in column_data.items()
}


result_df = create_result_dataframe(
    profiles,
    percentile,
)


# ============================================================
# RESULT TABLE
# ============================================================

st.subheader(
    "5. 96-Block Percentile Result"
)


st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
    height=550,
)


# ============================================================
# GRAPH
# ============================================================

st.subheader(
    "6. Percentile Profile"
)


chart_df = result_df.set_index(
    "Time"
)


chart_columns = [
    column
    for column in result_df.columns
    if column not in [
        "Block",
        "Time",
    ]
]


if chart_columns:

    st.line_chart(
        chart_df[chart_columns],
        use_container_width=True,
    )


# ============================================================
# INDIVIDUAL COLUMN GRAPH
# ============================================================

with st.expander(
    "View individual column profile",
    expanded=False,
):

    graph_column = st.selectbox(
        "Select column",
        options=list(
            column_data.keys()
        ),
        key="individual_graph_column",
    )


    graph_result_column = (
        f"{graph_column} P{percentile:g}"
    )


    individual_chart = (
        result_df
        .set_index("Time")
        [[graph_result_column]]
    )


    st.line_chart(
        individual_chart,
        use_container_width=True,
    )


# ============================================================
# DAYS × 96 MATRIX
# ============================================================

st.subheader(
    "7. Reshaped Days × 96 Data"
)


matrix_column = st.selectbox(
    "Select column to view its Days × 96 matrix",
    options=list(
        column_data.keys()
    ),
    key="daily_matrix_column",
)


matrix = column_data[
    matrix_column
]["reshaped"]


total_matrix_days = matrix.shape[0]


st.caption(
    f"Full matrix shape: "
    f"{total_matrix_days:,} days × "
    f"{BLOCKS_PER_DAY} blocks"
)


# ------------------------------------------------------------
# IMPORTANT FREEZE PROTECTION
# ------------------------------------------------------------
#
# Do NOT render thousands/millions of rows in Streamlit.
# Calculation uses the complete matrix, but UI only displays
# a limited preview.
#

rows_to_show = min(
    total_matrix_days,
    MAX_MATRIX_PREVIEW_ROWS,
)


preview_matrix = matrix[
    :rows_to_show
]


block_columns = [
    f"Block_{i:02d}"
    for i in range(
        1,
        BLOCKS_PER_DAY + 1,
    )
]


matrix_preview_df = pd.DataFrame(
    preview_matrix,
    columns=block_columns,
)


matrix_preview_df.insert(
    0,
    "Day",
    np.arange(
        1,
        rows_to_show + 1,
    ),
)


st.dataframe(
    matrix_preview_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)


if total_matrix_days > MAX_MATRIX_PREVIEW_ROWS:

    st.info(
        f"Only the first {MAX_MATRIX_PREVIEW_ROWS:,} "
        f"days are displayed to keep the page responsive. "
        f"The percentile calculation uses all "
        f"{total_matrix_days:,} complete days."
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "8. Download Result"
)


download_col1, download_col2 = (
    st.columns(2)
)


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

with download_col2:

    try:

        excel_data = create_excel_file(
            result_df
        )

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
            f"Excel download unavailable: {exc}"
        )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Calculation methodology",
    expanded=False,
):

    st.markdown(
        """
        ### Processing logic

        **1. Upload**

        CSV, XLSX and XLS files are supported.

        **2. Null handling**

        Blank / NaN values are replaced with **0**.

        Existing zeroes remain zero and are included.

        **3. Column selection**

        Multiple numeric columns can be selected.

        **4. Reshape**

        Each selected column is reshaped into:

        ```
        Days × 96
        ```

        Every row represents one day.

        Every column represents one 15-minute block.

        **5. Incomplete day**

        If the number of rows is not exactly divisible by 96,
        the incomplete final portion is excluded.

        **6. Percentile**

        The selected percentile is calculated independently
        for each of the 96 blocks:

        ```
        np.percentile(
            daily_matrix,
            percentile,
            axis=0
        )
        ```

        **7. Output**

        The result always contains exactly 96 blocks:

        ```
        Block 1  → 00:00
        Block 2  → 00:15
        Block 3  → 00:30
        ...
        Block 95 → 23:30
        Block 96 → 23:45
        ```
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "96-Block Percentile Calculator | "
    "15-minute resolution | "
    "Multiple columns | "
    "Null values treated as zero"
)
````
