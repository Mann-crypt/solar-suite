import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from io import BytesIO


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CMV Profile Generator",
    page_icon="☀️",
    layout="wide",
)


# ==========================================================
# FUNCTIONS
# ==========================================================

def clean_dataframe(df, min_data_ratio=0.30):
    """
    Apply the same cleaning logic used in the original script.
    """

    df = df.copy()

    # ------------------------------------------------------
    # Detect Date column
    # ------------------------------------------------------

    date_col = None

    for col in df.columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("_", " ")
        )

        if normalized in ["date", "datetime", "timestamp"]:
            date_col = col
            break

    # ------------------------------------------------------
    # Set Date as index if available
    # ------------------------------------------------------

    if date_col is not None:

        date_values = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        if date_values.notna().sum() > 0:

            df[date_col] = date_values
            df = df.set_index(date_col)

    # ------------------------------------------------------
    # Convert possible numeric columns
    # ------------------------------------------------------

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ------------------------------------------------------
    # Drop columns having insufficient data
    # ------------------------------------------------------

    min_data_requirement = int(
        len(df) * min_data_ratio
    )

    df = df.dropna(
        axis=1,
        thresh=min_data_requirement
    )

    # ------------------------------------------------------
    # Remove unwanted columns
    #
    # > 1200
    # OR
    # maximum < 800
    # ------------------------------------------------------

    drop_above = []

    for col in df.columns:

        values = df[col].dropna()

        if len(values) == 0:
            drop_above.append(col)
            continue

        if (
            (values > 1200).any()
            or values.max() < 800
        ):
            drop_above.append(col)

    df = df.drop(
        columns=drop_above,
        errors="ignore"
    )

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    df = df.fillna(0)

    # ------------------------------------------------------
    # Remove constant non-zero values appearing
    # for more than 2 consecutive blocks
    # ------------------------------------------------------

    for col in df.columns:

        s = df[col]

        if not pd.api.types.is_numeric_dtype(s):
            continue

        group = (
            s != s.shift()
        ).cumsum()

        group_size = group.map(
            group.value_counts()
        )

        df[col] = s.mask(
            (group_size > 2) & (s != 0),
            0
        )

    # ------------------------------------------------------
    # Remove completely zero columns
    # ------------------------------------------------------

    zero_cols = df.columns[
        (df == 0).all()
    ]

    df = df.drop(
        columns=zero_cols
    )

    return df


def calculate_percentile_profiles(
    df,
    percentile=95
):
    """
    Reshape data into days x 96 blocks and
    calculate percentile profile for every column.
    """

    if len(df) < 96:
        raise ValueError(
            "At least 96 rows are required."
        )

    days = len(df) // 96

    usable_length = days * 96

    df = df.iloc[
        :usable_length
    ]

    result = {}

    for col in df.columns:

        data = df[col].to_numpy(
            dtype=float
        )

        reshaped = data.reshape(
            days,
            96
        )

        perc_values = np.percentile(
            reshaped,
            percentile,
            axis=0
        )

        result[col] = perc_values

    return pd.DataFrame(result)


def remove_repeated_values(df):
    """
    Convert consecutive repeated percentile
    values into zero, matching the original logic.
    """

    df = df.copy()

    for col in df.columns:

        s = df[col]

        group = (
            s != s.shift()
        ).cumsum()

        group_size = group.map(
            group.value_counts()
        )

        df[col] = s.mask(
            (group_size > 2) & (s != 0),
            0
        )

    return df


def smooth_profile(
    average,
    first_window=15,
    second_window=7,
    polyorder=3,
    threshold=4.9
):
    """
    Apply the original two-stage
    Savitzky-Golay smoothing.
    """

    a = np.asarray(
        average,
        dtype=float
    )

    # First smoothing
    s = savgol_filter(
        a,
        window_length=first_window,
        polyorder=polyorder
    )

    s = np.clip(
        s,
        0,
        None
    )

    # Remove very small values
    s = np.where(
        s < threshold,
        0,
        s
    )

    # Second smoothing
    s = savgol_filter(
        s,
        window_length=second_window,
        polyorder=polyorder
    )

    s = np.clip(
        s,
        0,
        None
    )

    return s


def make_percentile_chart(
    percentile_df,
    selected_columns
):
    """
    Plot percentile profile of every column.
    """

    x = np.arange(1, 97)

    fig = go.Figure()

    for col in percentile_df.columns:

        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentile_df[col],
                mode="lines",
                name=str(col),
                visible=True
                if col in selected_columns
                else "legendonly",
            )
        )

    fig.update_layout(
        height=600,
        template="streamlit",
        hovermode="x unified",
        xaxis_title="15 Minute Block",
        yaxis_title="95th Percentile Power",
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


def make_final_chart(
    average,
    smooth
):
    """
    Plot average and smoothed profile.
    """

    x = np.arange(1, 97)

    fig = go.Figure([
        go.Scatter(
            x=x,
            y=average,
            name="Average 95th Percentile",
            line=dict(
                width=3
            )
        ),
        go.Scatter(
            x=x,
            y=smooth,
            name="Smooth Profile",
            line=dict(
                width=3,
                color="#00c6ff"
            )
        )
    ])

    fig.update_layout(
        height=550,
        template="streamlit",
        hovermode="x unified",
        xaxis_title="15 Minute Block",
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
# TITLE
# ==========================================================

st.title("☀️ CMV Profile Generator")

st.caption(
    "Generate a 95th-percentile CMV profile, "
    "select columns for averaging, and apply "
    "Savitzky-Golay smoothing."
)


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CMV Excel Workbook",
    type=["xlsx"],
)


if uploaded_file is None:

    st.info(
        "Upload your CMV profile Excel workbook "
        "to begin."
    )

    st.stop()


# ==========================================================
# READ EXCEL
# ==========================================================

try:

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheets = excel_file.sheet_names

    if "CMV_Curve" in sheets:
        default_sheet = sheets.index(
            "CMV_Curve"
        )
    else:
        default_sheet = 0

    sheet_name = st.selectbox(
        "Select CMV sheet",
        sheets,
        index=default_sheet,
    )

    df_original = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


# ==========================================================
# ORIGINAL DATA
# ==========================================================

with st.expander(
    "View Original Data"
):

    st.dataframe(
        df_original.head(20),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# CLEAN DATA
# ==========================================================

try:

    df = clean_dataframe(
        df_original
    )

except Exception as e:

    st.error(
        f"Data cleaning failed: {e}"
    )

    st.stop()


# ==========================================================
# DATA SUMMARY
# ==========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    len(df)
)

col2.metric(
    "Available Columns",
    len(df.columns)
)

col3.metric(
    "Available Days",
    len(df) // 96
)


# ==========================================================
# DATE INDEX
# ==========================================================

if isinstance(
    df.index,
    pd.DatetimeIndex
):

    st.success(
        "Date column detected and set as index."
    )


# ==========================================================
# CALCULATE PERCENTILE
# ==========================================================

try:

    percentile_df = (
        calculate_percentile_profiles(
            df,
            percentile=95
        )
    )

except Exception as e:

    st.error(
        f"Unable to calculate percentile profiles: {e}"
    )

    st.stop()


# ==========================================================
# REPEATED VALUE CLEANING
# ==========================================================

percentile_df = (
    remove_repeated_values(
        percentile_df
    )
)


# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.divider()

st.subheader(
    "📊 Select Columns for Final Average"
)

st.caption(
    "The graph shows the 95th-percentile profile "
    "of every available column. Select the columns "
    "you want to include in the final average."
)


selected_columns = st.multiselect(
    "Columns included in final average",
    options=list(
        percentile_df.columns
    ),
    default=list(
        percentile_df.columns
    ),
)


# ==========================================================
# PERCENTILE GRAPH
# ==========================================================

st.subheader(
    "95th Percentile Profile by Column"
)

fig = make_percentile_chart(
    percentile_df,
    selected_columns
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================================
# SELECTION SUMMARY
# ==========================================================

selected_count = len(
    selected_columns
)

excluded_columns = [
    col
    for col in percentile_df.columns
    if col not in selected_columns
]

col1, col2 = st.columns(2)

col1.metric(
    "Columns Included",
    selected_count
)

col2.metric(
    "Columns Excluded",
    len(excluded_columns)
)


if excluded_columns:

    st.warning(
        "Excluded columns: "
        + ", ".join(
            map(
                str,
                excluded_columns
            )
        )
    )


# ==========================================================
# VALIDATE SELECTION
# ==========================================================

if not selected_columns:

    st.error(
        "Please select at least one column "
        "for the final average."
    )

    st.stop()


# ==========================================================
# FINAL AVERAGE
# ==========================================================

df_selected = percentile_df[
    selected_columns
].copy()

average = (
    df_selected.mean(
        axis=1
    )
)


# ==========================================================
# SMOOTHING SETTINGS
# ==========================================================

st.divider()

st.subheader(
    "⚙️ Smoothing Settings"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    first_window = st.number_input(
        "First Window",
        min_value=5,
        max_value=31,
        value=15,
        step=2,
    )

with col2:

    second_window = st.number_input(
        "Second Window",
        min_value=5,
        max_value=31,
        value=7,
        step=2,
    )

with col3:

    polyorder = st.number_input(
        "Polynomial Order",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

with col4:

    threshold = st.number_input(
        "Zero Threshold",
        min_value=0.0,
        max_value=50.0,
        value=4.9,
        step=0.1,
    )


# ==========================================================
# VALIDATE SAVITZKY-GOLAY SETTINGS
# ==========================================================

if (
    first_window <= polyorder
    or second_window <= polyorder
):

    st.error(
        "Window length must be greater than "
        "the polynomial order."
    )

    st.stop()


if first_window % 2 == 0:

    st.error(
        "First Window must be an odd number."
    )

    st.stop()


if second_window % 2 == 0:

    st.error(
        "Second Window must be an odd number."
    )

    st.stop()


# ==========================================================
# SMOOTH
# ==========================================================

try:

    smooth = smooth_profile(
        average,
        first_window=first_window,
        second_window=second_window,
        polyorder=polyorder,
        threshold=threshold,
    )

except Exception as e:

    st.error(
        f"Smoothing failed: {e}"
    )

    st.stop()


# ==========================================================
# FINAL GRAPH
# ==========================================================

st.divider()

st.subheader(
    "📈 Final Generated Curve"
)

fig = make_final_chart(
    average,
    smooth
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================================
# FINAL RESULT
# ==========================================================

result = pd.DataFrame({
    "Block": np.arange(1, 97),
    "Average": average,
    "Smooth Profile": smooth,
})

st.subheader(
    "Generated Curve"
)

st.dataframe(
    result,
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# DOWNLOAD EXCEL
# ==========================================================

output = BytesIO()

with pd.ExcelWriter(
    output,
    engine="openpyxl"
) as writer:

    # Original percentile profiles
    percentile_df.to_excel(
        writer,
        sheet_name="Percentile Profiles",
        index=False
    )

    # Final result
    result.to_excel(
        writer,
        sheet_name="CMV_Curve",
        index=False
    )

    # Selected columns
    df_selected.to_excel(
        writer,
        sheet_name="Selected Columns",
        index=False
    )

output.seek(0)


st.download_button(
    "⬇️ Download Generated CMV Excel",
    data=output,
    file_name="CMV_Profile_Generated.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True,
)
