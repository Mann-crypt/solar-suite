import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from io import BytesIO


# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="CMV Curve Generator",
    page_icon="☀️",
    layout="wide",
)


# ==========================================================
# FUNCTIONS
# ==========================================================

def find_date_column(df):
    """Find a likely Date column."""

    candidates = [
        "Date",
        "date",
        "Datetime",
        "DateTime",
        "Timestamp",
        "Time",
    ]

    for col in df.columns:
        if str(col).strip() in candidates:
            return col

    # Try detecting datetime-like columns
    for col in df.columns:
        try:
            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            if converted.notna().mean() > 0.7:
                return col

        except Exception:
            pass

    return None


def clean_data(
    df,
    min_data_requirement=0.30,
    upper_limit=1200,
    lower_limit=800,
):
    """Apply the original cleaning logic."""

    df = df.copy()

    # ------------------------------------------------------
    # Remove columns with insufficient data
    # ------------------------------------------------------

    threshold = int(
        len(df) * min_data_requirement
    )

    df = df.dropna(
        axis=1,
        thresh=threshold,
    )

    # ------------------------------------------------------
    # Remove invalid columns
    # > 1200 anywhere
    # Maximum below 800
    # ------------------------------------------------------

    drop_columns = []

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if (
            (values > upper_limit).any()
            or values.max() < lower_limit
        ):
            drop_columns.append(col)

    df = df.drop(
        columns=drop_columns,
        errors="ignore",
    )

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    df = df.fillna(0)

    # ------------------------------------------------------
    # Convert constant non-zero runs > 2 blocks to zero
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
            0,
        )

    # ------------------------------------------------------
    # Remove completely zero columns
    # ------------------------------------------------------

    zero_columns = df.columns[
        (df == 0).all()
    ]

    df = df.drop(
        columns=zero_columns
    )

    return df, drop_columns, list(zero_columns)


def calculate_percentiles(
    df,
    blocks_per_day=96,
):
    """Calculate 95th percentile profile for every column."""

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        raise ValueError(
            "No numeric generation columns found."
        )

    days = len(numeric_df) // blocks_per_day

    if days < 1:
        raise ValueError(
            "Not enough data for 96-block daily reshaping."
        )

    usable_length = (
        days * blocks_per_day
    )

    numeric_df = numeric_df.iloc[
        :usable_length
    ]

    result = {}

    for col in numeric_df.columns:

        data = numeric_df[col].to_numpy(
            dtype=float
        )

        reshaped = data.reshape(
            days,
            blocks_per_day,
        )

        percentile = np.percentile(
            reshaped,
            95,
            axis=0,
        )

        result[col] = percentile

    return pd.DataFrame(result)


def remove_repeated_values(
    df
):
    """Replace consecutive identical percentile values with zero."""

    df = df.copy()

    for col in df.columns:

        values = df[col]

        df[col] = values.where(
            values.diff().fillna(1) != 0,
            0,
        )

    return df


def smooth_profile(
    average,
):
    """Apply the original smoothing logic."""

    average = np.asarray(
        average,
        dtype=float,
    )

    # First smoothing
    smooth = savgol_filter(
        average,
        window_length=15,
        polyorder=3,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    smooth = np.where(
        smooth < 4.9,
        0,
        smooth,
    )

    # Second smoothing
    smooth = savgol_filter(
        smooth,
        window_length=7,
        polyorder=3,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    return smooth


def create_percentile_chart(
    percentile_df,
    dropped_columns=None,
):
    """Interactive percentile chart with hover highlighting."""

    dropped_columns = dropped_columns or []

    fig = go.Figure()

    x = np.arange(
        1,
        len(percentile_df) + 1,
    )

    for col in percentile_df.columns:

        is_dropped = col in dropped_columns

        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentile_df[col],
                name=str(col),
                mode="lines",
                line=dict(
                    width=1.5 if not is_dropped else 1,
                    color=(
                        "rgba(150,150,150,0.25)"
                        if is_dropped
                        else None
                    ),
                ),
                opacity=(
                    0.25
                    if is_dropped
                    else 1
                ),
                legendgroup=str(col),
                hovertemplate=(
                    f"{col}<br>"
                    "Block: %{x}<br>"
                    "95th Percentile: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=600,
        template="streamlit",
        hovermode="x unified",
        xaxis_title="15-Minute Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            y=1.05,
            x=0,
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=13,
        ),
    )

    # Highlight hovered trace
    fig.update_traces(
        selector=dict(type="scatter"),
        line=dict(width=1.5),
    )

    return fig


def create_final_chart(
    percentile_df,
    selected_columns,
    smooth,
):
    """Final average + smoothed profile."""

    x = np.arange(
        1,
        len(percentile_df) + 1,
    )

    average = percentile_df[
        selected_columns
    ].mean(axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=average,
            name="95th Percentile Average",
            mode="lines",
            line=dict(
                width=3,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=smooth,
            name="Smooth Profile",
            mode="lines",
            line=dict(
                width=4,
                color="#00c6ff",
            ),
        )
    )

    fig.update_layout(
        height=500,
        template="streamlit",
        hovermode="x unified",
        xaxis_title="15-Minute Block",
        yaxis_title="Power",
        legend=dict(
            orientation="h",
            y=1.05,
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
# PAGE
# ==========================================================

st.title("☀️ CMV Curve Generator")

st.caption(
    "Generate a 95th-percentile CMV profile "
    "from historical generation data."
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
        "Upload an Excel workbook to continue."
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

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


sheet_name = st.selectbox(
    "Select Source Sheet",
    sheets,
)


try:

    df_original = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
    )

except Exception as e:

    st.error(
        f"Unable to read selected sheet: {e}"
    )

    st.stop()


# ==========================================================
# DATE INDEX
# ==========================================================

date_column = find_date_column(
    df_original
)

if date_column:

    try:

        df_original[date_column] = pd.to_datetime(
            df_original[date_column],
            errors="coerce",
        )

        df_original = df_original.set_index(
            date_column
        )

        st.success(
            f"Date column detected: {date_column}"
        )

    except Exception:

        st.warning(
            "Date column detected but could not be "
            "converted reliably."
        )


# ==========================================================
# DATA PREVIEW
# ==========================================================

with st.expander(
    "View Source Data"
):

    st.dataframe(
        df_original.head(20),
        use_container_width=True,
    )


# ==========================================================
# CLEANING SETTINGS
# ==========================================================

st.subheader(
    "1. Data Cleaning"
)

col1, col2, col3 = st.columns(3)

with col1:

    min_requirement = st.slider(
        "Minimum Data Requirement",
        0.10,
        1.00,
        0.30,
        0.05,
        format="%.0f%%",
    )

with col2:

    upper_limit = st.number_input(
        "Upper Power Limit",
        value=1200.0,
    )

with col3:

    lower_limit = st.number_input(
        "Minimum Valid Peak",
        value=800.0,
    )


# ==========================================================
# CLEAN DATA
# ==========================================================

try:

    df_clean, auto_dropped, zero_dropped = clean_data(
        df_original,
        min_data_requirement=min_requirement,
        upper_limit=upper_limit,
        lower_limit=lower_limit,
    )

except Exception as e:

    st.error(
        f"Cleaning failed: {e}"
    )

    st.stop()


st.write(
    f"Remaining columns: **{len(df_clean.columns)}**"
)

if auto_dropped:

    with st.expander(
        "Automatically Removed Columns"
    ):

        st.write(auto_dropped)


# ==========================================================
# PERCENTILE CALCULATION
# ==========================================================

try:

    percentile_df = calculate_percentiles(
        df_clean,
        96,
    )

    percentile_df = remove_repeated_values(
        percentile_df
    )

except Exception as e:

    st.error(
        f"Percentile calculation failed: {e}"
    )

    st.stop()


# ==========================================================
# PERCENTILE GRAPH
# ==========================================================

st.subheader(
    "2. 95th Percentile Profile by Column"
)

st.caption(
    "Hover over the legend or a curve to inspect "
    "individual column profiles."
)

fig = create_percentile_chart(
    percentile_df
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.subheader(
    "3. Select Columns for Final Average"
)

st.caption(
    "Unselect columns that should not contribute "
    "to the final CMV profile."
)

all_columns = list(
    percentile_df.columns
)

selected_columns = st.multiselect(
    "Columns included in final average",
    options=all_columns,
    default=all_columns,
)


if not selected_columns:

    st.warning(
        "Select at least one column."
    )

    st.stop()


dropped_by_user = [
    col
    for col in all_columns
    if col not in selected_columns
]


# ==========================================================
# SELECTED COLUMN SUMMARY
# ==========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Columns",
    len(all_columns),
)

col2.metric(
    "Included",
    len(selected_columns),
)

col3.metric(
    "Excluded",
    len(dropped_by_user),
)


if dropped_by_user:

    with st.expander(
        "Excluded Columns"
    ):

        st.write(
            dropped_by_user
        )


# ==========================================================
# FINAL AVERAGE
# ==========================================================

average = percentile_df[
    selected_columns
].mean(axis=1)


# ==========================================================
# SMOOTH
# ==========================================================

smooth = smooth_profile(
    average
)


# ==========================================================
# FINAL PROFILE
# ==========================================================

st.subheader(
    "4. Final CMV Profile"
)

final_fig = create_final_chart(
    percentile_df,
    selected_columns,
    smooth,
)

st.plotly_chart(
    final_fig,
    use_container_width=True,
)


# ==========================================================
# FINAL DATA
# ==========================================================

result = pd.DataFrame({
    "Block": np.arange(
        1,
        len(smooth) + 1,
    ),
    "95th Percentile Average": average,
    "Smooth Profile": smooth,
})


with st.expander(
    "View Final Profile Data"
):

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# EXCEL OUTPUT
# ==========================================================

st.subheader("5. Update Original Workbook")

st.caption(
    "Adds/replaces the CMV_Curve, Percentile_Data and "
    "Column_Selection sheets while keeping all other sheets unchanged."
)

if st.button(
    "💾 Add CMV Curve to Workbook",
    type="primary",
    use_container_width=True,
):

    try:
        # --------------------------------------------------
        # Load ORIGINAL uploaded workbook
        # --------------------------------------------------

        output = BytesIO(uploaded_file.getvalue())

        # --------------------------------------------------
        # Open existing workbook and add/replace sheets
        # --------------------------------------------------

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:

            # ==============================================
            # CMV CURVE
            # ==============================================

            final_output = pd.DataFrame({
                "Block": np.arange(1, len(smooth) + 1),
                "95th Percentile Average": average,
                "Smooth Profile": smooth,
            })

            final_output.to_excel(
                writer,
                sheet_name="CMV_Curve",
                index=False,
            )

            # ==============================================
            # PERCENTILE DATA
            # ==============================================

            percentile_output = percentile_df.copy()

            percentile_output.insert(
                0,
                "Block",
                np.arange(1, len(percentile_output) + 1),
            )

            percentile_output.to_excel(
                writer,
                sheet_name="Percentile_Data",
                index=False,
            )

            # ==============================================
            # COLUMN SELECTION
            # ==============================================

            selection_df = pd.DataFrame({
                "Column": all_columns,
                "Included in Average": [
                    col in selected_columns
                    for col in all_columns
                ],
            })

            selection_df.to_excel(
                writer,
                sheet_name="Column_Selection",
                index=False,
            )

        # --------------------------------------------------
        # Reset buffer
        # --------------------------------------------------

        output.seek(0)

        # --------------------------------------------------
        # Create download filename
        # --------------------------------------------------

        original_name = uploaded_file.name

        if original_name.lower().endswith(".xlsx"):
            download_name = (
                original_name[:-5] + "_Updated.xlsx"
            )
        else:
            download_name = (
                original_name + "_Updated.xlsx"
            )

        st.success(
            "Workbook updated successfully. "
            "All existing sheets have been preserved."
        )

        # --------------------------------------------------
        # Download
        # --------------------------------------------------

        st.download_button(
            "⬇️ Download Updated Workbook",
            data=output,
            file_name=download_name,
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Unable to update workbook: {e}"
        )
