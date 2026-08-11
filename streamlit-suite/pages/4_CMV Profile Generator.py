import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from io import BytesIO
from scipy.signal import savgol_filter


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="CMV Curve Generator",
    page_icon="📈",
    layout="wide",
)


# ==========================================================
# FUNCTIONS
# ==========================================================

def find_date_column(df):
    """Find a likely date/time column."""

    possible_names = [
        "Date",
        "Datetime",
        "Date Time",
        "Timestamp",
        "Time",
    ]

    normalized = {
        str(col).strip().lower().replace("_", " "): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


# ----------------------------------------------------------
# CLEAN DATA
# ----------------------------------------------------------

def prepare_data(
    df,
    min_data_requirement,
    min_cap,
    max_cap,
):
    """
    Clean raw CMV data.

    min_data_requirement:
        Percentage of non-empty values required.

    min_cap:
        Minimum acceptable generation value.

    max_cap:
        Maximum acceptable generation value.
    """

    df = df.copy()

    # ------------------------------------------------------
    # Date handling
    # ------------------------------------------------------

    date_col = find_date_column(df)

    if date_col is not None:

        parsed_date = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

        if parsed_date.notna().any():

            df[date_col] = parsed_date
            df = df.set_index(date_col)

    # ------------------------------------------------------
    # Convert numeric columns where possible
    # ------------------------------------------------------

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(df[col]):

            converted = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if converted.notna().sum() > 0:
                df[col] = converted

    # ------------------------------------------------------
    # Numeric columns only
    # ------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.empty:
        raise ValueError(
            "No numeric columns were found."
        )

    # ------------------------------------------------------
    # Minimum data requirement
    # ------------------------------------------------------

    threshold = int(
        np.ceil(
            len(numeric_df)
            * min_data_requirement
            / 100
        )
    )

    numeric_df = numeric_df.dropna(
        axis=1,
        thresh=threshold,
    )

    if numeric_df.empty:
        raise ValueError(
            f"No columns satisfy the "
            f"{min_data_requirement}% minimum "
            "data requirement."
        )

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    numeric_df = numeric_df.fillna(0)

    # ------------------------------------------------------
    # Remove columns outside generation caps
    # ------------------------------------------------------

    columns_to_drop = []

    for col in numeric_df.columns:

        values = numeric_df[col].to_numpy(
            dtype=float
        )

        if len(values) == 0:
            columns_to_drop.append(col)
            continue

        maximum = np.nanmax(values)

        if maximum > max_cap or maximum < min_cap:
            columns_to_drop.append(col)

    numeric_df = numeric_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # ------------------------------------------------------
    # Remove completely zero columns
    # ------------------------------------------------------

    zero_columns = numeric_df.columns[
        (numeric_df == 0).all()
    ]

    numeric_df = numeric_df.drop(
        columns=zero_columns,
        errors="ignore",
    )

    if numeric_df.empty:
        raise ValueError(
            "No usable generation columns remain "
            "after cleaning."
        )

    return numeric_df


# ----------------------------------------------------------
# PERCENTILE
# ----------------------------------------------------------

def calculate_percentiles(
    df,
    blocks=96,
):
    """Calculate 95th percentile for each 15-minute block."""

    usable_rows = (
        len(df) // blocks
    ) * blocks

    if usable_rows < blocks:
        raise ValueError(
            f"At least {blocks} rows are required."
        )

    df = df.iloc[:usable_rows].copy()

    days = usable_rows // blocks

    result = {}

    for col in df.columns:

        values = df[col].to_numpy(
            dtype=float
        )

        reshaped = values.reshape(
            days,
            blocks,
        )

        result[col] = np.percentile(
            reshaped,
            95,
            axis=0,
        )

    return pd.DataFrame(result)


# ----------------------------------------------------------
# CONSTANT BLOCK REMOVAL
# ----------------------------------------------------------

def remove_constant_blocks(percentile_df):

    result = percentile_df.copy()

    for col in result.columns:

        s = result[col]

        group = (
            s.ne(s.shift())
            .cumsum()
        )

        group_size = group.map(
            group.value_counts()
        )

        result[col] = s.mask(
            (group_size > 2) & (s != 0),
            0,
        )

    return result


# ----------------------------------------------------------
# SMOOTHING
# ----------------------------------------------------------

def generate_smooth_profile(
    average,
    window_length,
    polynomial_order,
    threshold=4.9,
):
    """Generate smooth CMV profile."""

    values = np.asarray(
        average,
        dtype=float,
    )

    # First smoothing
    smooth = savgol_filter(
        values,
        window_length=window_length,
        polyorder=polynomial_order,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    # Remove very small generation
    smooth = np.where(
        smooth < threshold,
        0,
        smooth,
    )

    # Second smoothing
    second_window = min(
        7,
        window_length,
    )

    if second_window % 2 == 0:
        second_window -= 1

    second_window = max(
        second_window,
        polynomial_order + 2,
    )

    if second_window % 2 == 0:
        second_window += 1

    if second_window <= len(smooth):
        smooth = savgol_filter(
            smooth,
            window_length=second_window,
            polyorder=polynomial_order,
        )

    return np.clip(
        smooth,
        0,
        None,
    )


# ----------------------------------------------------------
# LEGEND HIGHLIGHT
# ----------------------------------------------------------

def add_legend_highlight(fig):

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    fig.update_traces(
        hoverinfo="x+y+name",
        selector=dict(type="scatter"),
    )

    return fig


# ----------------------------------------------------------
# PERCENTILE CHART
# ----------------------------------------------------------

def make_percentile_chart(
    percentile_df,
    selected_columns,
):

    fig = go.Figure()

    x = np.arange(
        1,
        len(percentile_df) + 1,
    )

    for col in percentile_df.columns:

        selected = (
            col in selected_columns
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=percentile_df[col],
                name=str(col),
                mode="lines",
                line=dict(
                    width=3 if selected else 1.2
                ),
                opacity=1 if selected else 0.35,
                legendgroup=str(col),
            )
        )

    fig.update_layout(
        height=550,
        template="streamlit",
        xaxis_title="15 Minute Block",
        yaxis_title="95th Percentile Power",
        margin=dict(
            l=20,
            r=20,
            t=90,
            b=20,
        ),
    )

    return add_legend_highlight(fig)


# ----------------------------------------------------------
# FINAL CMV CHART
# ----------------------------------------------------------

def make_final_chart(
    average,
    smooth,
):

    x = np.arange(
        1,
        len(average) + 1,
    )

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
        height=550,
        template="streamlit",
        xaxis_title="15 Minute Block",
        yaxis_title="Power",
        margin=dict(
            l=20,
            r=20,
            t=90,
            b=20,
        ),
    )

    return add_legend_highlight(fig)


# ==========================================================
# HEADER
# ==========================================================

st.title("📈 CMV Curve Generator")

st.caption(
    "Generate a 95th-percentile CMV generation profile "
    "from historical generation data."
)

st.divider()


# ==========================================================
# EXCEL IMPORT
# ==========================================================

st.subheader("1. Import Excel Workbook")

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx"],
    key="cmv_excel_upload",
)

if uploaded_file is None:

    st.info(
        "Upload an Excel workbook to begin."
    )

    st.stop()


# ==========================================================
# SHEET SELECTION
# ==========================================================

try:

    excel_file = pd.ExcelFile(
        uploaded_file
    )

    sheet_names = excel_file.sheet_names

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


selected_sheet = st.selectbox(
    "Select CMV source sheet",
    sheet_names,
)


# ==========================================================
# READ SOURCE SHEET
# ==========================================================

try:

    raw_df = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet,
    )

except Exception as e:

    st.error(
        f"Unable to read selected sheet: {e}"
    )

    st.stop()


if raw_df.empty:

    st.error(
        "The selected sheet is empty."
    )

    st.stop()


st.success(
    f"Loaded {selected_sheet}: "
    f"{raw_df.shape[0]:,} rows × "
    f"{raw_df.shape[1]:,} columns."
)


# ==========================================================
# CLEANING SETTINGS
# ==========================================================

st.subheader("2. Cleaning Settings")

clean_col1, clean_col2, clean_col3 = st.columns(3)

with clean_col1:

    min_data_requirement = st.slider(
        "Minimum Data Requirement",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        format="%d%%",
        help=(
            "Minimum percentage of non-empty "
            "values required for a column."
        ),
    )

with clean_col2:

    min_cap = st.number_input(
        "Minimum Generation Cap",
        min_value=0.0,
        value=800.0,
        step=10.0,
        help=(
            "Columns whose maximum generation "
            "is below this value will be removed."
        ),
    )

with clean_col3:

    max_cap = st.number_input(
        "Maximum Generation Cap",
        min_value=0.0,
        value=1200.0,
        step=10.0,
        help=(
            "Columns whose maximum generation "
            "is above this value will be removed."
        ),
    )


if min_cap >= max_cap:

    st.error(
        "Minimum Generation Cap must be "
        "less than Maximum Generation Cap."
    )

    st.stop()


min_data_threshold = int(
    np.ceil(
        len(raw_df)
        * min_data_requirement
        / 100
    )
)


st.caption(
    f"Minimum data: **{min_data_requirement}%** "
    f"→ at least **{min_data_threshold:,} / "
    f"{len(raw_df):,} rows**"
)

st.caption(
    f"Generation range: "
    f"**{min_cap:,.0f} to {max_cap:,.0f}**"
)


# ==========================================================
# CLEAN DATA
# ==========================================================

try:

    clean_df = prepare_data(
        raw_df,
        min_data_requirement,
        min_cap,
        max_cap,
    )

except Exception as e:

    st.error(
        f"Cleaning failed: {e}"
    )

    st.stop()


# ==========================================================
# CLEANING SUMMARY
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Original Columns",
    raw_df.shape[1],
)

c2.metric(
    "Usable Columns",
    clean_df.shape[1],
)

c3.metric(
    "Rows",
    len(clean_df),
)

c4.metric(
    "Minimum Requirement",
    f"{min_data_requirement}%",
)


# ==========================================================
# DATE INFORMATION
# ==========================================================

if isinstance(
    clean_df.index,
    pd.DatetimeIndex,
):

    st.success(
        "Date column detected and used as the DataFrame index."
    )


# ==========================================================
# CLEAN DATA PREVIEW
# ==========================================================

with st.expander(
    "View Cleaned Data"
):

    st.dataframe(
        clean_df.head(20),
        use_container_width=True,
    )


# ==========================================================
# PERCENTILE CALCULATION
# ==========================================================

try:

    percentile_df = calculate_percentiles(
        clean_df,
        blocks=96,
    )

    percentile_df = (
        remove_constant_blocks(
            percentile_df
        )
    )

except Exception as e:

    st.error(
        f"Unable to calculate 95th percentile: {e}"
    )

    st.stop()


# ==========================================================
# COLUMN SELECTION
# ==========================================================

st.subheader(
    "3. Select Columns for Average"
)

all_columns = list(
    percentile_df.columns
)

selected_columns = st.multiselect(
    "Columns included in final average",
    options=all_columns,
    default=all_columns,
    help=(
        "Select the generation columns "
        "that should contribute to the "
        "final CMV average."
    ),
)

if not selected_columns:

    st.warning(
        "Select at least one column."
    )

    st.stop()


# ==========================================================
# PERCENTILE GRAPH
# ==========================================================

st.subheader(
    "4. 95th Percentile Profiles"
)

st.caption(
    "All cleaned columns are shown here. "
    "Selected columns are highlighted. "
    "Hover over a legend item to highlight its curve."
)

fig_percentile = make_percentile_chart(
    percentile_df,
    selected_columns,
)

st.plotly_chart(
    fig_percentile,
    use_container_width=True,
)


# ==========================================================
# SELECTED DATA
# ==========================================================

selected_percentile_df = (
    percentile_df[
        selected_columns
    ]
    .copy()
)


# ==========================================================
# FINAL AVERAGE
# ==========================================================

average = (
    selected_percentile_df
    .mean(axis=1)
    .to_numpy()
)


# ==========================================================
# SMOOTHING SETTINGS
# ==========================================================

st.subheader(
    "5. Smoothing Settings"
)

smooth_col1, smooth_col2 = st.columns(2)

max_window = len(average)

if max_window % 2 == 0:
    max_window -= 1

max_window = min(
    max_window,
    51,
)

if max_window < 5:

    st.error(
        "Not enough blocks for smoothing."
    )

    st.stop()


with smooth_col1:

    window_options = list(
        range(
            5,
            max_window + 1,
            2,
        )
    )

    window_length = st.selectbox(
        "Window Length",
        options=window_options,
        index=(
            window_options.index(15)
            if 15 in window_options
            else 0
        ),
        help=(
            "Must be an odd number. "
            "Larger values produce stronger smoothing."
        ),
    )


with smooth_col2:

    polynomial_order = st.selectbox(
        "Polynomial Order",
        options=[
            i
            for i in range(
                1,
                min(
                    8,
                    window_length,
                ),
            )
        ],
        index=2 if window_length >= 5 else 0,
        help=(
            "Higher values preserve more local "
            "curve shape."
        ),
    )


st.caption(
    f"Smoothing configuration: "
    f"Window Length = **{window_length}**, "
    f"Polynomial Order = **{polynomial_order}**"
)


# ==========================================================
# GENERATE SMOOTH PROFILE
# ==========================================================

smooth = generate_smooth_profile(
    average,
    window_length,
    polynomial_order,
)


# ==========================================================
# FINAL CMV CURVE
# ==========================================================

st.subheader(
    "6. Generated CMV Curve"
)

st.caption(
    "Only the final average and smooth profile "
    "are shown here."
)

fig_final = make_final_chart(
    average,
    smooth,
)

st.plotly_chart(
    fig_final,
    use_container_width=True,
)


# ==========================================================
# RESULT TABLE
# ==========================================================

final_output = pd.DataFrame({
    "Block": np.arange(
        1,
        len(smooth) + 1,
    ),
    "95th Percentile Average": average,
    "Smooth Profile": smooth,
})


with st.expander(
    "View Generated Curve Data"
):

    st.dataframe(
        final_output,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# COLUMN SELECTION TABLE
# ==========================================================

selection_df = pd.DataFrame({
    "Column": all_columns,
    "Included in Average": [
        col in selected_columns
        for col in all_columns
    ],
})


with st.expander(
    "View Column Selection"
):

    st.dataframe(
        selection_df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# EXCEL OUTPUT
# ==========================================================

st.subheader(
    "7. Update Original Workbook"
)

st.caption(
    "The original workbook is preserved. "
    "The generated CMV results are added as separate sheets."
)


if st.button(
    "💾 Add CMV Curve & Download Updated Workbook",
    type="primary",
    use_container_width=True,
):

    try:

        # --------------------------------------------------
        # Start from ORIGINAL workbook
        # --------------------------------------------------

        output = BytesIO(
            uploaded_file.getvalue()
        )

        # --------------------------------------------------
        # CMV CURVE
        # --------------------------------------------------

        cmv_output = pd.DataFrame({
            "Block": np.arange(
                1,
                len(smooth) + 1,
            ),
            "95th Percentile Average": average,
            "Smooth Profile": smooth,
        })

        # --------------------------------------------------
        # PERCENTILE DATA
        # --------------------------------------------------

        percentile_output = (
            percentile_df.copy()
        )

        percentile_output.insert(
            0,
            "Block",
            np.arange(
                1,
                len(percentile_output) + 1,
            ),
        )

        # --------------------------------------------------
        # COLUMN SELECTION
        # --------------------------------------------------

        selection_output = pd.DataFrame({
            "Column": all_columns,
            "Included in Average": [
                col in selected_columns
                for col in all_columns
            ],
        })

        # --------------------------------------------------
        # SETTINGS
        # --------------------------------------------------

        settings_output = pd.DataFrame({
            "Setting": [
                "Source Sheet",
                "Minimum Data Requirement",
                "Minimum Generation Cap",
                "Maximum Generation Cap",
                "Window Length",
                "Polynomial Order",
                "Rows",
                "Original Columns",
                "Usable Columns",
            ],
            "Value": [
                selected_sheet,
                f"{min_data_requirement}%",
                min_cap,
                max_cap,
                window_length,
                polynomial_order,
                len(raw_df),
                raw_df.shape[1],
                clean_df.shape[1],
            ],
        })

        # --------------------------------------------------
        # UPDATE ORIGINAL WORKBOOK
        # --------------------------------------------------

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:

            cmv_output.to_excel(
                writer,
                sheet_name="CMV_Curve",
                index=False,
            )

            percentile_output.to_excel(
                writer,
                sheet_name="Percentile_Data",
                index=False,
            )

            selection_output.to_excel(
                writer,
                sheet_name="Column_Selection",
                index=False,
            )

            settings_output.to_excel(
                writer,
                sheet_name="CMV_Settings",
                index=False,
            )

        output.seek(0)

        # --------------------------------------------------
        # DOWNLOAD NAME
        # --------------------------------------------------

        original_name = uploaded_file.name

        if original_name.lower().endswith(".xlsx"):

            download_name = (
                original_name[:-5]
                + "_Updated.xlsx"
            )

        else:

            download_name = (
                original_name
                + "_Updated.xlsx"
            )

        st.success(
            "CMV Curve added successfully. "
            "All existing workbook sheets were preserved."
        )

        st.download_button(
            "⬇️ Download Updated Workbook",
            data=output.getvalue(),
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
