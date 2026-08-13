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
    page_icon="📡",
    layout="wide",
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown(
    """
    <h1 style='text-align:center;
    background:linear-gradient(90deg,#00c6ff,#0072ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-size:40px;
    font-weight:800;'>
    ⚡ Solar Suite
    </h1>

    <p style='text-align:center;
    color:gray;
    font-size:14px;'>
    Forecast Correction Platform
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()


# ==========================================================
# GLOBAL CSS
# ==========================================================

st.markdown(
    """
    <style>

    div[data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 12px 20px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================

DEFAULT_STATE = {
    "cmv_average": None,
    "cmv_smooth": None,
    "cmv_selected_columns": None,
    "cmv_settings_applied": False,
    "cmv_generated": False,
    "cmv_source_sheet": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# BASIC HELPERS
# ==========================================================

def find_date_column(df):

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

        key = name.lower()

        if key in normalized:
            return normalized[key]

    return None


# ==========================================================
# DATA CLEANING
# ==========================================================

def prepare_data(
    df,
    min_data_requirement,
    min_cap,
    max_cap,
):

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

            df = df.set_index(
                date_col
            )

    # ------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------

    for col in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):

            converted = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            if converted.notna().sum() > 0:

                df[col] = converted

    # ------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.empty:

        raise ValueError(
            "No numeric columns were found "
            "in the selected sheet."
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
            f"No columns satisfy the selected "
            f"{min_data_requirement}% minimum "
            "data requirement."
        )

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    numeric_df = numeric_df.fillna(0)

    # ------------------------------------------------------
    # Generation cap filtering
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

        if (
            maximum < min_cap
            or maximum > max_cap
        ):

            columns_to_drop.append(col)

    numeric_df = numeric_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # ------------------------------------------------------
    # Remove zero columns
    # ------------------------------------------------------

    zero_columns = numeric_df.columns[
        (numeric_df == 0).all()
    ]

    numeric_df = numeric_df.drop(
        columns=zero_columns,
        errors="ignore",
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    if numeric_df.empty:

        raise ValueError(
            "No usable generation columns remain "
            "after applying the cleaning settings."
        )

    return numeric_df


# ==========================================================
# PERCENTILE CALCULATION
# ==========================================================

def calculate_percentiles(
    df,
    blocks=96,
):

    usable_rows = (
        len(df) // blocks
    ) * blocks

    if usable_rows < blocks:

        raise ValueError(
            f"At least {blocks} rows are required "
            "for percentile calculation."
        )

    df = df.iloc[
        :usable_rows
    ].copy()

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


# ==========================================================
# REMOVE CONSTANT BLOCKS
# ==========================================================

def remove_constant_blocks(
    percentile_df,
):

    result = percentile_df.copy()

    for col in result.columns:

        series = result[col]

        groups = (
            series
            .ne(series.shift())
            .cumsum()
        )

        group_size = groups.map(
            groups.value_counts()
        )

        result[col] = series.mask(
            (group_size > 2)
            & (series != 0),
            0,
        )

    return result


# ==========================================================
# SMOOTHING
# ==========================================================

def validate_window(
    window,
    data_length,
):

    window = int(window)

    if window % 2 == 0:
        window -= 1

    max_allowed = min(
        data_length,
        51,
    )

    if max_allowed % 2 == 0:
        max_allowed -= 1

    window = min(
        window,
        max_allowed,
    )

    window = max(
        window,
        3,
    )

    return window


def generate_smooth_profile(
    average,
    window_length_1,
    polynomial_order_1,
    use_second_smoothing=False,
    window_length_2=None,
    polynomial_order_2=None,
    threshold=4.9,
):

    values = np.asarray(
        average,
        dtype=float,
    )

    # ------------------------------------------------------
    # First smoothing
    # ------------------------------------------------------

    window_length_1 = validate_window(
        window_length_1,
        len(values),
    )

    polynomial_order_1 = min(
        int(polynomial_order_1),
        window_length_1 - 1,
    )

    smooth = savgol_filter(
        values,
        window_length=window_length_1,
        polyorder=polynomial_order_1,
    )

    smooth = np.clip(
        smooth,
        0,
        None,
    )

    # ------------------------------------------------------
    # Threshold
    # ------------------------------------------------------

    smooth = np.where(
        smooth < threshold,
        0,
        smooth,
    )

    # ------------------------------------------------------
    # Second smoothing
    # ------------------------------------------------------

    if (
        use_second_smoothing
        and window_length_2 is not None
        and polynomial_order_2 is not None
    ):

        window_length_2 = validate_window(
            window_length_2,
            len(values),
        )

        polynomial_order_2 = min(
            int(polynomial_order_2),
            window_length_2 - 1,
        )

        smooth = savgol_filter(
            smooth,
            window_length=window_length_2,
            polyorder=polynomial_order_2,
        )

        smooth = np.clip(
            smooth,
            0,
            None,
        )

    return smooth


# ==========================================================
# CACHED OPERATIONS
# ==========================================================

@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def get_sheet_names(
    file_bytes,
):

    excel_file = pd.ExcelFile(
        BytesIO(file_bytes)
    )

    return excel_file.sheet_names


@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def load_source_sheet(
    file_bytes,
    sheet_name,
):

    return pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
    )


@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def cached_prepare_data(
    df,
    min_data_requirement,
    min_cap,
    max_cap,
):

    return prepare_data(
        df,
        min_data_requirement,
        min_cap,
        max_cap,
    )


@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def cached_percentiles(
    df,
):

    percentile_df = calculate_percentiles(
        df,
        blocks=96,
    )

    return remove_constant_blocks(
        percentile_df
    )


@st.cache_data(
    show_spinner=False,
    max_entries=10,
)
def cached_smoothing(
    average,
    window_length_1,
    polynomial_order_1,
    use_second_smoothing,
    window_length_2,
    polynomial_order_2,
):

    return generate_smooth_profile(
        average=np.asarray(average),
        window_length_1=window_length_1,
        polynomial_order_1=polynomial_order_1,
        use_second_smoothing=use_second_smoothing,
        window_length_2=window_length_2,
        polynomial_order_2=polynomial_order_2,
    )


# ==========================================================
# PLOTLY HELPERS
# ==========================================================

def add_legend_highlight(
    fig,
):

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
        selector=dict(
            type="scatter"
        ),
    )

    return fig


import streamlit.components.v1 as components
import json


def make_percentile_chart(
    percentile_df,
    selected_columns,
):
    """
    Interactive 95th percentile chart.

    Legend hover:
    - Hovered profile becomes strongly highlighted.
    - Other profiles fade.
    - Selected profiles remain slightly stronger in normal state.
    """

    x = np.arange(
        1,
        len(percentile_df) + 1,
    )

    traces = []

    for col in percentile_df.columns:

        selected = col in selected_columns

        traces.append({
            "x": x.tolist(),
            "y": percentile_df[col].tolist(),
            "name": str(col),
            "type": "scatter",
            "mode": "lines",

            "line": {
                "width": 3 if selected else 1.2
            },

            "opacity": (
                1.0
                if selected
                else 0.25
            ),

            "legendgroup": str(col),

            "hovertemplate":
                f"{col}<br>"
                "Block: %{x}<br>"
                "Power: %{y:.2f}"
                "<extra></extra>",
        })

    chart_data = json.dumps(traces)

    chart_layout = json.dumps({
        "height": 550,

        "template": "plotly_white",

        "hovermode": "x unified",

        "xaxis": {
            "title": "15 Minute Block"
        },

        "yaxis": {
            "title": "95th Percentile Power"
        },

        "margin": {
            "l": 20,
            "r": 20,
            "t": 80,
            "b": 20
        },

        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,

            "bgcolor": "rgba(0,0,0,0)",

            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },

        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    })

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

        <style>

            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            #chart {{
                width: 100%;
                height: 550px;
            }}

        </style>

    </head>

    <body>

        <div id="chart"></div>

        <script>

            const data = {chart_data};

            const layout = {chart_layout};

            const config = {{
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: [
                    "lasso2d",
                    "select2d"
                ]
            }};

            Plotly.newPlot(
                "chart",
                data,
                layout,
                config
            ).then(function(gd) {{

                /*
                 * Save original opacity.
                 */
                const originalOpacity =
                    data.map(trace =>
                        trace.opacity
                    );

                const originalWidth =
                    data.map(trace =>
                        trace.line.width
                    );

                /*
                 * LEGEND HOVER
                 */
                gd.on(
                    "plotly_legendhover",
                    function(event) {{

                        const index =
                            event.curveNumber;

                        const opacity =
                            data.map(
                                (_, i) =>
                                    i === index
                                        ? 1
                                        : 0.08
                            );

                        const width =
                            data.map(
                                (_, i) =>
                                    i === index
                                        ? 5
                                        : 1
                            );

                        Plotly.restyle(
                            gd,
                            {{
                                opacity: opacity,
                                "line.width": width
                            }}
                        );

                    }}
                );

                /*
                 * LEGEND UNHOVER
                 */
                gd.on(
                    "plotly_legendunhover",
                    function() {{

                        Plotly.restyle(
                            gd,
                            {{
                                opacity:
                                    originalOpacity,

                                "line.width":
                                    originalWidth
                            }}
                        );

                    }}
                );

            }});

        </script>

    </body>

    </html>
    """

    components.html(
        html,
        height=570,
        scrolling=False,
    )


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
            t=100,
            b=20,
        ),
    )

    return add_legend_highlight(
        fig
    )


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "📡 CMV Curve Generator"
)

st.caption(
    "Generate a 95th-percentile CMV generation "
    "profile from historical generation data."
)


# ==========================================================
# 1. EXCEL IMPORT
# ==========================================================

st.subheader(
    "Import Excel Workbook"
)

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


file_bytes = uploaded_file.getvalue()


# ==========================================================
# SHEET SELECTION
# ==========================================================

try:

    sheet_names = get_sheet_names(
        file_bytes
    )

except Exception as e:

    st.error(
        f"Unable to read workbook: {e}"
    )

    st.stop()


selected_sheet = st.selectbox(
    "Select CMV source sheet",
    sheet_names,
)


# Reset results when a new sheet is selected

if (
    st.session_state.cmv_source_sheet
    != selected_sheet
):

    st.session_state.cmv_source_sheet = (
        selected_sheet
    )

    st.session_state.cmv_average = None
    st.session_state.cmv_smooth = None
    st.session_state.cmv_generated = False
    st.session_state.cmv_selected_columns = None


# ==========================================================
# READ SOURCE SHEET
# ==========================================================

try:

    raw_df = load_source_sheet(
        file_bytes,
        selected_sheet,
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
# 2. CLEANING SETTINGS
# ==========================================================

st.subheader(
    "Cleaning Settings"
)


with st.form(
    "cleaning_settings_form"
):

    clean_col1, clean_col2, clean_col3 = (
        st.columns(3)
    )

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
                "is below this value are removed."
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
                "is above this value are removed."
            ),
        )

    apply_cleaning = st.form_submit_button(
        "🔄 Apply Cleaning Settings",
        type="primary",
        use_container_width=True,
    )


if min_cap >= max_cap:

    st.error(
        "Minimum Generation Cap must be "
        "less than Maximum Generation Cap."
    )

    st.stop()


# ==========================================================
# CLEAN DATA
# ==========================================================

try:

    clean_df = cached_prepare_data(
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


min_data_threshold = int(
    np.ceil(
        len(raw_df)
        * min_data_requirement
        / 100
    )
)

st.caption(
    f"Minimum data requirement: "
    f"**{min_data_requirement}%** "
    f"→ at least "
    f"**{min_data_threshold:,} / "
    f"{len(raw_df):,} rows**"
)

st.caption(
    f"Generation range: "
    f"**{min_cap:,.0f} to "
    f"{max_cap:,.0f}**"
)


# ==========================================================
# DATE INFORMATION
# ==========================================================

if isinstance(
    clean_df.index,
    pd.DatetimeIndex,
):

    st.success(
        "Date column detected and used "
        "as the DataFrame index."
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

    percentile_df = cached_percentiles(
        clean_df
    )

except Exception as e:

    st.error(
        f"Unable to calculate 95th percentile: {e}"
    )

    st.stop()


# ==========================================================
# 3. COLUMN SELECTION
# ==========================================================

st.subheader(
    "Select Columns for Average"
)

all_columns = list(
    percentile_df.columns
)


if (
    st.session_state.cmv_selected_columns
    is None
):

    st.session_state.cmv_selected_columns = (
        all_columns.copy()
    )


with st.form(
    "column_selection_form"
):

    selected_columns = st.multiselect(
        "Columns included in final average",
        options=all_columns,
        default=st.session_state.cmv_selected_columns,
        help=(
            "Select the generation columns "
            "that should contribute to the "
            "final CMV average."
        ),
    )

    apply_selection = st.form_submit_button(
        "✓ Apply Column Selection",
        type="primary",
        use_container_width=True,
    )


if apply_selection:

    if not selected_columns:

        st.warning(
            "Select at least one column."
        )

    else:

        st.session_state.cmv_selected_columns = (
            selected_columns
        )

        # New selection means new average

        st.session_state.cmv_generated = False
        st.session_state.cmv_average = None
        st.session_state.cmv_smooth = None


selected_columns = (
    st.session_state.cmv_selected_columns
)


if not selected_columns:

    st.warning(
        "Select at least one column."
    )

    st.stop()


# ==========================================================
# 4. PERCENTILE GRAPH
# ==========================================================

st.subheader(
    "95th Percentile Profiles"
)

st.caption(
    "All cleaned columns are shown here. "
    "Selected columns are highlighted."
)


with st.expander(
    "📈 View 95th Percentile Profiles"
):

    fig_percentile = make_percentile_chart(
        percentile_df,
        selected_columns,
    )

    st.plotly_chart(
        fig_percentile,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


# ==========================================================
# SELECTED DATA
# ==========================================================

selected_percentile_df = (
    percentile_df[
        selected_columns
    ].copy()
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
# 5. SMOOTHING SETTINGS
# ==========================================================

st.subheader(
    "Smoothing Settings"
)


max_window_1 = min(
    len(average),
    51,
)

if max_window_1 % 2 == 0:

    max_window_1 -= 1


if max_window_1 < 3:

    st.error(
        "Not enough blocks for smoothing."
    )

    st.stop()


default_window_1 = min(
    15,
    max_window_1,
)

if default_window_1 % 2 == 0:

    default_window_1 -= 1


max_window_2 = min(
    len(average),
    51,
)

if max_window_2 % 2 == 0:

    max_window_2 -= 1


default_window_2 = min(
    7,
    max_window_2,
)

if default_window_2 % 2 == 0:

    default_window_2 -= 1


with st.form(
    "smoothing_form"
):

    use_second_smoothing = st.checkbox(
        "Apply Second Smoothing",
        value=True,
        help=(
            "Enable a second Savitzky-Golay "
            "smoothing stage."
        ),
    )

    st.markdown("")

    smooth_col1, smooth_col2 = (
        st.columns(2)
    )

    # ------------------------------------------------------
    # FIRST
    # ------------------------------------------------------

    with smooth_col1:

        st.markdown(
            "**First Smoothing**"
        )

        window_length_1 = st.number_input(
            "Window Length",
            min_value=3,
            max_value=max_window_1,
            value=default_window_1,
            step=2,
        )

        polynomial_order_1 = st.number_input(
            "Polynomial Order",
            min_value=1,
            max_value=max(
                1,
                window_length_1 - 1,
            ),
            value=min(
                3,
                window_length_1 - 1,
            ),
            step=1,
        )

    # ------------------------------------------------------
    # SECOND
    # ------------------------------------------------------

    with smooth_col2:

        st.markdown(
            "**Second Smoothing**"
        )

        if use_second_smoothing:

            window_length_2 = st.number_input(
                "Second Window Length",
                min_value=3,
                max_value=max_window_2,
                value=default_window_2,
                step=2,
            )

            polynomial_order_2 = st.number_input(
                "Second Polynomial Order",
                min_value=1,
                max_value=max(
                    1,
                    window_length_2 - 1,
                ),
                value=min(
                    3,
                    window_length_2 - 1,
                ),
                step=1,
            )

        else:

            window_length_2 = None
            polynomial_order_2 = None

            st.info(
                "Second smoothing is disabled."
            )

    generate_curve = st.form_submit_button(
        "⚡ Generate CMV Curve",
        type="primary",
        use_container_width=True,
    )


# ==========================================================
# GENERATE ONLY ON BUTTON CLICK
# ==========================================================

if generate_curve:

    with st.spinner(
        "⚡ Generating CMV curve..."
    ):

        smooth = cached_smoothing(
            tuple(average),
            int(window_length_1),
            int(polynomial_order_1),
            bool(use_second_smoothing),
            (
                int(window_length_2)
                if use_second_smoothing
                else None
            ),
            (
                int(polynomial_order_2)
                if use_second_smoothing
                else None
            ),
        )

    st.session_state.cmv_average = (
        average.copy()
    )

    st.session_state.cmv_smooth = (
        smooth.copy()
    )

    st.session_state.cmv_generated = True


# ==========================================================
# GENERATED RESULT
# ==========================================================

if st.session_state.cmv_generated:

    average = (
        st.session_state.cmv_average
    )

    smooth = (
        st.session_state.cmv_smooth
    )

    st.subheader(
        "Generated CMV Curve"
    )

    st.caption(
        "Only the 95th percentile average "
        "and final smooth profile are shown here."
    )

    fig_final = make_final_chart(
        average,
        smooth,
    )

    st.plotly_chart(
        fig_final,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    # ------------------------------------------------------
    # RESULT TABLE
    # ------------------------------------------------------

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
# 6. EXCEL OUTPUT
# ==========================================================

if st.session_state.cmv_generated:

    st.subheader(
        "Update Original Workbook"
    )

    st.caption(
        "The original workbook is preserved. "
        "The generated CMV results are added "
        "as separate sheets."
    )

    if st.button(
        "💾 Add CMV Curve & Download Updated Workbook",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Preparing updated workbook..."
            ):

                output = BytesIO(
                    uploaded_file.getvalue()
                )

                # --------------------------------------------------
                # CMV
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
                # PERCENTILE
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
                # SELECTION
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
                        "First Window Length",
                        "First Polynomial Order",
                        "Second Smoothing",
                        "Second Window Length",
                        "Second Polynomial Order",
                        "Rows",
                        "Original Columns",
                        "Usable Columns",
                    ],
                    "Value": [
                        selected_sheet,
                        f"{min_data_requirement}%",
                        min_cap,
                        max_cap,
                        window_length_1,
                        polynomial_order_1,
                        (
                            "Enabled"
                            if use_second_smoothing
                            else "Disabled"
                        ),
                        (
                            window_length_2
                            if use_second_smoothing
                            else ""
                        ),
                        (
                            polynomial_order_2
                            if use_second_smoothing
                            else ""
                        ),
                        len(raw_df),
                        raw_df.shape[1],
                        clean_df.shape[1],
                    ],
                })

                # --------------------------------------------------
                # WRITE
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

            # ------------------------------------------------------
            # DOWNLOAD NAME
            # ------------------------------------------------------

            original_name = uploaded_file.name

            if original_name.lower().endswith(
                ".xlsx"
            ):

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
                "All existing workbook sheets "
                "were preserved."
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
