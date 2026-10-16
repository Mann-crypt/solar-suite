
from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"
PAGE_SIZE = 20
ALL_USERS = "All Users"


# =========================================================
# CSS
# =========================================================

st.html("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

.dash-header {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 20px;
}

.dash-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb, #6366f1);
    color: white;
    font-size: 22px;
}

.dash-title {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.7px;
    line-height: 1.2;
}

.dash-subtitle {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 4px;
}

.section-heading {
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 2px;
}

.section-caption {
    font-size: 11px;
    opacity: 0.65;
    margin-bottom: 8px;
}

div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 10px 14px;
    min-height: 78px;
}

div[data-testid="stMetricLabel"] {
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 23px;
    font-weight: 800;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.22) !important;
    border-radius: 12px !important;
    overflow: hidden;
    background: var(--secondary-background-color);
    margin-bottom: 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.025);
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.55) !important;
}

div[data-testid="stExpander"] details summary {
    padding: 10px 12px !important;
    min-height: 60px;
}

div[data-testid="stExpander"] details summary p {
    font-size: 12px;
    font-weight: 650;
    line-height: 1.5;
}

div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
    padding: 0 12px 12px 12px !important;
}

.record-title {
    font-size: 14px;
    font-weight: 800;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.record-state {
    font-size: 11px;
    opacity: 0.72;
    margin-top: 5px;
}

.record-info {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px solid rgba(128,128,128,0.16);
}

.record-poc {
    font-size: 11px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.record-capacity {
    font-size: 13px;
    font-weight: 800;
    white-space: nowrap;
}

.record-label {
    font-size: 10px;
    opacity: 0.65;
    margin-bottom: 2px;
}

.record-value {
    font-size: 12px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.detail-heading {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 800;
    opacity: 0.65;
    margin: 10px 0 5px 0;
}

.detail-item {
    padding: 7px 0;
    border-bottom: 1px solid rgba(128,128,128,0.13);
}

.result-caption {
    font-size: 11px;
    opacity: 0.65;
    margin-top: 3px;
    margin-bottom: 8px;
}

.chart-heading {
    font-size: 13px;
    font-weight: 750;
    margin-bottom: 0;
}

.page-indicator {
    text-align: center;
    font-size: 12px;
    font-weight: 700;
    padding-top: 9px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 34px;
}

div[data-testid="stPlotlyChart"] {
    margin-top: -5px;
    margin-bottom: -12px;
}

hr {
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .dash-title {
        font-size: 22px;
    }

    div[data-testid="stMetric"] {
        padding: 8px;
        min-height: 72px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 18px;
    }
}
</style>
""")


# =========================================================
# HELPERS
# =========================================================

def normalize_header(value):
    return "".join(
        str(value).strip().casefold().split()
    ).replace("_", "")


def find_column(df, candidates=(), startswith=None):
    lookup = {
        normalize_header(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_header(candidate)
        if key in lookup:
            return lookup[key]

    if startswith:
        prefix = normalize_header(startswith)

        for col in df.columns:
            if normalize_header(col).startswith(prefix):
                return col

    return None


def clean(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def safe_html(value):
    return escape(clean(value))


def unique_sorted(series):
    values = {
        clean(value)
        for value in series
        if clean(value)
    }
    return sorted(values, key=str.casefold)


def reset_page():
    st.session_state["intimation_page"] = 1


def format_capacity(value):
    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return "—"

    return f"{number:,.2f} MW"


def status_style(value):
    value = clean(value).casefold()

    if any(word in value for word in [
        "complete", "checked", "verified",
        "approved", "done", "pass"
    ]):
        return "color:#15803d;"

    if any(word in value for word in [
        "pending", "awaiting", "in progress", "hold"
    ]):
        return "color:#b45309;"

    if any(word in value for word in [
        "reject", "error", "fail", "issue", "not checked"
    ]):
        return "color:#dc2626;"

    return ""


# =========================================================
# LOAD EXCEL
# =========================================================

@st.cache_data(show_spinner="Loading records...")
def load_data(file_path, modified_time):
    del modified_time

    raw = pd.read_excel(file_path)
    raw.columns = [str(col).strip() for col in raw.columns]

    priority_source = find_column(
        raw,
        [
            "Priority as per 003 sheet",
            "Priority as per Amresh Sir 28.07.2023",
        ],
    )

    if priority_source is None:
        priority_source = find_column(
            raw,
            startswith="Priority",
        )

    aliases = {
        "Lat": ["Lat", "Latitude"],
        "Long": ["Long", "Longitude", "Lon"],
        "PSS": ["PSS", "PSS Name"],
        "State": ["State"],
        "Capacity (MW)": [
            "Capacity (MW)",
            "Capacity MW",
            "Capacity",
        ],
        "POC1": ["POC1", "POC 1"],
        "POC 2": ["POC 2", "POC2"],
        "POC 3": ["POC 3", "POC3"],
        "Check Status": ["Check Status", "Status"],
    }

    output = pd.DataFrame(index=raw.index)

    for target, candidates in aliases.items():
        source = find_column(raw, candidates)
        output[target] = raw[source] if source else ""

    output["Priority"] = (
        raw[priority_source]
        if priority_source
        else ""
    )

    text_columns = [
        "Lat", "Long", "PSS", "State",
        "Priority", "POC1", "POC 2", "POC 3",
        "Check Status",
    ]

    for column in text_columns:
        output[column] = output[column].map(clean)

    output["Capacity (MW)"] = pd.to_numeric(
        output["Capacity (MW)"],
        errors="coerce",
    )

    output = output[
        [
            "Lat", "Long", "PSS", "State",
            "Capacity (MW)", "Priority",
            "POC1", "POC 2", "POC 3",
            "Check Status",
        ]
    ]

    has_data = (
        output["PSS"].ne("")
        | output["State"].ne("")
        | output["POC1"].ne("")
    )

    return output.loc[has_data].reset_index(drop=True)


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="dash-header">
    <div class="dash-logo">⚡</div>
    <div>
        <div class="dash-title">Intimation Dashboard</div>
        <div class="dash-subtitle">
            PSS records, capacity and status monitoring
        </div>
    </div>
</div>
""")

if not EXCEL_PATH.exists():
    st.error(
        f"Excel file not found: {EXCEL_PATH}. "
        "Place intimation.xlsx in the repository root."
    )
    st.stop()

try:
    df = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Unable to read intimation.xlsx: {error}")
    st.stop()

if df.empty:
    st.warning("No usable records found in the Excel file.")
    st.stop()


# =========================================================
# FILTER STATE
# =========================================================

if "intimation_poc1" not in st.session_state:
    st.session_state["intimation_poc1"] = ALL_USERS

if "intimation_states" not in st.session_state:
    st.session_state["intimation_states"] = []

if "intimation_pss_search" not in st.session_state:
    st.session_state["intimation_pss_search"] = ""

if "intimation_page" not in st.session_state:
    st.session_state["intimation_page"] = 1


# =========================================================
# FILTERS
# =========================================================

overview_slot = st.container()

st.html("""
<div class="section-heading">Filters</div>
<div class="section-caption">
    Filters are connected and update the overview and records
</div>
""")

poc_col, pss_col, state_col = st.columns(
    [1, 1.4, 1.2],
    gap="medium",
)

with pss_col:
    st.text_input(
        "Search PSS",
        placeholder="Enter PSS name...",
        key="intimation_pss_search",
        on_change=reset_page,
    )

pss_query = st.session_state["intimation_pss_search"].strip()

if pss_query:
    pss_mask = df["PSS"].str.contains(
        pss_query,
        case=False,
        na=False,
        regex=False,
    )
else:
    pss_mask = pd.Series(True, index=df.index)


# POC1 options depend on PSS search and selected states.
selected_states = st.session_state["intimation_states"]

poc_mask = pss_mask.copy()

if selected_states:
    poc_mask &= df["State"].isin(selected_states)

poc_values = unique_sorted(df.loc[poc_mask, "POC1"])
poc_options = [ALL_USERS] + [
    value for value in poc_values
    if value != ALL_USERS
]

if st.session_state["intimation_poc1"] not in poc_options:
    st.session_state["intimation_poc1"] = ALL_USERS

with poc_col:
    st.selectbox(
        "POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]


# State options depend on PSS search and selected POC1.
state_mask = pss_mask.copy()

if selected_poc != ALL_USERS:
    state_mask &= df["POC1"].eq(selected_poc)

available_states = unique_sorted(df.loc[state_mask, "State"])

current_states = st.session_state["intimation_states"]
valid_states = [
    state for state in current_states
    if state in available_states
]

if current_states != valid_states:
    st.session_state["intimation_states"] = valid_states

with state_col:
    st.multiselect(
        "State",
        options=available_states,
        key="intimation_states",
        placeholder="All states",
        on_change=reset_page,
    )

selected_states = st.session_state["intimation_states"]


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()

if pss_query:
    filtered_df = filtered_df[
        filtered_df["PSS"].str.contains(
            pss_query,
            case=False,
            na=False,
            regex=False,
        )
    ]

if selected_poc != ALL_USERS:
    filtered_df = filtered_df[
        filtered_df["POC1"].eq(selected_poc)
    ]

if selected_states:
    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_states)
    ]

filtered_df = filtered_df.reset_index(drop=True)


# =========================================================
# COMPACT OVERVIEW
# Only two metrics and compact charts.
# =========================================================

with overview_slot:
    st.html("""
    <div class="section-heading">Overview</div>
    <div class="section-caption">
        Summary and trends for the selected records
    </div>
    """)

    total_capacity = filtered_df["Capacity (MW)"].sum(
        min_count=1
    )

    if pd.isna(total_capacity):
        total_capacity = 0.0

    metric_left, metric_right = st.columns(2, gap="medium")

    metric_left.metric(
        "Matching Records",
        f"{len(filtered_df):,}",
    )

    metric_right.metric(
        "Total Capacity",
        f"{total_capacity:,.2f} MW",
    )

    if not filtered_df.empty:
        graph_left, graph_right = st.columns(2, gap="medium")

        # Capacity by state: compact horizontal bar chart.
        with graph_left:
            st.markdown(
                '<div class="chart-heading">Capacity by State</div>',
                unsafe_allow_html=True,
            )

            capacity_state = (
                filtered_df[
                    filtered_df["State"].ne("")
                    & filtered_df["Capacity (MW)"].notna()
                ]
                .groupby("State", as_index=False)["Capacity (MW)"]
                .sum()
                .sort_values("Capacity (MW)", ascending=False)
                .head(7)
                .sort_values("Capacity (MW)", ascending=True)
            )

            if not capacity_state.empty:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        capacity_state,
                        x="Capacity (MW)",
                        y="State",
                        orientation="h",
                        color_discrete_sequence=["#6366f1"],
                    )

                    fig.update_layout(
                        height=210,
                        margin=dict(l=0, r=8, t=5, b=0),
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9),
                        xaxis_title=None,
                        yaxis_title=None,
                    )

                    fig.update_traces(
                        marker_line_width=0,
                        hovertemplate=(
                            "%{y}<br>%{x:,.2f} MW<extra></extra>"
                        ),
                    )

                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.13)",
                        zeroline=False,
                    )
                    fig.update_yaxes(showgrid=False)

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                else:
                    st.bar_chart(
                        capacity_state.set_index("State")[
                            ["Capacity (MW)"]
                        ],
                        horizontal=True,
                        height=210,
                    )
            else:
                st.caption("No capacity data available.")

        # Check status: compact donut chart.
        with graph_right:
            st.markdown(
                '<div class="chart-heading">Check Status</div>',
                unsafe_allow_html=True,
            )

            status_data = filtered_df["Check Status"].replace(
                "", "Not specified"
            )

            status_summary = (
                status_data.value_counts()
                .rename_axis("Status")
                .reset_index(name="Records")
                .head(6)
            )

            if not status_summary.empty:
                if PLOTLY_AVAILABLE:
                    fig = px.pie(
                        status_summary,
                        names="Status",
                        values="Records",
                        hole=0.66,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )

                    fig.update_traces(
                        textposition="inside",
                        textinfo="percent",
                        hovertemplate=(
                            "%{label}<br>%{value:,} records"
                            "<extra></extra>"
                        ),
                    )

                    fig.update_layout(
                        height=210,
                        margin=dict(l=0, r=0, t=5, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.10,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=8),
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                else:
                    st.bar_chart(
                        status_summary.set_index("Status")[["Records"]],
                        height=210,
                    )
            else:
                st.caption("No status data available.")

        # Priority distribution: compact horizontal bar chart.
        st.markdown(
            '<div class="chart-heading">Priority Distribution</div>',
            unsafe_allow_html=True,
        )

        priority_data = filtered_df["Priority"].replace(
            "", "Not specified"
        )

        priority_summary = (
            priority_data.value_counts()
            .rename_axis("Priority")
            .reset_index(name="Records")
            .head(6)
            .sort_values("Records", ascending=True)
        )

        if not priority_summary.empty:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    priority_summary,
                    x="Records",
                    y="Priority",
                    orientation="h",
                    color_discrete_sequence=["#0d9488"],
                )

                fig.update_layout(
                    height=170,
                    margin=dict(l=0, r=8, t=3, b=0),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9),
                    xaxis_title=None,
                    yaxis_title=None,
                )

                fig.update_traces(
                    marker_line_width=0,
                    hovertemplate=(
                        "%{y}<br>%{x} records<extra></extra>"
                    ),
                )

                fig.update_xaxes(
                    showgrid=True,
                    gridcolor="rgba(128,128,128,0.13)",
                    zeroline=False,
                )
                fig.update_yaxes(showgrid=False)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.bar_chart(
                    priority_summary.set_index("Priority")[["Records"]],
                    horizontal=True,
                    height=170,
                )

    else:
        st.info("No records match the selected filters.")

    st.divider()


# =========================================================
# RECORDS
# =========================================================

st.html("""
<div class="section-heading">Records</div>
<div class="section-caption">
    Each collapsed card shows only PSS, capacity, state and POC1
</div>
""")

if filtered_df.empty:
    st.stop()

total_records = len(filtered_df)
total_pages = max(1, math.ceil(total_records / PAGE_SIZE))

current_page = max(
    1,
    min(st.session_state["intimation_page"], total_pages),
)
st.session_state["intimation_page"] = current_page

start = (current_page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, total_records)

st.html(f"""
<div class="result-caption">
    Showing {start + 1:,}–{end:,} of {total_records:,} records
</div>
""")

page_df = filtered_df.iloc[start:end]


# =========================================================
# CARDS
# =========================================================

left_cards, right_cards = st.columns(2, gap="medium")

for index, (_, record) in enumerate(page_df.iterrows()):
    target = left_cards if index % 2 == 0 else right_cards

    pss = clean(record["PSS"]) or "Unnamed PSS"
    state = clean(record["State"]) or "Not specified"
    poc1 = clean(record["POC1"]) or "Not specified"
    capacity = format_capacity(record["Capacity (MW)"])

    with target:
        # The collapsed header shows the PSS and capacity only.
        with st.expander(
            f"{pss}  ·  {capacity}",
            expanded=False,
        ):
            # Visible when the card is expanded.
            st.html(f"""
            <div class="record-title">{safe_html(pss)}</div>

            <div class="record-info">
                <div>
                    <div class="record-label">State</div>
                    <div class="record-value">{safe_html(state)}</div>
                </div>

                <div style="text-align:right;">
                    <div class="record-label">Capacity</div>
                    <div class="record-capacity">{safe_html(capacity)}</div>
                </div>
            </div>

            <div style="margin-top:10px;">
                <div class="record-label">POC1</div>
                <div class="record-value">{safe_html(poc1)}</div>
            </div>

            <div class="detail-heading">Additional Details</div>
            """)

            details = [
                ("Priority", record["Priority"]),
                ("POC 2", record["POC 2"]),
                ("POC 3", record["POC 3"]),
                ("Latitude", record["Lat"]),
                ("Longitude", record["Long"]),
                ("Check Status", record["Check Status"]),
            ]

            detail_left, detail_right = st.columns(2, gap="small")

            for detail_index, (label, value) in enumerate(details):
                detail_col = (
                    detail_left
                    if detail_index % 2 == 0
                    else detail_right
                )

                with detail_col:
                    st.html(f"""
                    <div class="detail-item">
                        <div class="record-label">{safe_html(label)}</div>
                        <div class="record-value"
                             style="{status_style(value) if label == 'Check Status' else ''}">
                            {safe_html(value) or "—"}
                        </div>
                    </div>
                    """)


# =========================================================
# PAGINATION
# =========================================================

st.divider()

prev_col, page_col, next_col = st.columns(
    [1, 1.2, 1],
    gap="small",
)

with prev_col:
    st.button(
        "← Previous",
        key="intimation_prev_btn",
        disabled=current_page <= 1,
        use_container_width=True,
        on_click=lambda: st.session_state.update(
            intimation_page=max(
                1,
                st.session_state["intimation_page"] - 1,
            )
        ),
    )

with page_col:
    st.markdown(
        f'<div class="page-indicator">'
        f'Page {current_page} of {total_pages}'
        f'</div>',
        unsafe_allow_html=True,
    )

with next_col:
    st.button(
        "Next →",
        key="intimation_next_btn",
        disabled=current_page >= total_pages,
        use_container_width=True,
        on_click=lambda: st.session_state.update(
            intimation_page=min(
                total_pages,
                st.session_state["intimation_page"] + 1,
            )
        ),
    )
