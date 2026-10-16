
from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
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
# AESTHETIC CSS
# =========================================================

st.html("""
<style>
.block-container {
    max-width: 1550px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.dash-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 5px;
}

.dash-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 48px;
    height: 48px;
    border-radius: 15px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 25px;
    box-shadow: 0 5px 16px rgba(79,70,229,0.22);
}

.dash-title {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.8px;
    line-height: 1.2;
}

.dash-subtitle {
    font-size: 12px;
    opacity: 0.68;
    margin-top: 5px;
}

.section-heading {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -0.3px;
    margin-bottom: 2px;
}

.section-caption {
    font-size: 11px;
    opacity: 0.65;
    margin-bottom: 7px;
}

div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px;
    padding: 14px 15px;
    min-height: 95px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.025);
}

div[data-testid="stMetricLabel"] {
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.23) !important;
    border-radius: 13px !important;
    overflow: hidden;
    background: var(--secondary-background-color);
    margin-bottom: 0 !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.025);
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.65) !important;
}

div[data-testid="stExpander"] details summary {
    padding: 11px 12px !important;
    min-height: 57px;
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
    margin-top: 4px;
}

.record-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    border-top: 1px solid rgba(128,128,128,0.17);
    margin-top: 10px;
    padding-top: 9px;
}

.record-poc {
    font-size: 11px;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.record-capacity {
    font-size: 14px;
    font-weight: 800;
    white-space: nowrap;
}

.status-pill {
    display: inline-block;
    font-size: 10px;
    font-weight: 750;
    border-radius: 20px;
    padding: 4px 8px;
    margin-top: 8px;
    margin-right: 4px;
    background: rgba(128,128,128,0.12);
    border: 1px solid rgba(128,128,128,0.12);
}

.status-good {
    color: #15803d;
    background: rgba(34,197,94,0.10);
    border-color: rgba(34,197,94,0.22);
}

.status-pending {
    color: #b45309;
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.24);
}

.status-alert {
    color: #dc2626;
    background: rgba(239,68,68,0.10);
    border-color: rgba(239,68,68,0.22);
}

.detail-heading {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 800;
    opacity: 0.65;
    margin-bottom: 7px;
}

.detail-item {
    padding: 7px 0;
    border-bottom: 1px solid rgba(128,128,128,0.13);
}

.detail-label {
    font-size: 10px;
    opacity: 0.65;
    margin-bottom: 2px;
}

.detail-value {
    font-size: 12px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.result-caption {
    font-size: 12px;
    opacity: 0.7;
    margin-top: 3px;
    margin-bottom: 8px;
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
    min-height: 36px;
}

div[data-testid="stPlotlyChart"] {
    border-radius: 12px;
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .dash-title {
        font-size: 22px;
    }

    div[data-testid="stMetric"] {
        padding: 9px;
        min-height: 80px;
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


def html_text(value):
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
        "complete", "checked", "verified", "approved", "done", "pass"
    ]):
        return "status-good"

    if any(word in value for word in [
        "pending", "awaiting", "in progress", "hold"
    ]):
        return "status-pending"

    if any(word in value for word in [
        "reject", "error", "fail", "issue", "not checked"
    ]):
        return "status-alert"

    return ""


def add_pagination_buttons(page, total_pages):
    """Update page number from button callbacks."""
    return


# =========================================================
# LOAD AND PREPARE EXCEL
# =========================================================

@st.cache_data(show_spinner="Loading intimation records...")
def load_data(file_path, modified_time):
    # modified_time ensures cache invalidation after Excel changes.
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

        if source is None:
            output[target] = ""
        else:
            output[target] = raw[source]

    if priority_source is not None:
        output["Priority"] = raw[priority_source]
    else:
        output["Priority"] = ""

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
            "Lat",
            "Long",
            "PSS",
            "State",
            "Capacity (MW)",
            "Priority",
            "POC1",
            "POC 2",
            "POC 3",
            "Check Status",
        ]
    ]

    has_identity = (
        output["PSS"].ne("")
        | output["State"].ne("")
        | output["POC1"].ne("")
    )

    return output.loc[has_identity].reset_index(drop=True)


# =========================================================
# CHART HELPERS
# =========================================================

def render_bar_chart(data, x, y, title, color=None):
    if data.empty:
        st.caption("No data available for this chart.")
        return

    if PLOTLY_AVAILABLE:
        fig = px.bar(
            data,
            x=x,
            y=y,
            title=title,
            color=color,
            text_auto=".2s",
            orientation="v",
        )

        fig.update_layout(
            height=310,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            title=dict(font=dict(size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.bar_chart(
            data.set_index(x)[y],
            height=300,
        )


def render_donut_chart(data, names, values, title):
    if data.empty:
        st.caption("No status data available.")
        return

    if PLOTLY_AVAILABLE:
        fig = px.pie(
            data,
            names=names,
            values=values,
            hole=0.62,
            title=title,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=(
                "%{label}<br>Records: %{value:,}"
                "<br>Share: %{percent}<extra></extra>"
            ),
        )

        fig.update_layout(
            height=310,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.bar_chart(
            data.set_index(names)[values],
            height=300,
        )


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="dash-header">
    <div class="dash-logo">⚡</div>
    <div>
        <div class="dash-title">Intimation Dashboard</div>
        <div class="dash-subtitle">
            Capacity intelligence · PSS records · Status tracking
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
# STATE INITIALIZATION
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
# Overview is rendered above filters using a placeholder.
# =========================================================

overview_slot = st.container()

st.html("""
<div class="section-heading">🎛️ Filters</div>
<div class="section-caption">
    Choose any combination — charts and records update together
</div>
""")

poc_col, pss_col, state_col = st.columns(
    [1, 1.5, 1.2],
    gap="medium",
)

with pss_col:
    st.text_input(
        "🔎 Search PSS",
        placeholder="Type a PSS name...",
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


# ---------- DYNAMIC POC OPTIONS ----------
selected_states = st.session_state["intimation_states"]

poc_option_mask = pss_mask.copy()

if selected_states:
    poc_option_mask &= df["State"].isin(selected_states)

poc_values = unique_sorted(df.loc[poc_option_mask, "POC1"])

poc_options = [ALL_USERS] + [
    value for value in poc_values
    if value != ALL_USERS
]

if st.session_state["intimation_poc1"] not in poc_options:
    st.session_state["intimation_poc1"] = ALL_USERS

with poc_col:
    st.selectbox(
        "👤 POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]


# ---------- DYNAMIC STATE OPTIONS ----------
state_option_mask = pss_mask.copy()

if selected_poc != ALL_USERS:
    state_option_mask &= df["POC1"].eq(selected_poc)

available_states = unique_sorted(
    df.loc[state_option_mask, "State"]
)

current_states = st.session_state["intimation_states"]

valid_states = [
    value for value in current_states
    if value in available_states
]

if valid_states != current_states:
    st.session_state["intimation_states"] = valid_states

with state_col:
    st.multiselect(
        "📍 State",
        options=available_states,
        key="intimation_states",
        placeholder="All states",
        on_change=reset_page,
    )

selected_states = st.session_state["intimation_states"]


# =========================================================
# FILTER DATA
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
# OVERVIEW METRICS + GRAPHS
# =========================================================

with overview_slot:
    st.html("""
    <div class="section-heading">📊 Overview</div>
    <div class="section-caption">
        Live analytics from the currently filtered records
    </div>
    """)

    total_capacity = filtered_df["Capacity (MW)"].sum(
        min_count=1
    )

    if pd.isna(total_capacity):
        total_capacity = 0.0

    unique_pss = filtered_df.loc[
        filtered_df["PSS"].ne(""), "PSS"
    ].nunique()

    m1, m2, m3 = st.columns(3, gap="medium")

    m1.metric(
        "📁 Matching Records",
        f"{len(filtered_df):,}",
    )

    m2.metric(
        "⚡ Total Capacity",
        f"{total_capacity:,.2f} MW",
    )

    m3.metric(
        "📍 Unique PSS",
        f"{unique_pss:,}",
    )

    st.markdown("")

    if filtered_df.empty:
        st.info("Select different filters to view analytics.")
    else:
        # ---------- CAPACITY BY STATE ----------
        chart_left, chart_right = st.columns(
            2,
            gap="medium",
        )

        with chart_left:
            st.markdown("##### ⚡ Capacity by State")

            capacity_state = (
                filtered_df.loc[
                    filtered_df["State"].ne("")
                    & filtered_df["Capacity (MW)"].notna()
                ]
                .groupby("State", as_index=False)["Capacity (MW)"]
                .sum()
                .sort_values("Capacity (MW)", ascending=False)
                .head(10)
                .sort_values("Capacity (MW)", ascending=True)
            )

            if not capacity_state.empty:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(
                        capacity_state,
                        x="Capacity (MW)",
                        y="State",
                        orientation="h",
                        text="Capacity (MW)",
                        color="Capacity (MW)",
                        color_continuous_scale=["#93C5FD", "#4F46E5"],
                    )

                    fig.update_traces(
                        texttemplate="%{x:,.1f}",
                        textposition="outside",
                        cliponaxis=False,
                    )

                    fig.update_layout(
                        height=320,
                        margin=dict(l=5, r=35, t=10, b=10),
                        coloraxis_showscale=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Capacity (MW)",
                        yaxis_title="",
                        font=dict(size=10),
                    )

                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.15)",
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
                    )
            else:
                st.caption("No capacity data available.")

        # ---------- CHECK STATUS DONUT ----------
        with chart_right:
            st.markdown("##### 🟢 Check Status Distribution")

            status_data = filtered_df.copy()

            status_data["Display Status"] = (
                status_data["Check Status"]
                .replace("", "Not specified")
            )

            status_summary = (
                status_data["Display Status"]
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Records")
                .head(8)
            )

            if not status_summary.empty:
                if PLOTLY_AVAILABLE:
                    fig = px.pie(
                        status_summary,
                        names="Status",
                        values="Records",
                        hole=0.60,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )

                    fig.update_traces(
                        textposition="inside",
                        textinfo="percent",
                        hovertemplate=(
                            "%{label}<br>Records: %{value:,}"
                            "<br>Share: %{percent}<extra></extra>"
                        ),
                    )

                    fig.update_layout(
                        height=320,
                        margin=dict(l=5, r=5, t=10, b=15),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.12,
                            xanchor="center",
                            x=0.5,
                        ),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                else:
                    st.bar_chart(
                        status_summary.set_index("Status")[
                            ["Records"]
                        ]
                    )
            else:
                st.caption("No status data available.")

        # ---------- PRIORITY DISTRIBUTION ----------
        st.markdown("##### 🎯 Priority Distribution")

        priority_data = filtered_df.copy()
        priority_data["Display Priority"] = (
            priority_data["Priority"].replace("", "Not specified")
        )

        priority_summary = (
            priority_data["Display Priority"]
            .value_counts()
            .rename_axis("Priority")
            .reset_index(name="Records")
            .head(10)
        )

        if not priority_summary.empty:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    priority_summary,
                    x="Priority",
                    y="Records",
                    text="Records",
                    color="Records",
                    color_continuous_scale=["#C4B5FD", "#6D28D9"],
                )

                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                )

                fig.update_layout(
                    height=300,
                    margin=dict(l=5, r=15, t=10, b=10),
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Priority",
                    yaxis_title="Records",
                    font=dict(size=10),
                )

                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(
                    gridcolor="rgba(128,128,128,0.15)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.bar_chart(
                    priority_summary.set_index("Priority")[
                        ["Records"]
                    ]
                )
        else:
            st.caption("No priority data available.")

    st.divider()


# =========================================================
# FIND RECORDS
# =========================================================

st.html("""
<div class="section-heading">🗂️ Find Records</div>
<div class="section-caption">
    Open any card to see complete record details
</div>
""")

if filtered_df.empty:
    st.info(
        "No records match these filters. "
        "Try changing POC1, PSS search or State."
    )
    st.stop()


# =========================================================
# PAGINATION
# =========================================================

total_records = len(filtered_df)
total_pages = max(1, math.ceil(total_records / PAGE_SIZE))

current_page = st.session_state["intimation_page"]
current_page = max(1, min(current_page, total_pages))
st.session_state["intimation_page"] = current_page

start = (current_page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, total_records)

st.html(f"""
<div class="result-caption">
    Showing <b>{start + 1:,}–{end:,}</b>
    of <b>{total_records:,}</b> matching records
</div>
""")

page_df = filtered_df.iloc[start:end]


# =========================================================
# RECORD CARDS
# =========================================================

left_cards, right_cards = st.columns(2, gap="medium")

for index, (_, record) in enumerate(page_df.iterrows()):
    target = left_cards if index % 2 == 0 else right_cards

    pss = clean(record["PSS"]) or "Unnamed PSS"
    state = clean(record["State"]) or "State unavailable"
    poc1 = clean(record["POC1"]) or "Not specified"
    capacity = format_capacity(record["Capacity (MW)"])

    status = clean(record["Check Status"]) or "Not specified"
    priority = clean(record["Priority"]) or "Not specified"

    status_class = status_style(status)

    with target:
        # One expander equals one complete card.
        # Details are integrated within the same border.
        with st.expander(
            f"⚡ {pss}  ·  {capacity}",
            expanded=False,
        ):
            st.html(f"""
            <div class="record-title">
                ⚡ {html_text(pss)}
            </div>

            <div class="record-state">
                📍 {html_text(state)}
            </div>

            <span class="status-pill {status_class}">
                {"✓" if status_class == "status-good" else
                 "◷" if status_class == "status-pending" else
                 "!" if status_class == "status-alert" else "●"}
                {html_text(status)}
            </span>

            <span class="status-pill">
                🎯 Priority: {html_text(priority)}
            </span>

            <div class="record-row">
                <div class="record-poc">
                    👤 <b>POC1</b><br>
                    {html_text(poc1)}
                </div>
                <div class="record-capacity">
                    {html_text(capacity)}
                </div>
            </div>

            <div style="height: 13px;"></div>
            <div class="detail-heading">📋 Record Details</div>
            """)

            details = [
                ("🎯 Priority", record["Priority"]),
                ("👤 POC 2", record["POC 2"]),
                ("👤 POC 3", record["POC 3"]),
                ("🌐 Latitude", record["Lat"]),
                ("🌐 Longitude", record["Long"]),
                ("✓ Check Status", record["Check Status"]),
            ]

            detail_left, detail_right = st.columns(2, gap="small")

            for detail_index, (label, value) in enumerate(details):
                detail_column = (
                    detail_left
                    if detail_index % 2 == 0
                    else detail_right
                )

                with detail_column:
                    st.html(f"""
                    <div class="detail-item">
                        <div class="detail-label">
                            {html_text(label)}
                        </div>
                        <div class="detail-value">
                            {html_text(value) or "—"}
                        </div>
                    </div>
                    """)


# =========================================================
# PAGINATION CONTROLS
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
