
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
# CUSTOM CSS
# =========================================================

st.html("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 1.15rem;
    padding-bottom: 2rem;
}

.dash-header {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 5px;
}

.dash-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 47px;
    height: 47px;
    border-radius: 15px;
    color: white;
    font-size: 24px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    box-shadow: 0 5px 16px rgba(79,70,229,0.20);
}

.dash-title {
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -0.8px;
    line-height: 1.2;
}

.dash-subtitle {
    font-size: 11px;
    opacity: 0.65;
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
    margin-bottom: 8px;
}

div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 13px;
    padding: 10px 13px;
    min-height: 78px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.025);
}

div[data-testid="stMetricLabel"] {
    font-size: 11px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.22) !important;
    border-radius: 13px !important;
    overflow: hidden;
    background: var(--secondary-background-color);
    margin-bottom: 0 !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.025);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 5px 15px rgba(79,70,229,0.07);
}

div[data-testid="stExpander"] details summary {
    padding: 10px 12px !important;
    min-height: 68px;
}

div[data-testid="stExpander"] details summary p {
    font-size: 12px;
    font-weight: 700;
    line-height: 1.65;
    overflow-wrap: anywhere;
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

.record-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    border-top: 1px solid rgba(128,128,128,0.16);
    margin-top: 10px;
    padding-top: 9px;
}

.record-poc {
    font-size: 11px;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.record-capacity {
    font-size: 15px;
    font-weight: 850;
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
    background: rgba(128,128,128,0.10);
    border: 1px solid rgba(128,128,128,0.14);
}

.status-good {
    color: #15803d;
    background: rgba(34,197,94,0.10);
    border-color: rgba(34,197,94,0.22);
}

.status-pending {
    color: #b45309;
    background: rgba(245,158,11,0.11);
    border-color: rgba(245,158,11,0.22);
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
    min-height: 44px;
}

.detail-label {
    font-size: 10px;
    opacity: 0.65;
    margin-bottom: 3px;
}

.detail-value {
    font-size: 12px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.result-caption {
    font-size: 11px;
    opacity: 0.7;
    margin-top: 3px;
    margin-bottom: 8px;
}

.page-indicator {
    text-align: center;
    font-size: 12px;
    font-weight: 750;
    padding-top: 9px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 35px;
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
        padding: 8px;
        min-height: 74px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 17px;
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


def format_capacity(value):
    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return "—"

    return f"{number:,.2f} MW"


def status_style(value):
    value = clean(value).casefold()

    if any(word in value for word in [
        "complete", "checked", "verified",
        "approved", "done", "pass",
    ]):
        return "status-good"

    if any(word in value for word in [
        "pending", "awaiting", "in progress", "hold",
    ]):
        return "status-pending"

    if any(word in value for word in [
        "reject", "error", "fail", "issue", "not checked",
    ]):
        return "status-alert"

    return ""


def reset_page():
    st.session_state["intimation_page"] = 1


# =========================================================
# LOAD EXCEL
# =========================================================

@st.cache_data(show_spinner="Loading intimation records...")
def load_data(file_path, modified_time):
    # Modified time invalidates cache when the workbook changes.
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

    output["Priority"] = (
        raw[priority_source]
        if priority_source is not None
        else ""
    )

    for column in [
        "Lat", "Long", "PSS", "State",
        "Priority", "POC1", "POC 2", "POC 3",
        "Check Status",
    ]:
        output[column] = output[column].map(clean)

    output["Capacity (MW)"] = pd.to_numeric(
        output["Capacity (MW)"],
        errors="coerce",
    )

    output = output[
        [
            "Lat", "Long", "PSS", "State",
            "Capacity (MW)", "Priority", "POC1",
            "POC 2", "POC 3", "Check Status",
        ]
    ]

    has_identity = (
        output["PSS"].ne("")
        | output["State"].ne("")
        | output["POC1"].ne("")
    )

    return output.loc[has_identity].reset_index(drop=True)


# =========================================================
# PAGE HEADER
# =========================================================

st.html("""
<div class="dash-header">
    <div class="dash-logo">⚡</div>
    <div>
        <div class="dash-title">Intimation Dashboard</div>
        <div class="dash-subtitle">
            PSS records · Capacity overview · Status tracking
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
# SESSION STATE
# =========================================================

defaults = {
    "intimation_poc1": ALL_USERS,
    "intimation_states": [],
    "intimation_pss_search": "",
    "intimation_page": 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# FILTERS
# =========================================================

overview_slot = st.container()

st.html("""
<div class="section-heading">🎛️ Filters</div>
<div class="section-caption">
    All filters work together and update the overview and records
</div>
""")

poc_col, pss_col, state_col = st.columns(
    [1.0, 1.45, 1.2],
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


# POC options depend on the search and selected states.
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
        "👤 POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]


# State options depend on the search and selected POC.
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
        "📍 State",
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
# =========================================================

with overview_slot:
    st.html("""
    <div class="section-heading">📊 Overview</div>
    <div class="section-caption">
        Summary and charts based on the current filters
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

    metric1, metric2, metric3 = st.columns(3, gap="small")

    metric1.metric(
        "📁 Matching Records",
        f"{len(filtered_df):,}",
    )

    metric2.metric(
        "⚡ Total Capacity",
        f"{total_capacity:,.2f} MW",
    )

    metric3.metric(
        "📍 Unique PSS",
        f"{unique_pss:,}",
    )

    if not filtered_df.empty:
        chart_left, chart_right = st.columns(
            2,
            gap="medium",
        )

        # ---------------------------------------------
        # CAPACITY BY STATE
        # ---------------------------------------------
        with chart_left:
            st.markdown("##### ⚡ Capacity by State")

            capacity_state = filtered_df[
                filtered_df["State"].ne("")
                & filtered_df["Capacity (MW)"].notna()
            ]

            capacity_state = (
                capacity_state
                .groupby("State", as_index=False)["Capacity (MW)"]
                .sum()
                .sort_values("Capacity (MW)", ascending=False)
                .head(8)
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
                        color_continuous_scale=[
                            "#BFDBFE", "#4F46E5"
                        ],
                    )

                    fig.update_traces(
                        texttemplate="%{x:,.1f}",
                        textposition="outside",
                        cliponaxis=False,
                    )

                    fig.update_layout(
                        height=235,
                        margin=dict(l=5, r=35, t=5, b=5),
                        coloraxis_showscale=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Capacity (MW)",
                        yaxis_title="",
                        font=dict(size=10),
                    )

                    fig.update_xaxes(
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.14)",
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
                        height=235,
                        horizontal=True,
                    )
            else:
                st.caption("No capacity data available.")

        # ---------------------------------------------
        # STATUS DISTRIBUTION
        # ---------------------------------------------
        with chart_right:
            st.markdown("##### 🟢 Check Status")

            status_data = filtered_df["Check Status"].replace(
                "", "Not specified"
            )

            status_summary = (
                status_data
                .value_counts()
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
                        hole=0.64,
                        color_discrete_sequence=(
                            px.colors.qualitative.Set2
                        ),
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
                        height=235,
                        margin=dict(l=5, r=5, t=5, b=12),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=10),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.13,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=9),
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
                        ],
                        height=235,
                    )
            else:
                st.caption("No status data available.")

    else:
        st.info("No records match the current filters.")

    st.divider()


# =========================================================
# RECORDS SECTION
# =========================================================

st.html("""
<div class="section-heading">🗂️ Find Records</div>
<div class="section-caption">
    Compact cards · Click a record to reveal all details
</div>
""")

if filtered_df.empty:
    st.stop()


# =========================================================
# PAGINATION
# =========================================================

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
    Showing <b>{start + 1:,}–{end:,}</b> of
    <b>{total_records:,}</b> matching records
</div>
""")

page_df = filtered_df.iloc[start:end]


# =========================================================
# EXPANDABLE CARDS
# COLLAPSED: PSS NAME + STATE + CAPACITY
# EXPANDED: ALL OTHER DETAILS
# =========================================================

left_cards, right_cards = st.columns(2, gap="medium")

for index, (_, record) in enumerate(page_df.iterrows()):
    target = left_cards if index % 2 == 0 else right_cards

    pss = clean(record["PSS"]) or "Unnamed PSS"
    state = clean(record["State"]) or "State unavailable"
    capacity = format_capacity(record["Capacity (MW)"])

    with target:
        with st.expander(
            f"⚡ {pss}  ·  📍 {state}  ·  {capacity}",
            expanded=False,
        ):
            status = clean(record["Check Status"]) or "Not specified"
            priority = clean(record["Priority"]) or "Not specified"
            status_class = status_style(status)

            st.html(f"""
            <div class="record-title">
                ⚡ {html_text(pss)}
            </div>

            <div class="record-state">
                📍 State: {html_text(state)}
            </div>

            <div class="record-row">
                <div class="record-poc">
                    ⚡ <b>Installed Capacity</b>
                </div>
                <div class="record-capacity">
                    {html_text(capacity)}
                </div>
            </div>

            <div style="height: 6px;"></div>

            <span class="status-pill {status_class}">
                {"✓" if status_class == "status-good" else
                 "◷" if status_class == "status-pending" else
                 "!" if status_class == "status-alert" else "●"}
                {html_text(status)}
            </span>

            <span class="status-pill">
                🎯 Priority: {html_text(priority)}
            </span>

            <div style="height: 13px;"></div>
            <div class="detail-heading">📋 Additional Details</div>
            """)

            details = [
                ("👤 POC1", record["POC1"]),
                ("👤 POC 2", record["POC 2"]),
                ("👤 POC 3", record["POC 3"]),
                ("🎯 Priority", record["Priority"]),
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
