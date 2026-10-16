
from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Intimation | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"
PAGE_SIZE = 20
ALL_USERS = "All Users"


# =========================================================
# AESTHETIC UI / CSS
# =========================================================

st.html("""
<style>
/* ---------- GLOBAL ---------- */
.block-container {
    max-width: 1500px;
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}

[data-testid="stAppViewContainer"] {
    background: var(--background-color);
}

div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

hr {
    margin: 0.65rem 0;
    opacity: 0.20;
}

/* ---------- HEADER ---------- */
.dash-header {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 3px;
}

.dash-logo {
    width: 43px;
    height: 43px;
    min-width: 43px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #2563eb, #6366f1);
    color: white;
    font-size: 21px;
    font-weight: 800;
}

.dash-title {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.7px;
    line-height: 1.25;
}

.dash-subtitle {
    font-size: 12px;
    opacity: 0.68;
    margin-top: 4px;
}

.section-heading {
    font-size: 16px;
    font-weight: 750;
    letter-spacing: -0.2px;
    margin-bottom: 0;
}

.section-caption {
    font-size: 11px;
    opacity: 0.65;
    margin-top: -5px;
}

/* ---------- METRIC CARDS ---------- */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.17);
    border-radius: 13px;
    padding: 14px 15px;
    min-height: 94px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.025);
    gap: 5px;
}

div[data-testid="stMetricLabel"] {
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.6px;
}

/* ---------- FILTERS ---------- */
div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"],
div[data-testid="stTextInput"] {
    margin-bottom: 0;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stTextInput"] label {
    font-size: 12px;
    font-weight: 650;
}

/* ---------- RECORD CARDS ---------- */
/* Native expander border is the card border.
   Details remain connected to the same card. */

div[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.22) !important;
    border-radius: 12px !important;
    overflow: hidden;
    background: var(--secondary-background-color);
    margin-bottom: 0 !important;
    box-shadow: 0 2px 7px rgba(0,0,0,0.025);
    transition: border-color 0.15s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.65) !important;
}

div[data-testid="stExpander"] details summary {
    padding: 11px 12px !important;
    min-height: 55px;
}

div[data-testid="stExpander"] details summary:hover {
    background: rgba(128,128,128,0.035);
}

div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
    padding: 0 12px 12px 12px !important;
}

div[data-testid="stExpander"] details summary p {
    font-size: 12px;
    line-height: 1.45;
    font-weight: 650;
}

.card-heading {
    font-size: 13px;
    font-weight: 800;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.card-state {
    font-size: 11px;
    opacity: 0.70;
    margin-top: 3px;
    overflow-wrap: anywhere;
}

.card-main-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-top: 11px;
    padding-top: 9px;
    border-top: 1px solid rgba(128,128,128,0.17);
}

.card-poc {
    font-size: 11px;
    line-height: 1.5;
    overflow-wrap: anywhere;
    min-width: 0;
}

.card-capacity {
    font-size: 13px;
    font-weight: 800;
    white-space: nowrap;
}

.card-detail-heading {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 750;
    opacity: 0.65;
    margin-bottom: 8px;
}

.detail-item {
    padding: 7px 0;
    border-bottom: 1px solid rgba(128,128,128,0.12);
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
    white-space: normal;
}

.detail-item:last-child {
    border-bottom: none;
}

/* ---------- RESULTS / PAGINATION ---------- */
.results-count {
    font-size: 12px;
    opacity: 0.72;
    padding-top: 7px;
}

.page-indicator {
    text-align: center;
    font-size: 12px;
    font-weight: 650;
    padding-top: 9px;
}

div.stButton > button {
    border-radius: 9px;
    font-size: 12px;
    font-weight: 650;
    min-height: 36px;
}

.empty-state {
    text-align: center;
    border: 1px dashed rgba(128,128,128,0.35);
    border-radius: 14px;
    padding: 35px 15px;
    font-size: 13px;
    opacity: 0.8;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 700px) {
    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .dash-title {
        font-size: 21px;
    }

    div[data-testid="stMetric"] {
        padding: 10px;
        min-height: 82px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 19px;
    }
}
</style>
""")


# =========================================================
# HELPERS
# =========================================================

def norm(value):
    return "".join(
        str(value).strip().casefold().split()
    ).replace("_", "")


def find_column(df, candidates=(), startswith=None):
    lookup = {norm(c): c for c in df.columns}

    for candidate in candidates:
        key = norm(candidate)
        if key in lookup:
            return lookup[key]

    if startswith:
        prefix = norm(startswith)
        for col in df.columns:
            if norm(col).startswith(prefix):
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
    values = {clean(v) for v in series if clean(v)}
    return sorted(values, key=str.casefold)


def reset_page():
    st.session_state["intimation_page"] = 1


def format_capacity(value):
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return f"{number:,.2f} MW"


# =========================================================
# LOAD EXCEL
# =========================================================

@st.cache_data(show_spinner="Loading intimation records...")
def load_data(file_path, modified_time):
    # modified_time changes when the Excel file changes.
    del modified_time

    raw = pd.read_excel(file_path)
    raw.columns = [str(c).strip() for c in raw.columns]

    # Priority preference:
    # 1. Priority as per 003 sheet
    # 2. Priority as per Amresh Sir 28.07.2023
    # 3. First column starting with Priority
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
        output[target] = (
            raw[source] if source is not None else ""
        )

    output["Priority"] = (
        raw[priority_source]
        if priority_source is not None
        else ""
    )

    for column in [
        "Lat", "Long", "PSS", "State", "Priority",
        "POC1", "POC 2", "POC 3", "Check Status",
    ]:
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

    # Remove completely unidentifiable rows.
    has_identity = (
        output["PSS"].ne("")
        | output["State"].ne("")
        | output["POC1"].ne("")
    )

    return output.loc[has_identity].reset_index(drop=True)


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="dash-header">
    <div class="dash-logo">I</div>
    <div>
        <div class="dash-title">Intimation Dashboard</div>
        <div class="dash-subtitle">
            Record management · Capacity overview · PSS directory
        </div>
    </div>
</div>
""")

if not EXCEL_PATH.exists():
    st.error(
        f"Could not find `{EXCEL_PATH.name}` at "
        f"`{EXCEL_PATH.parent}`. Place intimation.xlsx in "
        "your repository root."
    )
    st.stop()

try:
    df = load_data(
        str(EXCEL_PATH),
        EXCEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not read the Excel file: {exc}")
    st.stop()

if df.empty:
    st.warning("No usable records were found in the Excel file.")
    st.stop()


# =========================================================
# OVERVIEW SLOT
# It is populated later but remains above the filters.
# =========================================================

overview_slot = st.container()


# =========================================================
# FILTERS
# Each filter's options are based on the other active filters.
# =========================================================

st.html("""
<div class="section-heading">Filters</div>
""")

if "intimation_poc1" not in st.session_state:
    st.session_state["intimation_poc1"] = ALL_USERS

if "intimation_states" not in st.session_state:
    st.session_state["intimation_states"] = []

if "intimation_pss_search" not in st.session_state:
    st.session_state["intimation_pss_search"] = ""

if "intimation_page" not in st.session_state:
    st.session_state["intimation_page"] = 1


filter_poc_col, filter_pss_col, filter_state_col = st.columns(
    [1.0, 1.5, 1.2],
    gap="medium",
)

# PSS search first: it constrains all other available options.
with filter_pss_col:
    st.text_input(
        "Search PSS",
        placeholder="Search by PSS name...",
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


# ---------- POC OPTIONS ----------
# POC options depend on PSS search and selected States.

selected_states = st.session_state.get(
    "intimation_states", []
)

poc_mask = pss_mask.copy()

if selected_states:
    poc_mask &= df["State"].isin(selected_states)

poc_values = unique_sorted(df.loc[poc_mask, "POC1"])

poc_options = [ALL_USERS] + [
    value for value in poc_values
    if value != ALL_USERS
]

current_poc = st.session_state.get(
    "intimation_poc1", ALL_USERS
)

if current_poc not in poc_options:
    st.session_state["intimation_poc1"] = ALL_USERS

with filter_poc_col:
    st.selectbox(
        "POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]


# ---------- STATE OPTIONS ----------
# State options depend on PSS search and selected POC1.

state_mask = pss_mask.copy()

if selected_poc != ALL_USERS:
    state_mask &= df["POC1"].eq(selected_poc)

available_states = unique_sorted(
    df.loc[state_mask, "State"]
)

selected_states = st.session_state.get(
    "intimation_states", []
)

# Remove state selections that are no longer valid.
valid_states = [
    state for state in selected_states
    if state in available_states
]

if valid_states != selected_states:
    st.session_state["intimation_states"] = valid_states

with filter_state_col:
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
# OVERVIEW METRICS
# =========================================================

with overview_slot:
    st.html("""
    <div class="section-heading">Overview</div>
    <div class="section-caption">
        Live summary based on your selected filters
    </div>
    """)

    metric_records, metric_capacity, metric_pss = st.columns(
        3,
        gap="medium",
    )

    total_capacity = filtered_df["Capacity (MW)"].sum(
        min_count=1
    )

    if pd.isna(total_capacity):
        total_capacity = 0.0

    unique_pss_count = filtered_df.loc[
        filtered_df["PSS"].ne(""), "PSS"
    ].nunique()

    metric_records.metric(
        "Matching Records",
        f"{len(filtered_df):,}",
    )

    metric_capacity.metric(
        "Total Capacity",
        f"{total_capacity:,.2f} MW",
    )

    metric_pss.metric(
        "Unique PSS",
        f"{unique_pss_count:,}",
    )

    st.divider()


# =========================================================
# FIND RECORDS
# =========================================================

st.html("""
<div class="section-heading">Find Records</div>
""")

if filtered_df.empty:
    st.html("""
    <div class="empty-state">
        <div style="font-size:25px; margin-bottom:8px;">⌕</div>
        <b>No matching records</b>
        <div style="margin-top:5px;">
            Try changing the POC1, PSS search or State filters.
        </div>
    </div>
    """)
    st.stop()


# =========================================================
# PAGINATION STATE
# =========================================================

total_records = len(filtered_df)
total_pages = max(1, math.ceil(total_records / PAGE_SIZE))

current_page = st.session_state.get("intimation_page", 1)
current_page = max(1, min(current_page, total_pages))
st.session_state["intimation_page"] = current_page

start = (current_page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, total_records)

st.html(
    f"""
    <div class="results-count">
        Showing <b>{start + 1:,}–{end:,}</b>
        of <b>{total_records:,}</b> matching records
    </div>
    """
)

page_df = filtered_df.iloc[start:end]


# =========================================================
# COMPACT, CONNECTED, EXPANDABLE CARDS
# Two columns, 20 records per page.
# =========================================================

card_col_left, card_col_right = st.columns(
    2,
    gap="medium",
)

for index, (_, record) in enumerate(page_df.iterrows()):
    column = (
        card_col_left
        if index % 2 == 0
        else card_col_right
    )

    pss_raw = clean(record["PSS"]) or "Unnamed PSS"
    state_raw = clean(record["State"]) or "State unavailable"
    poc1_raw = clean(record["POC1"]) or "Not specified"

    capacity_raw = record["Capacity (MW)"]
    capacity_label = format_capacity(capacity_raw)

    # Compact summary shown on the clickable card header.
    # Use plain text for the expander label; HTML is escaped.
    summary_label = (
        f"{pss_raw}  ·  {state_raw}  ·  {capacity_label}"
    )

    with column:
        # One native expander = one card.
        # Details are inside the same card border.
        with st.expander(
            summary_label,
            expanded=False,
        ):
            st.html(f"""
            <div class="card-heading">
                {html_text(pss_raw)}
            </div>

            <div class="card-state">
                {html_text(state_raw)}
            </div>

            <div class="card-main-row">
                <div class="card-poc">
                    <b>POC1</b><br>
                    {html_text(poc1_raw)}
                </div>
                <div class="card-capacity">
                    {html_text(capacity_label)}
                </div>
            </div>

            <div style="height: 12px;"></div>

            <div class="card-detail-heading">
                Record details
            </div>
            """)

            # Display remaining information in two compact columns.
            detail_columns = st.columns(2, gap="small")

            details = [
                ("Priority", record["Priority"]),
                ("POC 2", record["POC 2"]),
                ("POC 3", record["POC 3"]),
                ("Latitude", record["Lat"]),
                ("Longitude", record["Long"]),
                ("Check Status", record["Check Status"]),
            ]

            for detail_index, (label, value) in enumerate(details):
                target = detail_columns[detail_index % 2]

                with target:
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

previous_col, page_col, next_col = st.columns(
    [1, 1.2, 1],
    gap="small",
)

with previous_col:
    st.button(
        "← Previous",
        key="intimation_previous",
        disabled=(current_page <= 1),
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
        key="intimation_next",
        disabled=(current_page >= total_pages),
        use_container_width=True,
        on_click=lambda: st.session_state.update(
            intimation_page=min(
                total_pages,
                st.session_state["intimation_page"] + 1,
            )
        ),
    )
