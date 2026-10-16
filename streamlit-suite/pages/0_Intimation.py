
from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
    layout="wide",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"
PAGE_SIZE = 20
ALL_POC_LABEL = "All Users"


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    .dashboard-heading {
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .dashboard-subtitle {
        font-size: 12px;
        opacity: 0.70;
        margin-bottom: 12px;
    }

    .overview-heading {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .intimation-card {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 3px;
        min-height: 82px;
        overflow-wrap: anywhere;
    }

    .intimation-card-title {
        font-size: 13px;
        font-weight: 750;
        line-height: 1.3;
        margin-bottom: 3px;
    }

    .intimation-card-state {
        font-size: 11px;
        opacity: 0.72;
        line-height: 1.25;
        margin-bottom: 6px;
    }

    .intimation-card-bottom {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }

    .intimation-card-poc {
        font-size: 11px;
        line-height: 1.25;
        flex: 1;
        min-width: 0;
        overflow-wrap: anywhere;
    }

    .intimation-card-capacity {
        font-size: 12px;
        font-weight: 750;
        white-space: nowrap;
    }

    div[data-testid="stExpander"] {
        margin-top: 0;
        margin-bottom: 7px;
    }

    div[data-testid="stExpander"] details summary {
        font-size: 11px;
        padding-top: 3px;
        padding-bottom: 3px;
    }

    div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
        padding-top: 3px;
        padding-bottom: 5px;
    }

    .detail-line {
        font-size: 11px;
        line-height: 1.55;
        overflow-wrap: anywhere;
    }

    .pagination-label {
        font-size: 12px;
        opacity: 0.75;
        text-align: center;
        padding-top: 7px;
    }

    div[data-testid="stMetric"] {
        padding: 9px 11px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 8px;
        gap: 2px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 11px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 23px;
    }

    @media (max-width: 700px) {
        .dashboard-heading {
            font-size: 21px;
        }

        .intimation-card {
            padding: 7px 8px;
        }
    }
</style>
""")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_header(value):
    """Normalize column headers for flexible matching."""
    return "".join(
        str(value).strip().casefold().split()
    ).replace("_", "")


def find_column(df, candidates, startswith=None):
    """Find a column using alternative header names."""
    normalized = {
        normalize_header(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalize_header(candidate)
        if key in normalized:
            return normalized[key]

    if startswith:
        prefix = normalize_header(startswith)
        for column in df.columns:
            if normalize_header(column).startswith(prefix):
                return column

    return None


def clean_text(value):
    """Convert missing values to an empty string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def unique_sorted(series):
    """Return unique, non-empty values in sorted order."""
    values = {
        clean_text(value)
        for value in series
        if clean_text(value)
    }
    return sorted(values, key=str.casefold)


def reset_page():
    st.session_state["intimation_page"] = 1


def safe_number(value):
    """Format a numeric value without raising errors."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return f"{number:,.2f}"


def safe_html(value):
    """Escape all spreadsheet text before displaying HTML."""
    return escape(clean_text(value))


# ============================================================
# LOAD AND PREPARE EXCEL DATA
# ============================================================

@st.cache_data(show_spinner="Loading intimation records...")
def load_data(file_path, modified_time):
    del modified_time  # Used to invalidate the cache when Excel changes.

    raw_df = pd.read_excel(file_path)
    raw_df.columns = [
        str(column).strip()
        for column in raw_df.columns
    ]

    # Prefer the specified Priority column, with fallbacks.
    priority_col = find_column(
        raw_df,
        [
            "Priority as per 003 sheet",
            "Priority as per Amresh Sir 28.07.2023",
        ],
    )

    if priority_col is None:
        priority_col = find_column(
            raw_df,
            [],
            startswith="Priority",
        )

    # Find alternative versions of expected headers.
    column_candidates = {
        "Lat": ["Lat", "Latitude"],
        "Long": ["Long", "Longitude", "Lon"],
        "PSS": ["PSS", "PSS Name"],
        "State": ["State"],
        "Capacity (MW)": ["Capacity (MW)", "Capacity MW", "Capacity"],
        "POC1": ["POC1", "POC 1"],
        "POC 2": ["POC 2", "POC2"],
        "POC 3": ["POC 3", "POC3"],
        "Check Status": ["Check Status", "Status"],
    }

    data = pd.DataFrame(index=raw_df.index)

    for target, candidates in column_candidates.items():
        source = find_column(raw_df, candidates)

        if source is not None:
            data[target] = raw_df[source]
        else:
            data[target] = ""

    if priority_col is not None:
        data["Priority"] = raw_df[priority_col]
    else:
        data["Priority"] = ""

    # These fields are displayed as text.
    text_columns = [
        "PSS",
        "State",
        "POC1",
        "POC 2",
        "POC 3",
        "Priority",
        "Check Status",
    ]

    for column in text_columns:
        data[column] = data[column].map(clean_text)

    # Numeric capacity is used for accurate overview totals.
    data["Capacity (MW)"] = pd.to_numeric(
        data["Capacity (MW)"],
        errors="coerce",
    )

    data["Lat"] = data["Lat"].map(clean_text)
    data["Long"] = data["Long"].map(clean_text)

    # Keep the expected output columns only.
    data = data[
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

    # Remove rows with no useful record identity.
    data = data[
        data["PSS"].ne("")
        | data["State"].ne("")
        | data["POC1"].ne("")
    ].copy()

    return data.reset_index(drop=True)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.html("""
<div class="dashboard-heading">Intimation Dashboard</div>
<div class="dashboard-subtitle">
    Search, filter and review intimation records
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
    st.error(f"Unable to load intimation.xlsx: {error}")
    st.stop()

if df.empty:
    st.warning("No records were found in the Excel file.")
    st.stop()


# ============================================================
# CREATE OVERVIEW SLOT ABOVE FILTERS
# The slot is filled after the connected filters are evaluated.
# ============================================================

overview_slot = st.container()


# ============================================================
# FILTERS — INTERCONNECTED / CASCADING
#
# POC1 options depend on PSS search + selected states.
# State options depend on PSS search + selected POC1.
# PSS search affects the available options for both filters.
# ============================================================

st.markdown("### Filters")

# Initialize persistent widget state.
if "intimation_poc1" not in st.session_state:
    st.session_state["intimation_poc1"] = ALL_POC_LABEL

if "intimation_states" not in st.session_state:
    st.session_state["intimation_states"] = []

if "intimation_pss_search" not in st.session_state:
    st.session_state["intimation_pss_search"] = ""

if "intimation_page" not in st.session_state:
    st.session_state["intimation_page"] = 1


filter_col1, filter_col2, filter_col3 = st.columns(
    [1.0, 1.5, 1.2],
    gap="small",
)

# PSS search is the starting point for option discovery.
with filter_col2:
    st.text_input(
        "Search PSS",
        placeholder="Type PSS name...",
        key="intimation_pss_search",
        on_change=reset_page,
    )

pss_query = st.session_state["intimation_pss_search"].strip()

# Build a PSS mask.
if pss_query:
    pss_mask = df["PSS"].str.contains(
        pss_query,
        case=False,
        na=False,
        regex=False,
    )
else:
    pss_mask = pd.Series(True, index=df.index)

# Get currently selected states before rendering POC1.
selected_states = st.session_state.get(
    "intimation_states", []
)

# POC1 options depend on search and selected states.
poc_discovery_mask = pss_mask.copy()

if selected_states:
    poc_discovery_mask &= df["State"].isin(selected_states)

poc_values = unique_sorted(
    df.loc[poc_discovery_mask, "POC1"]
)

poc_options = [ALL_POC_LABEL] + [
    value for value in poc_values
    if value != ALL_POC_LABEL
]

current_poc = st.session_state.get(
    "intimation_poc1", ALL_POC_LABEL
)

# Keep the current selection valid.
if current_poc not in poc_options:
    current_poc = ALL_POC_LABEL
    st.session_state["intimation_poc1"] = ALL_POC_LABEL

with filter_col1:
    st.selectbox(
        "POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]

# State options depend on PSS search and selected POC1.
state_discovery_mask = pss_mask.copy()

if selected_poc != ALL_POC_LABEL:
    state_discovery_mask &= df["POC1"].eq(selected_poc)

available_states = unique_sorted(
    df.loc[state_discovery_mask, "State"]
)

# Preserve only state selections still valid for this POC1 + PSS.
selected_states = st.session_state.get(
    "intimation_states", []
)

valid_selected_states = [
    state for state in selected_states
    if state in available_states
]

if valid_selected_states != selected_states:
    st.session_state["intimation_states"] = valid_selected_states

with filter_col3:
    st.multiselect(
        "State",
        options=available_states,
        key="intimation_states",
        placeholder="All states",
        on_change=reset_page,
    )

selected_states = st.session_state["intimation_states"]


# ============================================================
# APPLY ALL ACTIVE FILTERS TO THE DATA
# ============================================================

filtered_df = df.copy()

# PSS search
if pss_query:
    filtered_df = filtered_df[
        filtered_df["PSS"].str.contains(
            pss_query,
            case=False,
            na=False,
            regex=False,
        )
    ]

# POC1 selection
if selected_poc != ALL_POC_LABEL:
    filtered_df = filtered_df[
        filtered_df["POC1"].eq(selected_poc)
    ]

# State selection
if selected_states:
    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_states)
    ]

filtered_df = filtered_df.reset_index(drop=True)


# ============================================================
# OVERVIEW — DISPLAYED ABOVE FILTERS AND RECORDS
# ============================================================

with overview_slot:
    st.markdown(
        '<div class="overview-heading">Overview</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3, gap="small")

    total_capacity = filtered_df["Capacity (MW)"].sum(
        min_count=1
    )

    if pd.isna(total_capacity):
        total_capacity = 0.0

    unique_pss = filtered_df.loc[
        filtered_df["PSS"].ne(""), "PSS"
    ].nunique()

    metric1.metric(
        "Matching Records",
        f"{len(filtered_df):,}",
    )

    metric2.metric(
        "Total Capacity",
        f"{total_capacity:,.2f} MW",
    )

    metric3.metric(
        "Unique PSS",
        f"{unique_pss:,}",
    )

    st.divider()


# ============================================================
# FIND RECORDS
# ============================================================

st.markdown("### Find Records")

if filtered_df.empty:
    st.info(
        "No records match the selected filters. "
        "Try changing POC1, PSS search or State."
    )
    st.stop()

total_records = len(filtered_df)
total_pages = max(1, math.ceil(total_records / PAGE_SIZE))

# Clamp the page when filtering reduces the result count.
current_page = st.session_state.get("intimation_page", 1)
current_page = max(1, min(current_page, total_pages))
st.session_state["intimation_page"] = current_page

start_index = (current_page - 1) * PAGE_SIZE
end_index = min(start_index + PAGE_SIZE, total_records)

st.caption(
    f"Showing {start_index + 1:,}–{end_index:,} "
    f"of {total_records:,} records"
)


# ============================================================
# COMPACT RECORD CARDS — TWO COLUMNS
# ============================================================

page_df = filtered_df.iloc[start_index:end_index]

left_column, right_column = st.columns(2, gap="small")

for position, (_, record) in enumerate(page_df.iterrows()):
    target_column = (
        left_column if position % 2 == 0 else right_column
    )

    with target_column:
        pss = safe_html(record["PSS"]) or "Unnamed PSS"
        state = safe_html(record["State"]) or "State unavailable"
        poc1 = safe_html(record["POC1"]) or "Not specified"

        capacity_value = record["Capacity (MW)"]
        if pd.isna(capacity_value):
            capacity_label = "—"
        else:
            capacity_label = f"{capacity_value:,.2f} MW"

        card_html = f"""
        <div class="intimation-card">
            <div class="intimation-card-title">{pss}</div>
            <div class="intimation-card-state">{state}</div>
            <div class="intimation-card-bottom">
                <div class="intimation-card-poc">
                    <b>POC1:</b> {poc1}
                </div>
                <div class="intimation-card-capacity">
                    {safe_html(capacity_label)}
                </div>
            </div>
        </div>
        """

        st.html(card_html)

        with st.expander("View details", expanded=False):
            details = [
                ("Priority", record["Priority"]),
                ("POC 2", record["POC 2"]),
                ("POC 3", record["POC 3"]),
                ("Latitude", record["Lat"]),
                ("Longitude", record["Long"]),
                ("Check Status", record["Check Status"]),
            ]

            for label, value in details:
                value_text = safe_html(value) or "—"

                st.html(
                    f"""
                    <div class="detail-line">
                        <b>{escape(label)}:</b> {value_text}
                    </div>
                    """
                )


# ============================================================
# PAGINATION
# ============================================================

st.divider()

page_col1, page_col2, page_col3 = st.columns(
    [1, 2, 1],
    gap="small",
)

with page_col1:
    st.button(
        "← Previous",
        disabled=(current_page <= 1),
        use_container_width=True,
        on_click=lambda: st.session_state.update(
            intimation_page=max(
                1,
                st.session_state["intimation_page"] - 1,
            )
        ),
    )

with page_col2:
    st.markdown(
        f'<div class="pagination-label">'
        f'Page {current_page} of {total_pages}'
        f'</div>',
        unsafe_allow_html=True,
    )

with page_col3:
    st.button(
        "Next →",
        disabled=(current_page >= total_pages),
        use_container_width=True,
        on_click=lambda: st.session_state.update(
            intimation_page=min(
                total_pages,
                st.session_state["intimation_page"] + 1,
            )
        ),
    )
