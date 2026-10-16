
from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="📊",
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
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.dash-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 24px;
}

.dash-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 49px;
    height: 49px;
    flex-shrink: 0;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #6366f1);
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
    box-shadow: 0 5px 18px rgba(79,70,229,0.18);
}

.dash-title {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -0.8px;
    line-height: 1.25;
}

.dash-subtitle {
    font-size: 12px;
    opacity: 0.65;
    margin-top: 5px;
}

.section-heading {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.35px;
    margin-bottom: 3px;
}

.section-caption {
    font-size: 12px;
    opacity: 0.65;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 15px;
    padding: 17px 19px;
    min-height: 102px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.025);
}

div[data-testid="stMetricLabel"] {
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.22) !important;
    border-radius: 14px !important;
    overflow: hidden;
    background: var(--secondary-background-color);
    margin-bottom: 0 !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.025);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.65) !important;
    box-shadow: 0 5px 17px rgba(79,70,229,0.07);
}

div[data-testid="stExpander"] details summary {
    padding: 12px 14px !important;
    min-height: 62px;
}

div[data-testid="stExpander"] details summary p {
    font-size: 12px;
    font-weight: 700;
    line-height: 1.55;
}

div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
    padding: 0 14px 14px 14px !important;
}

.card-heading {
    font-size: 15px;
    font-weight: 800;
    line-height: 1.45;
    overflow-wrap: anywhere;
    margin-bottom: 12px;
}

.card-info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.card-info {
    padding: 10px 11px;
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 10px;
    background: rgba(128,128,128,0.035);
    min-width: 0;
}

.card-label {
    font-size: 10px;
    font-weight: 650;
    opacity: 0.65;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.55px;
}

.card-value {
    font-size: 12px;
    font-weight: 700;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.card-capacity {
    font-size: 14px;
    font-weight: 800;
}

.card-details-heading {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.68;
    margin-top: 18px;
    margin-bottom: 7px;
}

.detail-item {
    padding: 9px 0;
    border-bottom: 1px solid rgba(128,128,128,0.14);
}

.detail-label {
    font-size: 10px;
    opacity: 0.65;
    margin-bottom: 3px;
}

.detail-value {
    font-size: 12px;
    font-weight: 600;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.status-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 750;
    margin-top: 10px;
    border: 1px solid rgba(128,128,128,0.15);
    background: rgba(128,128,128,0.09);
}

.status-good {
    color: #15803d;
    background: rgba(34,197,94,0.10);
    border-color: rgba(34,197,94,0.22);
}

.status-pending {
    color: #b45309;
    background: rgba(245,158,11,0.11);
    border-color: rgba(245,158,11,0.23);
}

.status-alert {
    color: #dc2626;
    background: rgba(239,68,68,0.10);
    border-color: rgba(239,68,68,0.22);
}

.result-caption {
    font-size: 12px;
    opacity: 0.7;
    margin-top: 6px;
    margin-bottom: 12px;
}

.page-indicator {
    text-align: center;
    font-size: 12px;
    font-weight: 700;
    padding-top: 10px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 38px;
    transition: all 0.2s ease;
}

div[data-testid="stDivider"] {
    margin-top: 1.2rem;
    margin-bottom: 1.2rem;
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
        padding: 12px;
        min-height: 88px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }

    .card-info-grid {
        grid-template-columns: 1fr 1fr;
        gap: 7px;
    }
}
</style>
""")


# =========================================================
# HELPER FUNCTIONS
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


# =========================================================
# LOAD EXCEL
# =========================================================

@st.cache_data(show_spinner="Loading intimation records...")
def load_data(file_path, modified_time):
    # The modified time changes whenever the Excel file changes.
    del modified_time

    raw = pd.read_excel(file_path)
    raw.columns = [str(col).strip() for col in raw.columns]

    # Preferred priority source, with fallback.
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
# HEADER
# =========================================================

st.html("""
<div class="dash-header">
    <div class="dash-logo">ID</div>
    <div>
        <div class="dash-title">Intimation Dashboard</div>
        <div class="dash-subtitle">
            Manage and review PSS records, capacity and assignments
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
# OVERVIEW
# Only two metrics. No charts and no unique PSS card.
# =========================================================

st.html("""
<div class="section-heading">Overview</div>
<div class="section-caption">
    Summary of records matching your selected filters
</div>
""")

overview_records = st.empty()
overview_capacity = st.empty()


# =========================================================
# FILTERS
# =========================================================

st.divider()

st.html("""
<div class="section-heading">Filters</div>
<div class="section-caption">
    Filters are connected and update available options automatically
</div>
""")

poc_col, pss_col, state_col = st.columns(
    [1, 1.5, 1.2],
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


# ---------------------------------------------------------
# POC1 options depend on search and selected states
# ---------------------------------------------------------

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
        "POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]


# ---------------------------------------------------------
# State options depend on search and selected POC1
# ---------------------------------------------------------

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
# UPDATE OVERVIEW METRICS
# =========================================================

total_capacity = filtered_df["Capacity (MW)"].sum(min_count=1)

if pd.isna(total_capacity):
    total_capacity = 0.0

with overview_records.container():
    st.metric(
        label="Matching Records",
        value=f"{len(filtered_df):,}",
        help="Number of records matching the current filters.",
    )

with overview_capacity.container():
    st.metric(
        label="Total Capacity",
        value=f"{total_capacity:,.2f} MW",
        help="Sum of capacity for the currently filtered records.",
    )


# =========================================================
# RECORDS SECTION
# =========================================================

st.divider()

st.html("""
<div class="section-heading">Records</div>
<div class="section-caption">
    Select a card to view the remaining record details
</div>
""")

if filtered_df.empty:
    st.info(
        "No records match the selected filters. "
        "Try another PSS search, POC1 or State."
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
    of <b>{total_records:,}</b> records
</div>
""")

page_df = filtered_df.iloc[start:end]


# =========================================================
# CARD RENDERING
# Collapsed: PSS name, capacity, state and POC1 only.
# Expanded: remaining details within the same card.
# =========================================================

left_cards, right_cards = st.columns(2, gap="medium")

for index, (_, record) in enumerate(page_df.iterrows()):
    target = left_cards if index % 2 == 0 else right_cards

    pss = clean(record["PSS"]) or "Unnamed PSS"
    state = clean(record["State"]) or "Not specified"
    poc1 = clean(record["POC1"]) or "Not specified"
    capacity = format_capacity(record["Capacity (MW)"])

    status = clean(record["Check Status"]) or "Not specified"
    priority = clean(record["Priority"]) or "Not specified"

    status_class = status_style(status)

    with target:
        # Summary remains compact when the card is collapsed.
        # PSS and capacity appear in the expander heading.
        with st.expander(
            f"{pss}  ·  {capacity}",
            expanded=False,
        ):
            # These four fields are the main card information.
            st.html(f"""
            <div class="card-heading">
                {html_text(pss)}
            </div>

            <div class="card-info-grid">
                <div class="card-info">
                    <div class="card-label">Capacity</div>
                    <div class="card-value card-capacity">
                        {html_text(capacity)}
                    </div>
                </div>

                <div class="card-info">
                    <div class="card-label">State</div>
                    <div class="card-value">
                        📍 {html_text(state)}
                    </div>
                </div>

                <div class="card-info" style="grid-column: 1 / -1;">
                    <div class="card-label">POC1</div>
                    <div class="card-value">
                        👤 {html_text(poc1)}
                    </div>
                </div>
            </div>
            """)

            # Remaining information appears only after expansion.
            st.html(f"""
            <div class="card-details-heading">
                Additional Details
            </div>
            """)

            details = [
                ("Priority", record["Priority"]),
                ("POC 2", record["POC 2"]),
                ("POC 3", record["POC 3"]),
                ("Latitude", record["Lat"]),
                ("Longitude", record["Long"]),
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

            st.html(f"""
            <div class="detail-item">
                <div class="detail-label">Check Status</div>
                <div class="detail-value">
                    {html_text(status)}
                </div>
                <span class="status-pill {status_class}">
                    {html_text(status)}
                </span>
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
