
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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXCEL_PATH = Path(__file__).resolve().parent.parent / "intimation.xlsx"
PAGE_SIZE = 20
ALL_USERS = "All Users"


# =========================================================
# SVG ICONS
# =========================================================

BOLT_SVG = """
<svg viewBox="0 0 48 48" width="30" height="30"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs>
        <linearGradient id="boltGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#FFF176"/>
            <stop offset="48%" stop-color="#FFC107"/>
            <stop offset="100%" stop-color="#FF8F00"/>
        </linearGradient>
    </defs>
    <path d="M27 2 L9 27 H21 L17 46 L39 18 H26 Z"
          fill="url(#boltGradient)"
          stroke="#FFF8D6"
          stroke-width="1.2"
          stroke-linejoin="round"/>
</svg>
"""


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>
.block-container {
    max-width: 1500px;
    padding: 1.5rem 2rem 2.5rem;
}

.dash-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 28px;
}

.brand-mark {
    width: 58px;
    height: 58px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background:
        radial-gradient(circle at 25% 15%, #93C5FD 0%, transparent 42%),
        linear-gradient(140deg, #1D4ED8, #4F46E5 60%, #7C3AED);
    box-shadow:
        0 0 0 5px rgba(79,70,229,0.07),
        0 9px 24px rgba(79,70,229,0.24);
    position: relative;
    overflow: hidden;
}

.brand-mark::after {
    content: "";
    position: absolute;
    width: 65px;
    height: 65px;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50%;
    right: -36px;
    bottom: -37px;
}

.brand-mark svg {
    width: 35px;
    height: 35px;
    filter: drop-shadow(0 0 7px rgba(255,220,80,0.65));
    z-index: 1;
}

.dash-title {
    font-size: 29px;
    font-weight: 850;
    letter-spacing: -1px;
    line-height: 1.2;
}

.dash-subtitle {
    font-size: 12px;
    opacity: 0.67;
    margin-top: 6px;
}

.section-heading {
    font-size: 19px;
    font-weight: 820;
    letter-spacing: -0.45px;
    margin-bottom: 4px;
}

.section-caption {
    font-size: 12px;
    opacity: 0.65;
    margin-bottom: 15px;
}

/* ---------------- OVERVIEW CARDS ---------------- */

.overview-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
    margin: 10px 0 24px;
}

.overview-card {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    min-height: 153px;
    padding: 22px 24px;
    border: 1px solid rgba(128,128,128,0.17);
    border-radius: 19px;
    background: var(--secondary-background-color);
    box-shadow: 0 6px 22px rgba(0,0,0,0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.overview-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 27px rgba(0,0,0,0.07);
}

.overview-card::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -2;
    opacity: 0.8;
}

.overview-card.records::before {
    background: linear-gradient(
        120deg,
        rgba(37,99,235,0.11),
        transparent 75%
    );
}

.overview-card.capacity::before {
    background: linear-gradient(
        120deg,
        rgba(124,58,237,0.12),
        transparent 75%
    );
}

.overview-orb {
    position: absolute;
    width: 145px;
    height: 145px;
    border-radius: 50%;
    right: -33px;
    top: -47px;
    z-index: -1;
}

.records .overview-orb {
    background: radial-gradient(
        circle,
        rgba(59,130,246,0.20),
        rgba(59,130,246,0)
    );
}

.capacity .overview-orb {
    background: radial-gradient(
        circle,
        rgba(139,92,246,0.22),
        rgba(139,92,246,0)
    );
}

.overview-content {
    display: flex;
    align-items: center;
    gap: 15px;
    position: relative;
}

.overview-icon {
    width: 54px;
    height: 54px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 16px;
}

.records .overview-icon {
    background: linear-gradient(145deg, #2563EB, #60A5FA);
    box-shadow: 0 6px 16px rgba(37,99,235,0.22);
}

.capacity .overview-icon {
    background: linear-gradient(145deg, #6D28D9, #A78BFA);
    box-shadow: 0 6px 16px rgba(109,40,217,0.22);
}

.overview-icon svg {
    width: 31px;
    height: 31px;
}

.overview-label {
    font-size: 12px;
    font-weight: 700;
    opacity: 0.72;
    margin-bottom: 6px;
}

.overview-value {
    font-size: 28px;
    font-weight: 850;
    letter-spacing: -0.8px;
    line-height: 1.2;
    overflow-wrap: anywhere;
}

.overview-footnote {
    font-size: 11px;
    opacity: 0.65;
    margin-top: 15px;
}

/* ---------------- FILTERS ---------------- */

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stMultiSelect"] [data-baseweb="select"] {
    border-radius: 10px;
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    font-size: 12px;
    font-weight: 700;
}

/* ---------------- RECORD CARDS ---------------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(128,128,128,0.21) !important;
    border-radius: 16px !important;
    background: var(--secondary-background-color);
    box-shadow: 0 4px 15px rgba(0,0,0,0.025);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(99,102,241,0.62) !important;
    box-shadow: 0 7px 22px rgba(79,70,229,0.075);
}

.pss-card-head {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 15px;
    min-width: 0;
}

.pss-bolt {
    width: 43px;
    height: 43px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background:
        radial-gradient(circle at 30% 20%, #60A5FA, transparent 70%),
        linear-gradient(145deg, #1D4ED8, #6D28D9);
    box-shadow: 0 4px 13px rgba(79,70,229,0.22);
}

.pss-bolt svg {
    width: 27px;
    height: 27px;
    filter: drop-shadow(0 0 5px rgba(255,221,87,0.65));
}

.pss-name {
    font-size: 14px;
    font-weight: 850;
    line-height: 1.45;
    overflow-wrap: anywhere;
}

.pss-caption {
    font-size: 10px;
    opacity: 0.63;
    margin-top: 4px;
}

.pss-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px;
}

.pss-info {
    min-width: 0;
    padding: 11px;
    border: 1px solid rgba(128,128,128,0.14);
    border-radius: 11px;
    background: rgba(128,128,128,0.035);
}

.pss-info.capacity-info {
    background: rgba(99,102,241,0.055);
    border-color: rgba(99,102,241,0.17);
}

.pss-info.poc-info {
    grid-column: 1 / -1;
}

.pss-label {
    font-size: 10px;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: 0.55px;
    opacity: 0.63;
    margin-bottom: 5px;
}

.pss-value {
    font-size: 12px;
    font-weight: 750;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.pss-capacity {
    font-size: 14px;
    font-weight: 850;
}

.pss-details-heading {
    font-size: 11px;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    opacity: 0.66;
    margin: 16px 0 6px;
}

.pss-detail-row {
    padding: 9px 0;
    border-bottom: 1px solid rgba(128,128,128,0.13);
}

.pss-detail-label {
    font-size: 10px;
    opacity: 0.64;
    margin-bottom: 4px;
}

.pss-detail-value {
    font-size: 12px;
    font-weight: 650;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.status-pill {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 750;
    margin-top: 12px;
    background: rgba(128,128,128,0.09);
    border: 1px solid rgba(128,128,128,0.15);
}

.status-good {
    color: #15803d;
    background: rgba(34,197,94,0.10);
    border-color: rgba(34,197,94,0.23);
}

.status-pending {
    color: #b45309;
    background: rgba(245,158,11,0.11);
    border-color: rgba(245,158,11,0.24);
}

.status-alert {
    color: #dc2626;
    background: rgba(239,68,68,0.10);
    border-color: rgba(239,68,68,0.22);
}

.result-caption {
    font-size: 12px;
    opacity: 0.68;
    margin: 6px 0 12px;
}

.page-indicator {
    text-align: center;
    font-size: 12px;
    font-weight: 750;
    padding-top: 10px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 36px;
}

@media (max-width: 800px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    .dash-title {
        font-size: 24px;
    }

    .overview-card {
        padding: 17px;
    }

    .overview-value {
        font-size: 24px;
    }
}

@media (max-width: 600px) {
    .overview-grid {
        grid-template-columns: 1fr;
        gap: 12px;
    }

    .overview-card {
        min-height: 125px;
    }

    .dash-title {
        font-size: 22px;
    }

    .brand-mark {
        width: 49px;
        height: 49px;
    }

    .pss-grid {
        gap: 7px;
    }

    .pss-info {
        padding: 9px;
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
    # modified_time changes when the Excel file is updated.
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
        priority_source = find_column(raw, startswith="Priority")

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
        if priority_source is not None
        else ""
    )

    text_columns = [
        "Lat", "Long", "PSS", "State", "Priority",
        "POC1", "POC 2", "POC 3", "Check Status",
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
            "POC1", "POC 2", "POC 3", "Check Status",
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

st.html(f"""
<div class="dash-header">
    <div class="brand-mark">
        {BOLT_SVG}
    </div>
    <div>
        <div class="dash-title">Intimation Dashboard</div>
        <div class="dash-subtitle">
            PSS records, capacity and assignment overview
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

if "expanded_intimation_cards" not in st.session_state:
    st.session_state["expanded_intimation_cards"] = set()


# =========================================================
# FILTERS
# =========================================================

st.html("""
<div class="section-heading">Filters</div>
<div class="section-caption">
    Search PSS records and narrow results by POC1 and State
</div>
""")

filter_poc, filter_pss, filter_state = st.columns(
    [1, 1.5, 1.2],
    gap="medium",
)

with filter_pss:
    st.text_input(
        "Search PSS",
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


# ---------------------------------------------------------
# POC1 OPTIONS DEPEND ON SEARCH + SELECTED STATES
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

with filter_poc:
    st.selectbox(
        "POC1",
        options=poc_options,
        key="intimation_poc1",
        on_change=reset_page,
    )

selected_poc = st.session_state["intimation_poc1"]


# ---------------------------------------------------------
# STATE OPTIONS DEPEND ON SEARCH + SELECTED POC1
# ---------------------------------------------------------

state_option_mask = pss_mask.copy()

if selected_poc != ALL_USERS:
    state_option_mask &= df["POC1"].eq(selected_poc)

available_states = unique_sorted(
    df.loc[state_option_mask, "State"]
)

current_states = st.session_state["intimation_states"]

valid_states = [
    state for state in current_states
    if state in available_states
]

if valid_states != current_states:
    st.session_state["intimation_states"] = valid_states

with filter_state:
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
# OVERVIEW
# Two horizontal decorative cards; no charts.
# =========================================================

total_capacity = filtered_df["Capacity (MW)"].sum(min_count=1)

if pd.isna(total_capacity):
    total_capacity = 0.0

records_svg = """
<svg viewBox="0 0 40 40" width="31" height="31"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <rect x="8" y="6" width="24" height="29" rx="4"
          fill="none" stroke="white" stroke-width="2.4"/>
    <path d="M14 14 H26 M14 20 H26 M14 26 H23"
          stroke="white" stroke-width="2.4" stroke-linecap="round"/>
    <path d="M13 3 H27" stroke="white" stroke-width="2.4"
          stroke-linecap="round"/>
</svg>
"""

capacity_svg = """
<svg viewBox="0 0 40 40" width="33" height="33"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path d="M22 2 L8 23 H18 L15 38 L32 15 H22 Z"
          fill="#FFF176" stroke="white" stroke-width="1.3"
          stroke-linejoin="round"/>
</svg>
"""

st.html(f"""
<div class="section-heading">Overview</div>
<div class="section-caption">
    Live summary of your filtered records
</div>

<div class="overview-grid">
    <div class="overview-card records">
        <div class="overview-orb"></div>
        <div class="overview-content">
            <div class="overview-icon">
                {records_svg}
            </div>
            <div>
                <div class="overview-label">Matching Records</div>
                <div class="overview-value">{len(filtered_df):,}</div>
            </div>
        </div>
        <div class="overview-footnote">
            Records matching your current selection
        </div>
    </div>

    <div class="overview-card capacity">
        <div class="overview-orb"></div>
        <div class="overview-content">
            <div class="overview-icon">
                {capacity_svg}
            </div>
            <div>
                <div class="overview-label">Total Capacity</div>
                <div class="overview-value">{total_capacity:,.2f} MW</div>
            </div>
        </div>
        <div class="overview-footnote">
            Combined capacity of matching records
        </div>
    </div>
</div>
""")


# =========================================================
# RECORDS HEADER
# =========================================================

st.divider()

st.html("""
<div class="section-heading">PSS Records</div>
<div class="section-caption">
    Every card shows the key information. Expand a card for more details.
</div>
""")

if filtered_df.empty:
    st.info(
        "No records match the selected filters. "
        "Try changing the PSS search, POC1 or State."
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
# Collapsed: PSS, Capacity, State and POC1.
# Expanded: additional details inside the same container.
# =========================================================

card_left, card_right = st.columns(2, gap="medium")

for index, (_, record) in enumerate(page_df.iterrows()):
    card_column = card_left if index % 2 == 0 else card_right

    pss = clean(record["PSS"]) or "Unnamed PSS"
    state = clean(record["State"]) or "Not specified"
    poc1 = clean(record["POC1"]) or "Not specified"
    capacity = format_capacity(record["Capacity (MW)"])

    status = clean(record["Check Status"]) or "Not specified"
    priority = clean(record["Priority"]) or "Not specified"

    # Use a stable identifier based on the current result page.
    card_id = f"intimation_card_{start + index}"
    expanded_cards = st.session_state["expanded_intimation_cards"]
    is_expanded = card_id in expanded_cards

    with card_column:
        # Border wraps preview, expanded details and toggle button.
        with st.container(border=True):
            st.html(f"""
            <div class="pss-card-head">
                <div class="pss-bolt">
                    {BOLT_SVG}
                </div>
                <div style="min-width:0;">
                    <div class="pss-name">
                        {safe_html(pss)}
                    </div>
                    <div class="pss-caption">
                        Power Substation
                    </div>
                </div>
            </div>

            <div class="pss-grid">
                <div class="pss-info capacity-info">
                    <div class="pss-label">Capacity</div>
                    <div class="pss-value pss-capacity">
                        {safe_html(capacity)}
                    </div>
                </div>

                <div class="pss-info">
                    <div class="pss-label">State</div>
                    <div class="pss-value">
                        {safe_html(state)}
                    </div>
                </div>

                <div class="pss-info poc-info">
                    <div class="pss-label">POC1</div>
                    <div class="pss-value">
                        {safe_html(poc1)}
                    </div>
                </div>
            </div>
            """)

            if is_expanded:
                st.html(f"""
                <div class="pss-details-heading">
                    Additional Details
                </div>
                """)

                details = [
                    ("Priority", priority),
                    ("POC 2", record["POC 2"]),
                    ("POC 3", record["POC 3"]),
                    ("Latitude", record["Lat"]),
                    ("Longitude", record["Long"]),
                    ("Check Status", status),
                ]

                for label, value in details:
                    st.html(f"""
                    <div class="pss-detail-row">
                        <div class="pss-detail-label">
                            {safe_html(label)}
                        </div>
                        <div class="pss-detail-value">
                            {safe_html(value) or "—"}
                        </div>
                    </div>
                    """)

                badge_class = status_style(status)

                st.html(f"""
                <span class="status-pill {badge_class}">
                    {safe_html(status)}
                </span>
                """)

            def toggle_card(key=card_id):
                current = set(
                    st.session_state["expanded_intimation_cards"]
                )

                if key in current:
                    current.remove(key)
                else:
                    current.add(key)

                st.session_state["expanded_intimation_cards"] = current

            st.button(
                "Hide details ↑" if is_expanded else "View details ↓",
                key=f"toggle_{card_id}",
                use_container_width=True,
                on_click=toggle_card,
            )


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
