
from pathlib import Path
import html
import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intimation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.html("""
<style>
    .stApp {
        background:
            radial-gradient(ellipse at top left,
                rgba(59,130,246,.045), transparent 36%),
            #f6f8fc;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.15rem;
        padding-bottom: 1.5rem;
    }

    /* Dashboard header */
    .dashboard-header {
        padding: 19px 22px;
        margin-bottom: 20px;
        border-radius: 15px;
        background: linear-gradient(115deg, #172554, #1d4ed8 65%, #3b82f6);
        box-shadow: 0 6px 18px rgba(37,99,235,.15);
        color: white;
    }

    .header-chip {
        display: inline-block;
        padding: 4px 9px;
        margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 20px;
        background: rgba(255,255,255,.12);
        color: #dbeafe;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1px;
    }

    .dashboard-header h1 {
        color: white;
        margin: 0;
        font-size: clamp(23px, 2.3vw, 30px);
        font-weight: 800;
        line-height: 1.25;
        letter-spacing: -.5px;
    }

    .dashboard-header p {
        color: #dbeafe;
        font-size: 12px;
        margin: 6px 0 0;
    }

    /* Section headings */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 20px 0 10px;
        color: #1e293b;
        font-size: 15px;
        font-weight: 800;
    }

    .section-icon {
        display: inline-flex;
        width: 28px;
        height: 28px;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 14px;
    }

    /* Overview cards */
    .overview-card {
        position: relative;
        overflow: hidden;
        min-height: 92px;
        padding: 15px 17px;
        border-radius: 13px;
        color: white;
        box-shadow: 0 4px 12px rgba(15,23,42,.07);
    }

    .overview-card.records {
        background: linear-gradient(120deg, #1d4ed8, #3b82f6);
    }

    .overview-card.capacity {
        background: linear-gradient(120deg, #047857, #10b981);
    }

    .metric-label {
        color: rgba(255,255,255,.86);
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 6px;
        color: white;
        font-size: clamp(23px, 2.2vw, 29px);
        font-weight: 800;
        line-height: 1.3;
    }

    .metric-icon {
        position: absolute;
        top: 14px;
        right: 16px;
        font-size: 23px;
        opacity: .7;
    }

    /* Search recommendations */
    .suggestion-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .7px;
        margin: 0 0 5px 2px;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 9px;
        min-height: 42px;
        border-color: #d9e2ef;
        background: white;
    }

    /* Compact suggestion buttons */
    .suggestion-wrap div[data-testid="stButton"] button {
        min-height: 34px;
        height: auto;
        padding: 6px 9px;
        border: 1px solid #dbeafe;
        border-radius: 8px;
        background: #ffffff;
        color: #334155;
        font-size: 11px;
        font-weight: 650;
        text-align: left;
        line-height: 1.3;
        overflow-wrap: anywhere;
        box-shadow: 0 1px 3px rgba(15,23,42,.025);
    }

    .suggestion-wrap div[data-testid="stButton"] button:hover {
        background: #eff6ff;
        border-color: #93b4f5;
        color: #1d4ed8;
    }

    /* PSS card border */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e1e8f2 !important;
        border-radius: 13px !important;
        background: #ffffff !important;
        box-shadow: 0 3px 9px rgba(15,23,42,.035);
        transition: border-color .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #b8cdf1 !important;
        box-shadow: 0 5px 13px rgba(37,99,235,.06);
    }

    .pss-title-row {
        display: flex;
        align-items: center;
        gap: 9px;
        min-width: 0;
        margin-bottom: 10px;
    }

    .pss-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #dbeafe, #eff6ff);
        border: 1px solid #dbeafe;
        font-size: 16px;
    }

    .pss-heading {
        color: #172554;
        font-size: 13px;
        font-weight: 800;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .pss-subheading {
        color: #94a3b8;
        font-size: 9px;
        font-weight: 750;
        letter-spacing: .7px;
        margin-top: 2px;
        text-transform: uppercase;
    }

    .pss-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 8px;
    }

    .pss-info {
        min-width: 0;
        padding: 9px 10px;
        border: 1px solid #edf1f7;
        border-radius: 9px;
        background: #f8fafc;
    }

    .pss-info-label {
        margin-bottom: 4px;
        color: #94a3b8;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .6px;
        text-transform: uppercase;
    }

    .pss-info-value {
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .pss-info.capacity-tile {
        background: #eff6ff;
        border-color: #dbeafe;
    }

    .pss-info.capacity-tile .pss-info-value {
        color: #1d4ed8;
    }

    .pss-info.state-tile {
        background: #f0fdf4;
        border-color: #dcfce7;
    }

    .pss-info.state-tile .pss-info-value {
        color: #047857;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1px solid #e5ebf4 !important;
        border-radius: 9px !important;
        background: #fbfdff !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] summary {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        color: #334155 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
    }

    .detail-tile {
        min-height: 52px;
        margin-bottom: 8px;
        padding: 9px 10px;
        border: 1px solid #e7edf5;
        border-radius: 8px;
        background: #f1f5f9;
    }

    .detail-label {
        margin-bottom: 4px;
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .detail-value {
        color: #1e293b;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        background: #dbeafe;
        color: #1d4ed8;
        font-size: 10px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    /* Buttons and dropdowns */
    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
    }

    .stButton > button[kind="secondary"] {
        border-color: #dbe4f0;
        background: white;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 9px;
    }

    .results-caption {
        margin: 3px 0 11px;
        color: #64748b;
        font-size: 11px;
    }

    .page-label {
        padding-top: 7px;
        color: #64748b;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
    }

    .empty-state {
        padding: 25px 15px;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        background: white;
        text-align: center;
        color: #64748b;
    }

    .empty-state strong {
        display: block;
        margin-bottom: 5px;
        color: #334155;
        font-size: 15px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: .7rem;
            padding-right: .7rem;
        }

        .dashboard-header {
            padding: 15px;
        }

        .overview-card {
            padding: 12px;
            min-height: 83px;
        }

        .pss-heading {
            font-size: 12px;
        }
    }
</style>
""")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_column(columns, candidates=None, startswith=None):
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    if candidates:
        for candidate in candidates:
            key = normalize_column(candidate)
            if key in normalized:
                return normalized[key]

    if startswith:
        prefix = normalize_column(startswith)
        for column in columns:
            if normalize_column(column).startswith(prefix):
                return column

    return None


def clean_text(value, default="—"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "nat"}:
        return default

    return value


def safe_html(value):
    return html.escape(clean_text(value))


def numeric_value(value):
    if value is None:
        return 0.0

    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    try:
        value = str(value).replace(",", "").strip()
        value = re.sub(r"(?i)\s*mw\s*$", "", value)
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        return float(match.group()) if match else 0.0
    except (TypeError, ValueError):
        return 0.0


def display_capacity(value):
    number = numeric_value(value)

    if number.is_integer():
        return f"{int(number):,} MW"

    return f"{number:,.2f}".rstrip("0").rstrip(".") + " MW"


def detail_item(label, value, status=False):
    value_html = safe_html(value)

    if status:
        value_html = (
            f'<span class="status-pill">{value_html}</span>'
        )

    st.markdown(
        f"""
        <div class="detail-tile">
            <div class="detail-label">{safe_html(label)}</div>
            <div class="detail-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def copy_button(pss_name, card_key):
    # JSON encoding safely embeds the PSS name in JavaScript.
    encoded_name = json.dumps(str(pss_name), ensure_ascii=False)

    components.html(
        f"""
        <div style="
            display:flex;
            justify-content:flex-end;
            align-items:center;
            font-family:Arial,sans-serif;
        ">
            <button id="copy_{card_key}" title="Copy PSS name"
                style="
                    border:1px solid #dbeafe;
                    background:#eff6ff;
                    color:#1d4ed8;
                    border-radius:8px;
                    height:31px;
                    min-width:38px;
                    cursor:pointer;
                    font-size:16px;
                ">⚡</button>
            <span id="msg_{card_key}" style="
                display:none;
                margin-left:5px;
                font-size:10px;
                color:#047857;
                font-weight:700;
            ">Copied!</span>
        </div>
        <script>
        const btn = document.getElementById("copy_{card_key}");
        const msg = document.getElementById("msg_{card_key}");
        const pssText = {encoded_name};

        btn.addEventListener("click", async () => {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(pssText);
                }} else {{
                    const area = document.createElement("textarea");
                    area.value = pssText;
                    area.style.position = "fixed";
                    area.style.left = "-9999px";
                    document.body.appendChild(area);
                    area.select();
                    const ok = document.execCommand("copy");
                    area.remove();
                    if (!ok) throw new Error("Copy failed");
                }}

                btn.innerHTML = "✓";
                msg.style.display = "inline";

                setTimeout(() => {{
                    btn.innerHTML = "⚡";
                    msg.style.display = "none";
                }}, 1200);
            }} catch (error) {{
                msg.textContent = "Copy unavailable";
                msg.style.display = "inline";

                setTimeout(() => {{
                    msg.style.display = "none";
                    msg.textContent = "Copied!";
                }}, 1500);
            }}
        }});
        </script>
        """,
        height=38,
        scrolling=False,
    )


def render_pss_card(row, columns, card_key):
    pss_name = clean_text(row.get(columns["pss"]))
    state = clean_text(row.get(columns["state"])) if columns["state"] else "—"
    capacity = display_capacity(row.get(columns["capacity"])) if columns["capacity"] else "—"
    poc1 = clean_text(row.get(columns["poc1"])) if columns["poc1"] else "—"

    with st.container(border=True):
        heading_col, copy_col = st.columns(
            [5.5, 0.8],
            vertical_alignment="center",
            gap="small",
        )

        with heading_col:
            st.markdown(
                f"""
                <div class="pss-title-row">
                    <div class="pss-icon">⚡</div>
                    <div>
                        <div class="pss-heading">{safe_html(pss_name)}</div>
                        <div class="pss-subheading">Power Substation</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            copy_button(pss_name, card_key)

        st.markdown(
            f"""
            <div class="pss-info-grid">
                <div class="pss-info capacity-tile">
                    <div class="pss-info-label">Capacity</div>
                    <div class="pss-info-value">{safe_html(capacity)}</div>
                </div>
                <div class="pss-info state-tile">
                    <div class="pss-info-label">State</div>
                    <div class="pss-info-value">{safe_html(state)}</div>
                </div>
            </div>
            <div class="pss-info">
                <div class="pss-info-label">POC 1</div>
                <div class="pss-info-value">{safe_html(poc1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View site details", expanded=False):
            left, right = st.columns(2, gap="small")

            with left:
                detail_item(
                    "Priority",
                    row.get(columns["priority"])
                    if columns["priority"] else "—",
                )
                detail_item(
                    "POC 2",
                    row.get(columns["poc2"])
                    if columns["poc2"] else "—",
                )
                detail_item(
                    "Latitude",
                    row.get(columns["lat"])
                    if columns["lat"] else "—",
                )

            with right:
                detail_item(
                    "POC 3",
                    row.get(columns["poc3"])
                    if columns["poc3"] else "—",
                )
                detail_item(
                    "Longitude",
                    row.get(columns["long"])
                    if columns["long"] else "—",
                )
                detail_item(
                    "Check Status",
                    row.get(columns["status"])
                    if columns["status"] else "—",
                    status=True,
                )


def select_pss_recommendation(pss_name):
    # Callback executes before the next rerun, so widget state
    # can safely be updated here.
    st.session_state["intimation_pss_search"] = pss_name


# ============================================================
# LOAD EXCEL
# ============================================================

@st.cache_data(show_spinner="Loading intimation data...")
def load_data(file_path, modified_time):
    return pd.read_excel(file_path)


file_path = Path(__file__).resolve().parent.parent / "intimation.xlsx"

if not file_path.exists():
    st.error(
        f"Excel file not found: {file_path}. "
        "Ensure intimation.xlsx is in the project root."
    )
    st.stop()

try:
    df = load_data(str(file_path), file_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Unable to read intimation.xlsx: {exc}")
    st.stop()

if df.empty:
    st.warning("The Excel file is empty.")
    st.stop()


# ============================================================
# COLUMN MAPPING
# ============================================================

all_columns = list(df.columns)

columns = {
    "pss": find_column(
        all_columns,
        candidates=[
            "PSS", "PSS Name", "Name of PSS",
            "Substation", "Substation Name",
        ],
    ),
    "state": find_column(
        all_columns,
        candidates=["State", "PSS State", "State Name"],
    ),
    "capacity": find_column(
        all_columns,
        candidates=[
            "Capacity (MW)", "Capacity MW",
            "Capacity", "PSS Capacity (MW)",
        ],
    ),
    "poc1": find_column(
        all_columns,
        candidates=["POC1", "POC 1", "POC-1"],
    ),
    "poc2": find_column(
        all_columns,
        candidates=["POC 2", "POC2", "POC-2"],
    ),
    "poc3": find_column(
        all_columns,
        candidates=["POC 3", "POC3", "POC-3"],
    ),
    "lat": find_column(
        all_columns,
        candidates=["Lat", "Latitude"],
    ),
    "long": find_column(
        all_columns,
        candidates=["Long", "Longitude", "Lon"],
    ),
    "status": find_column(
        all_columns,
        candidates=["Check Status", "Status", "Intimation Status"],
    ),
}

columns["priority"] = find_column(
    all_columns,
    candidates=[
        "Priority as per 003 sheet",
        "Priority as per Amresh Sir 28.07.2023",
    ],
)

if columns["priority"] is None:
    columns["priority"] = find_column(
        all_columns,
        startswith="Priority",
    )

if columns["pss"] is None:
    st.error(
        "PSS column could not be identified. "
        f"Available columns: {all_columns}"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-chip">SOLAR SUITE · OPERATIONS</div>
        <h1>⚡ Intimation Dashboard</h1>
        <p>Power Substation Overview &nbsp;·&nbsp; Search, filter and inspect site details</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PSS SEARCH + LIVE RECOMMENDATIONS
# ============================================================

st.markdown(
    """
    <div class="section-heading" style="margin-top:5px">
        <span class="section-icon">⌕</span>
        Search & Filters
    </div>
    """,
    unsafe_allow_html=True,
)

# Search gets its own row so recommendations don't squeeze
# the POC1 and State dropdowns.
pss_search = st.text_input(
    "Search PSS",
    placeholder="Start typing a PSS name...",
    key="intimation_pss_search",
)

search_text = pss_search.strip()

# Recommendations match the typed letters anywhere in the PSS name.
# Use distinct names and preserve the original spelling.
all_pss_names = (
    df[columns["pss"]]
    .dropna()
    .astype(str)
    .str.strip()
)

all_pss_names = all_pss_names[all_pss_names != ""]
unique_pss_names = list(dict.fromkeys(all_pss_names.tolist()))

if search_text:
    matching_names = [
        name for name in unique_pss_names
        if search_text.casefold() in name.casefold()
    ]

    # Don't show a redundant recommendation when the exact name
    # is already selected/typed.
    exact_match_exists = any(
        name.casefold() == search_text.casefold()
        for name in matching_names
    )

    if matching_names and not exact_match_exists:
        suggestions = matching_names[:6]

        st.markdown(
            '<div class="suggestion-label">Matching PSS names</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="suggestion-wrap">', unsafe_allow_html=True)

        suggestion_cols = st.columns(3, gap="small")

        for index, suggestion in enumerate(suggestions):
            with suggestion_cols[index % 3]:
                st.button(
                    f"⌕  {suggestion}",
                    key=f"pss_suggestion_{index}_{normalize_column(suggestion)}",
                    use_container_width=True,
                    on_click=select_pss_recommendation,
                    args=(suggestion,),
                    help=f"Select {suggestion}",
                )

        st.markdown("</div>", unsafe_allow_html=True)

        if len(matching_names) > len(suggestions):
            st.caption(
                f"{len(matching_names)} matching names found. "
                "Refine your search to narrow the recommendations."
            )

    elif not matching_names:
        st.caption("No matching PSS names found.")

# Filter data by typed or selected PSS name.
search_df = df.copy()

if search_text:
    search_df = search_df[
        search_df[columns["pss"]]
        .fillna("")
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False,
            regex=False,
        )
    ]


# ============================================================
# CONNECTED POC1 + STATE FILTERS
# ============================================================

filter_col1, filter_col2 = st.columns(2, gap="medium")

with filter_col1:
    if columns["poc1"]:
        poc1_options = sorted(
            {
                clean_text(value)
                for value in search_df[columns["poc1"]].tolist()
                if clean_text(value) != "—"
            },
            key=str.casefold,
        )

        poc1_options = ["All"] + poc1_options

        # If the search changes and the previous selection is no
        # longer available, reset the dependent filter.
        if (
            st.session_state.get("intimation_poc1_filter")
            not in poc1_options
        ):
            st.session_state["intimation_poc1_filter"] = "All"

        poc1_choice = st.selectbox(
            "POC 1",
            options=poc1_options,
            key="intimation_poc1_filter",
        )
    else:
        poc1_choice = "All"
        st.selectbox(
            "POC 1",
            options=["All"],
            disabled=True,
            key="intimation_poc1_unavailable",
        )

state_source = search_df.copy()

if columns["poc1"] and poc1_choice != "All":
    state_source = state_source[
        state_source[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == poc1_choice
    ]

with filter_col2:
    if columns["state"]:
        state_options = sorted(
            {
                clean_text(value)
                for value in state_source[columns["state"]].tolist()
                if clean_text(value) != "—"
            },
            key=str.casefold,
        )

        state_options = ["All"] + state_options

        if (
            st.session_state.get("intimation_state_filter")
            not in state_options
        ):
            st.session_state["intimation_state_filter"] = "All"

        state_choice = st.selectbox(
            "State",
            options=state_options,
            key="intimation_state_filter",
        )
    else:
        state_choice = "All"
        st.selectbox(
            "State",
            options=["All"],
            disabled=True,
            key="intimation_state_unavailable",
        )


# Apply filters
filtered_df = search_df.copy()

if columns["poc1"] and poc1_choice != "All":
    filtered_df = filtered_df[
        filtered_df[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == poc1_choice
    ]

if columns["state"] and state_choice != "All":
    filtered_df = filtered_df[
        filtered_df[columns["state"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == state_choice
    ]

filtered_df = filtered_df.reset_index(drop=True)


# ============================================================
# OVERVIEW
# ============================================================

record_count = len(filtered_df)

if columns["capacity"]:
    total_capacity = filtered_df[columns["capacity"]].apply(
        numeric_value
    ).sum()
else:
    total_capacity = 0.0

overview1, overview2 = st.columns(2, gap="medium")

with overview1:
    st.markdown(
        f"""
        <div class="overview-card records">
            <div class="metric-icon">▦</div>
            <div class="metric-label">Matching Records</div>
            <div class="metric-value">{record_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with overview2:
    st.markdown(
        f"""
        <div class="overview-card capacity">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">Total Capacity</div>
            <div class="metric-value">{safe_html(display_capacity(total_capacity))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGINATION
# ============================================================

PAGE_SIZE = 20

filter_signature = (
    search_text.casefold(),
    poc1_choice,
    state_choice,
)

if st.session_state.get("intimation_filter_signature") != filter_signature:
    st.session_state["intimation_filter_signature"] = filter_signature
    st.session_state["intimation_page"] = 1

total_pages = max(1, (record_count + PAGE_SIZE - 1) // PAGE_SIZE)

current_page = st.session_state.get("intimation_page", 1)
current_page = min(max(1, current_page), total_pages)
st.session_state["intimation_page"] = current_page

start_index = (current_page - 1) * PAGE_SIZE
end_index = min(start_index + PAGE_SIZE, record_count)
page_df = filtered_df.iloc[start_index:end_index]


# ============================================================
# PSS CARDS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">▤</span>
        PSS Sites
    </div>
    """,
    unsafe_allow_html=True,
)

if record_count:
    st.markdown(
        f"""
        <div class="results-caption">
            Showing <b>{start_index + 1}–{end_index}</b>
            of <b>{record_count:,}</b> matching sites
            &nbsp;·&nbsp; 20 records per page
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = list(page_df.iterrows())

    for pair_start in range(0, len(records), 2):
        card_cols = st.columns(2, gap="medium")

        for offset in range(2):
            record_index = pair_start + offset

            if record_index >= len(records):
                continue

            original_index, row = records[record_index]

            with card_cols[offset]:
                render_pss_card(
                    row=row,
                    columns=columns,
                    card_key=f"pss_{current_page}_{original_index}",
                )

        st.markdown(
            '<div style="height:5px"></div>',
            unsafe_allow_html=True,
        )

    if total_pages > 1:
        st.markdown("---")

        prev_col, page_col, next_col = st.columns(
            [1, 2, 1],
            vertical_alignment="center",
        )

        with prev_col:
            if st.button(
                "← Previous",
                use_container_width=True,
                disabled=current_page <= 1,
                key="intimation_prev_page",
            ):
                st.session_state["intimation_page"] = current_page - 1
                st.rerun()

        with page_col:
            st.markdown(
                f'<div class="page-label">Page {current_page} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

        with next_col:
            if st.button(
                "Next →",
                use_container_width=True,
                disabled=current_page >= total_pages,
                key="intimation_next_page",
            ):
                st.session_state["intimation_page"] = current_page + 1
                st.rerun()

else:
    st.markdown(
        """
        <div class="empty-state">
            <strong>🔎 No matching PSS sites</strong>
            Try another PSS name or change the POC 1 / State filters.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:24px;
        padding-top:12px;
        border-top:1px solid #e2e8f0;
        text-align:center;
        color:#94a3b8;
        font-size:10px;
    ">
        SOLAR SUITE · INTIMATION DASHBOARD
    </div>
    """,
    unsafe_allow_html=True,
)
