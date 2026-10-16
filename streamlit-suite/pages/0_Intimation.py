
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
# CSS — CLEAN, SPACIOUS & COMPACT
# ============================================================

st.html("""
<style>
    .stApp {
        background: #f6f8fc;
    }

    .block-container {
        max-width: 1460px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }

    /* Header */
    .dashboard-header {
        padding: 20px 23px;
        margin-bottom: 19px;
        border-radius: 15px;
        color: white;
        background: linear-gradient(115deg, #172554, #1d4ed8 68%, #5288f5);
        box-shadow: 0 5px 16px rgba(37,99,235,.13);
    }

    .header-label {
        display: inline-block;
        padding: 4px 9px;
        margin-bottom: 8px;
        border: 1px solid #ffffff38;
        border-radius: 20px;
        background: #ffffff16;
        color: #dbeafe;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1px;
    }

    .dashboard-header h1 {
        margin: 0;
        color: white;
        font-size: clamp(24px, 2.3vw, 31px);
        font-weight: 800;
        letter-spacing: -.5px;
    }

    .dashboard-header p {
        margin: 6px 0 0;
        color: #dbeafe;
        font-size: 12px;
    }

    /* Section headings */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 17px 0 9px;
        color: #1e293b;
        font-size: 15px;
        font-weight: 800;
    }

    .section-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 29px;
        height: 29px;
        border-radius: 9px;
        background: #e7efff;
        color: #2459c4;
        font-size: 15px;
    }

    .helper-text {
        margin: 1px 0 10px;
        color: #718096;
        font-size: 11px;
    }

    /* Overview */
    .overview-card {
        min-height: 91px;
        padding: 15px 17px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 3px 10px #0f172a0d;
    }

    .overview-records {
        background: linear-gradient(120deg, #1d4ed8, #4383f5);
    }

    .overview-capacity {
        background: linear-gradient(120deg, #047857, #10b981);
    }

    .metric-label {
        color: #ffffffd9;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .8px;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 5px;
        color: white;
        font-size: clamp(22px, 2.3vw, 29px);
        font-weight: 800;
        line-height: 1.3;
    }

    /* Suggestions */
    .suggestion-caption {
        margin: 5px 0 7px;
        color: #64748b;
        font-size: 11px;
        font-weight: 650;
    }

    .suggestion-chip {
        padding: 7px 9px;
        border: 1px solid #e0e9f7;
        border-radius: 8px;
        background: #ffffff;
        color: #334155;
        font-size: 11px;
    }

    /* PSS cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e3e9f2 !important;
        border-radius: 13px !important;
        background: #ffffff !important;
        box-shadow: 0 2px 7px rgba(15,23,42,.025);
        transition: border-color .18s ease, box-shadow .18s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #b9cdf2 !important;
        box-shadow: 0 5px 14px rgba(37,99,235,.065);
    }

    .pss-head {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin: 2px 0 13px;
    }

    .pss-icon {
        flex: 0 0 36px;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #dbeafe;
        border-radius: 10px;
        background: linear-gradient(135deg, #dbeafe, #f5f8ff);
        font-size: 17px;
    }

    .pss-name {
        margin-top: 1px;
        color: #172554;
        font-size: 14px;
        font-weight: 800;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .pss-subtitle {
        margin-top: 3px;
        color: #94a3b8;
        font-size: 9px;
        font-weight: 750;
        letter-spacing: .8px;
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

    .pss-info.capacity {
        border-color: #dce8ff;
        background: #f0f5ff;
    }

    .pss-info.state {
        border-color: #d8f3e5;
        background: #f0fbf5;
    }

    .pss-info-label {
        margin-bottom: 4px;
        color: #8492a6;
        font-size: 9px;
        font-weight: 750;
        letter-spacing: .6px;
        text-transform: uppercase;
    }

    .pss-info-value {
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .pss-info.capacity .pss-info-value {
        color: #2459c4;
    }

    .pss-info.state .pss-info-value {
        color: #047857;
    }

    .poc-info {
        padding: 9px 10px;
        border: 1px solid #edf1f7;
        border-radius: 9px;
        background: #ffffff;
    }

    /* Details */
    div[data-testid="stExpander"] {
        margin-top: 10px;
        border: 1px solid #e8edf5 !important;
        border-radius: 9px !important;
        background: #fbfcff !important;
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
        min-height: 53px;
        margin-bottom: 7px;
        padding: 9px 10px;
        border: 1px solid #e8edf5;
        border-radius: 8px;
        background: #f4f7fb;
    }

    .detail-label {
        margin-bottom: 4px;
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .6px;
        text-transform: uppercase;
    }

    .detail-value {
        color: #1e293b;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .status-pill {
        display: inline-block;
        max-width: 100%;
        padding: 4px 8px;
        border-radius: 6px;
        background: #e0ecff;
        color: #1d4ed8;
        font-size: 10px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    /* Inputs and buttons */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border-radius: 9px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
        transition: all .15s ease;
    }

    .stButton > button[kind="secondary"] {
        border-color: #dce5f1;
        background: white;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #9db9ee;
        color: #1d4ed8;
    }

    .page-label {
        padding-top: 8px;
        color: #64748b;
        text-align: center;
        font-size: 12px;
        font-weight: 700;
    }

    .empty-state {
        padding: 26px 15px;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        background: white;
        color: #64748b;
        text-align: center;
    }

    .empty-state strong {
        display: block;
        margin-bottom: 5px;
        color: #334155;
        font-size: 15px;
    }

    /* Keep vertical rhythm consistent */
    div[data-testid="stVerticalBlock"] {
        gap: .65rem;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: .7rem;
            padding-right: .7rem;
        }

        .dashboard-header {
            padding: 16px;
        }

        .pss-name {
            font-size: 12px;
        }
    }
</style>
""")


# ============================================================
# HELPERS
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
    if pd.isna(value):
        return default

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "nat"}:
        return default

    return value


def safe_html(value):
    return html.escape(clean_text(value))


def numeric_value(value):
    if pd.isna(value):
        return 0.0

    try:
        text = str(value).replace(",", "").strip()
        text = re.sub(r"(?i)\s*mw\s*$", "", text)
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group()) if match else 0.0
    except (TypeError, ValueError):
        return 0.0


def display_capacity(value):
    number = numeric_value(value)

    if number.is_integer():
        return f"{int(number):,} MW"

    return f"{number:,.2f}".rstrip("0").rstrip(".") + " MW"


def render_copy_button(pss_name, card_key):
    # A separate iframe prevents copy clicks from opening the expander.
    pss_json = json.dumps(str(pss_name))

    components.html(
        f"""
        <div style="display:flex;justify-content:flex-end;
                    align-items:center;font-family:Arial,sans-serif;">
            <button id="copy_{card_key}" title="Copy PSS name"
                style="height:31px;min-width:38px;cursor:pointer;
                border:1px solid #dbeafe;border-radius:8px;
                background:#f0f6ff;color:#1d4ed8;font-size:16px;">
                ⚡
            </button>
            <span id="msg_{card_key}" style="display:none;
                margin-left:5px;color:#047857;font-size:10px;">
                Copied!
            </span>
        </div>
        <script>
        const btn = document.getElementById("copy_{card_key}");
        const msg = document.getElementById("msg_{card_key}");
        const value = {pss_json};

        btn.addEventListener("click", async () => {{
            try {{
                if (navigator.clipboard && window.isSecureContext) {{
                    await navigator.clipboard.writeText(value);
                }} else {{
                    const area = document.createElement("textarea");
                    area.value = value;
                    area.style.position = "fixed";
                    area.style.left = "-9999px";
                    document.body.appendChild(area);
                    area.select();
                    const ok = document.execCommand("copy");
                    area.remove();
                    if (!ok) throw new Error("Copy failed");
                }}

                btn.textContent = "✓";
                msg.style.display = "inline";
                setTimeout(() => {{
                    btn.textContent = "⚡";
                    msg.style.display = "none";
                }}, 1200);
            }} catch (e) {{
                msg.textContent = "Copy unavailable";
                msg.style.display = "inline";
                setTimeout(() => {{
                    msg.style.display = "none";
                    msg.textContent = "Copied!";
                }}, 1400);
            }}
        }});
        </script>
        """,
        height=36,
        scrolling=False,
    )


def detail_item(label, value, status=False):
    value_html = safe_html(value)

    if status:
        value_html = f'<span class="status-pill">{value_html}</span>'

    st.markdown(
        f"""
        <div class="detail-tile">
            <div class="detail-label">{safe_html(label)}</div>
            <div class="detail-value">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pss_card(row, columns, card_key):
    pss_name = clean_text(row.get(columns["pss"]))
    state = clean_text(row.get(columns["state"])) if columns["state"] else "—"
    capacity = display_capacity(row.get(columns["capacity"])) if columns["capacity"] else "—"
    poc1 = clean_text(row.get(columns["poc1"])) if columns["poc1"] else "—"

    with st.container(border=True):
        title_col, copy_col = st.columns(
            [5.5, 0.8],
            vertical_alignment="center",
            gap="small",
        )

        with title_col:
            st.markdown(
                f"""
                <div class="pss-head">
                    <div class="pss-icon">⚡</div>
                    <div>
                        <div class="pss-name">{safe_html(pss_name)}</div>
                        <div class="pss-subtitle">POWER SUBSTATION</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with copy_col:
            render_copy_button(pss_name, card_key)

        st.markdown(
            f"""
            <div class="pss-info-grid">
                <div class="pss-info capacity">
                    <div class="pss-info-label">Capacity</div>
                    <div class="pss-info-value">{safe_html(capacity)}</div>
                </div>
                <div class="pss-info state">
                    <div class="pss-info-label">State</div>
                    <div class="pss-info-value">{safe_html(state)}</div>
                </div>
            </div>
            <div class="poc-info">
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
                    row.get(columns["priority"]) if columns["priority"] else "—",
                )
                detail_item(
                    "POC 2",
                    row.get(columns["poc2"]) if columns["poc2"] else "—",
                )
                detail_item(
                    "Latitude",
                    row.get(columns["lat"]) if columns["lat"] else "—",
                )

            with right:
                detail_item(
                    "POC 3",
                    row.get(columns["poc3"]) if columns["poc3"] else "—",
                )
                detail_item(
                    "Longitude",
                    row.get(columns["long"]) if columns["long"] else "—",
                )
                detail_item(
                    "Check Status",
                    row.get(columns["status"]) if columns["status"] else "—",
                    status=True,
                )


# ============================================================
# SEARCH / SUGGESTION CALLBACKS
# ============================================================

def on_search_change():
    # Typing a new query clears any previously selected suggestion.
    st.session_state["intimation_selected_pss"] = ""


def choose_suggestion(name):
    st.session_state["intimation_selected_pss"] = name


def clear_suggestion():
    st.session_state["intimation_selected_pss"] = ""


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
    st.warning("The Excel file contains no records.")
    st.stop()


# ============================================================
# COLUMN MAPPING
# ============================================================

all_columns = list(df.columns)

columns = {
    "pss": find_column(
        all_columns,
        [
            "PSS",
            "PSS Name",
            "Name of PSS",
            "Substation",
            "Substation Name",
        ],
    ),
    "state": find_column(
        all_columns,
        ["State", "PSS State", "State Name"],
    ),
    "capacity": find_column(
        all_columns,
        [
            "Capacity (MW)",
            "Capacity MW",
            "Capacity",
            "PSS Capacity (MW)",
        ],
    ),
    "poc1": find_column(
        all_columns,
        ["POC1", "POC 1", "POC-1"],
    ),
    "poc2": find_column(
        all_columns,
        ["POC 2", "POC2", "POC-2"],
    ),
    "poc3": find_column(
        all_columns,
        ["POC 3", "POC3", "POC-3"],
    ),
    "lat": find_column(
        all_columns,
        ["Lat", "Latitude"],
    ),
    "long": find_column(
        all_columns,
        ["Long", "Longitude", "Lon"],
    ),
    "status": find_column(
        all_columns,
        ["Check Status", "Status", "Intimation Status"],
    ),
}

columns["priority"] = find_column(
    all_columns,
    [
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
    st.error("PSS column was not found in the Excel file.")
    st.write("Available columns:", all_columns)
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-label">SOLAR SUITE · OPERATIONS</div>
        <h1>⚡ Intimation Dashboard</h1>
        <p>Power Substation Overview · Search, filter and inspect site details</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SEARCH + LIVE RECOMMENDATIONS
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        <span class="section-icon">⌕</span>
        Search & Filters
    </div>
    <div class="helper-text">
        Type a PSS name to see matching recommendations, or use the filters.
    </div>
    """,
    unsafe_allow_html=True,
)

search_col, poc_col, state_col = st.columns(
    [2.2, 1.25, 1.25],
    gap="medium",
)

if "intimation_selected_pss" not in st.session_state:
    st.session_state["intimation_selected_pss"] = ""

with search_col:
    pss_search = st.text_input(
        "Search PSS",
        placeholder="Start typing a PSS name...",
        key="intimation_pss_search",
        on_change=on_search_change,
    )

query = pss_search.strip()

# Recommendations update with every typed character.
# Only show suggestions when the query contains at least one character.
if query:
    pss_names = (
        df[columns["pss"]]
        .dropna()
        .astype(str)
        .str.strip()
    )

    unique_names = sorted(
        {name for name in pss_names if name},
        key=str.casefold,
    )

    recommendations = [
        name for name in unique_names
        if query.casefold() in name.casefold()
    ]

    selected_pss = st.session_state.get("intimation_selected_pss", "")

    if selected_pss:
        st.caption(f"Selected PSS: **{selected_pss}**")

        if st.button(
            "✕ Clear selected PSS",
            key="intimation_clear_suggestion",
            type="secondary",
        ):
            clear_suggestion()
            st.rerun()

    elif recommendations:
        st.markdown(
            '<div class="suggestion-caption">Matching PSS recommendations</div>',
            unsafe_allow_html=True,
        )

        # Limit visible suggestions to keep the page uncluttered.
        visible_recommendations = recommendations[:8]

        suggestion_cols = st.columns(4, gap="small")

        for index, name in enumerate(visible_recommendations):
            with suggestion_cols[index % 4]:
                st.button(
                    f"⌕ {name}",
                    key=f"pss_suggestion_{index}_{hash(name)}",
                    help=f"Select {name}",
                    use_container_width=True,
                    on_click=choose_suggestion,
                    args=(name,),
                    type="secondary",
                )

        if len(recommendations) > len(visible_recommendations):
            st.caption(
                f"Showing {len(visible_recommendations)} of "
                f"{len(recommendations)} matches. Keep typing to narrow results."
            )

    else:
        st.caption("No matching PSS names found. Try another search.")


# ============================================================
# POC1 FILTER — DEPENDS ON SEARCH / SELECTED PSS
# ============================================================

selected_pss = st.session_state.get("intimation_selected_pss", "")

search_df = df.copy()

if selected_pss:
    search_df = search_df[
        search_df[columns["pss"]]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        == selected_pss.casefold()
    ]
elif query:
    search_df = search_df[
        search_df[columns["pss"]]
        .fillna("")
        .astype(str)
        .str.contains(query, case=False, na=False, regex=False)
    ]

with poc_col:
    if columns["poc1"]:
        poc1_options = sorted(
            {
                clean_text(value)
                for value in search_df[columns["poc1"]].tolist()
                if clean_text(value) != "—"
            },
            key=str.casefold,
        )

        poc1_choice = st.selectbox(
            "POC 1",
            ["All"] + poc1_options,
            key="intimation_poc1_filter",
        )
    else:
        poc1_choice = "All"
        st.selectbox(
            "POC 1",
            ["All"],
            key="intimation_poc1_unavailable",
            disabled=True,
        )


# ============================================================
# STATE FILTER — DEPENDS ON SEARCH + POC1
# ============================================================

state_source = search_df.copy()

if columns["poc1"] and poc1_choice != "All":
    state_source = state_source[
        state_source[columns["poc1"]]
        .fillna("")
        .astype(str)
        .str.strip()
        == poc1_choice
    ]

with state_col:
    if columns["state"]:
        state_options = sorted(
            {
                clean_text(value)
                for value in state_source[columns["state"]].tolist()
                if clean_text(value) != "—"
            },
            key=str.casefold,
        )

        state_choice = st.selectbox(
            "State",
            ["All"] + state_options,
            key="intimation_state_filter",
        )
    else:
        state_choice = "All"
        st.selectbox(
            "State",
            ["All"],
            key="intimation_state_unavailable",
            disabled=True,
        )


# ============================================================
# APPLY FILTERS
# ============================================================

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

total_capacity = (
    filtered_df[columns["capacity"]].apply(numeric_value).sum()
    if columns["capacity"]
    else 0.0
)

capacity_display = display_capacity(total_capacity)

metric1, metric2 = st.columns(2, gap="medium")

with metric1:
    st.markdown(
        f"""
        <div class="overview-card overview-records">
            <div class="metric-label">Matching Records</div>
            <div class="metric-value">{record_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        f"""
        <div class="overview-card overview-capacity">
            <div class="metric-label">Total Capacity</div>
            <div class="metric-value">{safe_html(capacity_display)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGINATION
# ============================================================

PAGE_SIZE = 20

filter_signature = (
    query.casefold(),
    selected_pss.casefold(),
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
        <div class="helper-text">
            Showing <b>{start_index + 1}–{end_index}</b>
            of <b>{record_count:,}</b> matching sites
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
                    row,
                    columns,
                    f"pss_{current_page}_{original_index}",
                )

        st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)

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
        margin-top:22px;
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
